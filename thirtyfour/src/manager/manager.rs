use std::collections::HashMap;
use std::fmt::Formatter;
use std::future::{Future, IntoFuture};
use std::net::{IpAddr, Ipv4Addr};
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex as StdMutex, MutexGuard as StdMutexGuard, Weak};
use std::time::Duration;

use serde_json::Value;
use tokio::sync::Mutex;
use url::Url;

use crate::Capabilities;
use crate::common::config::WebDriverConfig;
use crate::error::{WebDriverError, WebDriverResult};
use crate::extensions::query::IntoElementPoller;
use crate::session::DriverGuard;
use crate::session::create::start_session;
use crate::session::handle::SessionHandle;
use crate::session::http::create_reqwest_client;
use crate::web_driver::WebDriver;

use super::browser::{BrowserKind, detect_local_version};
use super::download::{DownloadConfig, Mirror, ensure_driver, resolve_version};
use super::error::ManagerError;
use super::process::{ManagedDriverProcess, SpawnConfig, SpawnContext, StdioMode};
use super::status::{
    DriverId, DriverLogCallback, DriverLogLine, DriverLogSubscription, Emitter, LogSubscribers,
    Status, StatusCallback, Subscription,
};
use super::version::DriverVersion;

const DEFAULT_DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(60);
const DEFAULT_READY_TIMEOUT: Duration = Duration::from_secs(30);

/// Default cache directory: `<cache_dir>/thirtyfour/drivers`, falling back to the
/// system temp dir if no cache dir is available.
fn default_cache_dir() -> PathBuf {
    dirs::cache_dir().unwrap_or_else(std::env::temp_dir).join("thirtyfour").join("drivers")
}

/// Auto-download and lifetime-managed local WebDriver process manager.
///
/// See the [module documentation](super) for the high-level usage. The two
/// common entry points are [`WebDriver::managed`] (one-shot, fresh manager
/// per call) and [`WebDriverManager::builder`] (for sharing one manager
/// across many sessions).
///
/// [`WebDriver::managed`]: crate::WebDriver::managed
pub struct WebDriverManager {
    pub(crate) cfg: ResolvedConfig,
    /// Configuration applied to every WebDriver session launched by this manager.
    pub(crate) web_driver_config: WebDriverConfig,
    /// HTTP client used for fetching upstream metadata and binaries.
    download_client: reqwest::Client,
    /// Map from `(browser, resolved_version)` to a Weak handle on a live driver.
    drivers: Mutex<HashMap<DriverKey, Weak<ManagedDriverProcess>>>,
    /// Cache for the local-browser version probe. The installed browser
    /// doesn't change for the lifetime of a manager, so probing
    /// `<browser> --version` once per `(browser, custom_binary)` is enough.
    /// Skips repeated subprocess spawns and avoids transient
    /// `Command::output` failures on busy CI runners.
    local_versions: Mutex<HashMap<(BrowserKind, Option<String>), String>>,
    /// Status-event emitter (also forwards to `tracing`).
    pub(crate) emitter: Emitter,
    /// Manager-wide driver-log subscribers — propagated into every spawned
    /// `ManagedDriverProcess` so subscribers added at any time see lines from
    /// every live driver.
    pub(crate) log_subscribers: LogSubscribers,
    /// Monotonic counter for [`DriverId`].
    next_driver_id: Arc<AtomicU64>,
}

impl std::fmt::Debug for WebDriverManager {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebDriverManager")
            .field("cfg", &self.cfg)
            .field("web_driver_config", &self.web_driver_config)
            .finish_non_exhaustive()
    }
}

#[derive(Clone)]
pub(crate) struct ResolvedConfig {
    pub version: DriverVersion,
    pub cache_dir: PathBuf,
    pub host: IpAddr,
    pub download_timeout: Duration,
    pub ready_timeout: Duration,
    pub offline: bool,
    pub mirror: Mirror,
    pub stdio: StdioMode,
    /// Windows only: give the driver process its own console window instead
    /// of suppressing it with `CREATE_NO_WINDOW`.
    pub show_console_window: bool,
    /// Per-browser driver-binary overrides. When a browser appears here,
    /// `ensure_driver` skips download/cache and spawns the supplied binary
    /// directly.
    pub driver_paths: HashMap<BrowserKind, PathBuf>,
}

impl std::fmt::Debug for ResolvedConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedConfig")
            .field("version", &self.version)
            .field("cache_dir", &self.cache_dir)
            .field("host", &self.host)
            .field("download_timeout", &self.download_timeout)
            .field("ready_timeout", &self.ready_timeout)
            .field("offline", &self.offline)
            .field("stdio", &self.stdio)
            .field("show_console_window", &self.show_console_window)
            .field("driver_paths", &self.driver_paths)
            .finish_non_exhaustive()
    }
}

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub(super) struct DriverKey {
    pub(super) browser: BrowserKind,
    pub(super) version: String,
    pub(super) host: IpAddr,
}

impl DriverGuard for ManagedDriverProcess {}

/// Per-session guard held by `SessionHandle::driver_guard`. Keeps the
/// underlying [`ManagedDriverProcess`] alive for the lifetime of the session,
/// and emits [`Status::SessionEnded`] when dropped.
pub(crate) struct SessionGuard {
    driver: StdMutex<Option<Arc<ManagedDriverProcess>>>,
    emitter: Emitter,
    browser: BrowserKind,
    session_id: String,
}

impl std::fmt::Debug for SessionGuard {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionGuard")
            .field("browser", &self.browser)
            .field("session_id", &self.session_id)
            .field("driver", &self.driver())
            .finish_non_exhaustive()
    }
}

impl SessionGuard {
    fn driver(&self) -> StdMutexGuard<'_, Option<Arc<ManagedDriverProcess>>> {
        self.driver.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub(crate) fn driver_id(&self) -> Option<DriverId> {
        self.driver().as_ref().map(|driver| driver.driver_id)
    }

    pub(crate) fn subscribe_log<F>(&self, f: F) -> Option<DriverLogSubscription>
    where
        F: Fn(&DriverLogLine) + Send + Sync + 'static,
    {
        self.driver().as_ref().map(|driver| driver.subscribe_log(f))
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        self.emitter.emit(Status::SessionEnded {
            browser: self.browser,
            session_id: self.session_id.clone(),
        });
    }
}

impl DriverGuard for SessionGuard {
    fn release(&self) -> Pin<Box<dyn Future<Output = WebDriverResult<()>> + Send + '_>> {
        Box::pin(async move {
            let driver = self.driver().take();
            if let Some(driver) = driver
                && let Some(process) = Arc::into_inner(driver)
            {
                process.shutdown().await.map_err(WebDriverError::from)?;
            }
            Ok(())
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Builder for [`WebDriverManager`]. The same type is used both via
/// `WebDriverManager::builder()` and (with capabilities preloaded) via
/// [`WebDriver::managed`]. See the [module documentation](super) for examples.
///
/// [`WebDriver::managed`]: crate::WebDriver::managed
#[derive(Default)]
pub struct WebDriverManagerBuilder {
    pub(crate) version: DriverVersion,
    pub(crate) cache_dir: Option<PathBuf>,
    pub(crate) host: Option<IpAddr>,
    pub(crate) download_timeout: Option<Duration>,
    pub(crate) ready_timeout: Option<Duration>,
    pub(crate) offline: Option<bool>,
    pub(crate) mirror: Option<Mirror>,
    pub(crate) stdio: Option<StdioMode>,
    pub(crate) show_console_window: Option<bool>,
    /// Per-browser driver-binary overrides registered via
    /// [`WebDriverManagerBuilder::driver_binary`].
    pub(crate) driver_paths: HashMap<BrowserKind, PathBuf>,
    /// Status subscribers registered before `build`.
    pub(crate) status_subscribers: Vec<StatusCallback>,
    /// Driver-log subscribers registered before `build`.
    pub(crate) log_subscribers: Vec<DriverLogCallback>,
    /// Configuration applied to every WebDriver session launched by the manager.
    pub(crate) web_driver_config: WebDriverConfig,
    /// Set when constructed via `WebDriver::managed(caps)`.
    pub(crate) preloaded_caps: Option<Capabilities>,
}

impl std::fmt::Debug for WebDriverManagerBuilder {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WebDriverManagerBuilder")
            .field("version", &self.version)
            .field("cache_dir", &self.cache_dir)
            .field("host", &self.host)
            .field("download_timeout", &self.download_timeout)
            .field("ready_timeout", &self.ready_timeout)
            .field("offline", &self.offline)
            .field("stdio", &self.stdio)
            .field("show_console_window", &self.show_console_window)
            .field("driver_paths", &self.driver_paths)
            .field("status_subscribers", &self.status_subscribers.len())
            .field("log_subscribers", &self.log_subscribers.len())
            .field("web_driver_config", &self.web_driver_config)
            .finish_non_exhaustive()
    }
}

impl Clone for WebDriverManagerBuilder {
    fn clone(&self) -> Self {
        Self {
            version: self.version.clone(),
            cache_dir: self.cache_dir.clone(),
            host: self.host,
            download_timeout: self.download_timeout,
            ready_timeout: self.ready_timeout,
            offline: self.offline,
            mirror: self.mirror.clone(),
            stdio: self.stdio,
            show_console_window: self.show_console_window,
            driver_paths: self.driver_paths.clone(),
            status_subscribers: self.status_subscribers.iter().map(Arc::clone).collect(),
            log_subscribers: self.log_subscribers.iter().map(Arc::clone).collect(),
            web_driver_config: self.web_driver_config.clone(),
            preloaded_caps: self.preloaded_caps.clone(),
        }
    }
}

impl WebDriverManagerBuilder {
    /// Set the configuration used by every [`WebDriver`] session launched by
    /// the resulting manager.
    pub fn config(mut self, config: WebDriverConfig) -> Self {
        self.web_driver_config = config;
        self
    }

    /// Set the element poller used by every [`WebDriver`] session launched by
    /// the resulting manager.
    ///
    /// This replaces the poller in the current [`WebDriverConfig`], including
    /// a configuration previously supplied via
    /// [`WebDriverManagerBuilder::config`].
    pub fn poller(mut self, poller: Arc<dyn IntoElementPoller + Send + Sync>) -> Self {
        self.web_driver_config.poller = poller;
        self
    }

    /// Pick a driver version. See [`DriverVersion`] for variants.
    pub fn version(mut self, v: DriverVersion) -> Self {
        self.version = v;
        self
    }

    /// Use the latest stable driver from upstream metadata.
    pub fn latest(self) -> Self {
        self.version(DriverVersion::Latest)
    }

    /// Probe the locally-installed browser and use a matching driver. This is
    /// the default; calling it explicitly is a no-op.
    pub fn match_local(self) -> Self {
        self.version(DriverVersion::MatchLocalBrowser)
    }

    /// Read `browserVersion` from the capabilities supplied to `launch()` /
    /// `WebDriver::managed()`.
    pub fn from_caps(self) -> Self {
        self.version(DriverVersion::FromCapabilities)
    }

    /// Pin a specific version (full like `"126.0.6478.126"` or major-only like `"126"`).
    pub fn exact(self, v: impl Into<String>) -> Self {
        self.version(DriverVersion::Exact(v.into()))
    }

    /// Override the cache directory. Default: `<system cache dir>/thirtyfour/drivers`.
    pub fn cache_dir(mut self, p: PathBuf) -> Self {
        self.cache_dir = Some(p);
        self
    }

    /// Override the host the driver binds to. Default: `127.0.0.1`.
    pub fn host(mut self, h: IpAddr) -> Self {
        self.host = Some(h);
        self
    }

    /// Override the timeout for upstream metadata + binary downloads. Default: 60s.
    pub fn download_timeout(mut self, d: Duration) -> Self {
        self.download_timeout = Some(d);
        self
    }

    /// Override the timeout for waiting on driver `/status` readiness. Default: 30s.
    pub fn ready_timeout(mut self, d: Duration) -> Self {
        self.ready_timeout = Some(d);
        self
    }

    /// Refuse to download anything; require the driver to already be in cache.
    pub fn offline(self) -> Self {
        self.offline_mode(true)
    }

    /// Allow downloads. This is the default.
    pub fn online(self) -> Self {
        self.offline_mode(false)
    }

    /// Set offline mode explicitly.
    pub fn offline_mode(mut self, yes: bool) -> Self {
        self.offline = Some(yes);
        self
    }

    /// Override the Chrome-for-Testing metadata base URL.
    pub fn chrome_metadata_mirror(mut self, base: Url) -> Self {
        let m = self.mirror.get_or_insert_with(Mirror::default);
        m.chrome_metadata = base;
        self
    }

    /// Override the geckodriver binary download base URL.
    pub fn geckodriver_downloads_mirror(mut self, base: Url) -> Self {
        let m = self.mirror.get_or_insert_with(Mirror::default);
        m.geckodriver_downloads = base;
        self
    }

    /// Override the msedgedriver download host.
    pub fn edge_downloads_mirror(mut self, base: Url) -> Self {
        let m = self.mirror.get_or_insert_with(Mirror::default);
        m.edge_downloads = base;
        self
    }

    /// Set how driver-process stdout/stderr is handled. Default:
    /// [`StdioMode::Tracing`].
    pub fn stdio(mut self, mode: StdioMode) -> Self {
        self.stdio = Some(mode);
        self
    }

    /// Windows only: give the driver process its own console window.
    ///
    /// By default the driver is spawned with the `CREATE_NO_WINDOW` creation
    /// flag so that no console window flashes up — this matters for GUI apps
    /// built with `#![windows_subsystem = "windows"]`, where the console
    /// subprocess would otherwise pop open a visible console. Driver output
    /// is unaffected either way: stdout/stderr are still delivered according
    /// to [`WebDriverManagerBuilder::stdio`].
    ///
    /// No-op on non-Windows platforms.
    pub fn show_console_window(mut self, yes: bool) -> Self {
        self.show_console_window = Some(yes);
        self
    }

    /// Use an already-installed driver binary at `path` for the given
    /// browser instead of resolving and downloading one. Skips the
    /// version-resolution and download/cache flow entirely; the binary is
    /// spawned as-is. Bare command names (e.g. `"chromedriver"`) are
    /// resolved against the OS `PATH`.
    ///
    /// This is intended for environments that ship their own driver — CI
    /// images with pinned binaries, sandboxes with no network access, or
    /// users who simply prefer to manage driver versions themselves. If
    /// the binary doesn't match the installed browser's version, expect a
    /// runtime error from the driver when the session is started.
    ///
    /// Call once per browser to override drivers for a multi-browser
    /// manager:
    ///
    /// ```no_run
    /// # use thirtyfour::manager::{WebDriverManager, BrowserKind};
    /// let mgr = WebDriverManager::builder()
    ///     .driver_binary(BrowserKind::Chrome, "/usr/local/bin/chromedriver")
    ///     .driver_binary(BrowserKind::Firefox, "/usr/local/bin/geckodriver")
    ///     .build();
    /// ```
    pub fn driver_binary(mut self, browser: BrowserKind, path: impl Into<PathBuf>) -> Self {
        self.driver_paths.insert(browser, path.into());
        self
    }

    /// Register a closure to receive every [`Status`] event emitted by the
    /// resulting manager. Equivalent to calling
    /// [`WebDriverManager::subscribe`] right after `build`, except the
    /// subscriber is attached for the manager's whole lifetime — not removable.
    pub fn on_status<F>(mut self, f: F) -> Self
    where
        F: Fn(&Status) + Send + Sync + 'static,
    {
        self.status_subscribers.push(Arc::new(f));
        self
    }

    /// Register a closure to receive every [`DriverLogLine`] from drivers
    /// spawned by the resulting manager. Equivalent to
    /// [`WebDriverManager::on_driver_log`] applied right after `build`, except
    /// the subscriber is attached for the manager's whole lifetime — not
    /// removable.
    pub fn on_driver_log<F>(mut self, f: F) -> Self
    where
        F: Fn(&DriverLogLine) + Send + Sync + 'static,
    {
        self.log_subscribers.push(Arc::new(f));
        self
    }

    /// Build the manager.
    pub fn build(self) -> Arc<WebDriverManager> {
        let cfg = ResolvedConfig {
            version: self.version,
            cache_dir: self.cache_dir.unwrap_or_else(default_cache_dir),
            host: self.host.unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST)),
            download_timeout: self.download_timeout.unwrap_or(DEFAULT_DOWNLOAD_TIMEOUT),
            ready_timeout: self.ready_timeout.unwrap_or(DEFAULT_READY_TIMEOUT),
            offline: self.offline.unwrap_or(false),
            mirror: self.mirror.unwrap_or_default(),
            stdio: self.stdio.unwrap_or_default(),
            show_console_window: self.show_console_window.unwrap_or(false),
            driver_paths: self.driver_paths,
        };
        let emitter = Emitter::new();
        for cb in self.status_subscribers {
            // `forget` so the subscriber lives for the manager's lifetime.
            std::mem::forget(emitter.add_arc(cb));
        }
        let log_subscribers = LogSubscribers::new();
        for cb in self.log_subscribers {
            std::mem::forget(log_subscribers.add_arc(cb));
        }
        Arc::new(WebDriverManager {
            web_driver_config: self.web_driver_config,
            download_client: reqwest::Client::builder()
                .build()
                .expect("default reqwest client should always build"),
            cfg,
            drivers: Mutex::new(HashMap::new()),
            local_versions: Mutex::new(HashMap::new()),
            emitter,
            log_subscribers,
            next_driver_id: Arc::new(AtomicU64::new(0)),
        })
    }
}

impl IntoFuture for WebDriverManagerBuilder {
    type Output = WebDriverResult<WebDriver>;
    type IntoFuture = Pin<Box<dyn Future<Output = Self::Output> + Send>>;

    fn into_future(self) -> Self::IntoFuture {
        Box::pin(async move {
            let caps = self
                .preloaded_caps
                .clone()
                .ok_or_else(|| WebDriverError::from(ManagerError::NoCapabilities))?;
            self.build().launch(caps).await
        })
    }
}

impl WebDriverManager {
    /// Construct an empty builder.
    pub fn builder() -> WebDriverManagerBuilder {
        WebDriverManagerBuilder::default()
    }

    /// Register a closure to receive every [`Status`] event emitted by this
    /// manager. Returns an RAII guard; dropping it removes the subscriber.
    /// `mem::forget` keeps the subscriber alive for the manager's lifetime.
    ///
    /// Subscribers are also forwarded to the `tracing` ecosystem under the
    /// `thirtyfour::manager` target — they're additive, not a replacement.
    pub fn subscribe<F>(&self, f: F) -> Subscription
    where
        F: Fn(&Status) + Send + Sync + 'static,
    {
        self.emitter.add(f)
    }

    /// Register a closure to receive every [`DriverLogLine`] from drivers
    /// spawned (or to be spawned) by this manager. Returns an RAII guard;
    /// dropping it removes the subscriber.
    pub fn on_driver_log<F>(&self, f: F) -> DriverLogSubscription
    where
        F: Fn(&DriverLogLine) + Send + Sync + 'static,
    {
        self.log_subscribers.add(f)
    }

    pub(crate) fn mint_driver_id(&self) -> DriverId {
        DriverId::from_raw(self.next_driver_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Spawn (or reuse) the appropriate driver and start a session.
    ///
    /// If a driver process for the same `(browser, resolved_version, host)` is
    /// already alive (held by another `WebDriver`), it's reused. Otherwise a
    /// new one is downloaded if needed and spawned.
    pub async fn launch(
        self: &Arc<Self>,
        capabilities: impl Into<Capabilities>,
    ) -> WebDriverResult<WebDriver> {
        let caps: Capabilities = capabilities.into();
        let driver = self.ensure_driver(&caps).await.map_err(WebDriverError::from)?;
        let browser = driver.browser;
        let server_url: Url = driver
            .url()
            .parse()
            .map_err(|e| WebDriverError::ParseError(format!("invalid driver url: {e}")))?;

        let config = self.web_driver_config.clone();
        let client = create_reqwest_client(config.request_timeout);
        let client_arc: Arc<dyn crate::session::http::HttpClient> = Arc::new(client);
        self.emitter.emit(Status::SessionStarting {
            browser,
            url: server_url.to_string(),
        });
        let started = start_session(client_arc.as_ref(), &server_url, &config, caps).await?;
        let session_id = started.session_id.clone();
        self.emitter.emit(Status::SessionStarted {
            browser,
            session_id: session_id.to_string(),
            url: server_url.to_string(),
        });
        let guard: Arc<dyn DriverGuard> = Arc::new(SessionGuard {
            driver: StdMutex::new(Some(driver)),
            emitter: self.emitter.clone(),
            browser,
            session_id: session_id.to_string(),
        });
        let handle = SessionHandle::new_with_config_guard_and_caps(
            client_arc,
            server_url,
            session_id,
            config,
            Some(guard),
            started.capabilities,
        )?;
        Ok(WebDriver {
            handle: Arc::new(handle),
        })
    }

    /// Ensure a driver is running and return an `Arc<ManagedDriverProcess>`
    /// whose existence keeps the child alive.
    async fn ensure_driver(
        &self,
        caps: &Capabilities,
    ) -> Result<Arc<ManagedDriverProcess>, ManagerError> {
        let browser = BrowserKind::from_capabilities(caps)?;
        self.emitter.emit(Status::BrowserKindResolved {
            browser,
        });

        // Manual binary override: skip the resolve/download flow entirely
        // and spawn the supplied path directly. Version string for the
        // DriverKey + Status events is synthesized from the path so manual
        // entries don't collide with cache-keyed (browser, version) tuples.
        if let Some(binary) = self.cfg.driver_paths.get(&browser).cloned() {
            let version = format!("manual:{}", binary.display());
            return self.spawn_or_reuse(browser, version, &binary).await;
        }

        // For MatchLocalBrowser we probe the binary up front (potentially using a
        // capabilities-supplied path). Cached per `(browser, custom_binary)` so
        // we don't re-spawn `<browser> --version` for every session.
        let local = match self.cfg.version {
            DriverVersion::MatchLocalBrowser => {
                let custom = browser.binary_from_caps(caps);
                let key = (browser, custom.clone());
                let mut cache = self.local_versions.lock().await;
                let version = match cache.get(&key) {
                    Some(v) => v.clone(),
                    None => {
                        let v = detect_local_version(browser, custom.as_deref(), &self.emitter)?;
                        cache.insert(key, v.clone());
                        v
                    }
                };
                Some(version)
            }
            _ => None,
        };
        let caps_version = caps.get("browserVersion").and_then(Value::as_str);

        let download_cfg = DownloadConfig {
            cache_dir: self.cfg.cache_dir.clone(),
            mirror: self.cfg.mirror.clone(),
            download_timeout: self.cfg.download_timeout,
            offline: self.cfg.offline,
        };
        let resolved = resolve_version(
            &self.download_client,
            &download_cfg,
            browser,
            &self.cfg.version,
            local.as_deref(),
            caps_version,
            &self.emitter,
        )
        .await?;

        // Fast path: live driver matching `(browser, resolved, host)` already exists.
        {
            let key = DriverKey {
                browser,
                version: resolved.clone(),
                host: self.cfg.host,
            };
            let map = self.drivers.lock().await;
            if let Some(existing) = map.get(&key).and_then(Weak::upgrade) {
                self.emitter.emit(Status::DriverReused {
                    browser,
                    version: resolved,
                    url: existing.url(),
                });
                return Ok(existing);
            }
        }

        let driver_path =
            ensure_driver(&self.download_client, &download_cfg, browser, &resolved, &self.emitter)
                .await?;
        self.spawn_or_reuse(browser, resolved, &driver_path.binary).await
    }

    /// Cache-check, spawn, and register a managed driver process for
    /// `(browser, version, host)`.
    async fn spawn_or_reuse(
        &self,
        browser: BrowserKind,
        version: String,
        binary: &Path,
    ) -> Result<Arc<ManagedDriverProcess>, ManagerError> {
        let key = DriverKey {
            browser,
            version: version.clone(),
            host: self.cfg.host,
        };

        {
            let map = self.drivers.lock().await;
            if let Some(existing) = map.get(&key).and_then(Weak::upgrade) {
                self.emitter.emit(Status::DriverReused {
                    browser,
                    version: version.clone(),
                    url: existing.url(),
                });
                return Ok(existing);
            }
        }

        let process = ManagedDriverProcess::spawn(
            binary,
            browser,
            &SpawnConfig {
                host: self.cfg.host,
                ready_timeout: self.cfg.ready_timeout,
                stdio: self.cfg.stdio,
                show_console_window: self.cfg.show_console_window,
            },
            SpawnContext {
                driver_id: self.mint_driver_id(),
                version: &version,
                emitter: &self.emitter,
                manager_log_subscribers: self.log_subscribers.clone(),
            },
        )
        .await?;

        let arc = Arc::new(process);
        let mut map = self.drivers.lock().await;
        // Re-check after re-locking — another caller may have raced us.
        if let Some(existing) = map.get(&key).and_then(Weak::upgrade) {
            self.emitter.emit(Status::DriverReused {
                browser,
                version,
                url: existing.url(),
            });
            return Ok(existing);
        }
        map.insert(key, Arc::downgrade(&arc));
        Ok(arc)
    }
}
