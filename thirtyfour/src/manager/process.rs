use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::task::JoinHandle;
use tracing::{debug, warn};

use super::browser::BrowserKind;
use super::error::ManagerError;
use super::status::{
    DriverId, DriverLogLine, DriverLogSubscription, DriverStream, Emitter, LogSubscribers, Status,
};

/// How driver-process stdout/stderr is handled.
#[derive(Debug, Clone, Copy, Default)]
pub enum StdioMode {
    /// Pipe both streams and emit each line via `tracing::debug!` under the
    /// `thirtyfour::manager::driver` target. This is the default.
    #[default]
    Tracing,
    /// Inherit the parent process's stdio.
    Inherit,
    /// Discard both streams.
    Null,
}

impl StdioMode {
    fn to_stdio(self) -> Stdio {
        match self {
            StdioMode::Tracing => Stdio::piped(),
            StdioMode::Inherit => Stdio::inherit(),
            StdioMode::Null => Stdio::null(),
        }
    }
}

pub(crate) struct SpawnConfig {
    pub host: IpAddr,
    pub ready_timeout: Duration,
    pub stdio: StdioMode,
    /// Windows only: skip `CREATE_NO_WINDOW` so the driver gets its own
    /// console window. Only read on Windows.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub show_console_window: bool,
}

impl Default for SpawnConfig {
    fn default() -> Self {
        Self {
            host: IpAddr::V4(Ipv4Addr::LOCALHOST),
            ready_timeout: Duration::from_secs(30),
            stdio: StdioMode::default(),
            show_console_window: false,
        }
    }
}

/// A spawned driver process. Killed on drop.
pub(crate) struct ManagedDriverProcess {
    pub host: IpAddr,
    pub port: u16,
    pub browser: BrowserKind,
    /// Resolved driver version; carried so `DriverShutdown` can name it.
    pub version: String,
    /// Synthetic identifier unique within the parent manager.
    pub driver_id: DriverId,
    /// Per-process log subscribers. Cloneable so [`crate::WebDriver`] can
    /// register subscribers via the `DriverGuard` slot it already holds.
    pub log_subscribers: LogSubscribers,
    /// `None` after `Drop` has run.
    child: Option<Child>,
    /// Set when shutdown is initiated, so the stdio pump tasks can exit cleanly.
    shutdown: Arc<AtomicBool>,
    /// Handles for the stdout / stderr pump tasks (when stdio is `Tracing`).
    /// `Drop` aborts these so the runtime can shut down promptly — Windows
    /// pipe semantics mean a stuck `next_line().await` would otherwise block
    /// runtime shutdown indefinitely after `start_kill`.
    pump_handles: Vec<JoinHandle<()>>,
    /// Status emitter so `Drop` can fire `DriverShutdown`.
    emitter: Emitter,
}

impl std::fmt::Debug for ManagedDriverProcess {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ManagedDriverProcess")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("browser", &self.browser)
            .finish()
    }
}

/// Per-spawn context the manager threads in alongside the binary path.
pub(crate) struct SpawnContext<'a> {
    pub driver_id: DriverId,
    pub version: &'a str,
    pub emitter: &'a Emitter,
    /// Manager-wide log subscribers. The newly-spawned process retains a clone
    /// so its pump tasks can fan log lines out alongside per-process subscribers.
    pub manager_log_subscribers: LogSubscribers,
}

impl ManagedDriverProcess {
    /// Spawn the driver and poll until it reports ready.
    ///
    /// Retries are only attempted on the narrow case of a port-collision after
    /// the bind/release dance. Spawn failures (binary not found, exit on
    /// startup) and readiness timeouts surface immediately — retrying those
    /// would just multiply the wait by 3x for no benefit.
    pub(crate) async fn spawn(
        binary: &Path,
        browser: BrowserKind,
        cfg: &SpawnConfig,
        ctx: SpawnContext<'_>,
    ) -> Result<Self, ManagerError> {
        const MAX_PORT_ATTEMPTS: u8 = 3;
        let mut last_err: Option<ManagerError> = None;
        for attempt in 0..MAX_PORT_ATTEMPTS {
            let port = pick_port(cfg.host)?;
            match spawn_at_port(binary, browser, cfg, port, &ctx).await {
                Ok(p) => return Ok(p),
                Err(e) if is_port_in_use(&e) => {
                    debug!("driver port {port} already in use (attempt {attempt}): {e}");
                    last_err = Some(e);
                }
                Err(e) => return Err(e),
            }
        }
        Err(last_err.unwrap_or_else(|| ManagerError::Spawn("port allocation exhausted".into())))
    }

    /// Connection URL.
    pub(crate) fn url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }

    /// Subscribe to log lines from just this driver process. Returns an RAII
    /// guard whose drop unsubscribes; `mem::forget` keeps the subscriber alive
    /// for the rest of the process's lifetime.
    pub(crate) fn subscribe_log<F>(&self, f: F) -> DriverLogSubscription
    where
        F: Fn(&DriverLogLine) + Send + Sync + 'static,
    {
        self.log_subscribers.add(f)
    }

    /// Kill the driver and wait until the OS has reaped it.
    ///
    /// This consumes the process after the final managed-session lease is
    /// released. [`Drop`] remains the non-blocking fallback when explicit
    /// asynchronous cleanup is skipped or cancelled.
    pub(crate) async fn shutdown(mut self) -> std::io::Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        for handle in self.pump_handles.drain(..) {
            handle.abort();
        }
        if let Some(mut child) = self.child.take() {
            child.kill().await?;
        }
        Ok(())
    }
}

/// Heuristic: did this error come from the OS reporting that the chosen port
/// was already bound? Both "Address already in use" and Windows's "Only one
/// usage of each socket address" land here.
fn is_port_in_use(err: &ManagerError) -> bool {
    let msg = match err {
        ManagerError::Spawn(s) => s.as_str(),
        _ => return false,
    };
    let lower = msg.to_ascii_lowercase();
    lower.contains("address already in use")
        || lower.contains("only one usage of each socket address")
        || lower.contains("addrinuse")
}

async fn spawn_at_port(
    binary: &Path,
    browser: BrowserKind,
    cfg: &SpawnConfig,
    port: u16,
    ctx: &SpawnContext<'_>,
) -> Result<ManagedDriverProcess, ManagerError> {
    // chromedriver, geckodriver, msedgedriver, and safaridriver all accept --port=N.
    let mut cmd = Command::new(binary);
    cmd.arg(format!("--port={port}"));
    cmd.stdout(cfg.stdio.to_stdio());
    cmd.stderr(cfg.stdio.to_stdio());
    cmd.stdin(Stdio::null());
    cmd.kill_on_drop(true);

    // On Windows a console subprocess pops up its own console window whenever
    // the parent has none (GUI apps built with `windows_subsystem =
    // "windows"`). Suppress it with CREATE_NO_WINDOW unless the user opted in
    // via `show_console_window`. Explicitly-set stdio handles still work under
    // CREATE_NO_WINDOW, so every `StdioMode` behaves the same either way.
    #[cfg(windows)]
    if !cfg.show_console_window {
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    // chromedriver / msedgedriver bind only to loopback by default; pass
    // --allowed-ips when the user has configured a non-loopback host.
    if matches!(browser, BrowserKind::Chrome | BrowserKind::Edge)
        && cfg.host != IpAddr::V4(Ipv4Addr::LOCALHOST)
    {
        cmd.arg(format!("--allowed-ips={}", cfg.host));
    }

    // geckodriver's WebDriver-BiDi WebSocket binds Firefox's RemoteAgent
    // server to a fixed default port (9222). When the manager runs many
    // Firefox sessions back-to-back the new Firefox can't bind 9222 if the
    // previous Firefox hasn't fully released it yet — and the resulting
    // `webSocketUrl` capability still reports 9222, so connecting fails
    // with HTTP 404. Pass `--websocket-port=0` to ask geckodriver to pick a
    // free port; it propagates through to RemoteAgent and the returned
    // `webSocketUrl` reflects the actual bound port.
    if matches!(browser, BrowserKind::Firefox) {
        cmd.arg("--websocket-port=0");
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| ManagerError::Spawn(format!("spawn {}: {}", binary.display(), e)))?;
    let pid = child.id().unwrap_or(0);
    ctx.emitter.emit(Status::DriverProcessSpawned {
        browser,
        version: ctx.version.to_string(),
        pid,
        port,
        binary: PathBuf::from(binary),
    });

    let shutdown = Arc::new(AtomicBool::new(false));
    let log_subscribers = LogSubscribers::new();
    let mut pump_handles = Vec::new();
    if matches!(cfg.stdio, StdioMode::Tracing) {
        let line_ctx = LogLineContext {
            driver_id: ctx.driver_id,
            browser,
            version: ctx.version.to_string(),
            port,
        };
        if let Some(stdout) = child.stdout.take() {
            pump_handles.push(spawn_pump(
                DriverStream::Stdout,
                stdout,
                Arc::clone(&shutdown),
                line_ctx.clone(),
                log_subscribers.clone(),
                ctx.manager_log_subscribers.clone(),
            ));
        }
        if let Some(stderr) = child.stderr.take() {
            pump_handles.push(spawn_pump(
                DriverStream::Stderr,
                stderr,
                Arc::clone(&shutdown),
                line_ctx,
                log_subscribers.clone(),
                ctx.manager_log_subscribers.clone(),
            ));
        }
    }

    let ready_started = Instant::now();
    if let Err(e) = wait_until_ready(cfg.host, port, cfg.ready_timeout).await {
        let _ = child.kill().await;
        for h in &pump_handles {
            h.abort();
        }
        return Err(e);
    }
    let url = format!("http://{}:{}", cfg.host, port);
    ctx.emitter.emit(Status::DriverReady {
        browser,
        version: ctx.version.to_string(),
        url,
        elapsed: ready_started.elapsed(),
    });

    Ok(ManagedDriverProcess {
        host: cfg.host,
        port,
        browser,
        version: ctx.version.to_string(),
        driver_id: ctx.driver_id,
        log_subscribers,
        child: Some(child),
        shutdown,
        pump_handles,
        emitter: ctx.emitter.clone(),
    })
}

fn pick_port(host: IpAddr) -> Result<u16, ManagerError> {
    let listener = TcpListener::bind(SocketAddr::new(host, 0))
        .map_err(|e| ManagerError::Spawn(format!("bind ephemeral port: {e}")))?;
    let port =
        listener.local_addr().map_err(|e| ManagerError::Spawn(format!("local_addr: {e}")))?.port();
    drop(listener);
    Ok(port)
}

/// Context shared between every line dispatched by one pump task.
#[derive(Clone)]
struct LogLineContext {
    driver_id: DriverId,
    browser: BrowserKind,
    version: String,
    port: u16,
}

fn spawn_pump<R>(
    stream: DriverStream,
    reader: R,
    shutdown: Arc<AtomicBool>,
    ctx: LogLineContext,
    process_subs: LogSubscribers,
    manager_subs: LogSubscribers,
) -> JoinHandle<()>
where
    R: tokio::io::AsyncRead + Unpin + Send + 'static,
{
    let stream_label = match stream {
        DriverStream::Stdout => "stdout",
        DriverStream::Stderr => "stderr",
    };
    tokio::spawn(async move {
        let mut lines = BufReader::new(reader).lines();
        loop {
            if shutdown.load(Ordering::Relaxed) {
                break;
            }
            match lines.next_line().await {
                Ok(Some(line)) => {
                    debug!(target: "thirtyfour::manager::driver", stream = stream_label, line = %line);
                    let log = DriverLogLine {
                        driver_id: ctx.driver_id,
                        browser: ctx.browser,
                        version: ctx.version.clone(),
                        port: ctx.port,
                        stream,
                        line,
                    };
                    process_subs.dispatch(&log);
                    manager_subs.dispatch(&log);
                }
                Ok(None) => break,
                Err(e) => {
                    warn!(target: "thirtyfour::manager::driver", stream = stream_label, error = %e);
                    break;
                }
            }
        }
    })
}

async fn wait_until_ready(host: IpAddr, port: u16, timeout: Duration) -> Result<(), ManagerError> {
    let url = format!("http://{host}:{port}/status");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .map_err(|e| ManagerError::Spawn(e.to_string()))?;

    let deadline = Instant::now() + timeout;
    while Instant::now() < deadline {
        if let Ok(resp) = client.get(&url).send().await
            && resp.status().is_success()
            && let Ok(body) = resp.json::<serde_json::Value>().await
            && body
                .get("value")
                .and_then(|v| v.get("ready"))
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        {
            return Ok(());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Err(ManagerError::DriverNotReady(timeout))
}

impl Drop for ManagedDriverProcess {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        // Sync drop, so we can't await. `start_kill` issues SIGKILL (or
        // TerminateProcess on Windows) without blocking; `kill_on_drop(true)`
        // asks Tokio to reap the child on a best-effort basis when the Child
        // handle drops. Explicit `WebDriver::quit().await` uses `shutdown()`
        // above and waits for reaping instead.
        if let Some(mut child) = self.child.take()
            && let Err(e) = child.start_kill()
        {
            warn!(target: "thirtyfour::manager", error = %e, "failed to kill driver");
        }
        // Abort the stdio pump tasks. Without this, on Windows the pumps can
        // remain stuck on `next_line().await` after the child is killed
        // (anonymous pipe semantics don't always surface EOF cleanly), and the
        // multi-thread Tokio runtime drops by waiting for all spawned tasks to
        // finish — blocking process exit indefinitely.
        for h in self.pump_handles.drain(..) {
            h.abort();
        }
        self.emitter.emit(Status::DriverShutdown {
            browser: self.browser,
            version: self.version.clone(),
            port: self.port,
        });
    }
}
