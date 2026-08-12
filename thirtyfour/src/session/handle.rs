use arc_swap::ArcSwap;
use serde_json::Value;
use std::fmt::{Debug, Display, Formatter};
use std::future::Future;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::{Arc, LazyLock};
use std::time::Duration;
use tokio::sync::OnceCell;
use url::{ParseError, Url};

use crate::action_chain::ActionChain;
use crate::common::command::{Command, FormatRequestData};
use crate::common::config::WebDriverConfig;
use crate::common::cookie::Cookie;
use crate::common::print::PrintParameters;
use crate::error::WebDriverResult;
use crate::extensions::query::IntoElementPoller;
use crate::prelude::WebDriverError;
use crate::session::scriptret::ScriptRet;
use crate::support::base64_decode;
use crate::web_driver::AlreadyQuit;
use crate::{By, Capabilities, OptionRect, Rect, SessionId, WebDriverStatus, WebElement, support};
use crate::{IntoArcStr, IntoUrl};
use crate::{TimeoutConfiguration, WindowHandle};

use super::DriverGuard;
use super::http::{CmdResponse, HttpClient, run_webdriver_cmd};

/// The SessionHandle contains a shared reference to the HTTP client
/// to allow sending commands to the underlying WebDriver.
pub struct SessionHandle {
    /// The HTTP client for performing webdriver requests.
    pub client: Arc<dyn HttpClient>,
    /// The webdriver server URL.
    server_url: Arc<Url>,
    /// The session id for this webdriver session.
    session_id: SessionId,
    /// Session capabilities returned by `New Session` (vendor-prefixed
    /// fields like `goog:chromeOptions.debuggerAddress` and `se:cdp`
    /// are read from here for CDP WebSocket discovery; W3C `webSocketUrl`
    /// for BiDi).
    capabilities: Arc<Capabilities>,
    /// Atomically replaceable config used by this instance.
    config: ArcSwap<WebDriverConfig>,
    /// quit session flag
    quit: Arc<OnceCell<()>>,
    /// Lazily-connected WebDriver BiDi handle, shared by every clone of the
    /// `SessionHandle`. Initialised on first call to [`crate::WebDriver::bidi`].
    #[cfg(feature = "bidi")]
    bidi: Arc<OnceCell<crate::bidi::BiDi>>,
    /// Optional opaque guard that keeps an external resource (e.g. a managed
    /// `chromedriver` subprocess) alive for as long as this session exists.
    /// Declared **last** so it drops after `quit` has run in the `Drop` impl.
    driver_guard: Option<Arc<dyn DriverGuard>>,
}

impl Debug for SessionHandle {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SessionHandle")
            .field("session_id", &self.session_id)
            .field("config", &self.config)
            .finish()
    }
}

impl SessionHandle {
    /// Create a new `SessionHandle` storing the capabilities returned by
    /// `New Session`. Used internally by session-creation paths so vendor
    /// caps (e.g. `goog:chromeOptions`, `se:cdp`) survive into
    /// [`crate::cdp`] discovery. The optional [`DriverGuard`] ties this
    /// session's lifetime to an external resource (e.g. a managed driver
    /// subprocess).
    pub(crate) fn new_with_config_guard_and_caps(
        client: Arc<dyn HttpClient>,
        server_url: impl IntoUrl,
        session_id: SessionId,
        config: WebDriverConfig,
        driver_guard: Option<Arc<dyn DriverGuard>>,
        capabilities: Capabilities,
    ) -> WebDriverResult<Self> {
        Ok(Self {
            client,
            server_url: Arc::new(server_url.into_url()?),
            session_id,
            capabilities: Arc::new(capabilities),
            config: ArcSwap::from_pointee(config),
            quit: Arc::new(OnceCell::new()),
            #[cfg(feature = "bidi")]
            bidi: Arc::new(OnceCell::new()),
            driver_guard,
        })
    }

    /// Clone this session handle but attach the specified `WebDriverConfig`.
    ///
    /// See `WebDriver::clone_with_config()`.
    pub(crate) fn clone_with_config(self: &SessionHandle, config: WebDriverConfig) -> Self {
        Self {
            client: Arc::clone(&self.client),
            server_url: Arc::clone(&self.server_url),
            session_id: self.session_id.clone(),
            capabilities: Arc::clone(&self.capabilities),
            quit: Arc::clone(&self.quit),
            #[cfg(feature = "bidi")]
            bidi: Arc::clone(&self.bidi),
            config: ArcSwap::from_pointee(config),
            driver_guard: self.driver_guard.clone(),
        }
    }

    /// Cached BiDi handle. Used by [`crate::WebDriver::bidi`] to lazy-init
    /// once and reuse on subsequent calls.
    #[cfg(feature = "bidi")]
    pub(crate) fn bidi_cell(&self) -> &Arc<OnceCell<crate::bidi::BiDi>> {
        &self.bidi
    }

    /// The session id for this webdriver session.
    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    /// Capabilities returned by `New Session`. Empty if the WebDriver server
    /// returned an unrecognised response shape. Useful for reading
    /// vendor-prefixed fields like `goog:chromeOptions` or `se:cdp`.
    pub fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }

    /// The opaque driver guard (if this session was launched via the manager).
    /// Used internally by [`crate::WebDriver::driver_id`] /
    /// [`crate::WebDriver::on_driver_log`] to reach the underlying managed
    /// driver via [`DriverGuard::as_any`].
    ///
    /// Only the manager-feature side of the crate consumes this — without
    /// `feature = "manager"` the guard slot is always `None` (there are no
    /// `DriverGuard` implementors in scope), so the accessor would generate
    /// a dead-code warning under `cargo hack`.
    #[cfg(feature = "manager")]
    pub(crate) fn driver_guard(&self) -> Option<&Arc<dyn DriverGuard>> {
        self.driver_guard.as_ref()
    }

    /// The webdriver server URL this session is connected to.
    pub fn server_url(&self) -> &Url {
        &self.server_url
    }

    /// Return a snapshot of the configuration currently used by this session.
    ///
    /// The returned [`Arc`] keeps this snapshot alive if another task replaces
    /// the session configuration.
    pub fn config(&self) -> Arc<WebDriverConfig> {
        self.config.load_full()
    }

    /// Atomically replace the configuration used by this session.
    ///
    /// The new configuration is shared by all [`WebDriver`](crate::WebDriver)
    /// clones and [`WebElement`]s associated with this session. Existing
    /// [`ElementQuery`](crate::extensions::query::ElementQuery) and
    /// [`ElementWaiter`](crate::extensions::query::ElementWaiter) values keep
    /// the poller they were created with.
    ///
    /// Changing [`WebDriverConfig::request_timeout`] does not reconfigure the
    /// existing HTTP client. Its timeout is fixed when the client is created.
    pub fn set_config(&self, config: WebDriverConfig) {
        self.config.store(Arc::new(config));
    }

    /// Atomically replace the default element poller used by this session.
    ///
    /// The new poller is shared by all [`WebDriver`](crate::WebDriver) clones
    /// and [`WebElement`]s associated with this session. Existing
    /// [`ElementQuery`](crate::extensions::query::ElementQuery) and
    /// [`ElementWaiter`](crate::extensions::query::ElementWaiter) values keep
    /// the poller they were created with.
    pub fn set_poller(&self, poller: Arc<dyn IntoElementPoller + Send + Sync>) {
        self.config.rcu(|current| {
            let mut config = current.as_ref().clone();
            config.poller = Arc::clone(&poller);
            config
        });
    }

    /// Send the specified command to the webdriver server.
    pub async fn cmd(&self, command: impl FormatRequestData) -> WebDriverResult<CmdResponse> {
        let request_data = command.format_request(&self.session_id);
        let config = self.config();
        run_webdriver_cmd(&*self.client, &request_data, &self.server_url, config.as_ref()).await
    }

    /// Get the WebDriver status.
    ///
    /// # Example
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// let caps = DesiredCapabilities::chrome();
    /// let mut driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let status = driver.status().await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn status(&self) -> WebDriverResult<WebDriverStatus> {
        self.cmd(Command::Status).await?.value()
    }

    /// Close the current window or tab. This will close the session if no other windows exist.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// // Open a new tab.
    /// driver.new_tab().await?;
    ///
    /// // Get window handles and switch to the new tab.
    /// let handles = driver.windows().await?;
    /// driver.switch_to_window(handles[1].clone()).await?;
    ///
    /// // We are now controlling the new tab.
    /// driver.goto("https://www.rust-lang.org").await?;
    ///
    /// // Close the tab. This will return to the original tab.
    /// driver.close_window().await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn close_window(&self) -> WebDriverResult<()> {
        self.cmd(Command::CloseWindow).await?;
        Ok(())
    }

    /// Navigate to the specified URL.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// driver.goto("https://www.rust-lang.org").await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn goto(&self, url: impl IntoArcStr) -> WebDriverResult<()> {
        let url = url.into();

        let parse_url = |url: Arc<str>| Url::parse(&url).map(|_| url);
        let url = parse_url(url.clone())
            .or_else(|e| match e {
                ParseError::RelativeUrlWithoutBase => {
                    parse_url(("https://".to_string() + &*url).into())
                }
                e => Err(e),
            })
            .map_err(WebDriverError::InvalidUrl)?;
        self.cmd(Command::NavigateTo(url)).await?;
        Ok(())
    }

    /// Navigate to the specified URL. Alias of goto().
    pub async fn get(&self, url: impl IntoArcStr) -> WebDriverResult<()> {
        self.goto(url).await
    }

    /// Get the current URL.
    pub async fn current_url(&self) -> WebDriverResult<Url> {
        let r = self.cmd(Command::GetCurrentUrl).await?;
        let s: String = r.value()?;
        Url::parse(&s).map_err(|e| WebDriverError::ParseError(format!("invalid url: {s}: {e}")))
    }

    /// Get the page source as a String.
    pub async fn source(&self) -> WebDriverResult<String> {
        self.cmd(Command::GetPageSource).await?.value()
    }

    /// Get the page title as a String.
    pub async fn title(&self) -> WebDriverResult<String> {
        self.cmd(Command::GetTitle).await?.value()
    }

    /// Search for an element on the current page using the specified selector.
    ///
    /// **NOTE**: For more powerful element queries including polling and filters, see the
    ///           [`WebDriver::query`] method instead.
    ///
    /// [`WebDriver::query`]: crate::extensions::query::ElementQueryable::query
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let elem_button = driver.find(By::Id("my-element-id")).await?;
    /// let elem_text = driver.find(By::Name("my-text-input")).await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn find(self: &Arc<Self>, by: By) -> WebDriverResult<WebElement> {
        let r = self.cmd(Command::FindElement(by.into())).await?;
        r.element(self.clone())
    }

    /// Search for all elements on the current page that match the specified selector.
    ///
    /// **NOTE**: For more powerful element queries including polling and filters, see the
    ///           [`WebDriver::query`] method instead.
    ///
    /// [`WebDriver::query`]: crate::extensions::query::ElementQueryable::query
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let elems = driver.find_all(By::ClassName("section")).await?;
    /// for elem in elems {
    ///     assert!(elem.attr("class").await?.expect("Missing class on element").contains("section"));
    /// }
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn find_all(self: &Arc<Self>, by: By) -> WebDriverResult<Vec<WebElement>> {
        let r = self.cmd(Command::FindElements(by.into())).await?;
        r.elements(self.clone())
    }

    /// Execute the specified Javascript synchronously and return the result.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let ret = driver.execute(r#"
    ///     let elem = document.getElementById("button1");
    ///     elem.click();
    ///     return elem;
    ///     "#, Vec::new()
    /// ).await?;
    /// let elem_out: WebElement = ret.element()?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    /// To supply an element as an input argument to a script, use
    /// [`WebElement::to_json`] as follows:
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let elem = driver.find(By::Id("button1")).await?;
    /// let ret = driver.execute(r#"
    ///     arguments[0].innerHTML = arguments[1];
    ///     return arguments[0];
    ///     "#, vec![elem.to_json()?, serde_json::to_value("TESTING")?]
    /// ).await?;
    /// let elem_out = ret.element()?;
    /// assert_eq!(elem_out.element_id(), elem.element_id());
    /// assert_eq!(elem_out.text().await?, "TESTING");
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn execute(
        self: &Arc<Self>,
        script: impl IntoArcStr,
        args: impl Into<Arc<[Value]>>,
    ) -> WebDriverResult<ScriptRet> {
        let r = self.cmd(Command::ExecuteScript(script.into(), args.into())).await?;
        Ok(ScriptRet::new(self.clone(), r.value()?))
    }

    /// Execute the specified JavaScript asynchronously and return the result.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let ret = driver.execute_async(r#"
    ///     // Selenium automatically provides an extra argument which is a
    ///     // function that receives the return value(s).
    ///     let done = arguments[0];
    ///     window.setTimeout(() => {
    ///         let elem = document.getElementById("button1");
    ///         elem.click();
    ///         done(elem);
    ///     }, 1000);
    ///     "#, Vec::new()
    /// ).await?;
    /// let elem_out: WebElement = ret.element()?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    /// To supply an element as an input argument to a script, use
    /// [`WebElement::to_json`] as follows:
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let elem = driver.find(By::Id("button1")).await?;
    /// let args = vec![elem.to_json()?, serde_json::to_value("TESTING")?];
    /// let ret = driver.execute_async(r#"
    ///     // Selenium automatically provides an extra argument which is a
    ///     // function that receives the return value(s).
    ///     let done = arguments[2];
    ///     window.setTimeout(() => {
    ///         arguments[0].innerHTML = arguments[1];
    ///         done(arguments[0]);
    ///     }, 1000);
    ///     "#, args
    /// ).await?;
    /// let elem_out = ret.element()?;
    /// assert_eq!(elem_out.element_id(), elem.element_id());
    /// assert_eq!(elem_out.text().await?, "TESTING");
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn execute_async(
        self: &Arc<Self>,
        script: impl IntoArcStr,
        args: impl Into<Arc<[Value]>>,
    ) -> WebDriverResult<ScriptRet> {
        let r = self.cmd(Command::ExecuteAsyncScript(script.into(), args.into())).await?;
        Ok(ScriptRet::new(self.clone(), r.value()?))
    }

    /// Get the current window handle.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// // Get the current window handle.
    /// let handle = driver.window().await?;
    ///
    /// // Open a new tab.
    /// driver.new_tab().await?;
    ///
    /// // Get window handles and switch to the new tab.
    /// let handles = driver.windows().await?;
    /// driver.switch_to_window(handles[1].clone()).await?;
    ///
    /// // We are now controlling the new tab.
    /// driver.goto("https://www.rust-lang.org/").await?;
    /// assert_ne!(driver.window().await?, handle);
    ///
    /// // Switch back to original tab.
    /// driver.switch_to_window(handle.clone()).await?;
    /// assert_eq!(driver.window().await?, handle);
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn window(&self) -> WebDriverResult<WindowHandle> {
        let r = self.cmd(Command::GetWindowHandle).await?;
        Ok(WindowHandle::from(r.value::<String>()?))
    }

    /// Get all window handles for the current session.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// assert_eq!(driver.windows().await?.len(), 1);
    /// // Open a new tab.
    /// driver.new_tab().await?;
    ///
    /// // Get window handles and switch to the new tab.
    /// let handles = driver.windows().await?;
    /// assert_eq!(handles.len(), 2);
    /// driver.switch_to_window(handles[1].clone()).await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn windows(&self) -> WebDriverResult<Vec<WindowHandle>> {
        let r = self.cmd(Command::GetWindowHandles).await?;
        let handles: Vec<String> = r.value()?;
        Ok(handles.into_iter().map(WindowHandle::from).collect())
    }

    /// Maximize the current window.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// driver.maximize_window().await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn maximize_window(&self) -> WebDriverResult<()> {
        self.cmd(Command::MaximizeWindow).await?;
        Ok(())
    }

    /// Minimize the current window.
    ///
    /// # Example:
    /// ```no_run
    /// # // Minimize is not currently working on Chrome, but does work
    /// # // on Firefox/geckodriver.
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// driver.minimize_window().await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn minimize_window(&self) -> WebDriverResult<()> {
        self.cmd(Command::MinimizeWindow).await?;
        Ok(())
    }

    /// Make the current window fullscreen.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// driver.fullscreen_window().await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn fullscreen_window(&self) -> WebDriverResult<()> {
        self.cmd(Command::FullscreenWindow).await?;
        Ok(())
    }

    /// Get the current window rectangle, in pixels.
    ///
    /// The returned Rect struct has members `x`, `y`, `width`, `height`,
    /// all i32.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// use thirtyfour::Rect;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// driver.set_window_rect(0, 0, 600, 400).await?;
    /// let rect = driver.get_window_rect().await?;
    /// assert_eq!(rect, Rect::new(0, 0, 600, 400));
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn get_window_rect(&self) -> WebDriverResult<Rect> {
        self.cmd(Command::GetWindowRect).await?.value()
    }

    /// Set the current window rectangle, in pixels.
    ///
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// driver.set_window_rect(0, 0, 500, 400).await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn set_window_rect(
        &self,
        x: i64,
        y: i64,
        width: u32,
        height: u32,
    ) -> WebDriverResult<()> {
        let rect = OptionRect {
            x: Some(x),
            y: Some(y),
            width: Some(width as i64),
            height: Some(height as i64),
        };
        self.cmd(Command::SetWindowRect(rect)).await?;
        Ok(())
    }

    /// Go back. This is equivalent to clicking the browser's back button.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// driver.back().await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn back(&self) -> WebDriverResult<()> {
        self.cmd(Command::Back).await?;
        Ok(())
    }

    /// Go forward. This is equivalent to clicking the browser's forward button.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// driver.forward().await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn forward(&self) -> WebDriverResult<()> {
        self.cmd(Command::Forward).await?;
        Ok(())
    }

    /// Refresh the current page.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// driver.refresh().await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn refresh(&self) -> WebDriverResult<()> {
        self.cmd(Command::Refresh).await?;
        Ok(())
    }

    /// Get all timeouts for the current session.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// use thirtyfour::TimeoutConfiguration;
    /// use std::time::Duration;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let timeouts = driver.get_timeouts().await?;
    /// println!("Page load timeout = {:?}", timeouts.page_load());
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn get_timeouts(&self) -> WebDriverResult<TimeoutConfiguration> {
        self.cmd(Command::GetTimeouts).await?.value()
    }

    /// Set all timeouts for the current session.
    ///
    /// **NOTE:** Setting the implicit wait timeout to a non-zero value will interfere with the use
    /// of [`WebDriver::query`] and [`WebElement::wait_until`].
    /// It is therefore recommended to use these methods (which provide polling
    /// and explicit waits) instead rather than increasing the implicit wait timeout.
    ///
    /// [`WebDriver::query`]: crate::extensions::query::ElementQueryable::query
    /// [`WebElement::wait_until`]: crate::extensions::query::ElementWaitable::wait_until
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// use thirtyfour::TimeoutConfiguration;
    /// use std::time::Duration;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// // Setting timeouts to None means those timeout values will not be updated.
    /// let timeouts = TimeoutConfiguration::new(None, Some(Duration::new(11, 0)), None);
    /// driver.update_timeouts(timeouts).await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn update_timeouts(&self, timeouts: TimeoutConfiguration) -> WebDriverResult<()> {
        self.cmd(Command::SetTimeouts(timeouts)).await?;
        Ok(())
    }

    /// Set the implicit wait timeout.
    ///
    /// This is how long the WebDriver will wait when querying elements.
    /// By default, this is set to 0 seconds.
    ///
    /// **NOTE:** Setting the implicit wait timeout to a non-zero value will interfere with the use
    /// of [`WebDriver::query`] and [`WebElement::wait_until`].
    /// It is therefore recommended to use these methods (which provide polling
    /// and explicit waits) instead rather than increasing the implicit wait timeout.
    ///
    /// [`WebDriver::query`]: crate::extensions::query::ElementQueryable::query
    /// [`WebElement::wait_until`]: crate::extensions::query::ElementWaitable::wait_until
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// use thirtyfour::TimeoutConfiguration;
    /// use std::time::Duration;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let delay = Duration::new(11, 0);
    /// driver.set_implicit_wait_timeout(delay).await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn set_implicit_wait_timeout(&self, time_to_wait: Duration) -> WebDriverResult<()> {
        let timeouts = TimeoutConfiguration::new(None, None, Some(time_to_wait));
        self.update_timeouts(timeouts).await
    }

    /// Set the script timeout.
    ///
    /// This is how long the WebDriver will wait for a Javascript script to execute.
    /// By default, this is set to 60 seconds.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// use thirtyfour::TimeoutConfiguration;
    /// use std::time::Duration;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let delay = Duration::new(11, 0);
    /// driver.set_script_timeout(delay).await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn set_script_timeout(&self, time_to_wait: Duration) -> WebDriverResult<()> {
        let timeouts = TimeoutConfiguration::new(Some(time_to_wait), None, None);
        self.update_timeouts(timeouts).await
    }

    /// Set the page load timeout.
    ///
    /// This is how long the WebDriver will wait for the page to finish loading.
    /// By default, this is set to 60 seconds.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// use thirtyfour::TimeoutConfiguration;
    /// use std::time::Duration;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let delay = Duration::new(11, 0);
    /// driver.set_page_load_timeout(delay).await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn set_page_load_timeout(&self, time_to_wait: Duration) -> WebDriverResult<()> {
        let timeouts = TimeoutConfiguration::new(None, Some(time_to_wait), None);
        self.update_timeouts(timeouts).await
    }

    /// Create a new action chain for this session.
    ///
    /// Action chains can be used to simulate more complex user input actions
    /// involving key combinations, mouse movements, mouse click, right-click,
    /// and more.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let elem_text = driver.find(By::Name("input1")).await?;
    /// let elem_button = driver.find(By::Id("button-set")).await?;
    ///
    /// driver.action_chain()
    ///     .send_keys_to_element(&elem_text, "thirtyfour")
    ///     .move_to_element_center(&elem_button)
    ///     .click()
    ///     .perform()
    ///     .await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub fn action_chain(self: &Arc<SessionHandle>) -> ActionChain {
        ActionChain::new(self.clone())
    }

    /// Create a new action chain for this session.
    /// Set custom delays for key and pointer actions
    ///
    /// The [`Duration`] is the time before an action is executed in the chain.
    ///
    /// `key_delay` defaults to 0ms, `pointer_delay` defaults to 250ms
    pub fn action_chain_with_delay(
        self: &Arc<SessionHandle>,
        key_delay: Option<Duration>,
        pointer_delay: Option<Duration>,
    ) -> ActionChain {
        ActionChain::new_with_delay(self.clone(), key_delay, pointer_delay)
    }

    /// Get all cookies.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let cookies = driver.get_all_cookies().await?;
    /// for cookie in &cookies {
    ///     println!("Got cookie: {}", cookie.value);
    /// }
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn get_all_cookies(&self) -> WebDriverResult<Vec<Cookie>> {
        self.cmd(Command::GetAllCookies).await?.value()
    }

    /// Get the specified cookie.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let cookie = driver.get_named_cookie("key").await?;
    /// println!("Got cookie: {}", cookie.value);
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn get_named_cookie(&self, name: impl IntoArcStr) -> WebDriverResult<Cookie> {
        self.cmd(Command::GetNamedCookie(name.into())).await?.value()
    }

    /// Delete the specified cookie.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// driver.delete_cookie("key").await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn delete_cookie(&self, name: impl IntoArcStr) -> WebDriverResult<()> {
        self.cmd(Command::DeleteCookie(name.into())).await?;
        Ok(())
    }

    /// Delete all cookies.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// driver.delete_all_cookies().await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn delete_all_cookies(&self) -> WebDriverResult<()> {
        self.cmd(Command::DeleteAllCookies).await?;
        Ok(())
    }

    /// Add the specified cookie.
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// driver.goto("https://wikipedia.org").await?;
    /// let mut cookie = Cookie::new("key", "value");
    /// cookie.set_domain("wikipedia.org");
    /// cookie.set_path("/");
    /// cookie.set_same_site(SameSite::Lax);
    /// driver.add_cookie(cookie.clone()).await?;
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn add_cookie(&self, cookie: Cookie) -> WebDriverResult<()> {
        self.cmd(Command::AddCookie(cookie)).await?;
        Ok(())
    }

    /// Print the current window and return it as a PDF.
    pub async fn print_page(&self, parameters: PrintParameters) -> WebDriverResult<Vec<u8>> {
        base64_decode(&self.print_page_base64(parameters).await?)
    }

    /// Print the current window and return it as a PDF, base64 encoded.
    pub async fn print_page_base64(&self, parameters: PrintParameters) -> WebDriverResult<String> {
        self.cmd(Command::PrintPage(parameters)).await?.value()
    }

    /// Take a screenshot of the current window and return it as PNG, base64 encoded.
    pub async fn screenshot_as_png_base64(&self) -> WebDriverResult<String> {
        self.cmd(Command::TakeScreenshot).await?.value()
    }

    /// Take a screenshot of the current window and return it as PNG bytes.
    pub async fn screenshot_as_png(&self) -> WebDriverResult<Vec<u8>> {
        base64_decode(&self.screenshot_as_png_base64().await?)
    }

    /// Take a screenshot of the current window and write it to the specified filename.
    pub async fn screenshot(&self, path: &Path) -> WebDriverResult<()> {
        let png = self.screenshot_as_png().await?;
        support::write_file(path, png).await?;
        Ok(())
    }

    /// List the log types the driver knows about
    /// (`GET /session/{id}/log/types`).
    ///
    /// This is a legacy Selenium endpoint that is **not** part of the W3C
    /// WebDriver specification. Modern `chromedriver` rejects it in W3C
    /// mode with `"Cannot call non W3C standard command while in W3C
    /// mode"`; older `chromedriver`s (and Selenium Grid) still serve it.
    /// `geckodriver` does not.
    ///
    /// Use [`get_log`][SessionHandle::get_log] /
    /// [`browser_log`][SessionHandle::browser_log] directly — those work
    /// against modern `chromedriver`.
    pub async fn get_log_types(&self) -> WebDriverResult<Vec<String>> {
        self.cmd(Command::GetLogTypes).await?.value()
    }

    /// Drain the named log buffer (`POST /session/{id}/log`).
    ///
    /// The most useful log type is `"browser"` — that is where
    /// `chromedriver` reports `console.*` output and uncaught JavaScript
    /// errors. See [`browser_log`][SessionHandle::browser_log] for a
    /// type-shorthand wrapper.
    ///
    /// Each call **drains** the buffer: subsequent calls return only
    /// entries produced since the previous call. Not part of W3C
    /// WebDriver — `geckodriver` does not implement it. Modern
    /// `chromedriver` accepts this endpoint even in W3C mode provided
    /// the session was created with the `goog:loggingPrefs` capability;
    /// see [`set_browser_log_level`][sblp].
    ///
    /// [sblp]: crate::ChromiumLikeCapabilities::set_browser_log_level
    pub async fn get_log(&self, log_type: &str) -> WebDriverResult<Vec<crate::BrowserLogEntry>> {
        self.cmd(Command::GetLog(log_type.into())).await?.value()
    }

    /// Drain the `"browser"` log buffer — shorthand for
    /// `get_log("browser")`.
    ///
    /// Requires `goog:loggingPrefs.browser` to have been set on session
    /// creation; see
    /// [`ChromiumLikeCapabilities::set_browser_log_level`][sblp]. Not
    /// supported by `geckodriver`.
    ///
    /// [sblp]: crate::ChromiumLikeCapabilities::set_browser_log_level
    pub async fn browser_log(&self) -> WebDriverResult<Vec<crate::BrowserLogEntry>> {
        self.get_log("browser").await
    }

    /// Set the current window name.
    ///
    /// Useful for switching between windows/tabs using [`WebDriver::switch_to_named_window`]
    ///
    /// [`WebDriver::switch_to_named_window`]: SessionHandle::switch_to_named_window
    ///
    /// # Example:
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// // Get the current window handle.
    /// let handle = driver.window().await?;
    /// driver.set_window_name("main").await?;
    ///
    /// // Open a new tab.
    /// let new_handle = driver.new_tab().await?;
    ///
    /// // Get window handles and switch to the new tab.
    /// driver.switch_to_window(new_handle).await?;
    ///
    /// // We are now controlling the new tab.
    /// driver.goto("https://www.rust-lang.org").await?;
    /// assert_ne!(driver.window().await?, handle);
    ///
    /// // Switch back to original tab using window name.
    /// driver.switch_to_named_window("main").await?;
    /// assert_eq!(driver.window().await?, handle);
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn set_window_name(
        self: &Arc<SessionHandle>,
        window_name: impl Display,
    ) -> WebDriverResult<()> {
        let script = format!(r#"window.name = "{}""#, window_name);
        self.execute(script, Vec::new()).await?;
        Ok(())
    }

    /// Execute the specified function in a new browser tab, closing the tab when complete.
    ///
    /// The return value will be that of the supplied function, unless an error occurs while
    /// opening or closing the tab.
    ///
    /// ```no_run
    /// # use thirtyfour::prelude::*;
    /// # use thirtyfour::support::block_on;
    /// #
    /// # fn main() -> WebDriverResult<()> {
    /// #     block_on(async {
    /// #         let caps = DesiredCapabilities::chrome();
    /// #         let driver = WebDriver::new("http://localhost:4444", caps).await?;
    /// let window_title = driver.in_new_tab(|| async {
    ///     driver.goto("https://www.google.com").await?;
    ///     driver.title().await
    /// }).await?;
    /// assert_eq!(window_title, "Google");
    /// #         driver.quit().await?;
    /// #         Ok(())
    /// #     })
    /// # }
    /// ```
    pub async fn in_new_tab<F, Fut, T>(&self, f: F) -> WebDriverResult<T>
    where
        F: FnOnce() -> Fut + Send,
        Fut: Future<Output = WebDriverResult<T>> + Send,
        T: Send,
    {
        let handle = self.window().await?;

        // Open new tab.
        let tab_handle = self.new_tab().await?;
        self.switch_to_window(tab_handle).await?;

        let result = f().await;

        // Close tab.
        self.close_window().await?;
        self.switch_to_window(handle).await?;

        result
    }

    pub(crate) async fn quit(&self) -> WebDriverResult<()> {
        self.quit
            .get_or_try_init(|| async { self.cmd(Command::DeleteSession).await.map(drop) })
            .await?;
        if let Some(driver_guard) = &self.driver_guard {
            driver_guard.release().await?;
        }
        Ok(())
    }

    pub(crate) fn leak(&self) -> Result<(), AlreadyQuit> {
        self.quit.set(()).map_err(|_| AlreadyQuit(()))
    }
}

// "SyncDrop" only runs if not manually quit
impl Drop for SessionHandle {
    #[track_caller]
    fn drop(&mut self) {
        if self.quit.initialized() {
            return;
        }

        tracing::warn!(
            "WebDriver was not quit properly — falling back to synchronous teardown, \
             which blocks the async executor. Call `WebDriver::quit().await` instead."
        );

        struct SessionDropGuard(SessionHandle);

        impl Deref for SessionDropGuard {
            type Target = SessionHandle;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl DerefMut for SessionDropGuard {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl Drop for SessionDropGuard {
            fn drop(&mut self) {
                if !self.0.quit.initialized() {
                    static ALWAYS_INIT: LazyLock<Arc<OnceCell<()>>> =
                        LazyLock::new(|| Arc::new(OnceCell::new_with(Some(()))));
                    self.0.quit = Arc::clone(&ALWAYS_INIT);
                    debug_assert!(self.0.quit.initialized())
                }
            }
        }

        let mut this = SessionDropGuard(Self {
            client: Arc::clone(&self.client),
            server_url: Arc::clone(&self.server_url),
            quit: Arc::clone(&self.quit),
            session_id: self.session_id.clone(),
            capabilities: Arc::clone(&self.capabilities),
            config: ArcSwap::from(self.config.load_full()),
            #[cfg(feature = "bidi")]
            bidi: Arc::clone(&self.bidi),
            // The guard stays on the *original* SessionHandle so it drops after
            // this DropGuard's quit() future has run. The reconstructed handle
            // here is just the bits needed to issue the DELETE /session HTTP call.
            driver_guard: None,
        });

        // Run the DELETE /session call on a freshly-built single-thread tokio
        // runtime, on a dedicated OS thread, joined synchronously so Drop
        // blocks until cleanup finishes. The new thread has no tokio context,
        // so building a runtime on it is fine even when the user is inside
        // their own runtime. We also rebuild the HttpClient because the IO
        // drivers from the original runtime may already be gone.
        //
        // Why NOT reuse `support::GLOBAL_RT` here: `GLOBAL_RT` is a single
        // `current_thread` runtime whose core lives on one background thread.
        // Routing every `SessionHandle::Drop` through it serialises every
        // cleanup HTTP call onto that one thread / IO driver, even when the
        // Drops fire from independent OS threads. Under concurrent test load
        // (cargo runs integration test binaries in parallel; each binary can
        // have multiple sessions in flight) the queued cleanups slow each
        // other down enough to push later session creates past
        // chromedriver's readiness deadline, surfacing as
        // `DevToolsActivePort file doesn't exist`, "driver did not become
        // ready within 30s", and Windows chromedriver.exe sharing
        // violations. A per-Drop fresh runtime has its own independent
        // IO driver — concurrent Drops don't fight for one core. The
        // runtime build is in the microsecond range; cheap relative to
        // the HTTP DELETE it wraps.
        let _ = std::thread::spawn(move || {
            let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
                Ok(rt) => rt,
                Err(err) => {
                    tracing::warn!(
                        %err,
                        "failed to build cleanup runtime; WebDriver session may leak \
                         until the driver times it out"
                    );
                    return;
                }
            };
            rt.block_on(async move {
                this.client = this.client.new().await;
                if let Err(err) = this.quit().await {
                    tracing::warn!(
                        %err,
                        "WebDriver sync-Drop cleanup failed; the session may leak \
                         until the driver times it out"
                    );
                }
            });
        })
        .join();
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use http::{Request, Response};

    use super::*;
    use crate::extensions::query::{
        ElementPollerNoWait, ElementPollerWithTimeout, IntoElementPoller,
    };
    use crate::session::http::{Body, HttpClient};
    use crate::{Capabilities, WebDriver};

    struct UnusedClient;

    #[async_trait::async_trait]
    impl HttpClient for UnusedClient {
        async fn send(&self, _: Request<Body<'_>>) -> WebDriverResult<Response<Bytes>> {
            unreachable!("configuration tests do not issue WebDriver commands")
        }

        async fn new(&self) -> Arc<dyn HttpClient> {
            Arc::new(Self)
        }
    }

    fn test_driver() -> WebDriver {
        let handle = Arc::new(
            SessionHandle::new_with_config_guard_and_caps(
                Arc::new(UnusedClient),
                "http://localhost:4444",
                SessionId::from("config-test-session"),
                WebDriverConfig::default(),
                None,
                Capabilities::new(),
            )
            .unwrap(),
        );
        handle.leak().unwrap();
        WebDriver {
            handle,
        }
    }

    #[test]
    fn config_updates_are_shared_by_driver_clones() {
        let driver = test_driver();
        let clone = driver.clone();
        let poller: Arc<dyn IntoElementPoller + Send + Sync> = Arc::new(ElementPollerNoWait);

        driver.set_poller(Arc::clone(&poller));

        let config = clone.config();
        assert!(Arc::ptr_eq(&config.poller, &poller));
        assert_eq!(config.request_timeout, Duration::from_secs(120));

        let replacement_poller: Arc<dyn IntoElementPoller + Send + Sync> =
            Arc::new(ElementPollerWithTimeout::default());
        let replacement = WebDriverConfig::builder()
            .keep_alive(false)
            .poller(Arc::clone(&replacement_poller))
            .request_timeout(Duration::from_secs(42))
            .build()
            .unwrap();

        clone.set_config(replacement);

        let config = driver.config();
        assert!(!config.keep_alive);
        assert_eq!(config.request_timeout, Duration::from_secs(42));
        assert!(Arc::ptr_eq(&config.poller, &replacement_poller));
    }

    #[test]
    fn session_handle_setters_update_the_shared_config() {
        let driver = test_driver();
        let handle = Arc::clone(driver.handle());
        let poller: Arc<dyn IntoElementPoller + Send + Sync> = Arc::new(ElementPollerNoWait);

        handle.set_poller(Arc::clone(&poller));
        assert!(Arc::ptr_eq(&driver.config().poller, &poller));

        let replacement = WebDriverConfig::builder()
            .keep_alive(false)
            .request_timeout(Duration::from_secs(7))
            .build()
            .unwrap();
        handle.set_config(replacement);

        let config = driver.config();
        assert!(!config.keep_alive);
        assert_eq!(config.request_timeout, Duration::from_secs(7));
    }
}
