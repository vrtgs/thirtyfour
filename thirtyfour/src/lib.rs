//! Thirtyfour is a Selenium / WebDriver library for Rust, for automated website UI testing.
//!
//! It supports the W3C WebDriver v1 spec.
//! Tested with Chrome and Firefox, although any W3C-compatible WebDriver
//! should work.
//!
//! ## Getting Started
//!
//! Check out [The Book](https://stevepryde.github.io/thirtyfour/) 📚!
//!
//! ## Features
//!
//! - All W3C WebDriver and WebElement methods supported
//! - Async / await support (tokio only)
//! - Create new browser session directly via WebDriver (e.g. chromedriver)
//! - Create new browser session via Selenium Standalone or Grid
//! - Find elements (via all common selectors e.g. Id, Class, CSS, Tag, XPath)
//! - Send keys to elements, including key-combinations
//! - Execute Javascript
//! - Action Chains
//! - Get and set cookies
//! - Switch to frame/window/element/alert
//! - Shadow DOM support
//! - Alert support
//! - Capture / Save screenshot of browser or individual element as PNG
//! - Chrome DevTools Protocol (CDP) — typed commands via [`WebDriver::cdp`]
//!   (feature `cdp`, on by default), plus optional WebSocket-based event
//!   subscription via the `cdp-events` feature. See the [`cdp`] module for
//!   details.
//! - Powerful query interface (the recommended way to find elements) with explicit waits and various predicates
//! - Component Wrappers (similar to `Page Object Model`)
//! - Error- and unwind-aware browser-test cleanup via [`testing::run_browser_test`]
//!
//! ## Feature Flags
//!
//! * `rustls-tls`: (Default) Use rustls to provide TLS support (via reqwest).
//! * `native-tls`: Use native TLS (via reqwest).
//! * `component`: (Default) Enable the `Component` derive macro
//!   (via thirtyfour-macros) and export [`prelude::Component`] /
//!   [`prelude::ElementResolver`] from [`prelude`].
//! * `cdp`: (Default) Typed Chrome DevTools Protocol commands via
//!   [`WebDriver::cdp`] / [`WebElement::cdp_remote_object_id`] /
//!   [`WebElement::cdp_backend_node_id`].
//! * `cdp-events`: WebSocket-backed CDP event subscription. Enables
//!   [`cdp::CdpSession`] and pulls in `tokio-tungstenite`. Off by default.
//! * `bidi`: WebDriver BiDi (W3C bidirectional protocol) — typed commands
//!   and event subscription via [`WebDriver::bidi`]. Pulls in
//!   `tokio-tungstenite`. Off by default.
//!
//! ## Example
//!
//! The following example assumes you have a compatible version of Chrome
//! installed. [`WebDriver::managed`] auto-downloads the matching
//! `chromedriver`, starts it locally, and shuts it down when the session is
//! dropped.
//!
//! ```no_run
//! use thirtyfour::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let caps = DesiredCapabilities::chrome();
//!     let driver = WebDriver::managed(caps).await?;
//!
//!     // Navigate to https://wikipedia.org.
//!     driver.goto("https://wikipedia.org").await?;
//!     let elem_form = driver
//!         .query(By::Id("search-form"))
//!         .desc("Wikipedia search form")
//!         .single()
//!         .await?;
//!
//!     // Querying from an element scopes the search to its subtree.
//!     let elem_text = elem_form
//!         .query(By::Id("searchInput"))
//!         .desc("Wikipedia search input")
//!         .single()
//!         .await?;
//!
//!     // Type in the search terms.
//!     elem_text.send_keys("selenium").await?;
//!
//!     // Click the search button.
//!     let elem_button = elem_form
//!         .query(By::Css("button[type='submit']"))
//!         .desc("Wikipedia search button")
//!         .single()
//!         .await?;
//!     elem_button.click().await?;
//!
//!     // Wait for the unique article heading before checking the title.
//!     driver
//!         .query(By::ClassName("firstHeading"))
//!         .desc("Wikipedia article heading")
//!         .single()
//!         .await?;
//!     assert_eq!(driver.title().await?, "Selenium - Wikipedia");
//!
//!     // explicitly close the browser.
//!     driver.quit().await?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ### The browser will not close automatically
//!
//! Rust does not have [async destructors](https://boats.gitlab.io/blog/post/poll-drop/),
//! and so whenever you forget to use [`WebDriver::quit`] the webdriver will have to block the executor
//! to drop itself and will also ignore errors while dropping, so if you know when a webdriver is no longer used
//! it is recommended to more or less "asynchronously drop" via a call to [`WebDriver::quit`] as in the above example.
//! This also allows you to catch errors during quitting, and possibly panic or report back to the user
//!
//! If you do not call [`WebDriver::quit`] **your async executor will be blocked** meaning no futures can run
//! while quitting. The synchronous fallback also emits a `tracing` warning so you can spot it in logs.
//!
//! ### Element queries and explicit waits
//!
//! [`WebDriver::query`] is the recommended way to find elements. It polls
//! until the element appears, supports filtering and chained alternatives,
//! and produces clearer error messages when nothing matches. Custom filter
//! functions are also supported.
//!
//! The [`WebElement::wait_until`] method provides explicit waits using a
//! variety of built-in predicates, plus an escape hatch for custom predicates.
//!
//! See the [`query`] documentation for more details and examples.
//!
//! [`WebDriver::query`]: extensions::query::ElementQueryable::query
//! [`WebElement::wait_until`]: extensions::query::ElementWaitable::wait_until
//! [`query`]: extensions::query
//!
//! ### Prefer stable selectors
//!
//! When you control the application under test, give important elements stable
//! `data-testid` hooks and select them with [`By::Testid`]:
//!
//! ```no_run
//! # use thirtyfour::prelude::*;
//! # async fn example(driver: &WebDriver) -> WebDriverResult<()> {
//! let save_button = driver
//!     .query(By::Testid("settings-save"))
//!     .desc("settings save button")
//!     .single()
//!     .await?;
//! # Ok(())
//! # }
//! ```
//!
//! Prefer app-owned test IDs first, then stable semantic CSS selectors. Match
//! visible text when the copy itself is part of the behavior under test, and
//! use XPath only when CSS cannot reasonably express the target. Examples
//! against third-party sites must use the stable selectors those sites expose.
//!
//! ### Components
//!
//! Components allow you to wrap a web component using smart element resolvers that can
//! automatically re-query stale elements, and much more.
//! The default `component` feature makes [`prelude::Component`] and
//! [`prelude::ElementResolver`] available from [`prelude`].
//!
//! ```ignore
//! #[derive(Debug, Clone, Component)]
//! pub struct CheckboxComponent {
//!     base: WebElement,
//!     #[by(tag = "label", first)]
//!     label: ElementResolver<WebElement>,
//!     #[by(css = "input[type='checkbox']")]
//!     input: ElementResolver<WebElement>,
//! }
//!
//! impl CheckBoxComponent {
//!     pub async fn label_text(&self) -> WebDriverResult<String> {
//!         let elem = self.label.resolve().await?;
//!         elem.text().await
//!     }
//!
//!     pub async fn is_ticked(&self) -> WebDriverResult<bool> {
//!         let elem = self.input.resolve().await?;
//!         let prop = elem.prop("checked").await?;
//!         Ok(prop.unwrap_or_default() == "true")
//!     }
//!
//!     pub async fn tick(&self) -> WebDriverResult<()> {
//!         if !self.is_ticked().await? {
//!             let elem = self.input.resolve().await?;
//!             elem.click().await?;
//!             assert!(self.is_ticked().await?);
//!         }
//!         Ok(())
//!     }
//! }
//! ```
//!
//! See the [`components`] documentation for more details.
//!

#![deny(missing_docs)]
#![allow(unknown_lints)]
#![warn(missing_debug_implementations, rustdoc::all)]
#![allow(clippy::needless_doctest_main)]

// Re-export StringMatch if needed.
pub use stringmatch;

// Export types at root level.
pub use common::cookie;
pub use common::{
    capabilities::{
        chrome::ChromeCapabilities,
        chromium::{ChromiumCapabilities, ChromiumLikeCapabilities},
        desiredcapabilities::*,
        edge::EdgeCapabilities,
        firefox::FirefoxCapabilities,
        ie::InternetExplorerCapabilities,
        opera::OperaCapabilities,
        safari::SafariCapabilities,
    },
    command::By,
    cookie::*,
    keys::*,
    log::{BrowserLogEntry, LoggingPrefsLogLevel},
    requestdata::*,
    types::*,
};
pub use web_driver::{WebDriver, WebDriverBuilder, WebDriverRunError};
pub use web_element::WebElement;

/// Allow importing the common types via `use thirtyfour::prelude::*`.
pub mod prelude {
    pub use crate::WebDriver;
    pub use crate::WebElement;
    #[cfg(feature = "component")]
    pub use crate::components::{Component, ElementResolver};
    pub use crate::error::{WebDriverError, WebDriverResult};
    pub use crate::extensions::query::{ElementPoller, ElementQueryable, ElementWaitable};
    pub use crate::session::scriptret::ScriptRet;
    pub use crate::{
        BrowserCapabilitiesHelper, By, Capabilities, CapabilitiesHelper, ChromiumLikeCapabilities,
        DesiredCapabilities,
    };
    pub use crate::{Cookie, Key, SameSite, TimeoutConfiguration, TypingData, WindowHandle};
}

/// Action chains allow for more complex user interactions with the keyboard and mouse.
pub mod action_chain;
/// Alert handling.
pub mod alert;
/// WebDriver BiDi (W3C bidirectional protocol) support — typed commands and
/// event subscription over a WebSocket. Negotiated via the `webSocketUrl`
/// capability. See [`bidi`] for the API.
#[cfg(feature = "bidi")]
pub mod bidi;
/// Chrome DevTools Protocol (CDP) support — typed commands, optional event
/// subscription via WebSocket. See [`cdp`] for the API.
#[cfg(feature = "cdp")]
pub mod cdp;
/// Common wrappers used by both async and sync implementations.
pub mod common;
/// Components and component wrappers.
pub mod components;
/// Error wrappers.
pub mod error;
/// Extensions for specific browsers.
pub mod extensions;
/// Auto-download and lifetime-managed local WebDriver process management.
///
/// Available only with the default `manager` feature. See the module docs for
/// details and examples.
#[cfg(feature = "manager")]
pub mod manager;
/// Everything related to driving the underlying WebDriver session.
pub mod session;
/// Miscellaneous support functions for `thirtyfour` tests.
pub mod support;
/// Helpers for running reliable browser tests.
pub mod testing;

mod js;
mod switch_to;
mod web_driver;
mod web_element;

const VERSION: &str = env!("CARGO_PKG_VERSION");
