use crate::error::{WebDriverError, WebDriverResult};
use crate::{WebDriver, WebDriverRunError};
use std::future::{Future, IntoFuture};

/// An error produced while setting up, running, or cleaning up a browser test.
///
/// Body and cleanup failures are retained together rather than allowing the
/// cleanup failure to hide the original test failure. The body error type is
/// generic so callers can use an application-specific error; it defaults to
/// [`WebDriverError`] for the common `WebDriverResult` test body.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum BrowserTestError<E = WebDriverError> {
    /// The future supplied to [`run_browser_test`] failed to create a session.
    #[error("failed to start browser session: {0}")]
    Setup(WebDriverError),
    /// The test body failed and browser cleanup succeeded.
    #[error("browser test failed: {0}")]
    Body(E),
    /// The test body succeeded but closing the browser failed.
    #[error("browser cleanup failed: {0}")]
    Cleanup(WebDriverError),
    /// Both the test body and browser cleanup failed.
    #[error("browser test failed: {body}; browser cleanup also failed: {cleanup}")]
    BodyAndCleanup {
        /// The original test-body failure.
        body: E,
        /// The failure returned by [`WebDriver::quit`].
        cleanup: WebDriverError,
    },
}

impl<E> From<WebDriverError> for BrowserTestError<E> {
    fn from(error: WebDriverError) -> Self {
        Self::Setup(error)
    }
}

impl<E> BrowserTestError<E> {
    /// Return the test-body error, if the body failed.
    pub fn body_error(&self) -> Option<&E> {
        match self {
            Self::Body(error)
            | Self::BodyAndCleanup {
                body: error,
                ..
            } => Some(error),
            Self::Setup(_) | Self::Cleanup(_) => None,
        }
    }

    /// Return the cleanup error, if closing the browser failed.
    pub fn cleanup_error(&self) -> Option<&WebDriverError> {
        match self {
            Self::Cleanup(error)
            | Self::BodyAndCleanup {
                cleanup: error,
                ..
            } => Some(error),
            Self::Setup(_) | Self::Body(_) => None,
        }
    }
}

/// Run an async browser test and always attempt asynchronous session cleanup.
///
/// `session` can be `WebDriver::managed(...)` when the `manager` feature is
/// enabled, [`WebDriver::builder`], [`WebDriver::new`], or any other future
/// whose output is a `WebDriverResult` containing a [`WebDriver`]. The runner
/// retains its own session handle while passing a clone into `test`, so
/// consuming the body handle cannot skip the final [`WebDriver::quit`] call.
///
/// # Error precedence
///
/// - A session-creation error returns [`BrowserTestError::Setup`].
/// - A body error with successful cleanup returns [`BrowserTestError::Body`].
/// - A successful body with a cleanup error returns
///   [`BrowserTestError::Cleanup`].
/// - If both fail, [`BrowserTestError::BodyAndCleanup`] preserves both errors.
///
/// # Panics
///
/// Panics from invoking `test` or polling its returned future are caught long
/// enough to await `WebDriver::quit`, then resumed with their original payload.
/// If cleanup also fails, that error is emitted through `tracing` before the
/// original panic resumes. A process compiled with `panic = "abort"` cannot run
/// cleanup after a panic.
///
/// The runner future must be driven to completion. Dropping it or aborting its
/// task can interrupt the body or cleanup future; Rust does not provide
/// asynchronous cancellation destructors.
///
/// [`WebDriver::builder`]: crate::WebDriver::builder
/// [`WebDriver::new`]: crate::WebDriver::new
/// [`WebDriver::quit`]: crate::WebDriver::quit
pub async fn run_browser_test<S, F, Fut, T, E>(
    session: S,
    test: F,
) -> Result<T, BrowserTestError<E>>
where
    S: IntoFuture<Output = WebDriverResult<WebDriver>>,
    F: FnOnce(WebDriver) -> Fut,
    Fut: Future<Output = Result<T, E>>,
{
    let driver = session.into_future().await.map_err(BrowserTestError::Setup)?;
    match driver.run_and_quit(test).await {
        Ok(value) => Ok(value),
        Err(WebDriverRunError::Body(body)) => Err(BrowserTestError::Body(body)),
        Err(WebDriverRunError::Cleanup(cleanup)) => Err(BrowserTestError::Cleanup(cleanup)),
        Err(WebDriverRunError::BodyAndCleanup {
            body,
            cleanup,
        }) => Err(BrowserTestError::BodyAndCleanup {
            body,
            cleanup,
        }),
    }
}

#[cfg(test)]
mod tests {
    use futures_util::FutureExt;
    use std::fmt::{Display, Formatter};
    use std::panic::AssertUnwindSafe;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use bytes::Bytes;
    use http::{Request, Response};

    use super::*;
    use crate::common::config::WebDriverConfig;
    use crate::session::handle::SessionHandle;
    use crate::session::http::{Body, HttpClient};
    use crate::{Capabilities, SessionId};

    #[derive(Debug, PartialEq, Eq)]
    struct BodyError(&'static str);

    impl Display for BodyError {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.write_str(self.0)
        }
    }

    impl std::error::Error for BodyError {}

    #[derive(Debug, Default)]
    struct ClientState {
        delete_attempts: AtomicUsize,
        fail_first_delete: AtomicBool,
        panic_first_delete: AtomicBool,
    }

    #[derive(Debug, Clone)]
    struct RecordingClient(Arc<ClientState>);

    #[async_trait::async_trait]
    impl HttpClient for RecordingClient {
        async fn send(&self, request: Request<Body<'_>>) -> WebDriverResult<Response<Bytes>> {
            assert_eq!(request.method(), http::Method::DELETE);
            assert_eq!(request.uri().path(), "/session/test-session");
            self.0.delete_attempts.fetch_add(1, Ordering::SeqCst);

            if self.0.panic_first_delete.swap(false, Ordering::SeqCst) {
                panic!("cleanup panic");
            }

            if self.0.fail_first_delete.swap(false, Ordering::SeqCst) {
                return Err(WebDriverError::RequestFailed("cleanup failed".to_string()));
            }

            Ok(Response::builder()
                .status(200)
                .body(Bytes::from_static(br#"{"value":null}"#))
                .expect("valid response"))
        }

        async fn new(&self) -> Arc<dyn HttpClient> {
            Arc::new(self.clone())
        }
    }

    fn test_driver(fail_first_delete: bool) -> (WebDriver, Arc<ClientState>) {
        test_driver_with_cleanup(fail_first_delete, false)
    }

    fn test_driver_with_cleanup(
        fail_first_delete: bool,
        panic_first_delete: bool,
    ) -> (WebDriver, Arc<ClientState>) {
        test_driver_with_guard(fail_first_delete, panic_first_delete, None)
    }

    fn test_driver_with_guard(
        fail_first_delete: bool,
        panic_first_delete: bool,
        driver_guard: Option<Arc<dyn crate::session::DriverGuard>>,
    ) -> (WebDriver, Arc<ClientState>) {
        let state = Arc::new(ClientState {
            delete_attempts: AtomicUsize::new(0),
            fail_first_delete: AtomicBool::new(fail_first_delete),
            panic_first_delete: AtomicBool::new(panic_first_delete),
        });
        let client: Arc<dyn HttpClient> = Arc::new(RecordingClient(Arc::clone(&state)));
        let handle = SessionHandle::new_with_config_guard_and_caps(
            client,
            "http://localhost:4444",
            SessionId::from("test-session"),
            WebDriverConfig::default(),
            driver_guard,
            Capabilities::new(),
        )
        .expect("valid test session");

        (
            WebDriver {
                handle: Arc::new(handle),
            },
            state,
        )
    }

    #[tokio::test]
    async fn returns_body_value_after_successful_cleanup() {
        let (driver, state) = test_driver(false);

        let result = driver.run_and_quit(|_| async { Ok::<_, BodyError>(42) }).await;

        assert_eq!(result.expect("operation succeeds"), 42);
        assert_eq!(state.delete_attempts.load(Ordering::SeqCst), 1);
    }

    #[derive(Debug)]
    struct RecordingGuard(Arc<AtomicUsize>);

    impl crate::session::DriverGuard for RecordingGuard {
        fn release(
            &self,
        ) -> std::pin::Pin<Box<dyn Future<Output = WebDriverResult<()>> + Send + '_>> {
            Box::pin(async move {
                tokio::task::yield_now().await;
                self.0.fetch_add(1, Ordering::SeqCst);
                Ok(())
            })
        }
    }

    #[tokio::test]
    async fn run_and_quit_awaits_driver_guard_release() {
        let releases = Arc::new(AtomicUsize::new(0));
        let guard: Arc<dyn crate::session::DriverGuard> =
            Arc::new(RecordingGuard(Arc::clone(&releases)));
        let (driver, state) = test_driver_with_guard(false, false, Some(guard));

        let value = driver
            .run_and_quit(|_| async { Ok::<_, BodyError>(42) })
            .await
            .expect("operation and cleanup succeed");

        assert_eq!(value, 42);
        assert_eq!(state.delete_attempts.load(Ordering::SeqCst), 1);
        assert_eq!(releases.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn setup_error_does_not_invoke_body() {
        let body_invoked = Arc::new(AtomicBool::new(false));
        let invoked = Arc::clone(&body_invoked);

        let result = run_browser_test(
            async {
                Err::<WebDriver, _>(WebDriverError::SessionCreateError("setup failed".to_string()))
            },
            move |_| {
                invoked.store(true, Ordering::SeqCst);
                async { Ok::<(), BodyError>(()) }
            },
        )
        .await;

        assert!(matches!(result, Err(BrowserTestError::Setup(_))));
        assert!(!body_invoked.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn body_error_is_returned_after_successful_cleanup() {
        let (driver, state) = test_driver(false);

        let result = run_browser_test(async { Ok(driver) }, |_| async {
            Err::<(), _>(BodyError("body failed"))
        })
        .await;

        assert!(matches!(result, Err(BrowserTestError::Body(BodyError("body failed")))));
        assert_eq!(state.delete_attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cleanup_error_is_returned_after_successful_body() {
        let (driver, state) = test_driver(true);

        let result =
            run_browser_test(async { Ok(driver) }, |_| async { Ok::<_, BodyError>(()) }).await;

        assert!(matches!(result, Err(BrowserTestError::Cleanup(_))));
        assert!(state.delete_attempts.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn body_and_cleanup_errors_are_both_retained() {
        let (driver, state) = test_driver(true);

        let result = run_browser_test(async { Ok(driver) }, |_| async {
            Err::<(), _>(BodyError("body failed"))
        })
        .await;

        let error = result.expect_err("both operations fail");
        assert_eq!(error.body_error(), Some(&BodyError("body failed")));
        assert!(error.cleanup_error().is_some());
        assert!(matches!(error, BrowserTestError::BodyAndCleanup { .. }));
        assert!(state.delete_attempts.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn cleanup_runs_when_invoking_body_panics() {
        let (driver, state) = test_driver(false);

        let result = AssertUnwindSafe(run_browser_test(
            async { Ok(driver) },
            |_| -> std::future::Ready<Result<(), BodyError>> { panic!("body invocation panic") },
        ))
        .catch_unwind()
        .await;

        assert!(result.is_err());
        assert_eq!(state.delete_attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cleanup_runs_when_polling_body_panics() {
        let (driver, state) = test_driver(false);

        let result = AssertUnwindSafe(run_browser_test(async { Ok(driver) }, |_| async {
            panic!("body future panic");
            #[allow(unreachable_code)]
            Ok::<(), BodyError>(())
        }))
        .catch_unwind()
        .await;

        assert!(result.is_err());
        assert_eq!(state.delete_attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn cleanup_failure_does_not_replace_body_panic() {
        let (driver, state) = test_driver(true);

        let result = AssertUnwindSafe(run_browser_test(async { Ok(driver) }, |_| async {
            panic!("original body panic");
            #[allow(unreachable_code)]
            Ok::<(), BodyError>(())
        }))
        .catch_unwind()
        .await;

        let panic = result.expect_err("the original panic resumes");
        assert_eq!(panic.downcast_ref::<&str>(), Some(&"original body panic"));
        assert!(state.delete_attempts.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn cleanup_panic_does_not_replace_body_panic() {
        let (driver, state) = test_driver_with_cleanup(false, true);

        let result = AssertUnwindSafe(run_browser_test(async { Ok(driver) }, |_| async {
            panic!("original body panic");
            #[allow(unreachable_code)]
            Ok::<(), BodyError>(())
        }))
        .catch_unwind()
        .await;

        let panic = result.expect_err("the original panic resumes");
        assert_eq!(panic.downcast_ref::<&str>(), Some(&"original body panic"));
        assert!(state.delete_attempts.load(Ordering::SeqCst) >= 1);
    }
}
