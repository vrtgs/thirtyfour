/// Code for starting a new session.
pub mod create;
/// The underlying session handle.
pub mod handle;
/// HTTP helpers for WebDriver commands.
pub mod http;
/// Helper for values returned from scripts.
pub mod scriptret;

/// Marker trait for an opaque value whose `Drop` impl tears down some external
/// resource that a [`SessionHandle`] depends on (e.g. a locally-spawned
/// `chromedriver` subprocess managed by [`crate::manager::WebDriverManager`]).
///
/// Implementations are stored on the session handle inside an
/// `Option<Arc<dyn DriverGuard>>` and are dropped *after* the session has been
/// quit, so any external resources outlive the `DELETE /session` HTTP call.
///
/// [`SessionHandle`]: handle::SessionHandle
pub trait DriverGuard: Send + Sync + std::fmt::Debug + 'static {
    /// Asynchronously release the external resource after the session has
    /// ended. Implementations must be idempotent because concurrent clones can
    /// call [`SessionHandle::quit`](handle::SessionHandle::quit) at the same
    /// time.
    ///
    /// The default does nothing. Resource-owning guards should retain a
    /// synchronous `Drop` fallback for callers that do not explicitly await
    /// session cleanup.
    #[doc(hidden)]
    fn release(
        &self,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = crate::error::WebDriverResult<()>> + Send + '_>,
    > {
        Box::pin(async { Ok(()) })
    }

    /// Downcast helper used by [`crate::WebDriver::driver_id`] and friends to
    /// reach concrete guard types (e.g. the manager's `SessionGuard`). Default
    /// returns a placeholder; types with no useful runtime info don't need to
    /// override.
    fn as_any(&self) -> &dyn std::any::Any {
        &()
    }
}
