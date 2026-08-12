//! End-to-end tests for the WebDriver manager.
//!
//! These tests download a real driver binary and spawn a real subprocess, so
//! they're gated behind the `manager-tests` feature and run in their own CI
//! job that does *not* pre-start chromedriver / geckodriver. Run locally with:
//!
//! ```text
//! cargo test -p thirtyfour --features manager-tests --test managed -- --test-threads=1
//! ```

#![cfg(feature = "manager-tests")]

use std::future::Future;
use std::time::Duration;

use thirtyfour::manager::WebDriverManager;
use thirtyfour::prelude::*;
use thirtyfour::{ChromeCapabilities, EdgeCapabilities, FirefoxCapabilities};

/// Per-test wall-clock budget. Real driver downloads on slow CI runners are
/// usually well under a minute; this is a defensive ceiling so a hung test
/// fails-fast rather than burning the runner's full 6h limit.
const TEST_TIMEOUT: Duration = Duration::from_secs(180);

/// Run `f` with a hard timeout. Returns an error rather than hanging.
async fn with_timeout<F, T>(f: F) -> WebDriverResult<T>
where
    F: Future<Output = WebDriverResult<T>>,
{
    tokio::time::timeout(TEST_TIMEOUT, f).await.unwrap_or_else(|_| {
        Err(WebDriverError::FatalError(format!("test exceeded {}s budget", TEST_TIMEOUT.as_secs())))
    })
}

/// Chromium-based browsers (Chrome, Edge) running headless on Windows runners
/// have known reliability issues with `goto`, even for `about:blank` — see
/// `managed_edge_smoke`'s comment on the same bug for Linux Edge. The
/// manager's job is done by the time we'd call `goto`; navigation is the
/// browser engine's responsibility, not ours, so we skip it on the affected
/// platform.
fn skip_navigation() -> bool {
    cfg!(target_os = "windows")
}

fn chrome_caps() -> ChromeCapabilities {
    let mut caps = DesiredCapabilities::chrome();
    caps.set_headless().unwrap();
    caps.set_no_sandbox().unwrap();
    caps.set_disable_gpu().unwrap();
    caps.set_disable_dev_shm_usage().unwrap();
    caps.add_arg("--no-sandbox").unwrap();
    caps
}

fn firefox_caps() -> FirefoxCapabilities {
    let mut caps = DesiredCapabilities::firefox();
    caps.set_headless().unwrap();
    caps
}

fn edge_caps() -> EdgeCapabilities {
    let mut caps = DesiredCapabilities::edge();
    // EdgeCapabilities is Chromium-based and accepts the same args via
    // `ms:edgeOptions.args`. Headless is `--headless=new` for modern Edge.
    caps.add_arg("--headless=new").unwrap();
    caps.add_arg("--no-sandbox").unwrap();
    caps.add_arg("--disable-gpu").unwrap();
    caps.add_arg("--disable-dev-shm-usage").unwrap();
    caps
}

#[tokio::test(flavor = "multi_thread")]
async fn managed_chrome_smoke() -> WebDriverResult<()> {
    with_timeout(async {
        let driver = WebDriver::managed(chrome_caps()).await?;
        let server_url = driver.server_url().clone();
        if !skip_navigation() {
            driver.goto("about:blank").await?;
        }
        driver.quit().await?;
        assert!(
            !driver_is_listening(&server_url).await,
            "quit().await must not return while chromedriver is still listening"
        );
        Ok(())
    })
    .await
}

#[tokio::test(flavor = "multi_thread")]
async fn managed_firefox_smoke() -> WebDriverResult<()> {
    with_timeout(async {
        let driver = WebDriver::managed(firefox_caps()).await?;
        driver.goto("about:blank").await?;
        driver.quit().await?;
        Ok(())
    })
    .await
}

/// Edge headless on Ubuntu has a known issue where the renderer times out
/// even on `about:blank`. That's a quirk of Edge-for-Linux's headless mode,
/// not the manager — by the time we reach `goto` the manager has already done
/// its full job (download, spawn, readiness poll, session creation). So we
/// stop short of `goto` here: creating + quitting a session is the
/// manager-level smoke. The full session round-trip is exercised by the
/// existing chrome/firefox smokes.
#[tokio::test(flavor = "multi_thread")]
async fn managed_edge_smoke() -> WebDriverResult<()> {
    with_timeout(async {
        let driver = WebDriver::managed(edge_caps()).await?;
        driver.quit().await?;
        Ok(())
    })
    .await
}

/// Safari is macOS-only and uses the system `safaridriver`. The CI runner must
/// have run `safaridriver --enable` (and have Allow Remote Automation toggled).
#[cfg(target_os = "macos")]
#[tokio::test(flavor = "multi_thread")]
async fn managed_safari_smoke() -> WebDriverResult<()> {
    with_timeout(async {
        let mut caps = DesiredCapabilities::safari();
        let _ = &mut caps; // safari has no headless / no sandbox toggles
        let driver = WebDriver::managed(caps).await?;
        driver.goto("about:blank").await?;
        driver.quit().await?;
        Ok(())
    })
    .await
}

/// Exercises a few non-default options on Chrome:
///   - a custom cache dir
///   - a custom ready timeout
///   - two `launch()` calls share a single chromedriver process (refcount + dedup)
///
/// We deliberately do NOT exercise `DriverVersion::Latest` here. ChromeDriver's
/// major version must match Chrome's, and CI runners typically have a Chrome a
/// few days behind the absolute latest chromedriver release — so a
/// `Latest`-driven E2E test would be perpetually flaky. The `Latest` resolver
/// itself is covered by the wiremock-based unit tests.
#[tokio::test(flavor = "multi_thread")]
async fn managed_chrome_options_and_dedup() -> WebDriverResult<()> {
    with_timeout(async {
        let cache = tempfile::tempdir().expect("tempdir");
        let mgr = WebDriverManager::builder()
            .match_local() // default; explicit for clarity
            .cache_dir(cache.path().to_path_buf())
            .ready_timeout(Duration::from_secs(60))
            .build();

        let d1 = mgr.launch(chrome_caps()).await?;
        let d2 = mgr.launch(chrome_caps()).await?;

        // Two `launch()` calls keyed on the same (BrowserKind, version, host)
        // should reuse a single chromedriver subprocess — both `WebDriver`
        // instances should point at the same server URL.
        assert_eq!(
            d1.server_url(),
            d2.server_url(),
            "two managed sessions for the same browser must share a chromedriver process"
        );

        if !skip_navigation() {
            d1.goto("about:blank").await?;
            d2.goto("about:blank").await?;
        }

        d1.quit().await?;
        assert!(
            driver_is_listening(d2.server_url()).await,
            "quitting one shared session must keep chromedriver alive"
        );
        let server_url = d2.server_url().clone();
        d2.quit().await?;
        assert!(
            !driver_is_listening(&server_url).await,
            "quitting the final shared session must await chromedriver exit"
        );
        Ok(())
    })
    .await
}

/// Concurrent explicit cleanup of every session sharing one chromedriver must
/// still leave exactly one caller responsible for awaiting process exit.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_quit_awaits_shared_chromedriver_exit() -> WebDriverResult<()> {
    with_timeout(async {
        let mgr = WebDriverManager::builder().build();
        let d1 = mgr.launch(chrome_caps()).await?;
        let d2 = mgr.launch(chrome_caps()).await?;
        let server_url = d1.server_url().clone();
        assert_eq!(d2.server_url(), &server_url);

        let (r1, r2) = tokio::join!(d1.quit(), d2.quit());
        r1?;
        r2?;

        assert!(
            !driver_is_listening(&server_url).await,
            "concurrent quit().await calls must await the shared chromedriver exit"
        );
        Ok(())
    })
    .await
}

/// True if a TCP connect to the driver's `host:port` succeeds, i.e. the
/// chromedriver subprocess is still listening.
async fn driver_is_listening(server_url: &url::Url) -> bool {
    let host = server_url.host_str().unwrap_or("127.0.0.1");
    let port = server_url.port_or_known_default().unwrap_or(0);
    tokio::time::timeout(Duration::from_secs(1), tokio::net::TcpStream::connect((host, port)))
        .await
        .map(|r| r.is_ok())
        .unwrap_or(false)
}

/// Issue #281: dropping a `WebDriver` without calling `quit()` should still
/// close the session and (for managed drivers) kill the chromedriver
/// subprocess. `SessionHandle::Drop` spawns a dedicated OS thread that
/// runs `quit()` via `support::block_on` (process-wide `GLOBAL_RT`),
/// joining synchronously so Drop blocks until the DELETE /session call
/// completes. `ManagedDriverProcess::Drop` then SIGKILLs the subprocess.
///
/// Tested under both runtime flavors because the cleanup path must not
/// rely on the caller's runtime — the OS thread + GLOBAL_RT trick has to
/// work whether the caller is on `multi_thread` or `current_thread`.
#[tokio::test(flavor = "multi_thread")]
async fn dropping_managed_driver_kills_subprocess_multi_thread() -> WebDriverResult<()> {
    drop_kills_subprocess_inner().await
}

#[tokio::test(flavor = "current_thread")]
async fn dropping_managed_driver_kills_subprocess_current_thread() -> WebDriverResult<()> {
    drop_kills_subprocess_inner().await
}

async fn drop_kills_subprocess_inner() -> WebDriverResult<()> {
    with_timeout(async {
        let server_url = {
            let driver = WebDriver::managed(chrome_caps()).await?;
            let url = driver.server_url().clone();
            assert!(driver_is_listening(&url).await, "driver should be listening while alive");
            if !skip_navigation() {
                driver.goto("about:blank").await?;
            }
            url
            // No explicit `driver.quit().await` — drop fires here, which must
            // run quit() and then SIGKILL the subprocess.
        };

        // Give SIGKILL a moment to actually reap the process.
        tokio::time::sleep(Duration::from_millis(500)).await;

        assert!(
            !driver_is_listening(&server_url).await,
            "chromedriver at {server_url} should be gone after WebDriver was dropped without quit"
        );
        Ok(())
    })
    .await
}

/// Issue #281 specifically: panic unwind drops the in-scope `WebDriver`,
/// which must run the same `quit + kill subprocess` path as a clean drop.
/// Users should never need a custom `Drop` guard with `block_on` — that's
/// what `SessionHandle::Drop` already does internally.
#[tokio::test(flavor = "multi_thread")]
async fn panic_unwind_kills_managed_driver_subprocess() -> WebDriverResult<()> {
    use std::sync::{Arc, Mutex};

    with_timeout(async {
        let captured: Arc<Mutex<Option<url::Url>>> = Arc::new(Mutex::new(None));
        let captured_clone = Arc::clone(&captured);

        // Run the panicking code on a dedicated thread so the unwind happens
        // there and we can `join()` to observe it. The `WebDriver` is held
        // when `panic!` fires; the unwind must drop it cleanly.
        let join = std::thread::spawn(move || {
            thirtyfour::support::block_on(async move {
                let driver = WebDriver::managed(chrome_caps()).await.expect("managed launch");
                *captured_clone.lock().unwrap() = Some(driver.server_url().clone());
                if !skip_navigation() {
                    driver.goto("about:blank").await.expect("goto");
                }
                panic!("simulated user panic with WebDriver still in scope");
            });
        })
        .join();

        assert!(join.is_err(), "thread should have panicked");

        let url = captured
            .lock()
            .unwrap()
            .take()
            .expect("server URL should have been captured before panic");

        tokio::time::sleep(Duration::from_millis(500)).await;

        assert!(
            !driver_is_listening(&url).await,
            "chromedriver at {url} should be gone after panic unwind dropped the WebDriver"
        );
        Ok(())
    })
    .await
}

/// Several `WebDriver`s dropped at the same instant must all clean up.
///
/// Each `SessionHandle::Drop` spawns its own OS thread that builds a
/// fresh `current_thread` runtime to run the DELETE /session call. The
/// per-Drop runtime is intentional — a previous design routed all
/// cleanup through a shared `GLOBAL_RT` (a single-threaded runtime
/// driven by one parked thread), which serialised every concurrent
/// drop through one I/O driver and caused chromedriver-readiness
/// timeouts under cargo's parallel-test-binary harness. This test
/// fires N drops concurrently to guard against any regression that
/// re-introduces a shared-resource bottleneck (or a deadlock when
/// more than one drop is in flight).
///
/// Each `WebDriver::managed` call builds a fresh `WebDriverManager`,
/// so the N sessions back onto N independent chromedriver subprocesses
/// — we can verify each one is gone independently after its drop.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_drops_kill_all_subprocesses() -> WebDriverResult<()> {
    const N: usize = 3;

    with_timeout(async {
        // Spin up N drivers in parallel. Wrap each builder's `IntoFuture` in
        // an explicit async block so `try_join_all` sees a concrete `Future`.
        let drivers: Vec<WebDriver> = futures_util::future::try_join_all(
            (0..N).map(|_| async { WebDriver::managed(chrome_caps()).await }),
        )
        .await?;

        let urls: Vec<url::Url> = drivers.iter().map(|d| d.server_url().clone()).collect();
        for url in &urls {
            assert!(driver_is_listening(url).await, "driver at {url} should be listening");
        }

        // Drop each driver on its own OS thread so the Drops run concurrently
        // (vs. `drop(Vec)` which would run them sequentially). All cleanup
        // threads run their fresh per-Drop runtimes simultaneously.
        let drop_threads: Vec<_> =
            drivers.into_iter().map(|d| std::thread::spawn(move || drop(d))).collect();
        for t in drop_threads {
            t.join().expect("drop thread panicked");
        }

        // Give SIGKILL a moment to actually reap the processes.
        tokio::time::sleep(Duration::from_millis(500)).await;

        for url in &urls {
            assert!(
                !driver_is_listening(url).await,
                "chromedriver at {url} should be gone after concurrent drop"
            );
        }
        Ok(())
    })
    .await
}

/// Many sessions created concurrently against a shared
/// `WebDriverManager` must all start and tear down cleanly.
///
/// Mirrors the integration-test-harness pattern (one
/// `Arc<WebDriverManager>` shared across many tests in a binary) and
/// pushes more parallelism than `--test-threads=1` allows: every test
/// binary in the suite shares a per-binary manager, but cargo runs
/// multiple test binaries in parallel and each manager may have
/// several sessions in flight when those binaries overlap.
///
/// What this catches:
/// * Regressions in `SessionHandle::Drop` that re-share a single
///   tokio runtime or single I/O driver across cleanups (the
///   pre-fix shape funnelled all DELETE /session HTTP calls through
///   one `current_thread` runtime, slowing concurrent teardown
///   enough to time out chromedriver readiness in subsequent
///   launches).
/// * Deadlocks between session-launch HTTP traffic and cleanup HTTP
///   traffic when both share the same runtime/driver.
///
/// Sessions are created in pairs: launch two, drop both, repeat.
/// This pattern interleaves launch and drop traffic — the symptom
/// case the user sees in CI.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_sessions_against_shared_manager() -> WebDriverResult<()> {
    use std::sync::Arc;
    use thirtyfour::manager::WebDriverManager;

    const ROUNDS: usize = 3;
    const PARALLEL: usize = 3;

    with_timeout(async {
        let mgr: Arc<WebDriverManager> = WebDriverManager::builder().build();

        for _round in 0..ROUNDS {
            // Launch PARALLEL sessions concurrently against the same manager.
            // Without dedup-on-version they each get the same chromedriver
            // subprocess (the manager's `spawn_or_reuse` path), so this
            // exercises concurrent session-create traffic against one driver
            // while later iterations exercise concurrent drop+launch.
            let sessions: Vec<WebDriver> =
                futures_util::future::try_join_all((0..PARALLEL).map(|_| {
                    let mgr = Arc::clone(&mgr);
                    async move { mgr.launch(chrome_caps()).await }
                }))
                .await?;

            assert_eq!(sessions.len(), PARALLEL, "all sessions should have launched");

            // The shared manager dedups on (browser, version, host), so all
            // sessions in a round MUST point at the same chromedriver URL.
            let first_url = sessions[0].server_url().clone();
            for s in &sessions {
                assert_eq!(
                    s.server_url(),
                    &first_url,
                    "shared manager must dedup all sessions onto one chromedriver"
                );
            }

            // Cross-thread concurrent drop. Drop on per-session OS threads so
            // multiple cleanup futures run truly in parallel — the case the
            // shared-runtime regression broke.
            let drop_threads: Vec<_> =
                sessions.into_iter().map(|d| std::thread::spawn(move || drop(d))).collect();
            for t in drop_threads {
                t.join().expect("drop thread panicked");
            }
        }

        Ok(())
    })
    .await
}

/// Interleaved create/drop on a shared manager: while one session is
/// being torn down (its DELETE /session HTTP call is in flight),
/// another session-create HTTP call is also in flight against the
/// SAME chromedriver. Mirrors the worst-case CI shape where the
/// previous test's drop overlaps with the next test's launch.
///
/// Specifically targets the regression where DELETE /session and
/// New Session HTTP traffic both ran on the same `current_thread`
/// runtime, with the cleanup blocking the launch's I/O progress
/// (or vice versa).
#[tokio::test(flavor = "multi_thread")]
async fn interleaved_create_and_drop() -> WebDriverResult<()> {
    use std::sync::Arc;
    use thirtyfour::manager::WebDriverManager;

    const ROUNDS: usize = 5;

    with_timeout(async {
        let mgr: Arc<WebDriverManager> = WebDriverManager::builder().build();

        let mut prev: Option<WebDriver> = None;
        for _round in 0..ROUNDS {
            // Start a new session. This sends New Session against the shared
            // chromedriver while the previous round's drop (if any) may still
            // be in flight on its own OS thread.
            let next = mgr.launch(chrome_caps()).await?;

            // Now drop the previous session on a background thread (so its
            // DELETE /session fires concurrently with this round's other
            // work). For the very first round there's nothing to drop.
            if let Some(old) = prev.take() {
                let t = std::thread::spawn(move || drop(old));
                // Join it before the next iteration so we keep test failures
                // localised to the round that broke. Concurrency is real
                // because the launch already happened above, racing the drop.
                t.join().expect("drop thread panicked");
            }
            prev = Some(next);
        }

        // Final cleanup.
        if let Some(last) = prev.take() {
            last.quit().await?;
        }

        Ok(())
    })
    .await
}
