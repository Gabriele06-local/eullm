//! Generation models on a real GGUF, over HTTP: what a request does to the
//! model that answers it. CPU, one thread, a tiny model copied under several
//! names into a temporary store so that every copy is a distinct model to the
//! server — a hard link would be one file, and one model.
//!
//! ```text
//! EULLM_GENERATION_TEST_MODEL=/path/to/stories260K.gguf \
//!     cargo test -p eullm-engine -- --ignored real_model_ --test-threads=1
//! ```

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::net::TcpListener;

use super::{AppState, auth};
use crate::models::ModelStore;

/// The GGUF every test here runs on.
fn test_model() -> PathBuf {
    std::env::var("EULLM_GENERATION_TEST_MODEL")
        .expect("set EULLM_GENERATION_TEST_MODEL to a GGUF file")
        .into()
}

/// A store holding a copy of the test model under each of `names`, laid out
/// the way a pull leaves one.
fn store_of_copies(dir: &Path, names: &[&str]) -> ModelStore {
    let source = test_model();
    for name in names {
        let model_dir = dir.join(name);
        std::fs::create_dir_all(&model_dir).expect("model dir");
        std::fs::copy(&source, model_dir.join("model.gguf")).expect("copy the test model");
        let manifest = json!({
            "id": name, "name": name, "description": "test copy", "languages": ["en"],
            "base": "test", "vram_gb": 1, "size_bytes": 0, "license": "MIT",
            "digest": "sha256:0", "pulled_at": "2026-10-01T00:00:00Z", "status": "ready",
            "gguf_file": "model.gguf",
        });
        std::fs::write(model_dir.join("manifest.json"), manifest.to_string()).expect("manifest");
    }
    ModelStore::at(dir.to_path_buf())
}

/// A server on 127.0.0.1:0, with its idle-unload loop running, over a store
/// of copies of the test model. The store goes when this does.
struct TestServer {
    base: String,
    state: Arc<AppState>,
    dir: PathBuf,
}

impl Drop for TestServer {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Small and on one thread: a CPU shared with other work, and a model of a
/// few hundred thousand parameters, need no more.
async fn start(names: &[&str], configure: impl FnOnce(&mut AppState)) -> TestServer {
    let dir = std::env::temp_dir().join(format!("eullm-resident-{}", uuid::Uuid::new_v4()));
    let store = store_of_copies(&dir, names);
    let absent = Path::new("/nonexistent/eullm-test/.env");
    let mut state = AppState::for_tests(store, auth::ApiKeys::load(absent).expect("no keys"));
    state.ctx_size = 2048;
    configure(&mut state);
    let state = Arc::new(state);
    tokio::spawn(Arc::clone(&state).run_idle_unload_loop());

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local addr");
    let app = super::api_router(Arc::clone(&state));
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await;
    });
    TestServer {
        base: format!("http://{addr}"),
        state,
        dir,
    }
}

impl TestServer {
    /// The generation models loaded right now, by name, sorted.
    async fn loaded(&self) -> Vec<String> {
        let mut names: Vec<String> = self
            .state
            .models
            .read()
            .await
            .by_recent_use()
            .iter()
            .map(|m| m.name.clone())
            .collect();
        names.sort();
        names
    }

    /// Which load of `name` is resident, to tell a model kept from one
    /// unloaded and loaded again.
    async fn load_id(&self, name: &str) -> Option<u64> {
        self.state.models.read().await.find(name).map(|m| m.id)
    }

    /// Wait up to `limit` for exactly `expected` to be loaded.
    async fn wait_for_loaded(&self, expected: &[&str], limit: Duration) {
        let start = Instant::now();
        loop {
            let loaded = self.loaded().await;
            if loaded == expected {
                return;
            }
            assert!(
                start.elapsed() < limit,
                "expected {expected:?} loaded within {limit:?}, still {loaded:?}"
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
    }

    /// POST `body` to `/api/generate`; the status and every JSON line of the
    /// answer — one for a plain response, one per event for a stream.
    async fn generate(&self, body: Value) -> (reqwest::StatusCode, Vec<Value>) {
        let response = reqwest::Client::new()
            .post(format!("{}/api/generate", self.base))
            .json(&body)
            .send()
            .await
            .expect("request");
        let status = response.status();
        let text = response.text().await.expect("body");
        let lines = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l}")))
            .collect();
        (status, lines)
    }
}

/// The text of a generation's answer, from its lines.
fn answer(lines: &[Value]) -> String {
    lines
        .iter()
        .filter_map(|l| l["response"].as_str())
        .collect()
}

/// The last line of a generation that ended as it should: done, for a reason
/// a generation ends with, and no error anywhere.
fn assert_finished(lines: &[Value]) {
    let errors: Vec<&Value> = lines.iter().filter(|l| l.get("error").is_some()).collect();
    assert!(
        errors.is_empty(),
        "an error after {} lines: {errors:?}",
        lines.len()
    );
    let last = lines.last().expect("at least one line");
    assert_eq!(last["done"], true, "{last}");
    assert!(
        matches!(last["done_reason"].as_str(), Some("stop" | "length")),
        "{last}"
    );
}

/// `keep_alive: 0` with a prompt used to unload the model before the
/// request reached it: the scheduler's thread was gone, and the request
/// failed with "Scheduler queue full". It is answered now, and the model
/// unloaded once the answer is over — streamed or not.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_keep_alive_zero_with_a_prompt_answers_then_unloads() {
    let server = start(&["tiny-a"], |_| {}).await;
    for stream in [false, true] {
        let (status, lines) = server
            .generate(json!({
                "model": "tiny-a", "prompt": "Once upon a time", "keep_alive": 0,
                "stream": stream, "options": { "num_predict": 24 },
            }))
            .await;
        assert_eq!(status, 200, "{lines:?}");
        assert_finished(&lines);
        assert!(!answer(&lines).is_empty(), "{lines:?}");
        server.wait_for_loaded(&[], Duration::from_secs(5)).await;
    }
}

/// keep_alive counts from the end of the answer. It used to count from its
/// start, so an answer longer than keep_alive lost its model halfway: the
/// idle-unload loop shut the scheduler down under it. A short request that
/// finishes beside a long one must not do it either: its deadline passes
/// while the long one is still answering.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_long_generation_outlives_a_short_keep_alive() {
    let tick = Duration::from_millis(25);
    let keep_alive = Duration::from_millis(25);
    let server = start(&["tiny-a"], |state| {
        state.idle_tick = tick;
        state.ctx_size = 8192;
        state.batch_size = 2;
    })
    .await;

    let started = Instant::now();
    let response = reqwest::Client::new()
        .post(format!("{}/api/generate", server.base))
        .json(&json!({
            "model": "tiny-a", "prompt": "Once upon a time",
            "keep_alive": keep_alive.as_secs_f64(), "stream": true,
            "options": { "num_predict": 3000 },
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);

    // Past keep_alive and a few ticks of the loop, for as long as the answer
    // is still coming, the model must still be there.
    let midway = keep_alive + 4 * tick;
    let mut short_one_done = false;
    let mut checks_past_midway = 0;
    let mut pending = String::new();
    let mut lines = Vec::new();
    let mut body = response.bytes_stream();
    while let Some(chunk) = body.next().await {
        pending.push_str(&String::from_utf8_lossy(&chunk.expect("chunk")));
        while let Some(end) = pending.find('\n') {
            let line: String = pending.drain(..=end).collect();
            lines.push(serde_json::from_str::<Value>(line.trim()).expect("a JSON line"));
        }
        if !short_one_done {
            let (status, short) = server
                .generate(json!({
                    "model": "tiny-a", "prompt": "Once", "stream": false,
                    "keep_alive": keep_alive.as_secs_f64(), "options": { "num_predict": 4 },
                }))
                .await;
            assert_eq!(status, 200, "{short:?}");
            assert_finished(&short);
            short_one_done = true;
        }
        let done = lines.last().is_some_and(|l| l["done"] == true);
        if started.elapsed() > midway && !done {
            assert_eq!(
                server.loaded().await,
                ["tiny-a"],
                "unloaded {:?} into an answer that was still coming",
                started.elapsed()
            );
            checks_past_midway += 1;
        }
    }
    assert_finished(&lines);
    assert!(
        checks_past_midway > 10,
        "the answer took {:?}, not long enough to outlast keep_alive; raise num_predict",
        started.elapsed()
    );

    // And once it is over, keep_alive runs out and the model goes.
    server.wait_for_loaded(&[], keep_alive + 10 * tick).await;
}

/// One model at a time, as before: each request for another model replaces
/// the resident one, and coming back loads it again. A request for the
/// resident model under another spelling of its name, or by the path of its
/// file, is answered by it without a reload.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_one_model_swaps_exactly_as_before() {
    let server = start(&["tiny-a", "tiny-b"], |state| {
        state.allow_model_paths = true
    })
    .await;
    let ask = |model: &str| {
        json!({
            "model": model, "prompt": "Once upon a time", "stream": false,
            "options": { "num_predict": 8 },
        })
    };

    let mut loads = Vec::new();
    for model in ["tiny-a", "tiny-b", "tiny-a"] {
        let (status, lines) = server.generate(ask(model)).await;
        assert_eq!(status, 200, "{lines:?}");
        assert_finished(&lines);
        assert_eq!(lines[0]["model"], model);
        assert_eq!(server.loaded().await, [model]);
        loads.push(server.load_id(model).await.expect("loaded"));
    }
    assert!(
        loads[0] != loads[2],
        "coming back to tiny-a loads it again: {loads:?}"
    );

    let kept = loads[2];
    let path = server.dir.join("tiny-a").join("model.gguf");
    for spelling in ["tiny:a", "TINY-A", path.to_str().expect("utf-8 path")] {
        let (status, lines) = server.generate(ask(spelling)).await;
        assert_eq!(status, 200, "{spelling}: {lines:?}");
        assert_finished(&lines);
        assert_eq!(lines[0]["model"], "tiny-a", "{spelling}");
        assert_eq!(
            server.load_id("tiny-a").await,
            Some(kept),
            "{spelling} reloaded it"
        );
    }
}

/// A sequential engine — every multimodal model, and `--batch-size 0` — is
/// freed when the last request running on it lets go. A swap used to take it
/// out of the slot and load the next model at once, sized against VRAM that
/// still held it. It waits now: the answer streaming from the old model is
/// finished in full, and only then is the new model loaded.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_swap_waits_for_a_sequential_engine_to_be_released() {
    let server = start(&["tiny-a", "tiny-b"], |state| {
        state.batch_size = 0;
        state.ctx_size = 4096;
    })
    .await;

    let response = reqwest::Client::new()
        .post(format!("{}/api/generate", server.base))
        .json(&json!({
            "model": "tiny-a", "prompt": "Once upon a time", "stream": true,
            "options": { "num_predict": 2000 },
        }))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 200);
    let mut body = response.bytes_stream();
    let first = body.next().await.expect("a first chunk").expect("chunk");
    assert!(!first.is_empty());

    let swap = {
        let base = server.base.clone();
        tokio::spawn(async move {
            let response = reqwest::Client::new()
                .post(format!("{base}/api/generate"))
                .json(&json!({
                    "model": "tiny-b", "prompt": "Once", "stream": false,
                    "options": { "num_predict": 4 },
                }))
                .send()
                .await
                .expect("request");
            let status = response.status();
            let line: Value = response.json().await.expect("json");
            (status, line, Instant::now())
        })
    };

    let mut text = String::from_utf8_lossy(&first).into_owned();
    while let Some(chunk) = body.next().await {
        text.push_str(&String::from_utf8_lossy(&chunk.expect("chunk")));
    }
    let a_ended = Instant::now();
    let lines: Vec<Value> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).expect("a JSON line"))
        .collect();
    assert_finished(&lines);

    let (status, line, b_answered) = swap.await.expect("the swap");
    assert_eq!(status, 200, "{line}");
    assert_finished(&[line]);
    assert!(
        b_answered >= a_ended,
        "tiny-b answered {:?} before tiny-a's answer was over",
        a_ended - b_answered
    );
    assert_eq!(server.loaded().await, ["tiny-b"]);
}

impl TestServer {
    /// `/api/version`, where the residency counters are.
    async fn version(&self) -> Value {
        reqwest::get(format!("{}/api/version", self.base))
            .await
            .expect("request")
            .json()
            .await
            .expect("json")
    }

    /// A streamed generation from `model` that has started: its first chunk
    /// is in, so the model is answering.
    async fn start_streaming(&self, model: &str, num_predict: u32) -> reqwest::Response {
        let response = reqwest::Client::new()
            .post(format!("{}/api/generate", self.base))
            .json(&json!({
                "model": model, "prompt": "Once upon a time", "stream": true,
                "options": { "num_predict": num_predict },
            }))
            .send()
            .await
            .expect("request");
        assert_eq!(response.status(), 200, "{model}");
        response
    }
}

/// The rest of a streamed answer, as JSON lines, and when it ended.
async fn read_to_end(response: reqwest::Response) -> (Vec<Value>, Instant) {
    let text = response.text().await.expect("body");
    let lines = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("{e}: {l}")))
        .collect();
    (lines, Instant::now())
}

fn short(model: &str) -> Value {
    json!({
        "model": model, "prompt": "Once upon a time", "stream": false,
        "options": { "num_predict": 8 },
    })
}

/// Two models resident at once, each answering — at the same time too —
/// without either being loaded again.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_two_models_answer_side_by_side() {
    let server = start(&["tiny-a", "tiny-b"], |state| state.max_loaded_models = 2).await;
    for model in ["tiny-a", "tiny-b"] {
        let (status, lines) = server.generate(short(model)).await;
        assert_eq!(status, 200, "{lines:?}");
        assert_finished(&lines);
    }
    assert_eq!(server.loaded().await, ["tiny-a", "tiny-b"]);
    let loads = (
        server.load_id("tiny-a").await,
        server.load_id("tiny-b").await,
    );

    let streamed = |model: &'static str| {
        json!({
            "model": model, "prompt": "Once upon a time", "stream": true,
            "options": { "num_predict": 200 },
        })
    };
    let ((status_a, a), (status_b, b)) = tokio::join!(
        server.generate(streamed("tiny-a")),
        server.generate(streamed("tiny-b"))
    );
    assert_eq!(status_a, 200);
    assert_eq!(status_b, 200);
    assert_finished(&a);
    assert_finished(&b);
    assert_eq!(a[0]["model"], "tiny-a");
    assert_eq!(b[0]["model"], "tiny-b");
    assert_eq!(
        (
            server.load_id("tiny-a").await,
            server.load_id("tiny-b").await
        ),
        loads,
        "neither was loaded again"
    );

    let version = server.version().await;
    assert_eq!(version["max_loaded_models"], 2);
    assert_eq!(version["loaded_models"], 2);
    assert_eq!(version["generation_evictions"], 0);
}

/// Past `--max-loaded-models`, the least recently used idle model makes
/// room, and only it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_third_model_evicts_the_least_recently_used() {
    let server = start(&["tiny-a", "tiny-b", "tiny-c"], |state| {
        state.max_loaded_models = 2
    })
    .await;
    // tiny-b is used before tiny-a is used again: tiny-b is the least
    // recently used when tiny-c comes.
    for model in ["tiny-a", "tiny-b", "tiny-a"] {
        let (status, lines) = server.generate(short(model)).await;
        assert_eq!(status, 200, "{lines:?}");
    }
    let kept = server.load_id("tiny-a").await;

    let (status, lines) = server.generate(short("tiny-c")).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_finished(&lines);
    assert_eq!(server.loaded().await, ["tiny-a", "tiny-c"]);
    assert_eq!(server.load_id("tiny-a").await, kept);
    let version = server.version().await;
    assert_eq!(version["generation_evictions"], 1);
    assert_eq!(version["loaded_models"], 2);
}

/// With every resident answering a request, a load that needs one of them
/// waits for one to finish; it does not cut another client's answer off, as
/// a swap of the one resident model does.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_eviction_waits_for_a_busy_model_instead_of_aborting_it() {
    let server = start(&["tiny-a", "tiny-b", "tiny-c"], |state| {
        state.max_loaded_models = 2;
        state.ctx_size = 8192;
    })
    .await;
    let started = Instant::now();
    let a = server.start_streaming("tiny-a", 3000).await;
    let b = server.start_streaming("tiny-b", 3000).await;
    let third = {
        let base = server.base.clone();
        tokio::spawn(async move {
            let response = reqwest::Client::new()
                .post(format!("{base}/api/generate"))
                .json(&short("tiny-c"))
                .send()
                .await
                .expect("request");
            let status = response.status();
            let line: Value = response.json().await.expect("json");
            (status, line, Instant::now())
        })
    };

    let ((a, a_ended), (b, b_ended)) = tokio::join!(read_to_end(a), read_to_end(b));
    assert_finished(&a);
    assert_finished(&b);
    let (status, line, c_answered) = third.await.expect("the third request");
    assert_eq!(status, 200, "{line}");
    assert_finished(&[line]);
    // Compared on the client's clocks, across connections: the server frees a
    // model a moment before its client has read the end of the body, so a
    // few milliseconds either way say nothing. Not waiting would have
    // answered tiny-c a second or more before either answer was over.
    let first_free = a_ended.min(b_ended);
    let tolerance = Duration::from_millis(250);
    assert!(
        first_free.duration_since(started) > 4 * tolerance,
        "the answers took {:?}, too short to tell waiting from not; raise num_predict",
        first_free.duration_since(started)
    );
    assert!(
        c_answered + tolerance >= first_free,
        "tiny-c answered {:?} before a model it needed was free",
        first_free - c_answered
    );
    let loaded = server.loaded().await;
    assert!(
        loaded.len() == 2 && loaded.contains(&"tiny-c".to_string()),
        "{loaded:?}"
    );
    assert_eq!(server.version().await["generation_evictions"], 1);
}

/// When no resident finishes within the wait, the load gives up with a 503
/// a client can retry — and the answers that kept it waiting are not
/// touched.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_load_that_cannot_make_room_in_time_is_a_503() {
    let server = start(&["tiny-a", "tiny-b", "tiny-c"], |state| {
        state.max_loaded_models = 2;
        state.ctx_size = 8192;
        state.busy_wait = Duration::from_millis(200);
    })
    .await;
    let a = server.start_streaming("tiny-a", 3000).await;
    let b = server.start_streaming("tiny-b", 3000).await;

    let response = reqwest::Client::new()
        .post(format!("{}/api/generate", server.base))
        .json(&short("tiny-c"))
        .send()
        .await
        .expect("request");
    assert_eq!(response.status(), 503);
    assert_eq!(
        response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok()),
        Some("5")
    );
    let body: Value = response.json().await.expect("json");
    assert!(
        body["error"].as_str().is_some_and(|e| e.contains("tiny-c")),
        "{body}"
    );

    let ((a, _), (b, _)) = tokio::join!(read_to_end(a), read_to_end(b));
    assert_finished(&a);
    assert_finished(&b);
    assert_eq!(server.loaded().await, ["tiny-a", "tiny-b"]);

    // Retried once they are done, it goes through.
    let (status, lines) = server.generate(short("tiny-c")).await;
    assert_eq!(status, 200, "{lines:?}");
}

/// A request to a resident model takes no lock a load holds: it is
/// answered while another model is still loading.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_request_to_a_resident_model_is_not_held_by_a_load() {
    let gate = Arc::new(super::LoadGate {
        arrived: tokio::sync::Notify::new(),
        proceed: tokio::sync::Notify::new(),
    });
    let server = start(&["tiny-a", "tiny-b"], |state| {
        state.max_loaded_models = 2;
        state.load_gate = Some(Arc::clone(&gate));
    })
    .await;
    let spawn_request = |model: &'static str| {
        let base = server.base.clone();
        tokio::spawn(async move {
            reqwest::Client::new()
                .post(format!("{base}/api/generate"))
                .json(&short(model))
                .send()
                .await
                .expect("request")
                .status()
        })
    };

    let first = spawn_request("tiny-a");
    gate.arrived.notified().await;
    gate.proceed.notify_one();
    assert_eq!(first.await.expect("tiny-a"), 200);

    // tiny-b's load stops at the gate, holding the load lock...
    let loading = spawn_request("tiny-b");
    gate.arrived.notified().await;
    // ...and tiny-a answers all the same.
    let (status, lines) =
        tokio::time::timeout(Duration::from_secs(20), server.generate(short("tiny-a")))
            .await
            .expect("tiny-a answered while tiny-b was loading");
    assert_eq!(status, 200, "{lines:?}");
    assert_finished(&lines);
    assert!(!loading.is_finished(), "tiny-b was still loading");

    gate.proceed.notify_one();
    assert_eq!(loading.await.expect("tiny-b"), 200);
    assert_eq!(server.loaded().await, ["tiny-a", "tiny-b"]);
}

/// Each resident keeps its own keep_alive: one expiring leaves the others
/// alone, and one kept for good stays.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_each_model_expires_on_its_own() {
    let tick = Duration::from_millis(25);
    let server = start(&["tiny-a", "tiny-b"], |state| {
        state.max_loaded_models = 2;
        state.idle_tick = tick;
    })
    .await;
    let ask = |model: &str, keep_alive: Value| {
        json!({
            "model": model, "prompt": "Once", "stream": false, "keep_alive": keep_alive,
            "options": { "num_predict": 4 },
        })
    };
    let (status, _) = server.generate(ask("tiny-b", json!(-1))).await;
    assert_eq!(status, 200);
    let kept = server.load_id("tiny-b").await;
    let (status, _) = server.generate(ask("tiny-a", json!(0.3))).await;
    assert_eq!(status, 200);
    assert_eq!(server.loaded().await, ["tiny-a", "tiny-b"]);

    server
        .wait_for_loaded(&["tiny-b"], Duration::from_secs(5))
        .await;
    // Kept for good: still there well past tiny-a's keep_alive.
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(server.loaded().await, ["tiny-b"]);
    assert_eq!(server.load_id("tiny-b").await, kept);

    // Its own keep_alive changes with its own next request.
    let (status, _) = server.generate(ask("tiny-b", json!(0.1))).await;
    assert_eq!(status, 200);
    server.wait_for_loaded(&[], Duration::from_secs(5)).await;
}

/// An empty request with `keep_alive: 0` unloads the model it names and only
/// that one, without loading it first; a model still answering finishes its
/// answer and goes after.
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_an_empty_request_with_keep_alive_zero_unloads_only_that_model() {
    let server = start(&["tiny-a", "tiny-b", "tiny-c"], |state| {
        state.max_loaded_models = 2;
        state.ctx_size = 8192;
    })
    .await;
    let unload = |model: Option<&str>| {
        let mut body = json!({ "prompt": "", "keep_alive": 0 });
        if let Some(model) = model {
            body["model"] = json!(model);
        }
        body
    };
    for model in ["tiny-a", "tiny-b"] {
        let (status, _) = server.generate(short(model)).await;
        assert_eq!(status, 200);
    }

    let (status, lines) = server.generate(unload(Some("tiny-a"))).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_eq!(lines[0]["done_reason"], "unload");
    assert_eq!(
        server.loaded().await,
        ["tiny-b"],
        "unloaded before the answer"
    );

    // A model that is not loaded is not loaded to be unloaded.
    let (status, lines) = server.generate(unload(Some("tiny-c"))).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_eq!(lines[0]["done_reason"], "unload");
    assert_eq!(server.loaded().await, ["tiny-b"]);

    // Busy: its answer is finished, and the model goes after it.
    let answering = server.start_streaming("tiny-b", 3000).await;
    let (status, lines) = server.generate(unload(Some("tiny-b"))).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_eq!(server.loaded().await, ["tiny-b"], "still answering");
    let (lines, _) = read_to_end(answering).await;
    assert_finished(&lines);
    server.wait_for_loaded(&[], Duration::from_secs(5)).await;

    // With no model named, the most recently used one goes.
    for model in ["tiny-a", "tiny-c"] {
        let (status, _) = server.generate(short(model)).await;
        assert_eq!(status, 200);
    }
    let (status, lines) = server.generate(unload(None)).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_eq!(lines[0]["model"], "tiny-c");
    assert_eq!(server.loaded().await, ["tiny-a"]);
}

/// `/api/unload` with a model unloads that one and leaves the others; without
/// one, every generation model goes.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_unload_names_one_model_or_takes_them_all() {
    let server = start(&["tiny-a", "tiny-b"], |state| state.max_loaded_models = 2).await;
    for model in ["tiny-a", "tiny-b"] {
        let (status, _) = server.generate(short(model)).await;
        assert_eq!(status, 200);
    }
    let unload = |body: Option<Value>| {
        let url = format!("{}/api/unload", server.base);
        async move {
            let request = reqwest::Client::new().post(url);
            let request = match body {
                Some(body) => request.json(&body),
                None => request,
            };
            let response = request.send().await.expect("request");
            assert_eq!(response.status(), 200);
            response.json::<Value>().await.expect("json")
        }
    };

    let answer = unload(Some(json!({ "model": "tiny:a" }))).await;
    assert_eq!(answer["unloaded"], "tiny-a");
    assert_eq!(answer["unloaded_all"], json!(["tiny-a"]));
    assert_eq!(server.loaded().await, ["tiny-b"]);

    let (status, _) = server.generate(short("tiny-a")).await;
    assert_eq!(status, 200);
    let answer = unload(None).await;
    let mut all: Vec<String> = answer["unloaded_all"]
        .as_array()
        .expect("a list")
        .iter()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    all.sort();
    assert_eq!(all, ["tiny-a", "tiny-b"]);
    assert!(answer["unloaded"].is_string());
    assert!(server.loaded().await.is_empty());
}

/// `/api/ps` lists every resident with its own expiry, and `/api/tags` and
/// `/v1/models` list every resident as loaded; a request that loaded its
/// model says how long that took.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_every_resident_is_listed_with_its_own_expiry() {
    let server = start(&["tiny-a", "tiny-b"], |state| state.max_loaded_models = 2).await;
    let ask = |model: &str, keep_alive: Value| {
        json!({
            "model": model, "prompt": "Once", "stream": false, "keep_alive": keep_alive,
            "options": { "num_predict": 4 },
        })
    };
    let (status, loaded_a) = server.generate(ask("tiny-a", json!("10m"))).await;
    assert_eq!(status, 200);
    let (status, _) = server.generate(ask("tiny-b", json!(-1))).await;
    assert_eq!(status, 200);
    assert!(
        loaded_a[0]["load_duration"]
            .as_u64()
            .is_some_and(|ns| ns > 0),
        "{loaded_a:?}"
    );
    let (_, again) = server.generate(ask("tiny-a", json!("10m"))).await;
    assert_eq!(again[0]["load_duration"], 0, "already loaded: {again:?}");

    let ps: Value = reqwest::get(format!("{}/api/ps", server.base))
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    let models = ps["models"].as_array().expect("models");
    let names: Vec<&str> = models.iter().filter_map(|m| m["name"].as_str()).collect();
    assert_eq!(names, ["tiny-a", "tiny-b"], "most recently used first");
    let expires = |i: usize| {
        chrono::DateTime::parse_from_rfc3339(models[i]["expires_at"].as_str().expect("a date"))
            .expect("RFC 3339")
            .to_utc()
    };
    let in_ten_minutes = (expires(0) - chrono::Utc::now()).num_seconds();
    assert!((590..=600).contains(&in_ten_minutes), "{in_ten_minutes}");
    assert!(
        expires(1)
            .format("%Y")
            .to_string()
            .parse::<i32>()
            .expect("a year")
            > 2200,
        "kept for good"
    );
    for m in models {
        assert!(m["size"].as_u64().is_some_and(|b| b > 0), "{m}");
        assert!(m["context_length"].as_u64().is_some_and(|c| c > 0), "{m}");
        assert_eq!(m["eullm"]["slot"], "generation");
    }

    let tags: Value = reqwest::get(format!("{}/api/tags", server.base))
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    let mut loaded: Vec<&str> = tags["models"]
        .as_array()
        .expect("models")
        .iter()
        .filter(|m| m["loaded"] == true)
        .filter_map(|m| m["name"].as_str())
        .collect();
    loaded.sort();
    assert_eq!(loaded, ["tiny-a", "tiny-b"]);
    let openai: Value = reqwest::get(format!("{}/v1/models", server.base))
        .await
        .expect("request")
        .json()
        .await
        .expect("json");
    let ids: Vec<&str> = openai["data"]
        .as_array()
        .expect("data")
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert!(
        ids.contains(&"tiny-a") && ids.contains(&"tiny-b"),
        "{ids:?}"
    );
}

/// With `--default-model`, a request that names no model is answered by that
/// model, loaded for it beside the others, and not by the one used last.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "needs a GGUF model in EULLM_GENERATION_TEST_MODEL"]
async fn real_model_a_request_naming_no_model_goes_to_the_default_one() {
    let server = start(&["tiny-a", "tiny-b"], |state| {
        state.max_loaded_models = 2;
        let path = state
            .store
            .gguf_path("tiny-b")
            .expect("tiny-b in the store");
        state.default_model = Some(super::NamedModel {
            name: "tiny-b".into(),
            path,
        });
    })
    .await;
    let (status, _) = server.generate(short("tiny-a")).await;
    assert_eq!(status, 200);
    let unnamed = json!({
        "prompt": "Once upon a time", "stream": false, "options": { "num_predict": 8 },
    });
    let (status, lines) = server.generate(unnamed.clone()).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_finished(&lines);
    assert_eq!(lines[0]["model"], "tiny-b", "not tiny-a, the one used last");
    assert_eq!(server.loaded().await, ["tiny-a", "tiny-b"]);
    let kept = server.load_id("tiny-b").await;

    // Loaded now: the same load answers.
    let (status, lines) = server.generate(unnamed).await;
    assert_eq!(status, 200, "{lines:?}");
    assert_eq!(lines[0]["model"], "tiny-b");
    assert_eq!(server.load_id("tiny-b").await, kept);
}
