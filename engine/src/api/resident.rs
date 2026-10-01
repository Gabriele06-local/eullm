//! The generation models this server holds, and how each one is in use.
//!
//! A request holds a [`Lease`] on the model that answers it for as long as
//! its response lasts, streaming included. keep_alive is counted from the
//! moment the last lease on a model is released, and a model with a lease
//! out is never unloaded for being idle. It used to be counted from the start
//! of each request: an answer that took longer than its keep_alive lost its
//! model halfway through, and `keep_alive: 0` unloaded the model before the
//! request that asked for it had been submitted, which the scheduler then
//! reported as a full queue.
//!
//! Which resident gives way when another model has to load is decided here
//! too, by pure functions over a [`ResidentView`] of each: [`next_step`] and
//! [`eviction_order`].

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::Notify;

use super::KeepAlive;
use crate::inference::{InferenceEngine, SchedulerHandle};

/// How one loaded generation model is in use, shared by its entry among the
/// residents and by every lease on it. Nothing here is held across an await:
/// atomics, and mutexes held for one assignment each.
pub(crate) struct Usage {
    /// Leases out: requests whose response has not finished.
    in_flight: AtomicUsize,
    /// When a request on this model last started or finished, or when it
    /// loaded: what "least recently used" is measured by.
    last_used: parking_lot::Mutex<Instant>,
    /// When the model, once idle, becomes due for unloading: set by the last
    /// request to finish, from its keep_alive. `None`: no deadline.
    deadline: parking_lot::Mutex<Option<Instant>>,
    /// The last request to finish asked for `keep_alive: 0`.
    unload_when_idle: AtomicBool,
}

/// [`Usage`] read at one moment, for [`due`] and the eviction order.
#[derive(Debug, Clone, Copy)]
pub(crate) struct UsageView {
    pub(crate) in_flight: usize,
    pub(crate) last_used: Instant,
    pub(crate) deadline: Option<Instant>,
    pub(crate) unload_when_idle: bool,
}

impl UsageView {
    /// Nothing will unload it for being idle: no deadline, and its last
    /// request did not ask for `keep_alive: 0`.
    fn kept_forever(&self) -> bool {
        self.deadline.is_none() && !self.unload_when_idle
    }
}

impl Usage {
    /// A model nobody has used yet: no deadline, so it stays until a request
    /// on it finishes and sets one. That is also how the model `eullm run`
    /// starts with is kept until its first request.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            in_flight: AtomicUsize::new(0),
            last_used: parking_lot::Mutex::new(Instant::now()),
            deadline: parking_lot::Mutex::new(None),
            unload_when_idle: AtomicBool::new(false),
        })
    }

    /// One more request in flight on this model.
    ///
    /// Take it while holding a guard on the residents: an unload checks
    /// `in_flight` under their write guard, so a lease taken under the read
    /// guard is either seen there or finds the model gone. `default` is the
    /// server's `--keep-alive`, which a request's `KeepAlive::Default` stands
    /// for.
    pub(crate) fn lease(
        self: &Arc<Self>,
        keep_alive: KeepAlive,
        default: Option<Duration>,
        idle: &Arc<Notify>,
    ) -> Lease {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
        *self.last_used.lock() = Instant::now();
        Lease {
            usage: Arc::clone(self),
            keep_alive,
            default,
            idle: Arc::clone(idle),
        }
    }

    pub(crate) fn view(&self) -> UsageView {
        // `in_flight` first: a lease writes the deadline before it gives
        // back its count, so a view that reads 0 also reads that deadline.
        let in_flight = self.in_flight.load(Ordering::SeqCst);
        UsageView {
            in_flight,
            last_used: *self.last_used.lock(),
            deadline: *self.deadline.lock(),
            unload_when_idle: self.unload_when_idle.load(Ordering::SeqCst),
        }
    }
}

/// Whether a model should be unloaded now: no request is using it, and the
/// last one to finish asked for `keep_alive: 0` or left a deadline that has
/// passed.
pub(crate) fn due(usage: &UsageView, now: Instant) -> bool {
    usage.in_flight == 0
        && (usage.unload_when_idle || usage.deadline.is_some_and(|deadline| now >= deadline))
}

/// One request's claim on the model answering it, released when the response
/// is over: when the handler returns, or with the stream it was moved into.
pub(crate) struct Lease {
    usage: Arc<Usage>,
    keep_alive: KeepAlive,
    default: Option<Duration>,
    /// Woken when the last lease on a model is released, so the idle-unload
    /// loop acts on `keep_alive: 0` at once rather than at its next tick.
    idle: Arc<Notify>,
}

impl Drop for Lease {
    /// The request is over, so its keep_alive starts now. The last request to
    /// finish decides what happens next: each one replaces the deadline the
    /// one before it left.
    fn drop(&mut self) {
        let now = Instant::now();
        let (deadline, immediate) = match self.keep_alive.resolve(self.default) {
            // A duration too long for an `Instant` is a deadline that never
            // comes, not a panic in whatever drops the lease.
            KeepAlive::For(duration) => (now.checked_add(duration), false),
            KeepAlive::Immediate => (None, true),
            KeepAlive::Forever | KeepAlive::Default => (None, false),
        };
        *self.usage.last_used.lock() = now;
        *self.usage.deadline.lock() = deadline;
        self.usage
            .unload_when_idle
            .store(immediate, Ordering::SeqCst);
        if self.usage.in_flight.fetch_sub(1, Ordering::SeqCst) == 1 {
            self.idle.notify_waiters();
        }
    }
}

/// What a request needs from the model answering it: cloned handles, which
/// hold no lock — an `Arc` bump and a channel clone — and the lease that
/// counts the request as using the model.
pub(crate) struct SlotSnapshot {
    pub(crate) model_name: String,
    pub(crate) engine: Option<Arc<InferenceEngine>>,
    pub(crate) scheduler: Option<SchedulerHandle>,
    pub(crate) lease: Lease,
}

/// A generation model resident in this server.
pub(crate) struct LoadedModel {
    /// This load of the model, unique for the life of the process: what an
    /// unload names, so that a model loaded again under the same name in
    /// the meantime is not the one it takes out.
    pub(crate) id: u64,
    /// The name it was loaded under, normalized (`qwen3:8b` → `qwen3-8b`).
    pub(crate) name: String,
    /// Its GGUF. A request naming the same file another way is answered by
    /// this model rather than by a second copy of it.
    pub(crate) path: PathBuf,
    pub(crate) engine: Option<Arc<InferenceEngine>>,
    pub(crate) scheduler: Option<SchedulerHandle>,
    /// The model `eullm run` started with, whose terminal chat holds it:
    /// evicted after every other.
    pub(crate) launch: bool,
    pub(crate) usage: Arc<Usage>,
}

impl LoadedModel {
    /// A model just loaded, not yet in use; [`ResidentModels::insert`] gives
    /// it its id.
    pub(crate) fn new(
        name: String,
        path: PathBuf,
        engine: Option<Arc<InferenceEngine>>,
        scheduler: Option<SchedulerHandle>,
    ) -> Self {
        Self {
            id: 0,
            name,
            path,
            engine,
            scheduler,
            launch: false,
            usage: Usage::new(),
        }
    }

    fn view(&self) -> ResidentView {
        ResidentView {
            id: self.id,
            usage: self.usage.view(),
            launch: self.launch,
        }
    }
}

/// The generation models in memory, at most `--max-loaded-models` of them:
/// sixteen or fewer, so a list, searched in full.
#[derive(Default)]
pub(crate) struct ResidentModels {
    models: Vec<LoadedModel>,
    next_id: u64,
}

impl ResidentModels {
    pub(crate) fn insert(&mut self, mut model: LoadedModel) -> &LoadedModel {
        self.next_id += 1;
        model.id = self.next_id;
        self.models.push(model);
        self.models.last().expect("just pushed")
    }

    pub(crate) fn remove(&mut self, id: u64) -> Option<LoadedModel> {
        let index = self.models.iter().position(|m| m.id == id)?;
        Some(self.models.remove(index))
    }

    pub(crate) fn get(&self, id: u64) -> Option<&LoadedModel> {
        self.models.iter().find(|m| m.id == id)
    }

    /// The resident a request's `model` names: by the name it was loaded
    /// under, by an Ollama tag of it (`qwen3:8b`), or by its file's path or
    /// stem — the same matching a single loaded model always had.
    pub(crate) fn find(&self, requested: &str) -> Option<&LoadedModel> {
        let wanted = super::normalize_model_name(requested);
        self.models
            .iter()
            .find(|m| super::model_names_match(&m.name, &wanted))
    }

    /// The resident loaded from `path`'s file, under whatever name: one
    /// launched by its path and asked for by its store name, or a second name
    /// `eullm pull` hard-linked to the same weights.
    pub(crate) fn find_file(&self, path: &Path) -> Option<&LoadedModel> {
        self.models.iter().find(|m| super::same_file(&m.path, path))
    }

    /// The model a request that names none is answered by.
    pub(crate) fn most_recently_used(&self) -> Option<&LoadedModel> {
        self.models.iter().max_by_key(|m| m.usage.view().last_used)
    }

    /// Every resident, the most recently used first.
    pub(crate) fn by_recent_use(&self) -> Vec<&LoadedModel> {
        let mut models: Vec<&LoadedModel> = self.models.iter().collect();
        models.sort_by_key(|m| std::cmp::Reverse(m.usage.view().last_used));
        models
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.models.is_empty()
    }

    /// The residents as [`next_step`] and [`eviction_order`] read them.
    pub(crate) fn views(&self) -> Vec<ResidentView> {
        self.models.iter().map(LoadedModel::view).collect()
    }
}

/// What deciding which resident gives way needs to know about one.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ResidentView {
    pub(crate) id: u64,
    pub(crate) usage: UsageView,
    pub(crate) launch: bool,
}

/// What to do next to make room for a model about to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Step {
    /// There is room: load it.
    Load,
    /// Unload the resident at this index of the views first.
    Evict(usize),
}

/// The next step towards loading one more model beside `residents`, when at
/// most `max` may be resident: load when there is room, otherwise evict the
/// first resident in [`eviction_order`] — a busy one too, as a single model
/// has always been replaced even while it was answering.
pub(crate) fn next_step(residents: &[ResidentView], max: usize, now: Instant) -> Step {
    if residents.len() < max {
        return Step::Load;
    }
    eviction_order(residents, now)
        .first()
        .map_or(Step::Load, |&i| Step::Evict(i))
}

/// The order in which residents give way to another model, as indices into
/// `residents`: idle ones before busy ones; then one whose keep_alive is over
/// first, one with a keep_alive before one kept for good, any model before
/// the one `eullm run` started with, and the least recently used first.
pub(crate) fn eviction_order(residents: &[ResidentView], now: Instant) -> Vec<usize> {
    let mut order: Vec<usize> = (0..residents.len()).collect();
    order.sort_by_key(|&i| {
        let r = &residents[i];
        (
            r.usage.in_flight > 0,
            !due(&r.usage, now),
            r.usage.kept_forever(),
            r.launch,
            r.usage.last_used,
        )
    });
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::FutureExt;

    fn notify() -> Arc<Notify> {
        Arc::new(Notify::new())
    }

    #[test]
    fn lease_drop_sets_the_deadline_from_the_end_of_the_request() {
        let usage = Usage::new();
        let idle = notify();
        let keep = Duration::from_secs(60);
        let lease = usage.lease(KeepAlive::For(keep), None, &idle);
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(
            usage.view().deadline,
            None,
            "nothing counts down while the request runs"
        );

        let before = Instant::now();
        drop(lease);
        let after = Instant::now();
        let deadline = usage.view().deadline.expect("a deadline once it is over");
        assert!(deadline >= before + keep && deadline <= after + keep);
        assert!(!due(&usage.view(), after));
        assert!(due(&usage.view(), after + keep));
    }

    #[test]
    fn immediate_unloads_only_after_the_last_lease() {
        let usage = Usage::new();
        let idle = notify();
        let first = usage.lease(KeepAlive::Immediate, None, &idle);
        let second = usage.lease(KeepAlive::Immediate, None, &idle);
        let mut woken = std::pin::pin!(idle.notified());
        woken.as_mut().enable();

        drop(first);
        assert!(
            !due(&usage.view(), Instant::now()),
            "another request is still being answered"
        );
        assert!(woken.as_mut().now_or_never().is_none());

        drop(second);
        assert!(due(&usage.view(), Instant::now()));
        assert!(
            woken.as_mut().now_or_never().is_some(),
            "the idle-unload loop is woken when the last one ends"
        );
    }

    #[test]
    fn a_busy_slot_is_never_due() {
        let usage = Usage::new();
        let idle = notify();
        drop(usage.lease(KeepAlive::For(Duration::from_millis(1)), None, &idle));
        let much_later = Instant::now() + Duration::from_secs(3600);
        assert!(due(&usage.view(), much_later));

        let busy = usage.lease(KeepAlive::Immediate, None, &idle);
        assert!(
            !due(&usage.view(), much_later),
            "a model in use is not unloaded, whatever its deadline says"
        );
        drop(busy);
        assert!(due(&usage.view(), Instant::now()));
    }

    #[test]
    fn the_last_request_to_finish_decides() {
        let usage = Usage::new();
        let idle = notify();
        let unload = usage.lease(KeepAlive::Immediate, None, &idle);
        let keep = usage.lease(KeepAlive::Forever, None, &idle);
        drop(unload);
        drop(keep);
        assert!(!due(
            &usage.view(),
            Instant::now() + Duration::from_secs(3600)
        ));

        let keep = usage.lease(KeepAlive::Forever, None, &idle);
        let unload = usage.lease(KeepAlive::Immediate, None, &idle);
        drop(keep);
        drop(unload);
        assert!(due(&usage.view(), Instant::now()));
    }

    /// Without `--keep-alive`, a request that does not ask keeps the model
    /// for good, as every release before the flag did; with it, the flag's
    /// duration counts from the end of the request.
    #[test]
    fn a_request_without_keep_alive_takes_the_servers_default() {
        let usage = Usage::new();
        let idle = notify();
        drop(usage.lease(KeepAlive::Default, None, &idle));
        assert_eq!(usage.view().deadline, None);
        assert!(!due(
            &usage.view(),
            Instant::now() + Duration::from_secs(3600)
        ));

        drop(usage.lease(KeepAlive::Default, Some(Duration::from_secs(300)), &idle));
        assert!(usage.view().deadline.is_some());
    }

    fn resident(id: u64, last_used: Instant) -> ResidentView {
        ResidentView {
            id,
            usage: UsageView {
                in_flight: 0,
                last_used,
                deadline: None,
                unload_when_idle: false,
            },
            launch: false,
        }
    }

    #[test]
    fn at_one_model_the_resident_is_replaced_even_when_busy() {
        let now = Instant::now();
        let mut busy = resident(1, now);
        busy.usage.in_flight = 3;
        assert_eq!(next_step(&[busy], 1, now), Step::Evict(0));
        assert_eq!(next_step(&[], 1, now), Step::Load);
    }

    #[test]
    fn the_least_recently_used_idle_model_gives_way_first() {
        let t0 = Instant::now();
        let s = Duration::from_secs;
        let recent = resident(1, t0 + s(5));
        let old = resident(2, t0 + s(1));
        let mut busy_and_oldest = resident(3, t0);
        busy_and_oldest.usage.in_flight = 1;
        assert_eq!(
            eviction_order(&[recent, busy_and_oldest, old], t0 + s(10)),
            [2, 0, 1]
        );
        assert_eq!(next_step(&[recent, old], 2, t0 + s(10)), Step::Evict(1));
        assert_eq!(next_step(&[recent, old], 3, t0 + s(10)), Step::Load);
    }

    /// A model whose keep_alive is over goes first, whenever it was used; a
    /// model kept for good goes after any that will expire; the model `eullm
    /// run` started with goes last.
    #[test]
    fn expired_first_then_a_deadline_before_forever_then_the_launch_model() {
        let t0 = Instant::now();
        let s = Duration::from_secs;
        let now = t0 + s(100);
        let mut launch = resident(1, t0);
        launch.launch = true;
        let forever = resident(2, t0 + s(1));
        let mut later = resident(3, t0 + s(2));
        later.usage.deadline = Some(t0 + s(500));
        let mut expired = resident(4, t0 + s(90));
        expired.usage.deadline = Some(t0 + s(95));
        let mut unload_now = resident(5, t0 + s(99));
        unload_now.usage.unload_when_idle = true;
        assert_eq!(
            eviction_order(&[launch, forever, later, expired, unload_now], now),
            [3, 4, 2, 1, 0]
        );
    }

    #[test]
    fn a_resident_is_found_by_path_store_name_and_ollama_tag() {
        let mut residents = ResidentModels::default();
        let by_name = residents
            .insert(LoadedModel::new(
                "qwen3-8b".into(),
                "/m/qwen3-8b/Qwen3-8B-Q4_K_M.gguf".into(),
                None,
                None,
            ))
            .id;
        let launched_path = "/models/Ornith-1.0-35B-Q4_K_M.gguf";
        let by_path = residents
            .insert(LoadedModel::new(
                launched_path.into(),
                launched_path.into(),
                None,
                None,
            ))
            .id;
        let found = |name: &str| residents.find(name).map(|m| m.id);
        assert_eq!(found("qwen3-8b"), Some(by_name));
        assert_eq!(found("qwen3:8b"), Some(by_name));
        assert_eq!(found("QWEN3-8B"), Some(by_name));
        assert_eq!(found(launched_path), Some(by_path));
        assert_eq!(found("ornith-1.0-35b-q4_k_m"), Some(by_path));
        assert_eq!(found("Ornith-1.0-35B-Q5_K_M"), None, "another quant");
        assert_eq!(found("qwen3-14b"), None);
    }

    #[cfg(unix)]
    #[test]
    fn the_same_file_under_two_names_is_one_model() {
        let dir = std::env::temp_dir().join(format!("eullm-resident-{}", uuid::Uuid::new_v4()));
        for name in ["a", "b", "c"] {
            std::fs::create_dir_all(dir.join(name)).unwrap();
        }
        let weights = dir.join("a/model.gguf");
        std::fs::write(&weights, b"GGUF").unwrap();
        // What `eullm pull` does for a model pulled under a second name.
        std::fs::hard_link(&weights, dir.join("b/model.gguf")).unwrap();
        // A copy is another file, and another model.
        std::fs::write(dir.join("c/model.gguf"), b"GGUF").unwrap();

        let mut residents = ResidentModels::default();
        residents.insert(LoadedModel::new("a".into(), weights, None, None));
        let found = |path: &str| residents.find_file(&dir.join(path)).map(|m| m.name.clone());
        assert_eq!(found("b/model.gguf").as_deref(), Some("a"));
        assert_eq!(found("c/../a/model.gguf").as_deref(), Some("a"));
        assert_eq!(found("c/model.gguf"), None);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn residents_are_listed_most_recently_used_first() {
        let mut residents = ResidentModels::default();
        let idle = notify();
        for name in ["first", "second", "third"] {
            residents.insert(LoadedModel::new(name.into(), name.into(), None, None));
            std::thread::sleep(Duration::from_millis(2));
        }
        let second = residents.find("second").expect("loaded");
        drop(second.usage.lease(KeepAlive::Default, None, &idle));
        let names: Vec<&str> = residents
            .by_recent_use()
            .iter()
            .map(|m| m.name.as_str())
            .collect();
        assert_eq!(names, ["second", "third", "first"]);
        assert_eq!(
            residents.most_recently_used().map(|m| m.name.as_str()),
            Some("second")
        );
        let ids: Vec<u64> = residents.views().iter().map(|v| v.id).collect();
        assert_eq!(ids, [1, 2, 3], "every load its own id");
        let removed = residents.remove(2).expect("there");
        assert_eq!(removed.name, "second");
        assert!(residents.get(2).is_none());
    }

    /// `keep_alive` comes from a request body, and a Duration that fits is
    /// not always an `Instant` that does.
    #[test]
    fn a_keep_alive_past_the_end_of_time_never_expires() {
        let usage = Usage::new();
        let idle = notify();
        drop(usage.lease(KeepAlive::For(Duration::MAX), None, &idle));
        assert_eq!(usage.view().deadline, None);
        assert!(!due(&usage.view(), Instant::now()));
    }
}
