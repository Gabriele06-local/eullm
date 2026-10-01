//! The generation model this server holds, and how it is in use.
//!
//! A request holds a [`Lease`] on the model that answers it for as long as
//! its response lasts, streaming included. keep_alive is counted from the
//! moment the last lease on a model is released, and a model with a lease
//! out is never unloaded for being idle. It used to be counted from the start
//! of each request: an answer that took longer than its keep_alive lost its
//! model halfway through, and `keep_alive: 0` unloaded the model before the
//! request that asked for it had been submitted, which the scheduler then
//! reported as a full queue.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use tokio::sync::Notify;

use super::KeepAlive;
use crate::inference::{InferenceEngine, SchedulerHandle};

/// How one loaded generation model is in use, shared by the slot it sits in
/// and by every lease on it. Nothing here is held across an await: atomics,
/// and a mutex held for one assignment.
pub(crate) struct Usage {
    /// Leases out: requests whose response has not finished.
    in_flight: AtomicUsize,
    /// When the model, once idle, becomes due for unloading: set by the last
    /// request to finish, from its keep_alive. `None`: no deadline.
    deadline: parking_lot::Mutex<Option<Instant>>,
    /// The last request to finish asked for `keep_alive: 0`.
    unload_when_idle: AtomicBool,
}

/// [`Usage`] read at one moment, for [`due`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct UsageView {
    pub(crate) in_flight: usize,
    pub(crate) deadline: Option<Instant>,
    pub(crate) unload_when_idle: bool,
}

impl Usage {
    /// A model nobody has used yet: no deadline, so it stays until a request
    /// on it finishes and sets one. That is also how the model `eullm run`
    /// starts with is kept until its first request.
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            in_flight: AtomicUsize::new(0),
            deadline: parking_lot::Mutex::new(None),
            unload_when_idle: AtomicBool::new(false),
        })
    }

    /// One more request in flight on this model.
    ///
    /// Take it while holding the guard of the slot the model sits in: an
    /// unload checks `in_flight` under that slot's write guard, so a lease
    /// taken under the read guard is either seen there or finds the model
    /// gone. `default` is the server's `--keep-alive`, which a request's
    /// `KeepAlive::Default` stands for.
    pub(crate) fn lease(
        self: &Arc<Self>,
        keep_alive: KeepAlive,
        default: Option<Duration>,
        idle: &Arc<Notify>,
    ) -> Lease {
        self.in_flight.fetch_add(1, Ordering::SeqCst);
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
