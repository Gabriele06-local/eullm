//! Training: fine-tuning a model's weights in place through llama.cpp's
//! `ggml-opt`, the machinery behind llama.cpp's own `examples/training`.
//!
//! What upstream supports is what this supports, and upstream calls it "very
//! much WIP" (`examples/training/README.md`):
//!
//! - every weight tensor F32: the weight gradient (`OUT_PROD`) has no F16 or
//!   quantized kernel, on the CPU or on CUDA/HIP;
//! - no flash attention (`FLASH_ATTN_EXT` has no backward pass) and an F32 KV
//!   cache;
//! - context, logical batch and micro-batch of the same length — otherwise
//!   the K and V projections receive no gradient;
//! - the token embeddings never train (an upstream FIXME in `opt_init`).
//!
//! A context that has trained should not go on to serve: [`Trainer::new`]
//! changes the model's training context length to the context's own and
//! leaves the optimizer state in the context.

use std::cell::Cell;
use std::ffi::{c_void, CStr};
use std::ptr::NonNull;
use std::time::Instant;

use crate::context::LlamaContext;
use crate::token::LlamaToken;

/// Errors building the training data.
#[derive(Debug, Eq, PartialEq, thiserror::Error)]
pub enum OptError {
    /// Not one window of `n_ctx` tokens plus its next-token label fits.
    #[error("{have} tokens make no training window of {n_ctx}: at least {} are needed", n_ctx + 1)]
    TooFewTokens {
        /// Tokens given.
        have: usize,
        /// Window length asked for.
        n_ctx: usize,
    },
    /// A stride of zero would repeat the first window forever.
    #[error("the stride between windows must be at least one token")]
    ZeroStride,
    /// ggml returned no dataset.
    #[error("ggml could not allocate the training dataset")]
    Alloc,
    /// The context's batch or micro-batch differs from its length: llama.cpp
    /// would abort on the batch divisibility, or train without gradients for
    /// the K and V projections.
    #[error(
        "a training context needs n_ctx = n_batch = n_ubatch, this one has {n_ctx}, {n_batch}, \
         {n_ubatch} (llama.cpp rounds n_ctx up to a multiple of 256)"
    )]
    ContextShape {
        /// The context's length.
        n_ctx: u32,
        /// Its logical batch.
        n_batch: u32,
        /// Its micro-batch.
        n_ubatch: u32,
    },
    /// A learning rate or weight decay ggml refuses — with an abort, mid-run,
    /// at the first optimizer step.
    #[error("{0}")]
    Schedule(String),
    /// Windows of a different length than the context: ggml would read past
    /// the dataset and abort.
    #[error("the dataset's windows are {windows} tokens and the context is {n_ctx}")]
    WindowMismatch {
        /// The dataset's window length.
        windows: usize,
        /// The context's length.
        n_ctx: u32,
    },
}

/// The optimizer updating the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Optimizer {
    /// AdamW: two moments per trainable parameter.
    AdamW,
    /// Plain SGD with weight decay: no state beyond the gradient.
    Sgd,
}

impl Optimizer {
    fn raw(self) -> llama_cpp_sys_2::ggml_opt_optimizer_type {
        match self {
            Self::AdamW => llama_cpp_sys_2::GGML_OPT_OPTIMIZER_TYPE_ADAMW,
            Self::Sgd => llama_cpp_sys_2::GGML_OPT_OPTIMIZER_TYPE_SGD,
        }
    }

    /// Bytes of optimizer state per trainable F32 parameter, on top of its
    /// gradient.
    #[must_use]
    pub fn state_bytes_per_param(self) -> u64 {
        match self {
            Self::AdamW => 8,
            Self::Sgd => 0,
        }
    }
}

impl std::str::FromStr for Optimizer {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "adamw" => Ok(Self::AdamW),
            "sgd" => Ok(Self::Sgd),
            other => Err(format!("unknown optimizer {other:?}: adamw or sgd")),
        }
    }
}

impl std::fmt::Display for Optimizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::AdamW => "adamw",
            Self::Sgd => "sgd",
        })
    }
}

/// The learning rate of each epoch — llama.cpp's `lr_opt` (`common.cpp`),
/// so a run here and one with `llama-finetune` use the same schedule.
///
/// Constant at `lr0` unless `lr_min > 0`; then it halves geometrically from
/// `lr0` to `lr_min` over `decay_epochs` (all the epochs when unset) and
/// stays at `lr_min` after.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LrSchedule {
    /// Learning rate at the first epoch.
    pub lr0: f32,
    /// Floor reached after the decay; `<= 0` for a constant rate.
    pub lr_min: f32,
    /// Epochs over which the rate decays to `lr_min`.
    pub decay_epochs: f32,
    /// Weight decay, for both optimizers.
    pub weight_decay: f32,
    scale_epoch: f32,
}

impl LrSchedule {
    /// `lr_opt::init`, for a run of `epochs`.
    #[must_use]
    pub fn new(lr0: f32, lr_min: f32, decay_epochs: f32, weight_decay: f32, epochs: u32) -> Self {
        let mut s = Self {
            lr0,
            lr_min,
            decay_epochs,
            weight_decay,
            scale_epoch: 0.0,
        };
        if lr_min > 0.0 && lr_min < lr0 {
            let halvings = (lr0 / lr_min).ln() / std::f32::consts::LN_2;
            #[allow(clippy::cast_precision_loss)]
            let mut e = epochs as f32;
            if decay_epochs > 0.0 && decay_epochs < e {
                e = decay_epochs;
            } else {
                s.decay_epochs = e;
            }
            s.scale_epoch = halvings / e;
        }
        s
    }

    /// Whether ggml will accept every step of this schedule: a learning rate
    /// above zero at every epoch, and weight decay in `[0, 1]`. ggml checks
    /// both with assertions, so a bad value would abort the process at the
    /// first optimizer step rather than fail.
    ///
    /// # Errors
    ///
    /// [`OptError::Schedule`] naming the value.
    pub fn check(&self) -> Result<(), OptError> {
        if !(self.lr0.is_finite() && self.lr0 > 0.0) {
            return Err(OptError::Schedule(format!(
                "the learning rate must be above zero, got {}",
                self.lr0
            )));
        }
        if !(self.weight_decay.is_finite() && (0.0..=1.0).contains(&self.weight_decay)) {
            return Err(OptError::Schedule(format!(
                "weight decay must be between 0 and 1, got {}",
                self.weight_decay
            )));
        }
        if self.lr_min.is_nan() {
            return Err(OptError::Schedule("the learning-rate floor is NaN".into()));
        }
        Ok(())
    }

    /// `lr_opt::get_lr`.
    #[must_use]
    pub fn lr(&self, epoch: f32) -> f32 {
        if self.lr_min <= 0.0 {
            self.lr0
        } else if epoch >= self.decay_epochs {
            self.lr_min
        } else {
            self.lr0 * 0.5f32.powf(epoch * self.scale_epoch)
        }
    }
}

/// Windows of `n_ctx` tokens, `stride` apart, that a stream of `n_tokens`
/// gives — each needs one more token, for the label of its last position.
///
/// One more than llama.cpp's `common_opt_dataset_init`, whose
/// `(n - n_ctx - 1) / stride` leaves out the last window that fits.
#[must_use]
pub fn window_count(n_tokens: usize, n_ctx: usize, stride: usize) -> usize {
    if stride == 0 || n_ctx == 0 || n_tokens < n_ctx + 1 {
        0
    } else {
        (n_tokens - n_ctx - 1) / stride + 1
    }
}

/// Training data: windows of a token stream, each labelled with the same
/// stream shifted by one — next-token prediction.
#[derive(Debug)]
pub struct OptDataset {
    ptr: NonNull<llama_cpp_sys_2::ggml_opt_dataset>,
    n_ctx: usize,
    windows: usize,
}

impl OptDataset {
    /// Cut `tokens` into windows of `n_ctx`, `stride` apart.
    ///
    /// # Errors
    ///
    /// A zero stride, too few tokens for one window, or no allocation.
    pub fn from_tokens(
        tokens: &[LlamaToken],
        n_ctx: usize,
        stride: usize,
    ) -> Result<Self, OptError> {
        if stride == 0 {
            return Err(OptError::ZeroStride);
        }
        let windows = window_count(tokens.len(), n_ctx, stride);
        if windows == 0 {
            return Err(OptError::TooFewTokens {
                have: tokens.len(),
                n_ctx,
            });
        }
        #[allow(clippy::cast_possible_wrap)]
        let ptr = unsafe {
            llama_cpp_sys_2::ggml_opt_dataset_init(
                llama_cpp_sys_2::GGML_TYPE_I32,
                llama_cpp_sys_2::GGML_TYPE_I32,
                n_ctx as i64,
                n_ctx as i64,
                windows as i64,
                1,
            )
        };
        let ptr = NonNull::new(ptr).ok_or(OptError::Alloc)?;
        // SAFETY: ggml allocated an I32 data tensor and an I32 labels tensor
        // of `n_ctx * windows` elements each, in host memory, owned by the
        // dataset; every write below stays inside them because the last
        // window starts at `(windows - 1) * stride` and reads `n_ctx + 1`
        // tokens, which `window_count` guarantees exist.
        unsafe {
            let data = (*llama_cpp_sys_2::ggml_opt_dataset_data(ptr.as_ptr()))
                .data
                .cast::<i32>();
            let labels = (*llama_cpp_sys_2::ggml_opt_dataset_labels(ptr.as_ptr()))
                .data
                .cast::<i32>();
            for w in 0..windows {
                let start = w * stride;
                for i in 0..n_ctx {
                    *data.add(w * n_ctx + i) = tokens[start + i].0;
                    *labels.add(w * n_ctx + i) = tokens[start + i + 1].0;
                }
            }
        }
        Ok(Self {
            ptr,
            n_ctx,
            windows,
        })
    }

    /// Number of windows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.windows
    }

    /// Whether there are no windows (never, for a dataset that was built).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.windows == 0
    }

    /// Tokens per window.
    #[must_use]
    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }

    /// The tokens of window `w` and their labels, read back from ggml.
    #[must_use]
    pub fn window(&self, w: usize) -> Option<(Vec<i32>, Vec<i32>)> {
        if w >= self.windows {
            return None;
        }
        // SAFETY: as in `from_tokens`; reads only.
        unsafe {
            let data = (*llama_cpp_sys_2::ggml_opt_dataset_data(self.ptr.as_ptr()))
                .data
                .cast::<i32>();
            let labels = (*llama_cpp_sys_2::ggml_opt_dataset_labels(self.ptr.as_ptr()))
                .data
                .cast::<i32>();
            let off = w * self.n_ctx;
            Some((
                std::slice::from_raw_parts(data.add(off), self.n_ctx).to_vec(),
                std::slice::from_raw_parts(labels.add(off), self.n_ctx).to_vec(),
            ))
        }
    }
}

impl Drop for OptDataset {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::ggml_opt_dataset_free(self.ptr.as_ptr()) }
    }
}

struct OptResult(NonNull<llama_cpp_sys_2::ggml_opt_result>);

impl OptResult {
    fn new() -> Self {
        let ptr = unsafe { llama_cpp_sys_2::ggml_opt_result_init() };
        Self(NonNull::new(ptr).expect("ggml_opt_result_init returned null"))
    }

    fn read(&self) -> (i64, f64, f64, f64, f64) {
        let (mut n, mut loss, mut loss_unc, mut acc, mut acc_unc) = (0i64, 0.0, 0.0, 0.0, 0.0);
        unsafe {
            llama_cpp_sys_2::ggml_opt_result_ndata(self.0.as_ptr(), &mut n);
            if n > 0 {
                llama_cpp_sys_2::ggml_opt_result_loss(self.0.as_ptr(), &mut loss, &mut loss_unc);
                llama_cpp_sys_2::ggml_opt_result_accuracy(self.0.as_ptr(), &mut acc, &mut acc_unc);
            }
        }
        (n, loss, loss_unc, acc, acc_unc)
    }
}

impl Drop for OptResult {
    fn drop(&mut self) {
        unsafe { llama_cpp_sys_2::ggml_opt_result_free(self.0.as_ptr()) }
    }
}

/// Which weight tensors train: all of them, or those whose GGUF name matches
/// one of the patterns, where `*` stands for any run of characters
/// (`blk.*.ffn_*`, `output.weight`).
///
/// Training fewer tensors does not save the memory of their weights — those
/// are F32 whether they train or not — but it saves their gradients and
/// optimizer state, up to three quarters of the total with AdamW.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TensorFilter {
    patterns: Vec<String>,
}

impl TensorFilter {
    /// Every tensor trains.
    #[must_use]
    pub fn all() -> Self {
        Self::default()
    }

    /// Only tensors matching one of `patterns` train; none given means all.
    #[must_use]
    pub fn new(patterns: Vec<String>) -> Self {
        Self { patterns }
    }

    /// Whether the tensor named `name` trains.
    #[must_use]
    pub fn accepts(&self, name: &str) -> bool {
        self.patterns.is_empty() || self.patterns.iter().any(|p| glob_match(p, name))
    }

    /// The patterns, empty for all.
    #[must_use]
    pub fn patterns(&self) -> &[String] {
        &self.patterns
    }
}

/// Whether `name` matches `pattern`, where `*` is any run of characters
/// (including none) and everything else matches itself.
#[must_use]
pub fn glob_match(pattern: &str, name: &str) -> bool {
    let (p, n) = (pattern.as_bytes(), name.as_bytes());
    let (mut pi, mut ni) = (0, 0);
    let mut star: Option<(usize, usize)> = None;
    while ni < n.len() {
        if pi < p.len() && p[pi] == b'*' {
            star = Some((pi, ni));
            pi += 1;
        } else if pi < p.len() && p[pi] == n[ni] {
            pi += 1;
            ni += 1;
        } else if let Some((sp, sn)) = star {
            pi = sp + 1;
            ni = sn + 1;
            star = Some((sp, sn + 1));
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == b'*')
}

/// One pass over a dataset: training or evaluation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pass {
    /// Tokens the loss was measured on — ggml counts every label as a
    /// datapoint, so this is windows × window length, not windows.
    pub tokens: i64,
    /// Mean cross-entropy per token, nats.
    pub loss: f64,
    /// Its standard error.
    pub loss_unc: f64,
    /// Share of tokens whose most likely prediction was the label.
    pub accuracy: f64,
    /// Its standard error.
    pub accuracy_unc: f64,
    /// Wall-clock seconds.
    pub seconds: f64,
}

impl Pass {
    /// `exp(loss)`.
    #[must_use]
    pub fn perplexity(&self) -> f64 {
        self.loss.exp()
    }

    /// Tokens per second over the pass.
    #[must_use]
    pub fn tokens_per_second(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        if self.seconds > 0.0 {
            self.tokens as f64 / self.seconds
        } else {
            0.0
        }
    }
}

/// What the C callbacks read, at an address that does not move while the
/// optimizer context can call them.
struct CallbackState {
    schedule: LrSchedule,
    epoch: Cell<f32>,
    filter: TensorFilter,
    trainable_params: Cell<u64>,
    trainable_tensors: Cell<u32>,
}

/// Tensors llama.cpp's `opt_init` never trains, whatever a filter says
/// (FIXMEs in its `llama_set_param`). It asks the filter before it checks the
/// name, so a model whose output layer is its token embeddings (tied, as in
/// Qwen3-0.6B) offers them to the filter once more, as the output.
pub const NEVER_TRAINED: [&str; 2] = ["token_embd.weight", "rope_freqs.weight"];

unsafe extern "C" fn filter_trampoline(
    tensor: *const llama_cpp_sys_2::ggml_tensor,
    userdata: *mut c_void,
) -> bool {
    // SAFETY: `userdata` is the `CallbackState` boxed by `Trainer::new`,
    // alive for the whole `llama_opt_init` call that invokes this.
    let state = &*userdata.cast::<CallbackState>();
    let name_ptr = llama_cpp_sys_2::ggml_get_name(tensor);
    let name = if name_ptr.is_null() {
        ""
    } else {
        CStr::from_ptr(name_ptr).to_str().unwrap_or("")
    };
    let accepted = !NEVER_TRAINED.contains(&name) && state.filter.accepts(name);
    if accepted {
        #[allow(clippy::cast_sign_loss)]
        let n = llama_cpp_sys_2::ggml_nelements(tensor) as u64;
        state.trainable_params.set(state.trainable_params.get() + n);
        state
            .trainable_tensors
            .set(state.trainable_tensors.get() + 1);
    }
    accepted
}

unsafe extern "C" fn lr_trampoline(
    userdata: *mut c_void,
) -> llama_cpp_sys_2::ggml_opt_optimizer_params {
    // SAFETY: as above; called from `llama_opt_epoch` while the `Trainer`
    // that owns the state holds the context mutably.
    let state = &*userdata.cast::<CallbackState>();
    let mut params = llama_cpp_sys_2::ggml_opt_get_default_optimizer_params(std::ptr::null_mut());
    let lr = state.schedule.lr(state.epoch.get());
    params.adamw.alpha = lr;
    params.sgd.alpha = lr;
    params.adamw.wd = state.schedule.weight_decay;
    params.sgd.wd = state.schedule.weight_decay;
    params
}

/// Trains the weights of the model behind a context.
///
/// Holds the context mutably for as long as it lives, which is what keeps
/// the state the optimizer's callbacks point into alive and in place.
pub struct Trainer<'c, 'm> {
    ctx: &'c mut LlamaContext<'m>,
    state: Box<CallbackState>,
    optimizer: Optimizer,
}

impl std::fmt::Debug for Trainer<'_, '_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trainer")
            .field("optimizer", &self.optimizer)
            .field("trainable_params", &self.state.trainable_params.get())
            .finish_non_exhaustive()
    }
}

impl<'c, 'm> Trainer<'c, 'm> {
    /// Set the context up for training: the optimizer, its learning-rate
    /// schedule, and which tensors train (`llama_opt_init`).
    ///
    /// The context must have been created with `n_ctx == n_batch ==
    /// n_ubatch` — checked here, because llama.cpp would abort on one shape
    /// and silently train without K/V gradients on another — an F32 KV cache
    /// and flash attention off. It may be set up for training once.
    ///
    /// # Errors
    ///
    /// [`OptError::ContextShape`] for a context of another shape,
    /// [`OptError::Schedule`] for rates ggml would abort on.
    pub fn new(
        ctx: &'c mut LlamaContext<'m>,
        optimizer: Optimizer,
        schedule: LrSchedule,
        filter: TensorFilter,
    ) -> Result<Self, OptError> {
        schedule.check()?;
        let (n_ctx, n_batch, n_ubatch) = (ctx.n_ctx(), ctx.n_batch(), ctx.n_ubatch());
        if n_ctx != n_batch || n_batch != n_ubatch {
            return Err(OptError::ContextShape {
                n_ctx,
                n_batch,
                n_ubatch,
            });
        }
        let state = Box::new(CallbackState {
            schedule,
            epoch: Cell::new(0.0),
            filter,
            trainable_params: Cell::new(0),
            trainable_tensors: Cell::new(0),
        });
        let ud = std::ptr::addr_of!(*state).cast_mut().cast::<c_void>();
        let params = llama_cpp_sys_2::llama_opt_params {
            n_ctx_train: 0,
            param_filter: Some(filter_trampoline),
            param_filter_ud: ud,
            get_opt_pars: Some(lr_trampoline),
            get_opt_pars_ud: ud,
            optimizer_type: optimizer.raw(),
        };
        // SAFETY: both pointers are live; llama.cpp mutates the model (its
        // training context length, which tensors are parameters) through a
        // reference this crate otherwise treats as shared — the price of the
        // C API, and why a trained context should not go on to serve.
        unsafe {
            llama_cpp_sys_2::llama_opt_init(ctx.context.as_ptr(), ctx.model.model.as_ptr(), params);
        }
        Ok(Self {
            ctx,
            state,
            optimizer,
        })
    }

    /// Parameters that train (token embeddings never do).
    #[must_use]
    pub fn trainable_params(&self) -> u64 {
        self.state.trainable_params.get()
    }

    /// Tensors that train.
    #[must_use]
    pub fn trainable_tensors(&self) -> u32 {
        self.state.trainable_tensors.get()
    }

    /// The optimizer.
    #[must_use]
    pub fn optimizer(&self) -> Optimizer {
        self.optimizer
    }

    /// The learning rate epoch `epoch` trains at.
    #[must_use]
    pub fn lr(&self, epoch: u32) -> f32 {
        #[allow(clippy::cast_precision_loss)]
        self.state.schedule.lr(epoch as f32)
    }

    /// Train one epoch over every window of `data`, at epoch `epoch`'s
    /// learning rate. `progress` draws ggml's progress bar on stderr.
    ///
    /// # Errors
    ///
    /// [`OptError::WindowMismatch`] for windows not the context's length.
    pub fn train(
        &mut self,
        epoch: u32,
        data: &OptDataset,
        progress: bool,
    ) -> Result<Pass, OptError> {
        #[allow(clippy::cast_precision_loss)]
        self.state.epoch.set(epoch as f32);
        self.pass(data, true, progress)
    }

    /// Evaluate on every window of `data`, without changing the weights.
    ///
    /// # Errors
    ///
    /// [`OptError::WindowMismatch`] for windows not the context's length.
    pub fn evaluate(&mut self, data: &OptDataset, progress: bool) -> Result<Pass, OptError> {
        self.pass(data, false, progress)
    }

    fn pass(&mut self, data: &OptDataset, train: bool, progress: bool) -> Result<Pass, OptError> {
        let n_ctx = self.ctx.n_ctx();
        if data.n_ctx() != n_ctx as usize {
            return Err(OptError::WindowMismatch {
                windows: data.n_ctx(),
                n_ctx,
            });
        }
        let measured = OptResult::new();
        let unused = OptResult::new();
        let (result_train, result_eval) = if train {
            (&measured, &unused)
        } else {
            (&unused, &measured)
        };
        #[allow(clippy::cast_possible_wrap)]
        let split = if train { data.len() as i64 } else { 0 };
        let callback: llama_cpp_sys_2::ggml_opt_epoch_callback = if progress {
            Some(llama_cpp_sys_2::ggml_opt_epoch_callback_progress_bar)
        } else {
            None
        };
        let started = Instant::now();
        // SAFETY: the context is held mutably by `self`; the dataset and
        // both results outlive the call.
        unsafe {
            llama_cpp_sys_2::llama_opt_epoch(
                self.ctx.context.as_ptr(),
                data.ptr.as_ptr(),
                result_train.0.as_ptr(),
                result_eval.0.as_ptr(),
                split,
                callback,
                callback,
            );
        }
        if progress {
            eprintln!();
        }
        let (tokens, loss, loss_unc, accuracy, accuracy_unc) = measured.read();
        Ok(Pass {
            tokens,
            loss,
            loss_unc,
            accuracy,
            accuracy_unc,
            seconds: started.elapsed().as_secs_f64(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_cover_the_stream_and_the_last_one_fits() {
        assert_eq!(window_count(10, 4, 2), 3); // starts 0, 2, 4; 4+4+1 = 9 <= 10
        assert_eq!(window_count(5, 4, 1), 1);
        assert_eq!(window_count(4, 4, 1), 0);
        assert_eq!(window_count(100, 4, 0), 0);
    }

    #[test]
    fn dataset_labels_are_the_stream_shifted_by_one() {
        let tokens: Vec<LlamaToken> = (0..11).map(LlamaToken).collect();
        let ds = OptDataset::from_tokens(&tokens, 4, 3).unwrap();
        assert_eq!(ds.len(), 3);
        assert_eq!(ds.window(0), Some((vec![0, 1, 2, 3], vec![1, 2, 3, 4])));
        assert_eq!(ds.window(2), Some((vec![6, 7, 8, 9], vec![7, 8, 9, 10])));
        assert_eq!(ds.window(3), None);
    }

    #[test]
    fn dataset_refuses_what_makes_no_window() {
        let tokens: Vec<LlamaToken> = (0..4).map(LlamaToken).collect();
        assert_eq!(
            OptDataset::from_tokens(&tokens, 4, 1).unwrap_err(),
            OptError::TooFewTokens { have: 4, n_ctx: 4 }
        );
        assert_eq!(
            OptDataset::from_tokens(&tokens, 2, 0).unwrap_err(),
            OptError::ZeroStride
        );
    }

    #[test]
    fn schedule_matches_llama_cpp() {
        // Constant without a floor.
        let c = LrSchedule::new(1e-4, -1.0, -1.0, 0.0, 3);
        assert_eq!((c.lr(0.0), c.lr(2.0)), (1e-4, 1e-4));
        // Halving from 1e-4 to 1e-5 over 4 epochs: log2(10) halvings spread
        // evenly, the floor from epoch 4 on.
        let d = LrSchedule::new(1e-4, 1e-5, -1.0, 0.0, 4);
        assert!((d.lr(0.0) - 1e-4).abs() < 1e-9);
        assert!((d.lr(2.0) - 1e-4 * 0.1f32.sqrt()).abs() < 1e-8);
        assert_eq!(d.lr(4.0), 1e-5);
        assert_eq!(d.decay_epochs, 4.0);
        // A decay shorter than the run.
        let s = LrSchedule::new(1e-4, 1e-5, 2.0, 0.0, 10);
        assert_eq!(s.lr(2.0), 1e-5);
        assert!(s.lr(1.0) > 1e-5);
    }

    #[test]
    fn schedules_ggml_would_abort_on_are_refused() {
        assert!(LrSchedule::new(1e-4, -1.0, -1.0, 0.0, 2).check().is_ok());
        assert!(LrSchedule::new(1e-4, 1e-5, -1.0, 0.1, 2).check().is_ok());
        assert!(LrSchedule::new(0.0, -1.0, -1.0, 0.0, 2).check().is_err());
        assert!(LrSchedule::new(-1e-4, -1.0, -1.0, 0.0, 2).check().is_err());
        assert!(LrSchedule::new(f32::NAN, -1.0, -1.0, 0.0, 2)
            .check()
            .is_err());
        assert!(LrSchedule::new(1e-4, -1.0, -1.0, 1.5, 2).check().is_err());
        assert!(LrSchedule::new(1e-4, -1.0, -1.0, -0.1, 2).check().is_err());
    }

    #[test]
    fn glob_patterns() {
        assert!(glob_match("blk.*.ffn_*", "blk.3.ffn_down.weight"));
        assert!(glob_match("output.weight", "output.weight"));
        assert!(glob_match("*", ""));
        assert!(glob_match("blk.1*", "blk.1"));
        assert!(!glob_match("blk.1.*", "blk.10.attn_q.weight"));
        assert!(!glob_match("output.weight", "output_norm.weight"));
        assert!(glob_match("*.attn_*.weight", "blk.0.attn_q.weight"));
        let f = TensorFilter::new(vec!["blk.*.attn_*".into(), "output.weight".into()]);
        assert!(f.accepts("blk.2.attn_k.weight") && f.accepts("output.weight"));
        assert!(!f.accepts("blk.2.ffn_up.weight"));
        assert!(TensorFilter::all().accepts("anything"));
    }

    #[test]
    fn optimizers_parse_and_size() {
        assert_eq!("AdamW".parse::<Optimizer>(), Ok(Optimizer::AdamW));
        assert_eq!("sgd".parse::<Optimizer>(), Ok(Optimizer::Sgd));
        assert!("adam".parse::<Optimizer>().is_err());
        assert_eq!(Optimizer::AdamW.state_bytes_per_param(), 8);
        assert_eq!(Optimizer::Sgd.to_string(), "sgd");
    }
}
