//! How a model is laid out over several GPUs: `--split-mode` and
//! `--tensor-split`, llama.cpp's `llama_model_params` fields of the same
//! names (llama-server's `-sm` and `-ts`).
//!
//! The engine used llama.cpp's defaults until 15-10-2026: every layer whole
//! on one GPU, the layers dealt out by each GPU's free memory, so a request
//! runs through the GPUs one after another. On LUMI's MI250X that added no
//! throughput over one GCD (12-09-2026), and a second server on the same
//! GCDs doubled it (`docs/lumi/allocation-plan.md`). The tensor split cuts
//! each weight across the GPUs, which can make one request faster instead;
//! the `c03` campaign measures it. With one GPU visible none of this changes
//! anything.
//!
//! Not offered:
//! - `none`: the sizing (`fit::vram_bytes`) counts every visible GPU, and one
//!   GPU is chosen by making only it visible (`CUDA_VISIBLE_DEVICES`,
//!   `ROCR_VISIBLE_DEVICES`), as the campaign runner does;
//! - `row`: llama.cpp's CUDA backend, which is also its ROCm backend, no
//!   longer provides the split buffers it needs at the commit EuLLM pins, and
//!   the load fails ("device ROCm0 does not support split buffers", stock
//!   llama-bench on LUMI, 15-10-2026; only SYCL and Hexagon still have them);
//! - `--main-gpu`: llama.cpp reads it only with `none` and `row`.

use llama_cpp_2::model::params::{LlamaModelParams, LlamaSplitMode};

/// How the weights go across the GPUs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SplitMode {
    /// Whole layers per GPU, one GPU after another (llama.cpp's default).
    #[default]
    Layer,
    /// Weights and KV cache cut across the GPUs, which work on every layer
    /// together (llama.cpp's experimental tensor parallelism; not every
    /// architecture has it, and the load says so).
    Tensor,
}

impl SplitMode {
    fn llama(self) -> LlamaSplitMode {
        match self {
            SplitMode::Layer => LlamaSplitMode::Layer,
            SplitMode::Tensor => LlamaSplitMode::Tensor,
        }
    }

    fn name(self) -> &'static str {
        match self {
            SplitMode::Layer => "layer",
            SplitMode::Tensor => "tensor",
        }
    }
}

/// `--split-mode layer|tensor`.
pub fn parse_split_mode(s: &str) -> Result<SplitMode, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "layer" => Ok(SplitMode::Layer),
        "tensor" => Ok(SplitMode::Tensor),
        "none" => Err("`none` is not offered: make one GPU visible instead \
                       (CUDA_VISIBLE_DEVICES, ROCR_VISIBLE_DEVICES)"
            .to_string()),
        "row" => Err("`row` is not offered: llama.cpp's CUDA and ROCm backends no longer \
                      have the split buffers it needs, and the load fails; `tensor` cuts \
                      the weights across the GPUs instead"
            .to_string()),
        other => Err(format!("expected layer or tensor, got `{other}`")),
    }
}

/// `--tensor-split`'s value, one proportion per GPU (a type of its own, so
/// that clap takes the list as one value).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorSplit(pub Vec<f32>);

/// `--tensor-split 3,1`: the share of each GPU, as proportions in the order
/// the GPUs are listed. Separated by commas or slashes, as llama.cpp's.
pub fn parse_tensor_split(s: &str) -> Result<TensorSplit, String> {
    let split = s
        .split([',', '/'])
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| {
            p.parse::<f32>()
                .ok()
                .filter(|x| x.is_finite() && *x >= 0.0)
                .ok_or_else(|| format!("`{p}` is not a proportion (a number, 0 or more)"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if split.is_empty() || split.iter().all(|&x| x == 0.0) {
        return Err(format!("`{s}` gives no GPU a share"));
    }
    Ok(TensorSplit(split))
}

/// The flags together, as a load applies them.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GpuSplit {
    pub mode: SplitMode,
    /// Empty: llama.cpp splits by each GPU's free memory.
    pub tensor_split: Vec<f32>,
}

impl GpuSplit {
    /// The flags as `RuntimeOpts` holds them.
    pub fn from_flags(mode: SplitMode, tensor_split: Option<TensorSplit>) -> Self {
        GpuSplit {
            mode,
            tensor_split: tensor_split.map(|t| t.0).unwrap_or_default(),
        }
    }

    /// llama.cpp's defaults: nothing to apply, nothing to say.
    pub fn is_default(&self) -> bool {
        *self == GpuSplit::default()
    }

    /// The model's parameters with the split set.
    pub fn apply(&self, params: LlamaModelParams) -> Result<LlamaModelParams, String> {
        params
            .with_split_mode(self.mode.llama())
            .with_tensor_split(&self.tensor_split)
            .map_err(|e| format!("--tensor-split: {e}"))
    }

    /// For the load log: `tensor, 3:1`.
    pub fn describe(&self) -> String {
        let mut parts = vec![self.mode.name().to_string()];
        if !self.tensor_split.is_empty() {
            parts.push(
                self.tensor_split
                    .iter()
                    .map(|x| format!("{x}"))
                    .collect::<Vec<_>>()
                    .join(":"),
            );
        }
        parts.join(", ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_split_modes_parse_and_the_others_say_what_to_do_instead() {
        assert_eq!(parse_split_mode("layer"), Ok(SplitMode::Layer));
        assert_eq!(parse_split_mode("Tensor"), Ok(SplitMode::Tensor));
        assert!(parse_split_mode("none").unwrap_err().contains("VISIBLE_DEVICES"));
        assert!(parse_split_mode("row").unwrap_err().contains("split buffers"));
        assert!(parse_split_mode("columns").is_err());
    }

    #[test]
    fn a_tensor_split_is_proportions_with_at_least_one_share() {
        assert_eq!(parse_tensor_split("3,1"), Ok(TensorSplit(vec![3.0, 1.0])));
        assert_eq!(parse_tensor_split("1/1/2"), Ok(TensorSplit(vec![1.0, 1.0, 2.0])));
        assert_eq!(parse_tensor_split("0, 1"), Ok(TensorSplit(vec![0.0, 1.0])));
        assert!(parse_tensor_split("0,0").is_err());
        assert!(parse_tensor_split("1,-1").is_err());
        assert!(parse_tensor_split("a,b").is_err());
        assert!(parse_tensor_split("").is_err());
    }

    #[test]
    fn the_split_reaches_the_model_parameters() {
        assert!(GpuSplit::default().is_default());
        let split = GpuSplit::from_flags(SplitMode::Tensor, Some(TensorSplit(vec![3.0, 1.0])));
        assert!(!split.is_default());
        assert_eq!(split.describe(), "tensor, 3:1");
        let params = split.apply(LlamaModelParams::default()).unwrap();
        assert_eq!(params.split_mode(), Ok(LlamaSplitMode::Tensor));
        if llama_cpp_2::max_devices() >= 2 {
            assert_eq!(&params.tensor_split()[..2], &[3.0, 1.0]);
        }
        let plain = GpuSplit::default().apply(LlamaModelParams::default()).unwrap();
        assert_eq!(plain.split_mode(), Ok(LlamaSplitMode::Layer));
        assert!(plain.tensor_split().is_empty());
    }
}
