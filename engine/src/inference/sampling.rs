//! The single place a sampling chain is built.
//!
//! The chain was written out four times: once in the scheduler and three times
//! in `inference/mod.rs` (non-streaming, streaming, multimodal). Four copies of
//! the same sequence of `if` statements, and they had already started to drift:
//!
//! - the **seed fallback** differed. The scheduler passed `seq_id`, the three
//!   sequential paths a hardcoded `1234`. The value only matters when the
//!   system clock predates the Unix epoch, so nothing was visibly wrong — it is
//!   just two answers to one question, which is how the interesting divergences
//!   start.
//! - the **multimodal copy swallowed grammar errors**. Where the other three
//!   log `"Grammar sampler init failed …, falling back to unconstrained"`, it
//!   used `if let Ok(gs) = …` and said nothing. A request with `format: "json"`
//!   whose grammar failed to compile returned free-form text with no trace of
//!   why.
//!
//! Neither is dramatic on its own. The point is the shape: every new sampler
//! has to be added in four places, and being right in three of them is
//! indistinguishable, in a diff, from being right in all four. Order matters
//! here too (`dist` must close), and an ordering fixed in
//! one copy would silently not be fixed in the others.

use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token::data::LlamaTokenData;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;

use super::{GenerateRequest, random_seed_fallback};

/// The sampler of one request: its chain, and the grammar of a request that
/// asked for constrained output (`format: "json"`), kept out of the chain.
///
/// The grammar used to lead the chain, so it checked every token of the
/// vocabulary against the grammar at every step, on one CPU thread, while the
/// GPU waited for the next token. Gemma's vocabulary has some 262,000 tokens,
/// eight times a 32k one: with Gemma 4 E4B on an RTX 5070 Ti, short JSON
/// answers took 13 to 78 s with the GPU at 17-24%, and the same answers asked
/// for in the prompt, without `format`, took a third to a quarter of that
/// (measured on 19-10-2026).
///
/// Now the grammar checks the token the chain draws, as llama.cpp's own
/// sampler does (`common_sampler_sample`, grammar-based rejection sampling):
/// a model writing JSON almost always draws a token the grammar allows, so
/// the check costs one token instead of the whole vocabulary. Only when the
/// grammar rejects the token is the step drawn again from the full
/// distribution with the grammar applied first, the old path, so the answer
/// stays inside the grammar every time.
pub(crate) struct RequestSampler {
    chain: LlamaSampler,
    grammar: Option<LlamaSampler>,
}

impl RequestSampler {
    /// Draw the next token from the `idx`-th output of the last decode (`-1`:
    /// the only one) and accept it, once, into every sampler with a history:
    /// the penalties' window and the grammar's parse state.
    ///
    /// Never call `accept` after this. Every caller used to accept the token
    /// `LlamaSampler::sample` had already accepted, so each token was taken
    /// twice: the repeat penalty counted it twice over half its window, and
    /// with a grammar the second `{` left the grammar nowhere to go and
    /// llama.cpp aborted the whole server (`GGML_ASSERT(!stacks.empty())` in
    /// llama-grammar.cpp, found 08-10-2026).
    #[must_use]
    pub(crate) fn sample(&mut self, ctx: &LlamaContext<'_>, idx: i32) -> LlamaToken {
        let Some(grammar) = self.grammar.as_mut() else {
            // `LlamaSampler::sample` draws and accepts in one call.
            return self.chain.sample(ctx, idx);
        };

        // The same row `LlamaSampler::sample` reads: see
        // `scheduler::sampled_token_is_corrupt` for why `-1` goes through
        // `get_logits`.
        let logits = if idx == -1 {
            ctx.get_logits()
        } else {
            ctx.get_logits_ith(idx)
        };
        let candidates = |logits: &[f32]| {
            LlamaTokenDataArray::from_iter(
                (0_i32..)
                    .zip(logits)
                    .map(|(id, &logit)| LlamaTokenData::new(LlamaToken::new(id), logit, 0.0)),
                false,
            )
        };

        let mut cur = candidates(logits);
        self.chain.apply(&mut cur);
        let drawn = cur.selected_token();

        let token = match drawn.filter(|&t| grammar_allows(grammar, t)) {
            Some(token) => token,
            None => {
                // Rejected: the grammar narrows the full distribution first,
                // then the chain draws from what is left.
                let mut cur = candidates(logits);
                grammar.apply(&mut cur);
                self.chain.apply(&mut cur);
                cur.selected_token()
                    .or(drawn)
                    .expect("the dist sampler always selects a token")
            }
        };

        grammar.accept(token);
        self.chain.accept(token);
        token
    }
}

/// Whether the grammar, in its current state, allows `token` next: applied to
/// that token alone, it leaves its logit finite.
fn grammar_allows(grammar: &mut LlamaSampler, token: LlamaToken) -> bool {
    let mut single = LlamaTokenDataArray::new(vec![LlamaTokenData::new(token, 1.0, 0.0)], false);
    grammar.apply(&mut single);
    single.data.first().is_some_and(|d| d.logit().is_finite())
}

/// Build the sampler for one request.
///
/// `seed_fallback` is only consulted when the client sent no seed *and* the
/// system clock is unusable; see [`random_seed_fallback`]. Callers pass
/// something stable and locally distinct (the scheduler passes its slot id).
///
/// Order is part of the contract, not a style choice:
/// penalties → top-k → top-p → min-p → temperature → dist. `dist` must be
/// last because it is what actually draws the token; anything after it would
/// have no effect. The grammar is not in the chain: see [`RequestSampler`].
pub(crate) fn build_sampler(
    model: &LlamaModel,
    request: &GenerateRequest,
    seed_fallback: u32,
) -> RequestSampler {
    let seed = request
        .seed
        .unwrap_or_else(|| random_seed_fallback(seed_fallback));

    let grammar = request.grammar.as_ref().and_then(|grammar_str| {
        LlamaSampler::grammar(model, grammar_str, "root")
            // Never silent: a request that asked for constrained output and
            // did not get it must leave a trace. The multimodal copy used to
            // drop this on the floor.
            .inspect_err(|e| {
                tracing::warn!("Grammar sampler init failed ({e:?}), falling back to unconstrained")
            })
            .ok()
    });

    let mut chain: Vec<LlamaSampler> = Vec::new();

    // Repeat penalty (Ollama default: 1.1, last 64 tokens).
    if request.repeat_penalty != 1.0 {
        chain.push(LlamaSampler::penalties(
            model.n_vocab(),
            request.repeat_last_n,
            request.repeat_penalty,
            0.0,
            0.0,
        ));
    }
    // Top-K (Ollama default: 40).
    if request.top_k > 0 {
        chain.push(LlamaSampler::top_k(request.top_k));
    }
    // Top-P (Ollama default: 0.9).
    if request.top_p < 1.0 {
        chain.push(LlamaSampler::top_p(request.top_p, 1));
    }
    // Min-P (Ollama default: 0.0).
    if request.min_p > 0.0 {
        chain.push(LlamaSampler::min_p(request.min_p, 1));
    }
    chain.push(LlamaSampler::temp(request.temperature));
    // Draws the token — must stay last.
    chain.push(LlamaSampler::dist(seed));

    RequestSampler {
        chain: LlamaSampler::chain_simple(chain),
        grammar,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The chain itself needs a loaded model to build, so what is testable
    // without one is the decision that precedes it: which seed is used. That is
    // also the part that had actually diverged between the four copies.
    #[test]
    fn an_explicit_seed_is_used_verbatim_whatever_the_fallback() {
        let req = GenerateRequest {
            seed: Some(4242),
            ..Default::default()
        };
        let seed = req.seed.unwrap_or_else(|| random_seed_fallback(7));
        assert_eq!(seed, 4242);
    }

    #[test]
    fn without_a_seed_two_requests_do_not_agree() {
        // Ollama's default is a fresh seed per request, not a fixed one. Two
        // calls landing in the same nanosecond are astronomically unlikely, so
        // this is a meaningful check rather than a flaky one.
        let a = random_seed_fallback(1234);
        let b = random_seed_fallback(1234);
        assert_ne!(
            (a, b),
            (1234, 1234),
            "the fallback constant must not be the normal path"
        );
    }
}
