"""On-policy distillation with a privileged teacher: the loss and its checks.

The student writes its own answer to its own prompt; the teacher scores the
same answer tokens behind a prompt that also holds the source (the ruling
the question came from). At every answer position the loss is the reverse
KL divergence from the student's next-token distribution to the teacher's,
summed over the vocabulary:

    KL(p_s || p_t) = sum_v p_s(v) (log p_s(v) - log p_t(v))

Reverse, because it is mode-seeking: the student is pulled towards what the
teacher would say at the places the student actually goes, not asked to
cover everything the teacher might say. On the student's own samples,
because training on another model's text taught our Ministral the teacher's
style instead of its knowledge (research report of 2026-10-05; Thinking
Machines, On-Policy Distillation, 2025; TESSY, arXiv 2604.14164).

Teacher and student must share a tokenizer: the loss compares their
distributions token by token. `same_vocabulary` refuses a pair that does
not (Qwen3-30B-A3B can teach Qwen3-4B, Ministral-3-14B can teach
Ministral 8B, Qwen3.8 can teach Qwen3.5).
"""

from __future__ import annotations


def reverse_kl(student_logits, teacher_logits, mask=None):
    """Mean over unmasked positions of KL(p_s || p_t); logits are [..., T, V]."""
    import torch

    v = min(student_logits.shape[-1], teacher_logits.shape[-1])
    s = torch.log_softmax(student_logits[..., :v].float(), dim=-1)
    t = torch.log_softmax(teacher_logits[..., :v].float(), dim=-1)
    kl = (s.exp() * (s - t)).sum(-1)
    if mask is None:
        return kl.mean()
    mask = mask.to(kl.dtype)
    return (kl * mask).sum() / mask.sum().clamp(min=1)


#: How many tokens one tokenizer may have beyond the other and still count as
#: the same vocabulary: Qwen3.8 is Qwen3.5 plus seven audio tokens.
MAX_EXTRA_TOKENS = 16

_PROBE = "Il ricorso è respinto ai sensi dell'art. 120 c.p.a., con condanna alle spese."


def same_vocabulary(tok_a, tok_b, probe: str = _PROBE) -> bool:
    """Whether two tokenizers give the same ids, so their distributions can be
    compared token by token.

    Three ways to pass: the same vocabulary; one vocabulary being the other
    plus a few tokens of its own, every shared token at the same id and the
    extra ones after all of them (Qwen3.8 adds seven audio tokens to Qwen3.5's
    248,044 + 26 -- text never produces them, so they carry no probability
    that matters); or, when a tokenizer cannot list its vocabulary, the same
    size and the same encoding of an ordinary legal sentence (Mistral-Small-
    3.2-24B and Ministral 8B differ in two unused special tokens).
    """
    same_text = (tok_a(probe, add_special_tokens=False)["input_ids"]
                 == tok_b(probe, add_special_tokens=False)["input_ids"])
    try:
        va, vb = tok_a.get_vocab(), tok_b.get_vocab()
    except (AttributeError, NotImplementedError):
        return len(tok_a) == len(tok_b) and same_text
    if va == vb:
        return True
    small, big = sorted((va, vb), key=len)
    extra = set(big) - set(small)
    if (extra and len(extra) <= MAX_EXTRA_TOKENS
            and all(big.get(t) == i for t, i in small.items())
            and min(big[t] for t in extra) > max(small.values())):
        return same_text
    return len(tok_a) == len(tok_b) and same_text
