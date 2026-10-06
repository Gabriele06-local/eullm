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
Ministral 8B).
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


def same_vocabulary(tok_a, tok_b, probe: str = "Il ricorso è respinto ai sensi dell'art. 120 "
                                                "c.p.a., con condanna alle spese.") -> bool:
    """Whether two tokenizers give the same ids: the same vocabulary, or at least
    the same encoding of an ordinary legal sentence and the same size."""
    try:
        if tok_a.get_vocab() == tok_b.get_vocab():
            return True
    except (AttributeError, NotImplementedError):
        pass
    return (len(tok_a) == len(tok_b)
            and tok_a(probe, add_special_tokens=False)["input_ids"]
            == tok_b(probe, add_special_tokens=False)["input_ids"])
