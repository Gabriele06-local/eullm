"""Decision models trained on your own decisions (Reflex, MVP 4).

A server started with `EULLM_DECISION_TRACES=<dir>` writes the decisions it
computes, and whoever learns the right answer writes feedback beside them.
This package turns that into a small decision model of your own:

    eullm-forge decisions build  <traces> -o <dataset>   # traces → examples
    eullm-forge decisions train  <dataset> -o <run>      # LoRA on the answer code
    eullm-forge decisions export <run> -o model.gguf     # merge → GGUF

The model is trained for the engine's codes readout, the prompt
`eullm serve --decision-model` shows any chat model (`prompt`), so the GGUF
serves unchanged. Whether it may replace the decision model in service is
for `bench/reflexbench/qualify.py` to say, not this package.

The RAG gate's sets are labelled already: `rag` writes them as traces
(`eullm-forge decisions import-rag`), the prompt built by the gate's own
code in bench/reflexbench, the split by question and by document.

`prompt`, `traces`, `teachers`, `dataset`, `metrics` and `rag` need only
the standard library; `train` needs torch, transformers and peft.
"""
