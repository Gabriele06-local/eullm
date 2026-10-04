#!/usr/bin/env python3
"""Publish a GGUF and its model card to a Hugging Face model repository.

    python forge/scripts/publish_hf.py \\
        --repo eullm/legal-it-8b \\
        --gguf $GG/legal-it-8b-grpo-ministral-v04/legal-it-8b-grpo-ministral-v04-q4_k_m.gguf \\
        --name legal-it-8b-Q4_K_M.gguf \\
        --card forge/model_cards/legal-it-8b/README.md

($GG is $WORK/eullm_runs/stage3/gguf.) The repository is created PRIVATE
unless ``--public`` is given: look at the page first, then make it public
from its settings (or run again with --public). What it checks before
sending anything, because a 5 GB upload is a slow way to find a mistake:

* the file starts with the GGUF magic, so a partial or wrong file is not
  published under a model's name;
* the published name carries the quantization (``-Q4_K_M.gguf``), the form
  `ollama run hf.co/<repo>:Q4_K_M` looks for;
* the card has its YAML header and names the published file's repository,
  so a card copied from the other model is caught.

It prints the file's SHA-256, which is also what Hugging Face shows as the
file's LFS id and what the engine catalog (catalog/v1/catalog.json) records.
Run it on a machine with internet access (a login node); the token comes
from `hf auth login` or HF_TOKEN, never from the command line.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

GGUF_MAGIC = b"GGUF"
_QUANT_NAME = re.compile(r"-(Q\d_K_[SML]|Q\d_\d|Q\d_K|F16|BF16)\.gguf$")


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def problems(repo: str, gguf: Path, name: str, card: Path) -> list[str]:
    """Everything wrong with the inputs, before anything is sent."""
    out = []
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        out.append(f"repository {repo!r} is not <owner>/<name>")
    if not gguf.is_file():
        out.append(f"no file at {gguf}")
    else:
        with gguf.open("rb") as f:
            if f.read(4) != GGUF_MAGIC:
                out.append(f"{gguf} is not a GGUF file (no GGUF magic)")
    if not _QUANT_NAME.search(name):
        out.append(f"published name {name!r} must end with the quantization, "
                   "e.g. legal-it-8b-Q4_K_M.gguf")
    if not card.is_file():
        out.append(f"no model card at {card}")
    else:
        text = card.read_text(encoding="utf-8")
        if not text.startswith("---\n") or "\nlicense:" not in text.split("\n---", 1)[0]:
            out.append(f"{card} has no YAML header with a license")
        model = repo.split("/", 1)[-1]
        if f"# {model}\n" not in text:
            out.append(f"{card} is not the card of {model} (no '# {model}' title)")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="<owner>/<name> on Hugging Face")
    ap.add_argument("--gguf", required=True, type=Path)
    ap.add_argument("--name", required=True, help="file name in the repository")
    ap.add_argument("--card", required=True, type=Path, help="README.md to publish")
    ap.add_argument("--public", action="store_true",
                    help="create the repository public (default: private)")
    ap.add_argument("--dry-run", action="store_true", help="check and hash, send nothing")
    args = ap.parse_args(argv)

    found = problems(args.repo, args.gguf, args.name, args.card)
    for p in found:
        print(f"[publish] {p}", file=sys.stderr)
    if found:
        return 2
    digest = sha256(args.gguf)
    size = args.gguf.stat().st_size
    print(f"[publish] {args.name}: {size:,} bytes, sha256 {digest}", flush=True)
    if args.dry_run:
        print("[publish] dry run: nothing sent")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    print(f"[publish] as {who.get('name')}", flush=True)
    api.create_repo(args.repo, repo_type="model", private=not args.public, exist_ok=True)
    api.upload_file(path_or_fileobj=str(args.gguf), path_in_repo=args.name,
                    repo_id=args.repo, commit_message=f"Add {args.name}")
    api.upload_file(path_or_fileobj=str(args.card), path_in_repo="README.md",
                    repo_id=args.repo, commit_message="Model card")
    remote = {f.path: f for f in api.list_repo_tree(args.repo, expand=True)}
    lfs = getattr(remote.get(args.name), "lfs", None)
    remote_sha = getattr(lfs, "sha256", None) if lfs else None
    if remote_sha != digest:
        print(f"[publish] the uploaded file's sha256 is {remote_sha}, not {digest}",
              file=sys.stderr)
        return 1
    print(f"[publish] ok: https://huggingface.co/{args.repo} "
          f"({'public' if args.public else 'PRIVATE'}), sha256 verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
