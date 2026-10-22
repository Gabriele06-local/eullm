#!/bin/sh
# EuLLM Decision Desk, in one command, on Linux and macOS:
#
#   curl -fsSL https://raw.githubusercontent.com/eullm/eullm/main/examples/decision-desk/start.sh | sh
#
# or, from a clone of the repository:
#
#   sh examples/decision-desk/start.sh
#
# Installs EuLLM if it is not installed (installer/install.sh: one binary in
# ~/.local/bin, no root), pulls the three models (about 7 GB the first time),
# starts `eullm serve` with them unless a server already answers, and opens
# the page. Ctrl+C stops the page and the server this script started.
#
# Environment variables (all optional):
#   DESK_SMALL=1   the Jev-Style 0.8B instead of the 2B, for a GPU of 8 GB
#   DESK_CHAT      the chat model (default gemma-4-e4b)
#   DESK_PORT      the page's port (default 8770)
#   EULLM_PORT     the server's port (default 11434)
#   DESK_DIR       where the server's log goes, and the page's files when this
#                  script runs from a pipe (default ~/.eullm/demos/decision-desk)
#   DESK_REF       the branch or tag those files come from (default main)
#
# Everything runs on this machine; the only downloads are EuLLM, the models
# and, from a pipe, the page's two files.

set -eu

REPO="eullm/eullm"

say()  { printf '%s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1; }

# Whether $1 answers HTTP within two seconds: python3 is needed for the page
# anyway, so it does the asking, whatever else is installed, and asks
# directly rather than through a proxy in the environment.
answers() {
  python3 -c 'import sys, urllib.request as u; u.build_opener(u.ProxyHandler({})).open(sys.argv[1], timeout=2)' "$1" \
    >/dev/null 2>&1
}

fetch() {
  # $1 url, $2 destination
  if need curl; then
    curl -fsSL --proto '=https' --tlsv1.2 --retry 3 -o "$2" "$1"
  elif need wget; then
    wget -q -O "$2" "$1"
  else
    die "neither curl nor wget is installed"
  fi
}

main() {
  need python3 || die "python3 is needed for the page: install it (sudo apt install python3, brew install python) and run this again"

  port=${EULLM_PORT:-11434}
  url="http://127.0.0.1:$port"
  chat=${DESK_CHAT:-gemma-4-e4b}
  embedding=qwen3-embedding-0.6b-gguf-q8_0
  if [ "${DESK_SMALL:-0}" = 1 ]; then
    decision_ref=hf.co/chaoliangUNSW/Jev-Style-0.8B-Decision-v3-GGUF:Q4_K_M
    decision=jev-style-0.8b-decision-v3-gguf-q4_k_m
  else
    decision_ref=hf.co/chaoliangUNSW/Jev-Style-2B-Decision-v3-GGUF:Q4_K_M
    decision=jev-style-2b-decision-v3-gguf-q4_k_m
  fi

  # 1. The page: next to this script in a clone, downloaded from a pipe.
  here=$(dirname "$0")
  if [ -f "$here/desk.py" ] && [ -f "$here/desk.html" ]; then
    dir=$(cd "$here" && pwd)
  else
    dir=${DESK_DIR:-$HOME/.eullm/demos/decision-desk}
    mkdir -p "$dir"
    raw="https://raw.githubusercontent.com/$REPO/${DESK_REF:-main}/examples/decision-desk"
    say "Downloading the page into $dir"
    fetch "$raw/desk.py" "$dir/desk.py"
    fetch "$raw/desk.html" "$dir/desk.html"
  fi

  # 2. EuLLM itself.
  if ! need eullm; then
    if [ -x "${EULLM_INSTALL_DIR:-$HOME/.local/bin}/eullm" ]; then
      PATH="${EULLM_INSTALL_DIR:-$HOME/.local/bin}:$PATH"
    else
      say "EuLLM is not installed: installing it"
      installer=$(mktemp)
      fetch "https://raw.githubusercontent.com/$REPO/main/installer/install.sh" "$installer"
      sh "$installer" </dev/null
      rm -f "$installer"
      PATH="${EULLM_INSTALL_DIR:-$HOME/.local/bin}:$PATH"
    fi
  fi
  need eullm || die "eullm is not in PATH after installing it"

  # 3. The three models: a model already pulled is not downloaded again.
  say ""
  say "Models: $decision decides, $embedding finds, $chat writes"
  eullm pull "$decision_ref" </dev/null
  eullm pull hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0 </dev/null
  eullm pull "$chat" </dev/null

  # 4. The server: one already answering is used as it is.
  serve_pid=
  if answers "$url/api/version"; then
    say ""
    say "Using the EuLLM server already running on $url. It needs"
    say "  --decision-model $decision --embedding-model $embedding"
    say "the page shows in red what it lacks."
  else
    logs=${DESK_DIR:-$HOME/.eullm/demos/decision-desk}
    mkdir -p "$logs"
    log="$logs/eullm-serve.log"
    say ""
    say "Starting eullm serve on $url (log: $log)"
    eullm serve --port "$port" \
      --decision-model "$decision" \
      --embedding-model "$embedding" \
      --default-model "$chat" </dev/null >"$log" 2>&1 &
    serve_pid=$!
    trap 'kill "$serve_pid" 2>/dev/null || true' EXIT INT TERM
    waited=0
    until answers "$url/api/version"; do
      kill -0 "$serve_pid" 2>/dev/null || die "eullm serve stopped; the end of $log says why"
      waited=$((waited + 1))
      [ "$waited" -le 300 ] || die "eullm serve did not answer in 5 minutes; see $log"
      sleep 1
    done
  fi

  # 5. The page. The chat model loads while it opens.
  say ""
  python3 "$dir/desk.py" --url "$url" --port "${DESK_PORT:-8770}" \
    --decision-model "$decision" --embedding-model "$embedding" --chat-model "$chat" </dev/null
}

# Everything is in main, called on the last line, so that a download cut
# off halfway through `curl | sh` runs nothing at all.
main "$@"
