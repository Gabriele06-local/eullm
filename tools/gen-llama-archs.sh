#!/usr/bin/env bash
# Regenerate engine/src/llama_archs.rs from the vendored llama.cpp.
#
# `llama.h` exposes no way to ask, at runtime, which model architectures the
# library was compiled with. The list therefore has to come from the source we
# compile — `LLM_ARCH_NAMES` in `src/llama-arch.cpp` — and the only way for it
# to stay true is to be derived from that file rather than maintained by hand.
#
# Run this after bumping the llama.cpp submodule. The engine CI job runs it and
# fails if the result differs from what is committed, so a bump that adds an
# architecture cannot land with the list silently stale.
set -euo pipefail

cd "$(dirname "$0")/.."
SRC="engine/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/src/llama-arch.cpp"
OUT="engine/src/llama_archs.rs"

[ -f "$SRC" ] || {
    echo "error: $SRC not found — is the llama.cpp submodule checked out?" >&2
    echo "       git submodule update --init --recursive" >&2
    exit 1
}

# Only the LLM_ARCH_NAMES block, so an unrelated `{ LLM_ARCH_*, "..." }`
# elsewhere in the file can never leak in. `(unknown)` is the sentinel the
# lookup returns on a miss, not an architecture, and its parentheses keep it
# out of the pattern anyway.
NAMES=$(
    awk '/LLM_ARCH_NAMES *= *\{/{f=1; next} f && /^};/{exit} f' "$SRC" \
    | grep -oE '\{ *LLM_ARCH_[A-Z0-9_]+, *"[a-z0-9_.-]+" *\}' \
    | sed 's/.*"\(.*\)".*/\1/' \
    | sort -u
)

COUNT=$(printf '%s\n' "$NAMES" | grep -c .)
# A parse that silently returns nothing would generate an empty list, and an
# empty list marks every model unsupported. Refuse instead.
[ "$COUNT" -ge 50 ] || {
    echo "error: parsed only $COUNT architectures from $SRC — the table's shape probably changed." >&2
    exit 1
}

{
    echo "//! Model architectures this build of llama.cpp can load."
    echo "//!"
    echo "//! GENERATED FILE — do not edit by hand. Regenerate with"
    echo "//! \`tools/gen-llama-archs.sh\` after bumping the llama.cpp submodule;"
    echo "//! the engine CI job regenerates it and fails on any difference, so the"
    echo "//! list cannot drift away from the code it describes."
    echo "//!"
    echo "//! It is read out of \`LLM_ARCH_NAMES\` in the vendored"
    echo "//! \`src/llama-arch.cpp\` because \`llama.h\` offers no way to enumerate"
    echo "//! them at runtime. Deriving it from the source the binary is compiled"
    echo "//! from is what makes it describe *this* build rather than some other."
    echo
    echo "/// Every \`general.architecture\` value this build understands, sorted."
    echo "pub const SUPPORTED_ARCHITECTURES: &[&str] = &["
    printf '%s\n' "$NAMES" | sed 's/.*/    "&",/'
    echo "];"
    echo
    echo "/// Whether this build can load a model declaring \`arch\`."
    echo "///"
    echo "/// Callers must distinguish \"declares an architecture we do not have\""
    echo "/// from \"declares none at all\": only the first is a reason to tell a"
    echo "/// user the model will not load."
    echo "pub fn is_supported(arch: &str) -> bool {"
    echo "    SUPPORTED_ARCHITECTURES.binary_search(&arch).is_ok()"
    echo "}"
    # Emitted, not appended by hand: this script rewrites the whole file, so a
    # test written into it directly survives only until the next bump. These
    # check the two ways the parse above can go wrong quietly — a short list
    # (marks every model unsupported) and an unsorted one (binary_search then
    # reports real architectures as unknown).
    cat <<'RUST_TESTS'

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_list_is_sorted_and_plausibly_long() {
        assert!(
            SUPPORTED_ARCHITECTURES.len() > 50,
            "only {} architectures — the generator's parse broke, and a short \
             list marks models unsupported that this build loads fine",
            SUPPORTED_ARCHITECTURES.len()
        );
        assert!(
            SUPPORTED_ARCHITECTURES.windows(2).all(|w| w[0] < w[1]),
            "not sorted, or holds a duplicate: `is_supported` binary-searches \
             and would answer wrongly"
        );
    }

    #[test]
    fn the_long_standing_architectures_are_there() {
        for arch in ["llama", "qwen3", "gemma3", "phi3"] {
            assert!(
                is_supported(arch),
                "{arch} missing — the parse read the wrong table"
            );
        }
        assert!(!is_supported(""));
        assert!(!is_supported("definitely-not-an-architecture"));
    }
}
RUST_TESTS
} > "$OUT"

echo "wrote $OUT ($COUNT architectures)"
