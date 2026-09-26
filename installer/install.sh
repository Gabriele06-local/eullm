#!/bin/sh
# EuLLM Engine installer for Linux and macOS.
#
#   curl -fsSL https://raw.githubusercontent.com/eullm/eullm/main/installer/install.sh | sh
#
# Picks the release binary that fits this machine, verifies it against the
# release's checksums.txt, and installs it as `eullm` in a per-user directory.
# No root, no sudo, nothing outside the install directory is touched.
#
# Environment variables (all optional):
#   EULLM_VERSION      Release to install, e.g. 0.7.9 (default: latest stable)
#   EULLM_INSTALL_DIR  Where to put the binary (default: ~/.local/bin)
#   EULLM_VARIANT      Force a build instead of detecting one: cpu, cuda,
#                      cuda-datacenter, vulkan, rocm-consumer, rocm-gfx90a,
#                      cix-p1
#
# The Windows counterpart is installer/install.ps1.

set -eu

REPO="eullm/eullm"

say()  { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }

need() { command -v "$1" >/dev/null 2>&1; }

download() {
  # $1 url, $2 destination
  if need curl; then
    curl -fL --proto '=https' --tlsv1.2 --retry 3 --progress-bar -o "$2" "$1"
  elif need wget; then
    wget -q --show-progress -O "$2" "$1"
  else
    die "neither curl nor wget is installed"
  fi
}

sha256_of() {
  if need sha256sum; then
    sha256sum "$1" | cut -d' ' -f1
  elif need shasum; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    die "neither sha256sum nor shasum is installed, cannot verify the download"
  fi
}

# Prints the NVIDIA driver major version and the compute capability of the
# first GPU as "<driver_major> <compute_cap>", or nothing without a usable
# nvidia-smi. compute_cap needs a driver from 2021 or later; older drivers
# print an error there, which lands in the "no usable GPU" branch.
nvidia_info() {
  need nvidia-smi || return 0
  nvidia-smi --query-gpu=driver_version,compute_cap --format=csv,noheader 2>/dev/null \
    | head -n1 \
    | awk -F', *' '$2 ~ /^[0-9]+\.[0-9]+$/ { split($1, v, "."); print v[1], $2 }'
}

# Linux x86_64: CUDA when an NVIDIA GPU with a recent enough driver is
# present, otherwise the CPU build. ROCm and Vulkan need libraries the
# binary does not bundle, so they are only chosen through EULLM_VARIANT.
detect_linux_x64() {
  info=$(nvidia_info)
  if [ -z "$info" ]; then
    echo cpu
    return
  fi
  driver=${info%% *}
  cap=${info#* }
  case "$cap" in
    8.0|9.0)
      # A100 / H100: the data-centre build, CUDA 12.4, driver 550+.
      if [ "$driver" -ge 550 ]; then echo cuda-datacenter; return; fi
      warn "NVIDIA driver $driver is older than 550, which the A100/H100 build needs; installing the CPU build"
      ;;
    8.6|8.9|12.0)
      # RTX 3000 / 4000 / 5000: the consumer build, CUDA 13.1, driver 580+.
      if [ "$driver" -ge 580 ]; then echo cuda; return; fi
      warn "NVIDIA driver $driver is older than 580, which the CUDA build needs; installing the CPU build (update the driver and re-run to get GPU support)"
      ;;
    *)
      warn "NVIDIA GPU with compute capability $cap is not covered by a prebuilt CUDA binary; installing the CPU build (EULLM_VARIANT=vulkan is an alternative)"
      ;;
  esac
  echo cpu
}

asset_for() {
  # $1 os, $2 arch, $3 variant
  case "$1/$2/$3" in
    linux/x86_64/cpu)             echo eullm-linux-x64 ;;
    linux/x86_64/cuda)            echo eullm-linux-x64-cuda-13.1 ;;
    linux/x86_64/cuda-datacenter) echo eullm-linux-x64-cuda-12.4-datacenter ;;
    linux/x86_64/vulkan)          echo eullm-linux-x64-vulkan ;;
    linux/x86_64/rocm-consumer)   echo eullm-linux-x64-rocm-consumer ;;
    linux/x86_64/rocm-gfx90a)     echo eullm-linux-x64-rocm-gfx90a ;;
    linux/aarch64/cpu)            echo eullm-linux-arm64 ;;
    linux/aarch64/cuda)           echo eullm-linux-arm64-cuda-13.1 ;;
    linux/aarch64/cix-p1)         echo eullm-linux-arm64-cix-p1 ;;
    darwin/arm64/cpu)             echo eullm-macos-arm64 ;;
    darwin/x86_64/cpu)            echo eullm-macos-x64 ;;
    *) return 1 ;;
  esac
}

main() {
  os=$(uname -s | tr '[:upper:]' '[:lower:]')
  arch=$(uname -m)
  case "$arch" in
    amd64) arch=x86_64 ;;
    arm64) [ "$os" = linux ] && arch=aarch64 ;;
  esac
  # Under Rosetta uname reports x86_64 on an Apple Silicon Mac; the native
  # build is the one that gets Metal.
  if [ "$os" = darwin ] && [ "$arch" = x86_64 ] \
     && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
    arch=arm64
  fi

  case "$os" in
    linux|darwin) ;;
    *) die "unsupported OS '$os' — on Windows use: irm https://raw.githubusercontent.com/$REPO/main/installer/install.ps1 | iex" ;;
  esac

  variant=${EULLM_VARIANT:-}
  if [ -z "$variant" ]; then
    if [ "$os/$arch" = linux/x86_64 ]; then
      variant=$(detect_linux_x64)
    elif [ "$os/$arch" = linux/aarch64 ] && [ -n "$(nvidia_info)" ]; then
      driver=$(nvidia_info | cut -d' ' -f1)
      if [ "$driver" -ge 580 ]; then variant=cuda; else variant=cpu; fi
    else
      # macOS arm64 is the Metal build already; the rest are CPU-only.
      variant=cpu
    fi
  fi

  asset=$(asset_for "$os" "$arch" "$variant") \
    || die "no '$variant' build for $os/$arch — see https://github.com/$REPO/releases/latest"

  if [ -n "${EULLM_VERSION:-}" ]; then
    base="https://github.com/$REPO/releases/download/EuLLM-v${EULLM_VERSION#v}"
  else
    base="https://github.com/$REPO/releases/latest/download"
  fi

  install_dir=${EULLM_INSTALL_DIR:-$HOME/.local/bin}

  tmp=$(mktemp -d)
  trap 'rm -rf "$tmp"' EXIT INT TERM

  say "Installing $asset (variant: $variant) into $install_dir"
  download "$base/checksums.txt" "$tmp/checksums.txt"
  download "$base/$asset" "$tmp/$asset"

  # checksums.txt lines look like "<hash>  <artifact-dir>/<file>", so match
  # on the file name at the end of the path, not on the whole path.
  expected=$(awk -v f="$asset" '{ n = $2; sub(/.*\//, "", n); if (n == f) { print $1; exit } }' "$tmp/checksums.txt")
  [ -n "$expected" ] || die "$asset is not listed in checksums.txt, refusing to install an unverified binary"
  actual=$(sha256_of "$tmp/$asset")
  [ "$expected" = "$actual" ] || die "checksum mismatch for $asset (expected $expected, got $actual)"
  say "Checksum OK"

  mkdir -p "$install_dir"
  chmod +x "$tmp/$asset"
  # Move into place in one step so a running `eullm` is never left
  # half-overwritten.
  mv -f "$tmp/$asset" "$install_dir/eullm"

  say ""
  say "EuLLM installed: $install_dir/eullm"
  case ":$PATH:" in
    *":$install_dir:"*) ;;
    *)
      say ""
      say "$install_dir is not in your PATH. Add it with:"
      say "  echo 'export PATH=\"$install_dir:\$PATH\"' >> ~/.profile && . ~/.profile"
      ;;
  esac
  say ""
  say "Try it:"
  say "  eullm run hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M"
}

main "$@"
