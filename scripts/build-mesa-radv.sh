#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

MESA_SRC_DIR="$REPO_ROOT/third_party/mesa"
BUILD_DIR="$REPO_ROOT/.cache/torch2vk/mesa-radv-build"
BUILD_VENV_DIR="$REPO_ROOT/.cache/torch2vk/mesa-build-venv"
PREFIX="$REPO_ROOT/.cache/torch2vk/mesa-radv"
SDK_ROOT="${TORCH2VK_VULKAN_SDK_ROOT:-$REPO_ROOT/.cache/torch2vk/vulkan-sdk}"
TOOLS_DIR="${TORCH2VK_BUILD_TOOLS_DIR:-$REPO_ROOT/.cache/torch2vk/bin}"
RPM_CACHE="${TORCH2VK_RPM_CACHE:-$REPO_ROOT/.cache/torch2vk/rpms}"
DNF_REPO_ARGS="${TORCH2VK_DNF_REPO_ARGS:---disablerepo=terra*}"
BUILD_TYPE="${MESA_BUILD_TYPE:-debugoptimized}"
CLEAN=0
RECONFIGURE=0
SETUP_ONLY=0
NO_INSTALL=0
SETUP_SDK=auto
EXTRA_MESON_OPTIONS=()
MESA_BUILD_PYTHON_REQUIREMENTS=(
  "meson>=1.4,<1.6"
  "mako"
  "ninja"
  "packaging"
  "setuptools"
  "PyYAML"
)

usage() {
  cat <<'USAGE'
Usage: scripts/build-mesa-radv.sh [options] [-- extra meson options...]

Build and install the repository Mesa fork as a local RADV-only Vulkan ICD.

Options:
  --build-dir DIR       Meson build directory
                       default: .cache/torch2vk/mesa-radv-build
  --build-venv DIR      Python venv for Mesa build tools
                       default: .cache/torch2vk/mesa-build-venv
  --prefix DIR          Local install prefix
                       default: .cache/torch2vk/mesa-radv
  --sdk-root DIR        Local SDK sysroot for Fedora Atomic/Bazzite deps
                       default: .cache/torch2vk/vulkan-sdk
  --tools-dir DIR       Local glslc/glslangValidator directory
                       default: .cache/torch2vk/bin
  --rpm-cache DIR       RPM download cache for local SDK setup
                       default: .cache/torch2vk/rpms
  --buildtype TYPE      Meson buildtype
                       default: debugoptimized
  --setup-sdk           Always prepare the local SDK before configuring Mesa
  --no-setup-sdk        Do not prepare the local SDK automatically
  --reconfigure         Reconfigure an existing Meson build directory
  --clean               Remove build dir and prefix before setup/build
  --setup-only          Configure Meson but do not compile
  --no-install          Compile but do not run ninja install
  -h, --help            Show this help

Environment:
  MESON                 Meson command. Default: local build venv meson
  NINJA                 Ninja command. Default: local build venv ninja
  NINJA_ARGS            Extra args passed to ninja, e.g. "-j16"
  MESA_BUILD_TYPE       Default buildtype if --buildtype is not provided
  LLVM_CONFIG           llvm-config path. Auto-detects ROCm LLVM if unset
  TORCH2VK_VULKAN_SDK_ROOT
                        Same as --sdk-root
  TORCH2VK_RPM_CACHE    Same as --rpm-cache
  TORCH2VK_DNF_REPO_ARGS
                        Extra dnf repo args, default: --disablerepo=terra*
  TORCH2VK_*_PACKAGE    Override package names used for local SDK setup
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-dir)
      BUILD_DIR=$2
      shift 2
      ;;
    --build-venv)
      BUILD_VENV_DIR=$2
      shift 2
      ;;
    --prefix)
      PREFIX=$2
      shift 2
      ;;
    --sdk-root)
      SDK_ROOT=$2
      shift 2
      ;;
    --tools-dir)
      TOOLS_DIR=$2
      shift 2
      ;;
    --rpm-cache)
      RPM_CACHE=$2
      shift 2
      ;;
    --buildtype)
      BUILD_TYPE=$2
      shift 2
      ;;
    --setup-sdk)
      SETUP_SDK=always
      shift
      ;;
    --no-setup-sdk)
      SETUP_SDK=never
      shift
      ;;
    --reconfigure)
      RECONFIGURE=1
      shift
      ;;
    --clean)
      CLEAN=1
      shift
      ;;
    --setup-only)
      SETUP_ONLY=1
      shift
      ;;
    --no-install)
      NO_INSTALL=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_MESON_OPTIONS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

BUILD_DIR=$(abs_path "$BUILD_DIR")
BUILD_VENV_DIR=$(abs_path "$BUILD_VENV_DIR")
PREFIX=$(abs_path "$PREFIX")
SDK_ROOT=$(abs_path "$SDK_ROOT")
TOOLS_DIR=$(abs_path "$TOOLS_DIR")
RPM_CACHE=$(abs_path "$RPM_CACHE")

prepend_env_path() {
  local var_name=$1
  local dir=$2
  local current=${!var_name:-}
  if [[ ! -d "$dir" ]]; then
    return
  fi
  case ":$current:" in
    *":$dir:"*) ;;
    *) export "$var_name=$dir${current:+:$current}" ;;
  esac
}

ensure_build_venv() {
  if [[ ! -x "$BUILD_VENV_DIR/bin/python" ]]; then
    mkdir -p -- "$(dirname -- "$BUILD_VENV_DIR")"
    if command -v uv >/dev/null 2>&1; then
      uv venv "$BUILD_VENV_DIR"
    elif command -v python3 >/dev/null 2>&1; then
      python3 -m venv "$BUILD_VENV_DIR"
    else
      echo "Missing uv or python3; cannot create Mesa build venv." >&2
      exit 1
    fi
  fi

  if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$BUILD_VENV_DIR/bin/python" "${MESA_BUILD_PYTHON_REQUIREMENTS[@]}"
  else
    "$BUILD_VENV_DIR/bin/python" -m pip install "${MESA_BUILD_PYTHON_REQUIREMENTS[@]}"
  fi
}

sdk_ready() {
  [[ -f "$SDK_ROOT/usr/include/vulkan/vulkan.h" ]] \
    && [[ -f "$SDK_ROOT/usr/include/spirv/unified1/spirv.hpp" ]] \
    && [[ -f "$SDK_ROOT/usr/lib64/pkgconfig/libdrm.pc" ]] \
    && [[ -f "$SDK_ROOT/usr/lib64/pkgconfig/expat.pc" ]] \
    && [[ -e "$SDK_ROOT/usr/lib64/libdrm_amdgpu.so.1" ]] \
    && [[ -e "$SDK_ROOT/usr/lib64/libexpat.so.1" ]]
}

require_command() {
  local name=$1
  if ! command -v "$name" >/dev/null 2>&1; then
    echo "Missing required command for local Mesa SDK setup: $name" >&2
    exit 1
  fi
}

download_rpm() {
  local package=$1
  dnf download "${DNF_REPOS[@]}" --destdir "$RPM_CACHE" "$package"
}

latest_rpm() {
  local pattern=$1
  find "$RPM_CACHE" -maxdepth 1 -type f -name "$pattern" -printf "%T@ %p\n" \
    | sort -n \
    | tail -1 \
    | cut -d' ' -f2-
}

extract_rpm_to_sdk() {
  local rpm_path=$1
  (cd "$SDK_ROOT" && rpm2cpio "$rpm_path" | cpio -idmu)
}

prepare_mesa_build_sdk() {
  require_command dnf
  require_command rpm2cpio
  require_command cpio

  mkdir -p -- "$RPM_CACHE" "$SDK_ROOT" "$TOOLS_DIR"
  read -r -a DNF_REPOS <<< "$DNF_REPO_ARGS"

  if [[ ! -x "$TOOLS_DIR/glslc" || ! -x "$TOOLS_DIR/glslangValidator" ]]; then
    download_rpm "${TORCH2VK_GLSLC_PACKAGE:-glslc}"
    download_rpm "${TORCH2VK_GLSLANG_PACKAGE:-glslang}"
    GLSLC_RPM=$(latest_rpm "glslc-*x86_64.rpm")
    GLSLANG_RPM=$(latest_rpm "glslang-*x86_64.rpm")
    if [[ -z "$GLSLC_RPM" ]]; then
      echo "Could not find downloaded x86_64 glslc rpm in $RPM_CACHE" >&2
      exit 1
    fi
    if [[ -z "$GLSLANG_RPM" ]]; then
      echo "Could not find downloaded x86_64 glslang rpm in $RPM_CACHE" >&2
      exit 1
    fi

    TMP_EXTRACT="$REPO_ROOT/.cache/torch2vk/rpm-extract/glslang"
    rm -rf -- "$TMP_EXTRACT"
    mkdir -p -- "$TMP_EXTRACT"
    (cd "$TMP_EXTRACT" && rpm2cpio "$GLSLC_RPM" | cpio -idmu ./usr/bin/glslc)
    (cd "$TMP_EXTRACT" && rpm2cpio "$GLSLANG_RPM" | cpio -idmu)
    cp "$TMP_EXTRACT/usr/bin/glslc" "$TOOLS_DIR/glslc"
    cp "$TMP_EXTRACT/usr/bin/glslangValidator" "$TOOLS_DIR/glslangValidator"
    chmod +x "$TOOLS_DIR/glslc" "$TOOLS_DIR/glslangValidator"
  fi

  if ! sdk_ready; then
    download_rpm "${TORCH2VK_VULKAN_HEADERS_PACKAGE:-vulkan-headers}"
    download_rpm "${TORCH2VK_VULKAN_LOADER_DEVEL_PACKAGE:-vulkan-loader-devel}"
    download_rpm "${TORCH2VK_SPIRV_TOOLS_DEVEL_PACKAGE:-spirv-tools-devel}"
    download_rpm "${TORCH2VK_SPIRV_HEADERS_DEVEL_PACKAGE:-spirv-headers-devel}"
    download_rpm "${TORCH2VK_LIBDRM_PACKAGE:-libdrm}"
    download_rpm "${TORCH2VK_LIBDRM_DEVEL_PACKAGE:-libdrm-devel}"
    download_rpm "${TORCH2VK_EXPAT_PACKAGE:-expat}"
    download_rpm "${TORCH2VK_EXPAT_DEVEL_PACKAGE:-expat-devel}"

    for pattern in \
      "vulkan-headers-*.rpm" \
      "vulkan-loader-devel-*x86_64.rpm" \
      "spirv-tools-devel-*x86_64.rpm" \
      "spirv-headers-devel-*.rpm" \
      "libdrm-[0-9]*x86_64.rpm" \
      "libdrm-devel-*x86_64.rpm" \
      "expat-[0-9]*x86_64.rpm" \
      "expat-devel-*x86_64.rpm"; do
      RPM_PATH=$(latest_rpm "$pattern")
      if [[ -z "$RPM_PATH" ]]; then
        echo "Could not find downloaded rpm matching $pattern in $RPM_CACHE" >&2
        exit 1
      fi
      extract_rpm_to_sdk "$RPM_PATH"
    done
  fi

  "$TOOLS_DIR/glslc" --version
  "$TOOLS_DIR/glslangValidator" --version
  if ! sdk_ready; then
    echo "Local Mesa SDK setup finished, but required files are still missing under $SDK_ROOT" >&2
    exit 1
  fi
  cat <<EOF
Mesa build SDK is ready:
  sdk_root:  $SDK_ROOT
  tools_dir: $TOOLS_DIR
  rpm_cache: $RPM_CACHE
EOF
}

configure_sdk_env() {
  prepend_env_path PATH "$TOOLS_DIR"
  prepend_env_path LD_LIBRARY_PATH "$SDK_ROOT/usr/lib64"
  prepend_env_path PKG_CONFIG_PATH "$SDK_ROOT/usr/share/pkgconfig"
  prepend_env_path PKG_CONFIG_PATH "$SDK_ROOT/usr/lib64/pkgconfig"
  export PKG_CONFIG_SYSROOT_DIR="$SDK_ROOT"
}

find_llvm_config() {
  if [[ -n "${LLVM_CONFIG:-}" ]]; then
    printf '%s\n' "$LLVM_CONFIG"
    return 0
  fi

  local candidate
  for candidate in \
    /usr/lib64/rocm/llvm/bin/llvm-config \
    /opt/rocm/llvm/bin/llvm-config \
    /usr/lib/llvm/bin/llvm-config \
    /usr/bin/llvm-config; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  command -v llvm-config 2>/dev/null || true
}

if [[ ! -f "$MESA_SRC_DIR/meson.build" ]]; then
  echo "Mesa source tree is missing at $MESA_SRC_DIR" >&2
  echo "Run: git submodule update --init --recursive third_party/mesa" >&2
  exit 1
fi

if [[ "$SETUP_SDK" == always || ( "$SETUP_SDK" == auto && ! sdk_ready ) ]]; then
  prepare_mesa_build_sdk
fi

if sdk_ready; then
  configure_sdk_env
elif [[ "$SETUP_SDK" == never ]]; then
  echo "warning: local Mesa SDK is incomplete at $SDK_ROOT; relying on host development files." >&2
fi

if [[ -z "${MESON:-}" || -z "${NINJA:-}" ]]; then
  ensure_build_venv
  prepend_env_path PATH "$BUILD_VENV_DIR/bin"
fi

LLVM_CONFIG_PATH=$(find_llvm_config)
if [[ -n "$LLVM_CONFIG_PATH" ]]; then
  export LLVM_CONFIG="$LLVM_CONFIG_PATH"
  LLVM_PREFIX=$("$LLVM_CONFIG_PATH" --prefix 2>/dev/null || true)
  prepend_env_path PATH "$(dirname -- "$LLVM_CONFIG_PATH")"
  if [[ -n "$LLVM_PREFIX" ]]; then
    prepend_env_path LD_LIBRARY_PATH "$LLVM_PREFIX/lib"
    prepend_env_path LD_LIBRARY_PATH "$LLVM_PREFIX/lib64"
  fi
fi

if [[ -n "${MESON:-}" ]]; then
  read -r -a MESON_CMD <<< "$MESON"
elif [[ -x "$BUILD_VENV_DIR/bin/meson" ]]; then
  MESON_CMD=("$BUILD_VENV_DIR/bin/meson")
else
  echo "Missing meson in $BUILD_VENV_DIR; set MESON to override." >&2
  exit 1
fi

if [[ -n "${NINJA:-}" ]]; then
  read -r -a NINJA_CMD <<< "$NINJA"
elif [[ -x "$BUILD_VENV_DIR/bin/ninja" ]]; then
  NINJA_CMD=("$BUILD_VENV_DIR/bin/ninja")
else
  echo "Missing ninja in $BUILD_VENV_DIR; set NINJA to override." >&2
  exit 1
fi

if command -v pkg-config >/dev/null 2>&1 && ! pkg-config --exists libelf; then
  echo "warning: pkg-config cannot find libelf; RGP dump requires Mesa to build with libelf." >&2
fi

if [[ "$CLEAN" == 1 ]]; then
  rm -rf -- "$BUILD_DIR" "$PREFIX"
fi

mkdir -p -- "$BUILD_DIR" "$PREFIX"

MESON_OPTIONS=(
  "--prefix=$PREFIX"
  "--libdir=lib"
  "--buildtype=$BUILD_TYPE"
  "-Dvulkan-drivers=amd"
  "-Dgallium-drivers="
  "-Dplatforms="
  "-Dopengl=false"
  "-Dglx=disabled"
  "-Degl=disabled"
  "-Dgbm=disabled"
  "-Dglvnd=disabled"
  "-Dgles1=disabled"
  "-Dgles2=disabled"
  "-Dllvm=enabled"
  "-Dshared-llvm=enabled"
  "-Dgallium-va=disabled"
  "-Dgallium-rusticl=false"
  "-Dmicrosoft-clc=disabled"
  "-Dspirv-to-dxil=false"
  "-Dvalgrind=disabled"
  "-Dlibunwind=disabled"
  "-Dxmlconfig=disabled"
  "-Dbuild-tests=false"
  "-Dbuild-radv-tests=false"
  "-Dbuild-aco-tests=false"
  "-Dvulkan-layers="
  "-Dtools="
  "-Dvideo-codecs="
)

if [[ ! -f "$BUILD_DIR/build.ninja" ]]; then
  "${MESON_CMD[@]}" setup "$BUILD_DIR" "$MESA_SRC_DIR" "${MESON_OPTIONS[@]}" "${EXTRA_MESON_OPTIONS[@]}"
elif [[ "$RECONFIGURE" == 1 || ${#EXTRA_MESON_OPTIONS[@]} -gt 0 ]]; then
  "${MESON_CMD[@]}" setup --reconfigure "$BUILD_DIR" "$MESA_SRC_DIR" "${MESON_OPTIONS[@]}" "${EXTRA_MESON_OPTIONS[@]}"
fi

if [[ "$SETUP_ONLY" == 1 ]]; then
  exit 0
fi

NINJA_EXTRA_ARGS=()
if [[ -n "${NINJA_ARGS:-}" ]]; then
  read -r -a NINJA_EXTRA_ARGS <<< "$NINJA_ARGS"
fi

"${NINJA_CMD[@]}" -C "$BUILD_DIR" "${NINJA_EXTRA_ARGS[@]}"

if [[ "$NO_INSTALL" != 1 ]]; then
  "${NINJA_CMD[@]}" -C "$BUILD_DIR" install
fi

ICD_DIR="$PREFIX/share/vulkan/icd.d"
ICD_FILE=$(find "$ICD_DIR" -maxdepth 1 -type f -name 'radeon_icd*.json' 2>/dev/null | sort | head -n 1 || true)
RADV_LIB=$(find "$PREFIX" -type f -name 'libvulkan_radeon.so*' 2>/dev/null | sort | head -n 1 || true)

if [[ -z "$ICD_FILE" || -z "$RADV_LIB" ]]; then
  echo "Mesa build finished, but RADV ICD artifacts were not found under $PREFIX" >&2
  echo "Expected radeon_icd*.json and libvulkan_radeon.so*" >&2
  exit 1
fi

cat <<EOF
Built local RADV Mesa:
  prefix: $PREFIX
  icd:    $ICD_FILE
  lib:    $RADV_LIB

Use it with:
  export VK_DRIVER_FILES=$ICD_FILE
  export VK_ICD_FILENAMES=$ICD_FILE
  export LD_LIBRARY_PATH=$PREFIX/lib:$PREFIX/lib64:\${LD_LIBRARY_PATH:-}
EOF
