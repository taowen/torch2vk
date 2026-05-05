#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

RUN_ID=$(date +%Y%m%d-%H%M%S)
ROOT="$REPO_ROOT/.cache/torch2vk/sqtt/$RUN_ID"
MESA_PREFIX="$REPO_ROOT/.cache/torch2vk/mesa-radv"
SDK_ROOT="${TORCH2VK_VULKAN_SDK_ROOT:-$REPO_ROOT/.cache/torch2vk/vulkan-sdk}"
BUILD_MESA=auto
TRACE_BUFFER_MIB=256
QUEUE_EVENTS=true
INSTRUCTION_TIMING=true
CAPTURE_FILTER=""
CAPTURE_FRAME=""
CAPTURE_SHADER=""
CAPTURE_DISPATCH=""
DRY_RUN=0
PRINT_ENV=0
COMMAND=()

usage() {
  cat <<'USAGE'
Usage: scripts/profile-sqtt.sh [options] -- command [args...]

Run a command with the repository Mesa RADV build selected and RADV SQTT/RGP
per-submit capture enabled.

Options:
  --root DIR              SQTT artifact root
                          default: .cache/torch2vk/sqtt/<timestamp>
  --mesa-prefix DIR       Local Mesa install prefix
                          default: .cache/torch2vk/mesa-radv
  --sdk-root DIR          Local SDK sysroot for Mesa runtime deps
                          default: .cache/torch2vk/vulkan-sdk
  --build-mesa            Always build/install local Mesa before running
  --no-build-mesa         Fail if the local Mesa ICD is missing
  --trace-buffer-mib N    RADV_THREAD_TRACE_BUFFER_SIZE in MiB
                          default: 256
  --queue-events BOOL     RADV_THREAD_TRACE_QUEUE_EVENTS
                          default: true
  --instruction-timing BOOL
                          RADV_THREAD_TRACE_INSTRUCTION_TIMING
                          default: true
  --capture-filter TEXT   Only capture submits whose profile label payload
                          contains TEXT, for example:
                          'frame=qwen3_asr.text_decode.0000;shader=...'
  --frame NAME            Shorthand filter field: frame=NAME
  --shader NAME           Shorthand filter field: shader=NAME
  --dispatch-index N      Shorthand filter field: dispatch=N
  --print-env             Print selected environment before running
  --dry-run               Print environment/command and exit
  -h, --help              Show this help

Example:
  scripts/profile-sqtt.sh --root .cache/torch2vk/sqtt/qwen3 -- \
    uv run pytest tests/test_qwen3_asr.py -k decode
USAGE
}

abs_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$PWD" "$1" ;;
  esac
}

bool_value() {
  case "$1" in
    true|false) printf '%s\n' "$1" ;;
    1|yes|on) printf 'true\n' ;;
    0|no|off) printf 'false\n' ;;
    *)
      echo "Invalid boolean value: $1" >&2
      exit 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      ROOT=$2
      shift 2
      ;;
    --mesa-prefix)
      MESA_PREFIX=$2
      shift 2
      ;;
    --sdk-root)
      SDK_ROOT=$2
      shift 2
      ;;
    --build-mesa)
      BUILD_MESA=always
      shift
      ;;
    --no-build-mesa)
      BUILD_MESA=never
      shift
      ;;
    --trace-buffer-mib)
      TRACE_BUFFER_MIB=$2
      shift 2
      ;;
    --queue-events)
      QUEUE_EVENTS=$(bool_value "$2")
      shift 2
      ;;
    --instruction-timing)
      INSTRUCTION_TIMING=$(bool_value "$2")
      shift 2
      ;;
    --capture-filter)
      CAPTURE_FILTER=$2
      shift 2
      ;;
    --frame)
      CAPTURE_FRAME=$2
      shift 2
      ;;
    --shader)
      CAPTURE_SHADER=$2
      shift 2
      ;;
    --dispatch-index)
      CAPTURE_DISPATCH=$2
      shift 2
      ;;
    --print-env)
      PRINT_ENV=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      PRINT_ENV=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      COMMAND=("$@")
      break
      ;;
    *)
      echo "Unknown option before --: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ${#COMMAND[@]} -eq 0 ]]; then
  echo "Missing command. Pass it after --." >&2
  usage >&2
  exit 2
fi

ROOT=$(abs_path "$ROOT")
MESA_PREFIX=$(abs_path "$MESA_PREFIX")
SDK_ROOT=$(abs_path "$SDK_ROOT")
TRACE_BUFFER_BYTES=$((TRACE_BUFFER_MIB * 1024 * 1024))

if [[ -n "$CAPTURE_FILTER" && ( -n "$CAPTURE_FRAME" || -n "$CAPTURE_SHADER" || -n "$CAPTURE_DISPATCH" ) ]]; then
  echo "Use either --capture-filter or --frame/--shader/--dispatch-index shorthands, not both." >&2
  exit 2
fi

if [[ -z "$CAPTURE_FILTER" ]]; then
  FILTER_PARTS=()
  if [[ -n "$CAPTURE_FRAME" ]]; then
    FILTER_PARTS+=("frame=$CAPTURE_FRAME")
  fi
  if [[ -n "$CAPTURE_SHADER" ]]; then
    FILTER_PARTS+=("shader=$CAPTURE_SHADER")
  fi
  if [[ -n "$CAPTURE_DISPATCH" ]]; then
    FILTER_PARTS+=("dispatch=$CAPTURE_DISPATCH")
  fi
  if [[ ${#FILTER_PARTS[@]} -gt 0 ]]; then
    CAPTURE_FILTER=$(IFS=';'; printf '%s' "${FILTER_PARTS[*]}")
  fi
fi

if [[ -z "$CAPTURE_FILTER" ]]; then
  echo "SQTT capture requires --capture-filter or --frame/--shader/--dispatch-index." >&2
  echo "Do not run per-submit SQTT globally; first use the normal profiler to choose a target dispatch." >&2
  exit 2
fi

find_radv_icd() {
  local prefix=$1
  local icd_dir="$prefix/share/vulkan/icd.d"
  local machine
  machine=$(uname -m)

  local candidates=(
    "$icd_dir/radeon_icd.$machine.json"
    "$icd_dir/radeon_icd.x86_64.json"
    "$icd_dir/radeon_icd.json"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  find "$icd_dir" -maxdepth 1 -type f -name 'radeon_icd*.json' 2>/dev/null | sort | head -n 1
}

rgp_file_key() {
  local path=$1
  stat -c '%i:%Y:%s' "$path"
}

snapshot_rgp_files() {
  local capture
  while IFS= read -r capture; do
    RGP_BEFORE["$capture"]=$(rgp_file_key "$capture")
  done < <(find /tmp -maxdepth 1 -type f -name '*.rgp' 2>/dev/null | sort)
}

copy_new_rgp_captures() {
  local copied=0
  local capture
  local key
  while IFS= read -r capture; do
    key=$(rgp_file_key "$capture")
    if [[ "${RGP_BEFORE[$capture]:-}" == "$key" ]]; then
      continue
    fi
    copied=$((copied + 1))
    cp -f -- "$capture" "$ROOT/rgp/submit${copied}.rgp"
    echo "Copied RGP capture: $capture -> $ROOT/rgp/submit${copied}.rgp"
  done < <(find /tmp -maxdepth 1 -type f -name '*.rgp' -printf '%T@ %p\n' 2>/dev/null | sort -n | cut -d' ' -f2-)

  if [[ "$copied" -eq 0 ]]; then
    echo "warning: no new or updated /tmp/*.rgp capture was observed" >&2
  fi
}

ICD_FILE=$(find_radv_icd "$MESA_PREFIX" || true)
if [[ "$BUILD_MESA" == always || ( "$BUILD_MESA" == auto && -z "$ICD_FILE" ) ]]; then
  "$SCRIPT_DIR/build-mesa-radv.sh" --prefix "$MESA_PREFIX"
  ICD_FILE=$(find_radv_icd "$MESA_PREFIX" || true)
fi

if [[ -z "$ICD_FILE" ]]; then
  echo "Local Mesa RADV ICD not found under $MESA_PREFIX" >&2
  echo "Run scripts/build-mesa-radv.sh first, or omit --no-build-mesa." >&2
  exit 1
fi

mkdir -p -- "$ROOT/driver" "$ROOT/rgp" "$ROOT/mesa-shader-cache"

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

prepend_env_path LD_LIBRARY_PATH "$MESA_PREFIX/lib"
prepend_env_path LD_LIBRARY_PATH "$MESA_PREFIX/lib64"
prepend_env_path LD_LIBRARY_PATH "$MESA_PREFIX/lib/x86_64-linux-gnu"
prepend_env_path LD_LIBRARY_PATH "$SDK_ROOT/usr/lib64"
prepend_env_path LIBGL_DRIVERS_PATH "$MESA_PREFIX/lib/dri"
prepend_env_path LIBGL_DRIVERS_PATH "$MESA_PREFIX/lib64/dri"
prepend_env_path LIBGL_DRIVERS_PATH "$MESA_PREFIX/lib/x86_64-linux-gnu/dri"

LLVM_CONFIG_PATH=$(find_llvm_config)
if [[ -n "$LLVM_CONFIG_PATH" ]]; then
  export LLVM_CONFIG="$LLVM_CONFIG_PATH"
  LLVM_LIB_DIR=$("$LLVM_CONFIG_PATH" --libdir 2>/dev/null || true)
  LLVM_PREFIX=$("$LLVM_CONFIG_PATH" --prefix 2>/dev/null || true)
  prepend_env_path PATH "$(dirname -- "$LLVM_CONFIG_PATH")"
  if [[ -n "$LLVM_LIB_DIR" ]]; then
    prepend_env_path LD_LIBRARY_PATH "$LLVM_LIB_DIR"
  fi
  if [[ -n "$LLVM_PREFIX" ]]; then
    prepend_env_path LD_LIBRARY_PATH "$LLVM_PREFIX/lib"
    prepend_env_path LD_LIBRARY_PATH "$LLVM_PREFIX/lib64"
  fi
fi

export VK_DRIVER_FILES="$ICD_FILE"
export VK_ICD_FILENAMES="$ICD_FILE"
export MESA_VK_TRACE=rgp
export MESA_VK_TRACE_PER_SUBMIT=true
export RADV_THREAD_TRACE_BUFFER_SIZE="$TRACE_BUFFER_BYTES"
export RADV_THREAD_TRACE_QUEUE_EVENTS="$QUEUE_EVENTS"
export RADV_THREAD_TRACE_INSTRUCTION_TIMING="$INSTRUCTION_TIMING"
export AGENTORCH_RADV_SQTT_PROFILE_TAG_FILTER="$CAPTURE_FILTER"
export AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR="$ROOT/driver"
export AGENTORCH_RADV_EXPORT_TAG="$RUN_ID"
export MESA_SHADER_CACHE_DIR="$ROOT/mesa-shader-cache"
export TORCH2VK_SQTT_ROOT="$ROOT"
export TORCH2VK_PROFILE_RUN_DIR="$ROOT"
export TORCH2VK_MESA_PREFIX="$MESA_PREFIX"

if [[ "$PRINT_ENV" == 1 ]]; then
  cat <<EOF
SQTT profile environment:
  TORCH2VK_SQTT_ROOT=$TORCH2VK_SQTT_ROOT
  TORCH2VK_MESA_PREFIX=$TORCH2VK_MESA_PREFIX
  SDK_ROOT=$SDK_ROOT
  VK_DRIVER_FILES=$VK_DRIVER_FILES
  VK_ICD_FILENAMES=$VK_ICD_FILENAMES
  LLVM_CONFIG=${LLVM_CONFIG:-}
  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}
  MESA_VK_TRACE=$MESA_VK_TRACE
  MESA_VK_TRACE_PER_SUBMIT=$MESA_VK_TRACE_PER_SUBMIT
  RADV_THREAD_TRACE_BUFFER_SIZE=$RADV_THREAD_TRACE_BUFFER_SIZE
  RADV_THREAD_TRACE_QUEUE_EVENTS=$RADV_THREAD_TRACE_QUEUE_EVENTS
  RADV_THREAD_TRACE_INSTRUCTION_TIMING=$RADV_THREAD_TRACE_INSTRUCTION_TIMING
  AGENTORCH_RADV_SQTT_PROFILE_TAG_FILTER=$AGENTORCH_RADV_SQTT_PROFILE_TAG_FILTER
  AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR=$AGENTORCH_RADV_DRIVER_ARTIFACTS_DIR
  MESA_SHADER_CACHE_DIR=$MESA_SHADER_CACHE_DIR
  TORCH2VK_PROFILE_RUN_DIR=$TORCH2VK_PROFILE_RUN_DIR
Command:
  ${COMMAND[*]}
EOF
fi

if [[ "$DRY_RUN" == 1 ]]; then
  exit 0
fi

declare -A RGP_BEFORE=()
snapshot_rgp_files

set +e
"${COMMAND[@]}"
STATUS=$?
set -e

POSTPROCESS_STATUS=0
if [[ "$STATUS" -eq 0 ]]; then
  PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m torch2vk.sqtt --root "$ROOT" || POSTPROCESS_STATUS=$?
else
  copy_new_rgp_captures
fi

if [[ -f "$ROOT/driver/capture-sequence.jsonl" ]]; then
  echo "SQTT capture records:"
  tail -n 20 "$ROOT/driver/capture-sequence.jsonl"
else
  echo "warning: no capture-sequence.jsonl was written under $ROOT/driver" >&2
fi

if [[ -f "$ROOT/driver/dispatch-sequence.jsonl" ]]; then
  echo "SQTT dispatch records:"
  tail -n 20 "$ROOT/driver/dispatch-sequence.jsonl"
else
  echo "warning: no dispatch-sequence.jsonl was written under $ROOT/driver" >&2
fi

if [[ "$STATUS" -ne 0 ]]; then
  exit "$STATUS"
fi
exit "$POSTPROCESS_STATUS"
