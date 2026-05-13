#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${PROJECT_ROOT}/.venv"
REPORT_PATH="${SCRIPT_DIR}/binding-report.md"
HEADER_PATH="${PROJECT_ROOT}/whisper.cpp/include/whisper.h"
REPORT_SCRIPT="${SCRIPT_DIR}/gen_report.py"
REBUILD=0
SAVE_STUBS=0

parse_args() {
 while [[ $# -gt 0 ]]; do
  case "$1" in
    -r|--rebuild)
      REBUILD=1
      ;;
    -s|--stubs)
      SAVE_STUBS=1
      ;;
    -h|--help)
      cat <<'EOF'
      Usage: ./report-bindings.sh [options]

      Options:
      -r, --rebuild   Reinstall the package in editable mode before generating the report
      -s, --stubs     Save generated stubs to coverage/stubs
      -h, --help      Show this help message
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
  shift
 done
}

ensure_venv() {
  if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
  fi
}

activate_venv() {
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
}

install_python_tools() {
  python -m pip install -U pip

  if [[ -f "${PROJECT_ROOT}/requirements-dev.txt" ]]; then
    python -m pip install -r "${PROJECT_ROOT}/requirements-dev.txt"
  else
    python -m pip install pybind11-stubgen pygccxml
  fi

  
}

# rebuild(){
#   python -m pip install -e "$PROJECT_ROOT"
# }
rebuild() {
  local python_bin cmake_args arg
  python_bin="$VENV_DIR/bin/python"
  cmake_args="${CMAKE_ARGS:-}"

  for arg in \
    "-DPython_EXECUTABLE=${python_bin}" \
    "-DPython3_EXECUTABLE=${python_bin}" \
    "-DPYTHON_EXECUTABLE=${python_bin}" \
    "-DBUILD_SHARED_LIBS=OFF" \
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON"
  do
    case " ${cmake_args} " in
      *" ${arg} "*) ;;
      *)
        cmake_args="${cmake_args:+${cmake_args} }${arg}"
        ;;
    esac
  done

  CMAKE_ARGS="${cmake_args}" \
  Python_EXECUTABLE="${python_bin}" \
  Python3_EXECUTABLE="${python_bin}" \
  PYTHON_EXECUTABLE="${python_bin}" \
  python -m pip install -e "$PROJECT_ROOT"
}



is_git_repo() {
  git -C "$PROJECT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

submodule_config_path() {
  local submodule_name="$1"
  git -C "$PROJECT_ROOT" config --file .gitmodules --get "submodule.${submodule_name}.path" 2>/dev/null
}

submodule_needs_sync() {
  local submodule_name="$1"
  local expected_url configured_url

  expected_url="$(git -C "$PROJECT_ROOT" config --file .gitmodules --get "submodule.${submodule_name}.url" 2>/dev/null || true)"
  configured_url="$(git -C "$PROJECT_ROOT" config --get "submodule.${submodule_name}.url" 2>/dev/null || true)"

  [[ -n "$expected_url" && "$expected_url" != "$configured_url" ]]
}

submodule_needs_update() {
  local submodule_name="$1"
  local submodule_path marker_path

  submodule_path="$(submodule_config_path "$submodule_name")"
  case "$submodule_name" in
    pybind11)
      marker_path="$PROJECT_ROOT/${submodule_path}/CMakeLists.txt"
      ;;
    whisper.cpp)
      marker_path="$PROJECT_ROOT/${submodule_path}/CMakeLists.txt"
      ;;
    *)
      return 1
      ;;
  esac

  [[ ! -f "$marker_path" ]]
}

ensure_required_submodules() {
  local submodule_name

  if [[ ! -f "$PROJECT_ROOT/.gitmodules" ]] || ! is_git_repo; then
    return
  fi

  for submodule_name in pybind11 whisper.cpp; do
    if submodule_needs_sync "$submodule_name"; then
      echo "Syncing git submodule ${submodule_name}"
      git -C "$PROJECT_ROOT" submodule sync --recursive -- "$submodule_name"
    fi

    if submodule_needs_update "$submodule_name"; then
      echo "Initializing git submodule ${submodule_name}"
      git -C "$PROJECT_ROOT" submodule update --init --recursive -- "$submodule_name"
    fi
  done
}

ensure_castxml() {
  if command -v castxml >/dev/null 2>&1; then
    return
  fi

  echo "castxml not found on PATH"

  if ! command -v apt-get >/dev/null 2>&1; then
    echo "Could not install castxml automatically."
    echo "Install castxml with your system package manager, then rerun this script."
    exit 1
  fi

  read -r -p "Install castxml with apt-get? [y/N] " install_castxml
  case "$install_castxml" in
    [Yy]|[Yy][Ee][Ss])
      echo "Installing castxml with apt-get..."
      sudo apt-get install -y castxml
      ;;
    *)
      echo "castxml is required. Install it, then rerun this script."
      exit 1
      ;;
  esac

  if ! command -v castxml >/dev/null 2>&1; then
    echo "castxml installation did not succeed"
    exit 1
  fi
}

verify_extension_import() {
  python -c "import _pywhispercpp" >/dev/null 2>&1
}

ensure_extension_available() {
  if verify_extension_import; then
    return
  fi

  echo "_pywhispercpp is not installed; installing it now"
  REBUILD=1
}

compare_bindings() {
  local args=(
    --report-path "$REPORT_PATH"
    --header-path "$HEADER_PATH"
    --project-root "$PROJECT_ROOT"
  )

  if [[ "$SAVE_STUBS" -eq 1 ]]; then
    args+=(--stubs)
  fi

  python "$REPORT_SCRIPT" "${args[@]}"
}

main() {
  parse_args "$@"
  ensure_venv
  activate_venv
  ensure_castxml
  install_python_tools
  ensure_required_submodules
  ensure_extension_available
  if [[ "$REBUILD" -eq 1 ]]; then
    rebuild
  fi
  verify_extension_import
  compare_bindings

  echo
  echo "Artifacts:"
  if [[ "$SAVE_STUBS" -eq 1 ]]; then
    echo "  stubs:  ${SCRIPT_DIR}/stubs"
  fi
  echo "  report: ${REPORT_PATH}"
}

main "$@"