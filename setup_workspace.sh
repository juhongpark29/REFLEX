#!/usr/bin/env bash
set -e

# Usage: bash setup_workspace.sh /workspace

if [ $# -lt 1 ]; then
  echo "Usage: $0 /path/to/workspace"
  exit 1
fi

WORKSPACE_ROOT="$1"

# Normalize to absolute path
WORKSPACE_ROOT="$(cd "$WORKSPACE_ROOT" && pwd)"

# Directory of this script (REFLEX root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC_OMNI="${SCRIPT_DIR}/OmniServe"
SRC_ATTACC="${SCRIPT_DIR}/AttAcc"

DST_OMNI="${WORKSPACE_ROOT}/OmniServe"
DST_ATTACC="${WORKSPACE_ROOT}/attacc_simulator"

echo "[REFLEX] Workspace root: ${WORKSPACE_ROOT}"
echo "[REFLEX] Script dir    : ${SCRIPT_DIR}"
echo

# Sanity checks
if [ ! -d "$SRC_OMNI" ]; then
  echo "[REFLEX][ERROR] Source directory not found: ${SRC_OMNI}"
  exit 1
fi

if [ ! -d "$SRC_ATTACC" ]; then
  echo "[REFLEX][ERROR] Source directory not found: ${SRC_ATTACC}"
  exit 1
fi

if [ ! -d "$DST_OMNI" ]; then
  echo "[REFLEX][INFO] Target OmniServe directory does not exist. Creating: ${DST_OMNI}"
  mkdir -p "$DST_OMNI"
fi

if [ ! -d "$DST_ATTACC" ]; then
  echo "[REFLEX][INFO] Target attacc_simulator directory does not exist. Creating: ${DST_ATTACC}"
  mkdir -p "$DST_ATTACC"
fi

echo "[REFLEX] Patching OmniServe → ${DST_OMNI}"
# Copy (overwrite existing files, keep directory structure)
cp -a "${SRC_OMNI}/." "${DST_OMNI}/"

echo "[REFLEX] Patching AttAcc → ${DST_ATTACC}"
cp -a "${SRC_ATTACC}/." "${DST_ATTACC}/"

echo
echo "[REFLEX] Patch completed successfully."