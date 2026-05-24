#!/usr/bin/env bash
# Refresh vendored Infinigen dependencies (see ../VENDORED.md).
# Normal clones already include these; use this only to pull newer upstream pins.
# After running, remove nested .git dirs before committing: rm -rf infinigen/infinigen_gpl/.git infinigen/OcMesher/.git
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="${ROOT}/infinigen"

INFINIGEN_GPL_COMMIT=10c1d76f5c35003e919be7265185c9c355e3b70c
OCMESHER_COMMIT=bb895a01b8f574da10be1df470210df7463a75c7
UPSTREAM=https://github.com/princeton-vl/infinigen.git

clone_at() {
  local url="$1" dest="$2" commit="$3"
  if [[ -d "${dest}/.git" ]]; then
    git -C "${dest}" fetch --depth 1 origin "${commit}"
    git -C "${dest}" checkout "${commit}"
  else
    git clone --depth 1 "${url}" "${dest}"
    git -C "${dest}" fetch --depth 1 origin "${commit}"
    git -C "${dest}" checkout "${commit}"
  fi
}

echo "==> infinigen_gpl"
clone_at https://github.com/princeton-vl/infinigen_gpl.git \
  "${PKG}/infinigen_gpl" "${INFINIGEN_GPL_COMMIT}"

echo "==> OcMesher"
clone_at https://github.com/princeton-vl/OcMesher.git \
  "${PKG}/OcMesher" "${OCMESHER_COMMIT}"

sparse_copy() {
  local relpath="$1"
  local dest="${PKG}/${relpath}"
  if [[ -d "${dest}" ]] && [[ -n "$(ls -A "${dest}" 2>/dev/null || true)" ]]; then
    echo "==> ${relpath} (already present)"
    return
  fi
  echo "==> ${relpath} (from upstream Infinigen)"
  local tmp
  tmp="$(mktemp -d)"
  git clone --depth 1 --filter=blob:none --sparse "${UPSTREAM}" "${tmp}/src"
  git -C "${tmp}/src" sparse-checkout set "infinigen/${relpath}"
  mkdir -p "$(dirname "${dest}")"
  cp -r "${tmp}/src/infinigen/${relpath}" "${dest}"
  rm -rf "${tmp}"
}

sparse_copy assets/objects/creatures/parts
sparse_copy assets/objects/creatures/insects/parts

echo "Done. Re-run your generate_indoors command from ${ROOT}."
