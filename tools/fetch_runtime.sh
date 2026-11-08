#!/usr/bin/env bash
# Fetch the CIX NOE runtime the image needs into vendor/ (not committed: CIX/Arm China binaries).
#   vendor/libnoe-3.1.2-py3-none-manylinux2014_aarch64.whl   from cix-noe-umd_3.1.2_arm64.deb
#   vendor/libaipudrv.so                                     libaipu_driver.so from cix-npu-onnxruntime_1.2.0_arm64.deb
# Both .deb files are public in Radxa's release radxa-pkg/cix-prebuilt 26Q2-2607 (matches KMD
# cix-npu-driver-dkms 6.2.0). UMD 3.x dlopen()s "libaipudrv.so", hence the rename.
# The onnxruntime .deb is ~136 MB; downloads are cached in $CACHE. Idempotent; checksums are verified.
set -euo pipefail
cd "$(dirname "$0")/.."
REL=https://github.com/radxa-pkg/cix-prebuilt/releases/download/26Q2-2607
CACHE=${CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/npu-zoo}
WHL=libnoe-3.1.2-py3-none-manylinux2014_aarch64.whl
declare -A SHA=(
  [vendor/$WHL]=65b549c2306a48925f4f0b4db4b46a1cf3d74f2f85ccd48262e3a2a5cef9bff4
  [vendor/libaipudrv.so]=12b31f2997d217fb8adf3de41e038e8472ce247a6558966e1bd80f991929aca3
)

ok() { [[ -f $1 ]] && echo "${SHA[$1]}  $1" | sha256sum -c --quiet - 2>/dev/null; }

extract() {  # extract <deb> <path inside deb> <dest>
  local tmp; tmp=$(mktemp -d)
  if command -v dpkg-deb >/dev/null; then
    dpkg-deb --fsys-tarfile "$1" | tar -x -C "$tmp" "./$2"
  else  # plain ar + tar (data.tar.{xz,zst,gz})
    (cd "$tmp" && ar x "$1" && tar -xf data.tar.* "./$2")
  fi
  install -m 0755 "$tmp/$2" "$3"
  rm -rf "$tmp"
}

fetch() {  # fetch <deb name> <path inside deb> <dest>
  ok "$3" && { echo "ok    $3"; return; }
  mkdir -p "$CACHE" vendor
  [[ -f $CACHE/$1 ]] || { echo "get   $1"; curl -fL --progress-bar -o "$CACHE/$1.part" "$REL/$1" && mv "$CACHE/$1.part" "$CACHE/$1"; }
  extract "$(realpath "$CACHE/$1")" "$2" "$3"
  ok "$3" || { echo "checksum mismatch: $3" >&2; exit 1; }
  echo "new   $3"
}

fetch cix-noe-umd_3.1.2_arm64.deb usr/share/cix/pypi/$WHL vendor/$WHL
fetch cix-npu-onnxruntime_1.2.0_arm64.deb usr/share/cix/lib/onnxruntime/libaipu_driver.so vendor/libaipudrv.so
