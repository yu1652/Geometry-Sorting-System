#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

trtexec="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if [[ ! -x "$trtexec" ]]; then
    printf 'TensorRT trtexec is unavailable: %s\n' "$trtexec" >&2
    exit 1
fi

for model in SHAPE tray_object_detector surface_mask; do
    if [[ ! -s "models/$model.onnx" ]]; then
        printf 'Missing model: models/%s.onnx\n' "$model" >&2
        exit 1
    fi
done
if [[ -f models/onnx-sha256.txt ]]; then
    (cd models && sha256sum -c onnx-sha256.txt)
fi

build() {
    local model="$1" engine="$2" staging
    shift 2
    if [[ -s "models/$engine" ]]; then
        printf 'Keeping existing engine: models/%s\n' "$engine"
        return
    fi
    staging="$(mktemp -d "models/.engine-build.XXXXXX")"
    if ! "$trtexec" --onnx="models/$model.onnx" --saveEngine="$staging/$engine" \
            --fp16 --workspace=256 --avgTiming=1 --buildOnly "$@" ||
            [[ ! -s "$staging/$engine" ]]; then
        printf 'Build failed for %s. Inspect %s before retrying.\n' "$model" "$staging" >&2
        return 1
    fi
    mv -n -- "$staging/$engine" "models/$engine"
    if [[ -e "$staging/$engine" ]]; then
        printf 'Existing engine appeared during build. Candidate kept at %s/%s\n' "$staging" "$engine" >&2
        return 1
    fi
    rmdir -- "$staging"
    sha256sum "models/$engine"
}

build SHAPE shape_legacy_fp16.engine
build tray_object_detector tray_object_detector_fp16.engine
build surface_mask surface_mask_fp16.engine \
    --minShapes=input:1x3x320x320 --optShapes=input:1x3x320x320 --maxShapes=input:1x3x320x320
