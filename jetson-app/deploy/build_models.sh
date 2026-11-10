#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")/.."

trtexec="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if [[ ! -x "$trtexec" ]]; then
    printf 'TensorRT trtexec is unavailable: %s\n' "$trtexec" >&2
    exit 1
fi

model_dir=models
sources=(SHAPE tray_object_detector surface_mask)
surface_shapes=(--minShapes=input:1x3x320x320 --optShapes=input:1x3x320x320 --maxShapes=input:1x3x320x320)
if [[ "${1:-}" == --linux-20261107 && $# == 1 ]]; then
    model_dir=models/linux_20261107
    sources=(shape_detector tray_object_detector)
elif [[ "${1:-}" == --butu-24 && $# == 1 ]]; then
    model_dir=models/butu_20261110
    sources=(shape_detector surface_mask)
elif [[ $# != 0 ]]; then
    printf 'Usage: %s [--linux-20261107 | --butu-24]\n' "$0" >&2
    exit 2
fi
for model in "${sources[@]}"; do
    if [[ ! -s "$model_dir/$model.onnx" ]]; then
        printf 'Missing model: %s/%s.onnx\n' "$model_dir" "$model" >&2
        exit 1
    fi
done
check_model_hash() {
    local directory="$1" source="$2" manifest entry
    manifest="$directory/onnx-sha256.txt"
    if [[ ! -s "$manifest" ]]; then
        printf 'Missing model checksum manifest: %s\n' "$manifest" >&2
        return 1
    fi
    if ! entry="$(awk -v source="$source" '
        { sub(/\r$/, "") }
        NF == 2 && ($2 == source || $2 == "*" source) { print; count++ }
        END { exit count != 1 }
    ' "$manifest")"; then
        printf 'Expected one checksum for %s in %s\n' "$source" "$manifest" >&2
        return 1
    fi
    (cd "$directory" && printf '%s\n' "$entry" | sha256sum -c -)
}

if [[ "$model_dir" == models/butu_20261110 ]]; then
    tray_dir=models/linux_20261107
    check_model_hash "$model_dir" shape_detector.onnx
    check_model_hash "$model_dir" surface_mask.onnx
    # This profile shares only the tray model with Linux 20261107.
    check_model_hash "$tray_dir" tray_object_detector.onnx
elif [[ -f "$model_dir/onnx-sha256.txt" ]]; then
    (cd "$model_dir" && sha256sum -c onnx-sha256.txt)
fi

build() {
    local model="$1" engine="$2" staging
    shift 2
    if [[ -s "$model_dir/$engine" ]]; then
        printf 'Keeping existing engine: %s/%s\n' "$model_dir" "$engine"
        return
    fi
    staging="$(mktemp -d "$model_dir/.engine-build.XXXXXX")"
    if ! "$trtexec" --onnx="$model_dir/$model.onnx" --saveEngine="$staging/$engine" \
            --fp16 --workspace=256 --avgTiming=1 --buildOnly "$@" ||
            [[ ! -s "$staging/$engine" ]]; then
        printf 'Build failed for %s. Inspect %s before retrying.\n' "$model" "$staging" >&2
        return 1
    fi
    mv -n -- "$staging/$engine" "$model_dir/$engine"
    if [[ -e "$staging/$engine" ]]; then
        printf 'Existing engine appeared during build. Candidate kept at %s/%s\n' "$staging" "$engine" >&2
        return 1
    fi
    rmdir -- "$staging"
    sha256sum "$model_dir/$engine"
}

if [[ "$model_dir" == models/butu_20261110 ]]; then
    build shape_detector shape_detector_fp16.engine
    build surface_mask surface_mask_fp16.engine "${surface_shapes[@]}"
    (model_dir="$tray_dir"; build tray_object_detector tray_object_detector_fp16.engine)
    exit 0
elif [[ "$model_dir" == models/linux_20261107 ]]; then
    build shape_detector shape_detector_fp16.engine
    build tray_object_detector tray_object_detector_fp16.engine
    exit 0
fi
build SHAPE shape_legacy_fp16.engine
build tray_object_detector tray_object_detector_fp16.engine
build surface_mask surface_mask_fp16.engine "${surface_shapes[@]}"
