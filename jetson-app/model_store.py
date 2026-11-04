"""Validated, atomic model release pointers for the offline vision app."""
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelRelease:
    slot: str
    version: str
    manifest_path: Path
    engine_path: Path
    manifest: dict


class ModelStore:
    SLOTS = ("side_detector", "surface_mask")
    PREPROCESS = {"side_detector": "legacy_stitched_v1", "surface_mask": "surface_rgb320_v1"}

    def __init__(self, root):
        self.root = Path(root)
        self.releases = self.root / "releases"
        self.active_path = self.root / "active.json"

    def read_active(self):
        if not self.active_path.exists():
            return {"active": {}, "previous": None}
        value = json.loads(self.active_path.read_text(encoding="utf8"))
        if not isinstance(value.get("active"), dict) or value.get("previous") is not None and not isinstance(value.get("previous"), dict):
            raise ValueError("active model pointer is malformed")
        return value

    def validate_manifest(self, path, slot):
        if slot not in self.SLOTS:
            raise ValueError("unknown model slot")
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("manifest must be a regular file")
        manifest = json.loads(path.read_text(encoding="utf8"))
        required = ("schema", "slot", "version", "engine", "sha256", "input_shape", "output_shape", "preprocess")
        if any(key not in manifest for key in required):
            raise ValueError("manifest is missing required fields")
        if manifest["schema"] != 1 or manifest["slot"] != slot or not manifest["version"]:
            raise ValueError("manifest schema, slot or version is invalid")
        if manifest["preprocess"] != self.PREPROCESS[slot]:
            raise ValueError("preprocessing is incompatible with this slot")
        expected_shapes = (([1, 3, 640, 640], [1, 30, 8400]) if slot == "side_detector"
                           else ([1, 3, 320, 320], [1, 1, 320, 320]))
        if manifest["input_shape"] != expected_shapes[0] or manifest["output_shape"] != expected_shapes[1]:
            raise ValueError("model input/output shape is incompatible with this slot")
        version = str(manifest["version"])
        if not isinstance(manifest["version"], str) or not version or len(version) > 80 or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for ch in version):
            raise ValueError("version must be a simple release identifier")
        engine_name = manifest["engine"]
        if not isinstance(engine_name, str) or Path(engine_name).is_absolute() or len(Path(engine_name).parts) != 1 or engine_name in (".", ".."):
            raise ValueError("engine must be a same-directory filename")
        engine = path.parent / engine_name
        if engine.is_symlink() or not engine.is_file():
            raise ValueError("engine must be a regular same-directory file")
        digest = hashlib.sha256(engine.read_bytes()).hexdigest()
        if digest.lower() != str(manifest["sha256"]).lower():
            raise ValueError("engine SHA-256 does not match manifest")
        if slot == "side_detector":
            expected = (Path(__file__).parent / "models" / "legacy_classes.txt").read_text(encoding="utf8").splitlines()
            if manifest.get("class_names") != expected:
                raise ValueError("class_names do not match the official legacy order")
        return ModelRelease(slot, str(manifest["version"]), path.resolve(), engine.resolve(), manifest)

    def install(self, release):
        self.releases.mkdir(parents=True, exist_ok=True)
        target = self.releases / (release.slot + "-" + release.version + "-" + release.manifest["sha256"].lower())
        if target.exists():
            self.validate_manifest(target / "manifest.json", release.slot)
            return target / "manifest.json"
        target.mkdir()
        try:
            shutil.copyfile(release.engine_path, target / release.manifest["engine"])
            manifest = dict(release.manifest)
            (target / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
            self.validate_manifest(target / "manifest.json", release.slot)
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        return target / "manifest.json"

    def resolve_active(self, active):
        if not isinstance(active, dict) or any(slot not in self.SLOTS for slot in active):
            raise ValueError("invalid active model set")
        result = {}
        for slot, record in active.items():
            path = Path(record["manifest_path"])
            if path.is_symlink() or self.releases.resolve() not in path.resolve().parents:
                raise ValueError("model pointer outside releases")
            release = self.validate_manifest(path, slot)
            if record.get("version") != release.version:
                raise ValueError("model pointer version mismatch")
            result[slot] = release
        return result

    def commit_active(self, active, previous):
        self.root.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix="active-", suffix=".tmp", dir=str(self.root))
        try:
            with os.fdopen(fd, "w", encoding="utf8") as stream:
                json.dump({"active": active, "previous": previous}, stream, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, str(self.active_path))
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
