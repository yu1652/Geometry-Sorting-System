"""Offline state and recoverable sample storage. No hardware dependencies."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid


@dataclass
class ReplayState:
    source: str = ''
    status: str = 'empty'

    def load(self, source: str):
        if not source:
            raise ValueError('请选择回放文件')
        self.source, self.status = source, 'ready'

    def play(self):
        if not self.source:
            raise ValueError('请先打开图片或录像')
        self.status = 'playing'

    def pause(self):
        if self.source:
            self.status = 'paused'

    def stop(self):
        if self.source:
            self.status = 'stopped'


def save_sample(root: Path, source: str, payload: bytes, **details) -> Path:
    if not payload:
        raise ValueError('没有可以保存的图像')
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    name = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:8]
    pending, final = root/('.pending_'+name), root/name
    pending.mkdir()
    try:
        (pending/'source.png').write_bytes(payload)
        meta = {**details, 'source': source, 'saved_at': datetime.now(timezone.utc).isoformat(),
                'hardware_enabled': False, 'mode': 'offline_replay'}
        (pending/'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf8')
        pending.rename(final)
    except Exception:
        shutil.rmtree(pending)
        raise
    return final
