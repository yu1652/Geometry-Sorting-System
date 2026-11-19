"""Provision missing sorter settings and launcher; no camera/serial access."""
import json
from pathlib import Path
import shutil
import os
import sys

import yaml

root = Path.home() / 'jetson-jp4-20261115'
app = root / 'jetson-app'
data = app / 'data'
data.mkdir(exist_ok=True)
sys.path.insert(0, str(app))
from live_console import validate_points

backup = Path.home() / 'jetson-full-20261115/migration_backup_20261114'
saved = backup / 'live_calibration.json'
destination = data / saved.name
if saved.is_file() and not destination.exists():
    value = json.loads(saved.read_text(encoding='utf8'))
    assert value['mode'] == 'live'
    validate_points(value['points'])
    with destination.open('xb') as stream:
        stream.write(saved.read_bytes())
    print('Recovered dated nine-point calibration; GUI confirmation remains required.')

capture = Path.home() / 'jetson_dualcam_capture/config.yaml'
devices = data / 'live_devices.json'
if capture.is_file() and not devices.exists():
    config = yaml.safe_load(capture.read_text(encoding='utf8'))
    sources = {role: config['cameras'][role]['device'] for role in ('left', 'right')}
    remaining = [str(p) for p in Path('/dev/v4l/by-path').glob('*video-index0')
                 if str(p) not in sources.values()]
    if len(remaining) == 1 and all(Path(p).exists() for p in sources.values()):
        sources['top'] = remaining[0]
        with devices.open('x', encoding='utf8') as stream:
            json.dump({'sources': sources}, stream, indent=2)
        print('Camera roles copied from saved capture configuration; verify previews before use.')

entry = ('[Desktop Entry]\nType=Application\nName=YLERS 分拣 JP4\n'
         'Exec=/bin/bash "%s" --fullscreen\nPath=%s\nTerminal=false\n' % (app / 'run_jp4.sh', app))
applications = Path.home() / '.local/share/applications'
applications.mkdir(parents=True, exist_ok=True)
(applications / 'ylers-sorter-jp4.desktop').write_text(entry, encoding='utf8')
for folder in ('Desktop', '桌面'):
    desktop = Path.home() / folder
    if desktop.is_dir():
        icon = desktop / 'ylers-sorter-jp4.desktop'
        icon.write_text(entry, encoding='utf8')
        os.chmod(str(icon), 0o755)
print('JP4 launcher provisioned; no motion or serial commands issued.')
