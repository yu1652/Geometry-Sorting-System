"""User-home-only installer for the desktop converter; no sudo or model edits."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

FILES = ('onnx_contract.py', 'side_model_store.py', 'side_model_manager.py',
         'model_transfer.py', 'run_model_converter.sh')


def _write(path, content, executable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_bytes(content)
        if executable:
            temporary.chmod(0o755)
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def install(payload, target, home=None):
    payload = Path(payload).resolve()
    target = Path(target).resolve()
    home = Path(home or Path.home()).resolve()
    if home not in target.parents or not (target / 'run_jp4.sh').is_file():
        raise ValueError('目标必须是当前用户主目录中的已安装 JP4 上位机目录')
    manifest = json.loads((payload / 'payload-manifest.json').read_text(encoding='utf8'))
    contents = {}
    for name in FILES:
        source = payload / name
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest['files'].get(name):
            raise ValueError('安装包校验失败：' + name)
        if (target / name).is_symlink():
            raise ValueError('目标代码文件不能是符号链接：' + name)
        contents[name] = content
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6]
    backup = target / 'data' / 'model_tools_backups' / stamp
    backup.mkdir(parents=True)
    # Paths are quoted according to the freedesktop Exec grammar.
    command = str(target / 'run_model_converter.sh').replace('\\', '\\\\').replace('"', '\\"').replace('`', '\\`').replace('$', '\\$')
    desktop = ('[Desktop Entry]\nType=Application\nVersion=1.0\n'
               'Name=ONNX 模型转换器\nComment=选择 ONNX 或接收 Windows 模型，在 Jetson 转换并管理\n'
               'Exec="' + command + '"\nPath=' + str(target) + '\n'
               'Icon=applications-graphics\nTerminal=false\nCategories=Utility;\n')
    shortcut = home / 'Desktop' / 'ONNX模型转换器.desktop'
    result = {'installed': True, 'app_root': str(target), 'backup': str(backup),
              'shortcut': str(shortcut), 'files': manifest['files']}
    receipt = json.dumps(result, ensure_ascii=False, indent=2).encode('utf8')
    writes = [(target / name, contents[name], name.endswith('.sh')) for name in FILES]
    writes.extend([(shortcut, desktop.encode('utf8'), True),
                   (home / '.local' / 'share' / 'applications' / 'ylers-model-converter.desktop', desktop.encode('utf8'), False),
                   (backup / 'receipt.json', receipt, False),
                   (home / 'Desktop' / '模型工具安装结果.json', receipt, False)])
    originals, changed = {}, []
    for destination, content, executable in writes:
        if destination.is_symlink():
            raise ValueError('安装目标不能是符号链接：' + str(destination))
        originals[destination] = (destination.read_bytes(), bool(destination.stat().st_mode & 0o111)) if destination.exists() else None
    for name in FILES:
        if (target / name).exists():
            shutil.copy2(str(target / name), str(backup / name))
    try:
        for destination, content, executable in writes:
            changed.append(destination)
            _write(destination, content, executable)
    except Exception:
        for destination in reversed(changed):
            original = originals[destination]
            if original is None:
                if destination.exists():
                    destination.unlink()
            elif not destination.exists() or destination.read_bytes() != original[0]:
                _write(destination, original[0], original[1])
        raise
    # Trust only our generated launcher; failure is harmless on non-GNOME desktops.
    if os.name != 'nt' and shutil.which('gio'):
        subprocess.run(['gio', 'set', str(shortcut), 'metadata::trusted', 'true'],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--app-root', type=Path, default=Path.home() / 'jetson-jp4-20261115' / 'jetson-app')
    args = parser.parse_args()
    result = install(Path(__file__).resolve().parent, args.app_root)
    print('已安装：' + result['shortcut'])
    print('原代码备份：' + result['backup'])
    print('当前活动模型、九点标定及计数未改动。')


if __name__ == '__main__':
    main()
