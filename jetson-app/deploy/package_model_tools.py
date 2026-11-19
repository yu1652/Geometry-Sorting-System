"""Build a small verified, user-home-only Jetson tools update."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent
FILES = ('onnx_contract.py', 'side_model_store.py', 'side_model_manager.py',
         'model_transfer.py', 'run_model_converter.sh', 'install_model_tools.py')

BOOTSTRAP = '''import hashlib, io, json, os, runpy, subprocess, urllib.request, zipfile
from pathlib import Path
base = {base!r}
expected = {digest!r}
home = Path.home()
target = home / 'jetson-jp4-20261115' / 'jetson-app'
def digest_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
protected = ['data/live_side_model.json', 'data/live_calibration.json']
before = {{name:digest_file(target/name) for name in protected}}
data = urllib.request.urlopen(base+'/model-tools.zip', timeout=30).read()
assert hashlib.sha256(data).hexdigest() == expected, 'Package SHA256 mismatch'
folder = home / '.cache' / 'ylers-model-tools' / expected[:16]
folder.mkdir(parents=True, exist_ok=True)
names = {names!r}
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    assert sorted(archive.namelist()) == sorted(names), 'Unexpected package files'
    for name in names:
        (folder/name).write_bytes(archive.read(name))
module = runpy.run_path(str(folder/'install_model_tools.py'))
result = module['install'](folder, target)
after = {{name:digest_file(target/name) for name in protected}}
assert before == after, 'Protected model/calibration data changed'
result.update(protected_before=before, protected_after=after, package_sha256=expected)
(home/'Desktop'/'model-tools-install-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
request = urllib.request.Request(base+'/report', data=json.dumps(result).encode('utf8'), headers={{'Content-Type':'application/json'}}, method='POST')
try:
    urllib.request.urlopen(request, timeout=10).read()
except Exception as error:
    print('Report saved locally; return failed:', error)
environment = os.environ.copy()
environment['OPENBLAS_CORETYPE'] = 'ARMV8'
environment['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.2/lib64:' + environment.get('LD_LIBRARY_PATH','')
with (folder/'converter-launch.log').open('w') as log:
    subprocess.Popen([str(target/'.venv_jp4/bin/python'),str(target/'side_model_manager.py')],cwd=str(target),env=environment,stdout=log,stderr=subprocess.STDOUT)
print('INSTALLED. Model and calibration checksums unchanged. Desktop converter opened.')
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--base-url', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    contents = {name: (ROOT / name).read_bytes() for name in FILES}
    contents['run_model_converter.sh'] = contents['run_model_converter.sh'].replace(b'\r\n', b'\n')
    manifest = {'schema': 1, 'files': {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}}
    contents['payload-manifest.json'] = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf8')
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(name, content)
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    (args.output / 'model-tools.zip').write_bytes(data)
    bootstrap = BOOTSTRAP.format(base=args.base_url.rstrip('/'), digest=digest, names=list(contents))
    (args.output / 'bootstrap.py').write_text(bootstrap, encoding='utf8')
    (args.output / 'package-receipt.json').write_text(json.dumps(dict(manifest, sha256=digest, bytes=len(data)), indent=2), encoding='utf8')
    print(json.dumps({'package': str(args.output / 'model-tools.zip'), 'sha256': digest, 'bytes': len(data)}))


if __name__ == '__main__':
    main()
