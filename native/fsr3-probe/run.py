"""Offline official-FSR3 compatibility probe. Never launches a player/backend."""
import argparse
import hashlib
import json
import os
import subprocess
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sdk', type=Path, default=os.environ.get('ANIMESCALER_FSR_SDK'))
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--width', type=int)
    parser.add_argument('--height', type=int)
    args = parser.parse_args()
    if not args.sdk:
        parser.error('Set --sdk or configure ANIMESCALER_FSR_SDK with the official FidelityFX SDK path')
    exe = Path(__file__).parent / 'build' / 'fsr3-probe.exe'
    dll = args.sdk / 'Kits' / 'FidelityFX' / 'signedbin' / 'amd_fidelityfx_upscaler_dx12.dll'
    if not exe.is_file() or not dll.is_file():
        raise SystemExit('Build the probe and provide the official SDK DLL first')
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / 'report.json'
    if report_path.exists():
        raise SystemExit('Report already exists; use a new output directory to preserve evidence')
    snapshot = folder / 'fsr3-probe.exe'
    shutil.copy2(exe, snapshot)
    with Image.open(args.input) as source:
        image = source.convert('RGBA')
    width, height = image.size
    output_width, output_height = args.width or width * 2, args.height or height * 2
    if not (0 < width <= output_width <= 4096 and 0 < height <= output_height <= 4096):
        raise SystemExit('Dimensions outside bounded probe limits')
    raw_input, raw_output = folder / 'input.rgba', folder / 'output.rgba'
    raw_input.write_bytes(image.tobytes())
    report = {
        'date': datetime.now(timezone.utc).isoformat(),
        'status': 'NOT_MEASURED',
        'scope': 'Single-frame FSR3 compatibility; no temporal quality or player performance claim',
        'input': str(args.input.resolve()),
        'dll': str(dll.resolve()),
        'dllSha256': hashlib.sha256(dll.read_bytes()).hexdigest(),
        'exeSha256': hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        'inputSha256': hashlib.sha256(raw_input.read_bytes()).hexdigest(),
    }
    command = [str(snapshot), str(dll.resolve()), str(raw_input), str(width), str(height), str(output_width), str(output_height), str(raw_output)]
    try:
        # Timeout terminates this child; never retries with a contaminated context.
        result = subprocess.run(command, capture_output=True, timeout=60)
        stdout, stderr = result.stdout.decode('utf-8', errors='replace'), result.stderr.decode('utf-8', errors='replace')
        (folder / 'stdout.txt').write_text(stdout, encoding='utf-8')
        (folder / 'stderr.txt').write_text(stderr, encoding='utf-8')
        report['exitCode'] = result.returncode
        report['command'] = command
        if result.returncode:
            report['status'] = 'PROBE_FAILED'
        else:
            report['native'] = json.loads(stdout)
            data = raw_output.read_bytes()
            if len(data) != output_width * output_height * 4:
                raise ValueError('Output byte count mismatch')
            output = Image.frombytes('RGBA', (output_width, output_height), data)
            output.save(folder / 'output.png')
            image.resize((output_width, output_height), Image.Resampling.BILINEAR).save(folder / 'bilinear.png')
            report['outputSha256'] = hashlib.sha256(data).hexdigest()
            report['status'] = 'PASS'
    except subprocess.TimeoutExpired:
        report['status'] = 'GPU_PROBE_TIMEOUT'
    except Exception as error:
        report['status'] = 'PROBE_FAILED'
        report['error'] = str(error)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
