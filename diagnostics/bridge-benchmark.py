"""Real binary HTTP roundtrip + quality against original ONNX reference."""
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import numpy as np

root = Path(__file__).resolve().parents[1]
connection = json.loads((root / 'native/connection.json').read_text())
fixture = root / 'diagnostics/native-fixture'
data = np.load(fixture / 'input.npy', allow_pickle=False)
reference = np.load(fixture / 'reference.npy', allow_pickle=False)
height, width = data.shape[2:]
rgba = np.full((height, width, 4), 255, dtype=np.uint8)
rgba[..., :3] = (data[0].transpose(1, 2, 0) * 255).round().astype(np.uint8)
header = struct.Struct('<6I')
samples, processing = [], []
for index in range(25):
    body = header.pack(0x31435341, index, width, height, width*2, height*2) + rgba.tobytes()
    request = urllib.request.Request(connection['url']+'/frame', data=body,
        headers={'Authorization': 'Bearer '+connection['token'], 'Content-Type': 'application/octet-stream'})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=60) as response:
        result = response.read()
        duration = (time.perf_counter()-start)*1000
        if index >= 5:
            samples.append(duration)
            processing.append(float(response.headers['X-Processing-Ms']))
    assert header.unpack_from(result) == (0x31435341, index, width*2, height*2, width*2, height*2)
actual = np.frombuffer(result, np.uint8, offset=24).reshape(height*2, width*2, 4)
assert (actual[..., 3] == 255).all()
delta = np.abs(actual[..., :3].astype(np.float32)/255 - reference[0].transpose(1,2,0).clip(0,1))
quality = {'meanAbsoluteError':float(delta.mean()),'maxAbsoluteError':float(delta.max())}
assert quality['meanAbsoluteError'] < .005 and quality['maxAbsoluteError'] < .05, quality
report = {'inputSha256':hashlib.sha256(data.tobytes()).hexdigest(), 'frames':20, 'input':[width,height], 'output':[width*2,height*2],
          'roundtripMedianMs':float(np.median(samples)), 'roundtripP95Ms':float(np.percentile(samples,95)),
          'serviceMedianMs':float(np.median(processing)), 'quality':quality,
          'scope':'HTTP + RGBA copies + ROCm inference; no browser capture or display'}
(root/'diagnostics/bridge-results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
