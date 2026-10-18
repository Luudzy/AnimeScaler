"""Measure optional JPEG98 distortion and bridge processing cost on held-out images."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import io
import json
import numpy as np
from PIL import Image
from server import ROOT, Renderer
from protocol import HEADER,MAGIC

renderer=Renderer()
folder=ROOT/'native/training-data/v2'
rows=[json.loads(x) for x in (folder/'manifest.jsonl').read_text().splitlines()]
results=[]
for row in [r for r in rows if r['split']=='test'][:8]:
    with Image.open(folder/row['file']) as source:
        source=source.convert('RGBA').resize((1280,720),Image.Resampling.LANCZOS)
        request=HEADER.pack(MAGIC,1,1280,720,1920,1080)+source.tobytes()
    renderer.render(request,'adore-lite')
    raw,raw_ms=renderer.render(request,'adore-lite')
    compressed,jpeg_ms=renderer.render(request,'adore-lite','jpeg')
    reference=np.frombuffer(raw,dtype=np.uint8,offset=24).reshape(1080,1920,4)[...,:3].astype(np.float32)
    actual=np.asarray(Image.open(io.BytesIO(compressed[24:])).convert('RGB'),dtype=np.float32)
    mse=float(np.mean((reference-actual)**2))
    results.append({'image':row['sha256'],'psnrDb':float(10*np.log10(255**2/max(mse,1e-9))),
                    'maeByte':float(np.mean(np.abs(reference-actual))),'rgbaBytes':len(raw),'jpegBytes':len(compressed),
                    'rgbaMs':raw_ms,'jpegMs':jpeg_ms})
report={'quality':98,'subsampling':'4:4:4','cases':results,'minimumPsnrDb':min(r['psnrDb'] for r in results)}
(ROOT/'diagnostics/transport-quality.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
