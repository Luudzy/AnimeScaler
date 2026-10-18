"""Plot the same frozen frames and ROI used for the5k still comparison."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]

def main():
    folders=[ROOT/'diagnostics/zenkai-real-video-frequency-3k',ROOT/'diagnostics/zenkai-real-video-skip-3k']
    reports=[json.loads((folders[0]/'report.json').read_text(encoding='utf-8')),json.loads((ROOT/'diagnostics/zenkai-real-video-skip-3k-complete/report.json').read_text(encoding='utf-8'))]
    initial=json.loads((folders[1]/'report.json').read_text(encoding='utf-8'))
    completed={Path(s['manifest']).parent.name for s in initial['sequences']}
    assert all(r['state']=='complete' and len(r['sequences'])==3 and sum(s['frameCount'] for s in r['sequences'])==144 for r in reports)
    destination=ROOT/'diagnostics/zenkai-real-video-comparison-crops-3k'
    if destination.exists():raise FileExistsError('Preserve generated comparisons')
    destination.mkdir();roi=(448,100,832,484);provenance=[]
    for a,b in zip(reports[0]['sequences'],reports[1]['sequences']):
        assert a['manifestSha256']==b['manifestSha256']
        clip=Path(a['manifest']).parent.name
        for frame in (0,24,47):
            entries=[('Reference',folders[0]/clip/f'{frame:04d}-reference.png'),
                ('Bilinear',folders[0]/clip/f'{frame:04d}-bilinear.png'),
                ('HistoricalV3',folders[0]/clip/f'{frame:04d}-adore-lite-v3.png'),
                ('Frequency3k',folders[0]/clip/f'{frame:04d}-zenkai-ablation-frequency-3k-continuation.png'),
                ('Identity skip3k',(folders[1] if clip in completed else ROOT/'diagnostics/zenkai-real-video-skip-3k-remainder1')/clip/f'{frame:04d}-zenkai-ablation-spab-identity-skip-3k-continuation.png')]
            fig,axes=plt.subplots(1,5,figsize=(20,4.5),dpi=100)
            for axis,(label,path) in zip(axes,entries):
                with Image.open(path) as im:axis.imshow(im.crop(roi),interpolation='nearest')
                axis.set_title(label);axis.axis('off')
            fig.suptitle(f'{clip}, frame{frame}, fixed ROI{roi}; RGB8-bit display crops')
            fig.tight_layout();out=destination/f'{clip}-{frame:04d}.png';fig.savefig(out);plt.close(fig)
            provenance.append({'clip':clip,'frame':frame,'roi':roi,'manifestSha256':a['manifestSha256'],
                'inputs':{label:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for label,path in entries},
                'output':str(out),'outputSha256':hashlib.sha256(out.read_bytes()).hexdigest()})
    (destination/'provenance.json').write_text(json.dumps({'scope':'Same0/24/47 frames and384px ROI as5k; still review only, not moving-video perceptual certification','comparisons':provenance},indent=2),encoding='utf-8')
    print('Nine frozen-frame comparisons generated')

if __name__=='__main__':main()
