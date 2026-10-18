"""Same frozen frame/ROI panels for the three completed factors; never selects frames by scores."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
FACTORS=('dataset-sampling','crop96','codec50')


def main():
    folders=[ROOT/'diagnostics/zenkai-real-video-skip-frequency-1k']+[
        ROOT/f'diagnostics/zenkai-real-video-factor-{factor}-1k' for factor in FACTORS]
    reports=[json.loads((folder/'report.json').read_text()) for folder in folders]
    assert all(r['state']=='complete' and len(r['sequences'])==3 and sum(s['frameCount'] for s in r['sequences'])==144 for r in reports)
    destination=ROOT/'diagnostics/zenkai-real-video-comparison-crops-factors-1k'
    if destination.exists():raise FileExistsError('Preserve original panels')
    destination.mkdir();roi=(448,100,832,484);provenance=[]
    for clip_index,sequence in enumerate(reports[0]['sequences']):
        assert all(report['sequences'][clip_index]['manifestSha256']==sequence['manifestSha256'] for report in reports)
        clip=Path(sequence['manifest']).parent.name
        for frame in (0,24,47):
            entries=[('Reference',folders[0]/clip/f'{frame:04d}-reference.png'),
                     ('Bilinear',folders[0]/clip/f'{frame:04d}-bilinear.png'),
                     ('HistoricalV3',folders[0]/clip/f'{frame:04d}-adore-lite-v3.png'),
                     ('Skip Frequency1k',folders[0]/clip/f'{frame:04d}-zenkai-ablation-spab-identity-skip-1k.png')]+[
                     (factor,folder/clip/f'{frame:04d}-zenkai-ablation-skip-factor-{factor}-1k.png')
                     for factor,folder in zip(FACTORS,folders[1:])]
            fig,axes=plt.subplots(2,4,figsize=(16,8.5),dpi=100)
            for axis,(label,path) in zip(axes.flat,entries):
                with Image.open(path) as im:axis.imshow(im.crop(roi),interpolation='nearest')
                axis.set_title(label);axis.axis('off')
            axes.flat[-1].axis('off')
            fig.suptitle(f'{clip}, frame{frame}, fixed ROI{roi}; same RGB8-bit display crops')
            fig.tight_layout();out=destination/f'{clip}-{frame:04d}.png';fig.savefig(out);plt.close(fig)
            provenance.append({'clip':clip,'frame':frame,'roi':roi,'manifestSha256':sequence['manifestSha256'],
                'inputs':{label:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for label,path in entries},
                'output':str(out),'outputSha256':hashlib.sha256(out.read_bytes()).hexdigest()})
    (destination/'provenance.json').write_text(json.dumps({'scope':'Same frozen0/24/47 frames and ROI; still panels, not moving/blind study',
        'unavailable':'Scheduler failed before final checkpoint; no output generated', 'comparisons':provenance},indent=2),encoding='utf-8')
    print('Nine frozen panels for three completed factors saved')


if __name__=='__main__':main()
