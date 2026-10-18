"""Preserved frame0/24/47 and ROI, matched codec25/50 visual panels."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]


def main():
    pair=json.loads((ROOT/'diagnostics/zenkai-codec-pair-3k/comparison.json').read_text())
    if 'codec50' not in pair['video']:raise ValueError('Matched video not measured')
    old=pair['video']['codec25'];new=pair['video']['codec50']
    operation=pair['operations']['video-codec50']
    new_folder=Path(operation['result']['path']).parent
    old_sources=[(ROOT/path,json.loads((ROOT/path).read_text())) for path in old['mergedSourceReports']]
    dest=ROOT/'diagnostics/zenkai-codec-pair-3k/crops-v2'
    if dest.exists() and any(dest.iterdir()):raise FileExistsError('Preserve matched panels')
    dest.mkdir(exist_ok=True);roi=(448,100,832,484);records=[]
    for sequence in new['sequences']:
        clip=Path(sequence['manifest']).parent.name
        for frame in (0,24,47):
            old_name=f'{frame:04d}-{pair["runs"]["codec25"]}.png'
            # Ignore files left in an incomplete sequence of the interrupted source.
            paths=[path.parent/clip/old_name for path,report in old_sources if any(
                s['manifestSha256']==sequence['manifestSha256'] and s['frameCount']==48 for s in report['sequences'])]
            valid=[p for p in paths if p.is_file()]
            if len(valid)!=1:raise ValueError('No unique preserved Codec25 frame')
            entries=[('Reference',new_folder/clip/f'{frame:04d}-reference.png'),
                     ('Bilinear',new_folder/clip/f'{frame:04d}-bilinear.png'),
                     ('Historical V3',new_folder/clip/f'{frame:04d}-adore-lite-v3.png'),
                     ('Codec25 3k',valid[0]),
                     ('Codec50 3k',new_folder/clip/f'{frame:04d}-{pair["runs"]["codec50"]}.png')]
            fig,axes=plt.subplots(1,5,figsize=(20,4.5),dpi=100)
            for axis,(label,path) in zip(axes,entries):
                with Image.open(path) as im:axis.imshow(im.crop(roi),interpolation='nearest')
                axis.set_title(label);axis.axis('off')
            fig.suptitle(f'{clip}, frame{frame}, fixed ROI{roi}; same RGB8 crops')
            fig.tight_layout();output=dest/f'{clip}-{frame:04d}.png';fig.savefig(output);plt.close(fig)
            records.append({'clip':clip,'frame':frame,'roi':roi,'manifestSha256':sequence['manifestSha256'],
                'inputs':{label:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for label,path in entries},
                'output':str(output),'outputSha256':hashlib.sha256(output.read_bytes()).hexdigest()})
    (dest/'provenance.json').write_text(json.dumps({'scope':'Fixed still crops, not blind or moving perceptual study',
        'comparisons':records},indent=2),encoding='utf-8')
    print('Nine matched codec3k panels saved')


if __name__=='__main__':main()
