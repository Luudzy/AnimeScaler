"""Compare completed native and spatial evaluation in a common RGB8 metric scope."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixels(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB')).astype(np.float32)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,action='append',required=True)
    parser.add_argument('--spatial',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    spatial=json.loads(args.spatial.read_text(encoding='utf-8'))
    if spatial['state']!='complete' or spatial['browserErrors']: raise ValueError('Spatial evaluation incomplete')
    spatial_cases={Path(row['manifest']).parent.name:row for row in spatial['sequences']}
    report={'state':'running','scope':'Common full-frame RGB8 MAE, mean per-frame PSNR and all-transition reference-compensated temporal MAE',
            'limitations':['No optical-flow alignment or cut exclusion in these common temporal metrics.', 'Metrics do not establish perceptual approval or broad modern-anime generalization.', 'CPU FP32 outputs rounded to RGB8; spatial outputs are actual WebGPU RGB8 readbacks.'],
            'humanOrBlindReview':'NOT EXECUTED','releaseApproved':False,
            'sourceReports':[{'path':str(args.spatial),'sha256':sha(args.spatial)}],'sequences':[]}
    names=['bilinear','adore-lite-v3','zenkai-animevideo-30k']
    for folder in args.evaluation:
        raw=json.loads((folder/'report.json').read_text(encoding='utf-8'))
        if raw['state']!='complete': raise ValueError('Native evaluation incomplete')
        report['sourceReports'].append({'path':str(folder/'report.json'),'sha256':sha(folder/'report.json')})
        for case in raw['sequences']:
            manifest_path=Path(case['manifest']); key=manifest_path.parent.name
            if sha(manifest_path)!=case['manifestSha256']: raise ValueError('Manifest changed after evaluation')
            manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
            spatial_key='/'+manifest_path.relative_to(Path.cwd()).as_posix()
            if spatial['manifestHashes'][spatial_key]!=case['manifestSha256']: raise ValueError('Spatial/native manifest mismatch')
            preview=folder/key; records={name:{'mae':[],'psnr':[],'temporal':[]} for name in names}; previous=None
            for index, frame in enumerate(manifest['frames']):
                reference=pixels(preview/f'{index:04d}-reference.png')
                hr=manifest_path.parent/frame['hr']
                if sha(hr)!=frame['hrSha256'] or not np.array_equal(reference,pixels(hr)): raise ValueError('Preview reference differs from verified HR')
                outputs={name:pixels(preview/f'{index:04d}-{name}.png') for name in names}
                for name, output in outputs.items():
                    if output.shape!=reference.shape: raise ValueError('Output geometry mismatch')
                    delta=output-reference; rows=records[name]
                    rows['mae'].append(float(np.mean(np.abs(delta),dtype=np.float64)/255))
                    mse=float(np.mean(delta*delta,dtype=np.float64)/255**2)
                    rows['psnr'].append(float(10*np.log10(1/max(1e-12,mse))))
                    if previous is not None:
                        old_reference,old_outputs=previous
                        residual=(output-old_outputs[name])-(reference-old_reference)
                        rows['temporal'].append(float(np.mean(np.abs(residual),dtype=np.float64)/255))
                previous=reference,outputs
            models={name:{'mae':float(np.mean(rows['mae'])),'psnr':float(np.mean(rows['psnr'])),
                          'referenceCompensatedTemporalMae':float(np.mean(rows['temporal']))} for name,rows in records.items()}
            models.update(spatial_cases[key]['models'])
            category='historical-anime' if key.startswith('namakura') else ('credits' if key.startswith('hero-150s') else '2d-animation')
            report['sequences'].append({'sequence':key,'category':category,'frames':len(manifest['frames']),'manifestSha256':case['manifestSha256'],'models':models})
            print(json.dumps({'summarized':key}),flush=True)
    report['state']='complete';report['pairedFrames']=sum(row['frames'] for row in report['sequences'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream: json.dump(report,stream,indent=2)


if __name__=='__main__':main()
