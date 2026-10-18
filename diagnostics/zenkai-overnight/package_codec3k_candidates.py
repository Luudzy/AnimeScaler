"""Locally package numerically validated research candidates, never a product release."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
ROOT=Path(__file__).resolve().parents[2]


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    pair=read(ROOT/'diagnostics/zenkai-codec-pair-3k/comparison.json')
    exports=read(ROOT/'diagnostics/zenkai-codec-pair-3k-exports-recovery1/pipeline.json')
    if pair['state']!='COMPLETE' or exports['state']!='COMPLETE':raise ValueError('Unfinished candidate operations')
    destination=ROOT/'models/zenkai/review-packages/codec3k'
    if destination.exists():raise FileExistsError('Preserve prior package')
    destination.mkdir(parents=True);packages={}
    for label,run in pair['runs'].items():
        operation=exports['results'][label+'-export'];runtime=exports['results'][label+'-runtime']
        if operation['state']!='PASS' or runtime['state']!='PASS':
            packages[label]={'state':'NOT PACKAGED','reason':'Numeric export/runtime parity incomplete'};continue
        folder=Path(operation['result']['folder']);e=read(folder/'export-report.json');r=read(runtime['result']['path'])
        checkpoint=Path(e['checkpoint'])
        if e['checkpointSha256']!=pair[label]['checkpointsSha256']['step-3000-master-fp32.pt'] or sha(checkpoint)!=e['checkpointSha256']:
            raise ValueError('Source artifact mismatch')
        if not e['parityApproved'] or not r['parityApproved'] or r['exportReportSha256']!=sha(folder/'export-report.json'):
            raise ValueError('Parity record mismatch')
        package=destination/run;package.mkdir()
        for name,record in e['artifacts'].items():
            if Path(name).name!=name or sha(folder/name)!=record['sha256']:raise ValueError('Export artifact mutated')
            shutil.copyfile(folder/name,package/name)
        shutil.copyfile(folder/'export-report.json',package/'export-report.json')
        shutil.copyfile(runtime['result']['path'],package/'rocm-parity.json')
        shutil.copyfile(checkpoint,package/'training-master-fp32.pt')
        (package/'config.json').write_text(json.dumps(pair[label]['config'],indent=2),encoding='utf-8')
        (package/'measurements.json').write_text(json.dumps({
            'quality':pair['quality'],'video':pair['video'],'performance':pair['performance'],
            'gradients':pair['gradients'],'training':pair[label],
            'scope':'Research screening; no promotion, release approval, hidden holdout or player FPS'},indent=2),encoding='utf-8')
        shutil.copyfile(ROOT/'native/vendor/espan/LICENSE',package/'ESPAN-MIT-NOTICE.txt')
        source=package/'source';source.mkdir()
        for name in ('zenkai_model.py','zenkai_ablation_skip.py','export_zenkai_skip.py'):
            shutil.copyfile(ROOT/'native'/name,source/name)
        quality=pair['quality']['cases'];key='skip-'+label+'-3k'
        mae={domain:100*(m[key]['mae']/m['adore-lite-v3']['mae']-1) for domain,m in quality.items()}
        card=[f'# {run} — experimental review artifact','',
            '**Technical export: PASS. Product quality advancement: FAIL at this screening. Not release-approved.**','',
            'This package can be uploaded as a private experimental artifact for review. It is not registered in the extension and does not replace Historical Adore V3. Original checkpoints remain intact.','',
            'Native x2, RGB NCHW, batch1, float values0–1. Input `input`, output `output`, dynamic H/W. FP32/FP16 ONNX opset17. The graph includes bilinear x2 plus the learned residual; do not add a second bilinear base. Clamp the final output to0–1. 720→1080 uses the existing x2→1440 then bilinear post-resize; no x1.5 head.','',
            f'Training:3000 total steps, seed929, channels32/depth5, codec fraction{pair[label]["config"]["codecFraction"]}, LR128/HR256, batch4×2, frequency loss, original5000-step cosine schedule. Training params{e["trainingParameters"]}; fused deploy params{e["deployParameters"]}. Family`ZenkaiIdentitySkipProbe` is distinct from ordinary`Zenkai`; preserve its identity bypass.','',
            'Numeric fusion/FP16/ONNX CPU parity and reloaded PyTorch ROCm parity passed six fixed shape/pattern cases. ONNX ROCm/TensorRT/WebGPU parity and end-to-end player FPS: NOT MEASURED. Do not infer support for a backend solely from ONNX export.','',
            'Fixed validation:160 samples,32 each clean/heavy/H264/H265/AV1. MAE relative to HistoricalV3 (positive is worse): '+', '.join(f'{d}:{v:+.4f}%' for d,v in mae.items())+'.','',
            'Same144 real frames from three48-frame360→720 sequences. Codec50 improves all clips against Codec25 but still loses HistoricalV3 in two; see measurements. Reference is a compressed WebRip, not established native4K ground truth. No blind preference/moving perceptual study executed. Final-hidden remains unopened.','',
            'Latency comes from completion-timed model/post-resize/total with50 warmups/500 distinct inputs; roughly18ms360→720 and67ms720→1080 on RX5500XT. It is not player FPS and the720p path does not meet a30fps inference budget here.','',
            'Weights trained locally; no official ESPAN pretrained checkpoint is relabeled. The ESPAN inspiration notice is included. This package does not assign a new public redistribution license to weights or dataset content. No training/reference images are bundled.','',
            'All member hashes are in`package-manifest.json`. `training-master-fp32.pt` is the unchanged unfused master; `deploy-fp32.pt` is fused weights for the explicit skip factory; ONNX files are standalone deploy graphs.']
        (package/'README.md').write_text('\n'.join(card)+'\n',encoding='utf-8')
        members={str(p.relative_to(package)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size}
                 for p in sorted(package.rglob('*')) if p.is_file()}
        manifest={'family':e['family'],'run':run,'step':3000,'status':'experimental',
                  'technicalExport':'PASS','qualityAdvancementAgainstHistoricalV3':'FAIL at measured screening',
                  'releaseApproved':False,'promoted':False,'members':members}
        (package/'package-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        archive=destination/(run+'.zip')
        with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(package.rglob('*')):
                if p.is_file():z.write(p,str(p.relative_to(package)))
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:raise ValueError('ZIP CRC failed')
            for name,record in members.items():
                if hashlib.sha256(z.read(name)).hexdigest()!=record['sha256']:raise ValueError('ZIP member SHA failed')
        packages[label]={'state':'PACKAGED EXPERIMENTAL','archive':str(archive),'sha256':sha(archive),
                         'bytes':archive.stat().st_size,'members':len(members),'qualityGate':'FAIL, not promoted'}
    (destination/'packages.json').write_text(json.dumps(packages,indent=2),encoding='utf-8')
    print(json.dumps(packages,indent=2))


if __name__=='__main__':main()
