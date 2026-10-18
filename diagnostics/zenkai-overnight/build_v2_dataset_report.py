"""Only selected complete V2 operations; individual regressions remain visible."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'native'))
from frozen_video_protocol import validate_protocol
from artifact_provenance import validate_artifact_pins
from build_codec3k_report import selected,table,pct
BASE=ROOT/'diagnostics/zenkai-v2-dataset-1k'
RUN='zenkai-ablation-skip-v2-dataset-1k'
REFERENCE='zenkai-ablation-spab-identity-skip-1k'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    pipeline=read(BASE/'pipeline.json');training=read(ROOT/'diagnostics/zenkai-training/v2-dataset-1k-training/operation.json')
    if pipeline['state']!='COMPLETE' or training['state']!='PASS':raise ValueError('V2 operations unfinished')
    results={k:selected(v) for k,v in pipeline['results'].items()}
    folder=ROOT/'models/zenkai/experiments'/RUN;cfg=read(folder/'config.json');train=read(folder/'report.json')
    if train['stepsExecuted']!=1000 or train['optimizerUpdates']!=1000 or train['samples']!=8000:raise ValueError('Wrong actual V2 budget')
    for name,pin in cfg['sourceFilesSha256'].items():
        if sha(folder/'source'/name)!=pin:raise ValueError('Archived V2 source changed')
    index=read(folder/'checkpoint-index.json')
    if index['lastCompleteStep']!=1000:raise ValueError('Incomplete indexed final state')
    for name,pin in index['checkpoints'].items():
        if sha(folder/name)!=pin:raise ValueError('Recovery state changed')
    domains=train['sampledTrainDomainCounts'];fraction=domains.get('illustration',0)/8000
    pairing=read(folder/'initial-pairing.json')
    if pairing['state']!='matched' or not pairing['baselineSamplerReproducedInActualStack']:raise ValueError('Unpaired source')
    pins={p.name:sha(p) for p in folder.glob('*.pt')}
    from zenkai_v2_dataset_protocol import load_candidate
    load_candidate(folder/'step-1000-master-fp32.pt')
    if (pins['step-1000-master-fp32.pt']!=training['result']['sha256']
            or pins['step-1000-training-state.pt']!=training['result']['stateSha256']):
        raise ValueError('Completed training hashes changed before reporting')
    report={'state':'COMPLETE','run':RUN,'trainingOperation':training,'config':cfg,'training':train,
        'initialPairing':pairing,'checkpointsSha256':pins,'operations':pipeline['results'],
        'quality':None,'video':None,'performance':None,'gradients':None,'releaseApproved':False,
        'promoted':False,'Stage2Started':False,'finalHiddenOpened':False}
    sampling=read(BASE/'sampling-analysis.json')
    if sampling['v2PlannedDomainDraws']!=domains or sampling['v2PlannedUniqueImages']!=train['distinctTrainImagesSampled']:
        raise ValueError('Actual completed sampler differs from preparation analysis')
    report['samplingAnalysis']={**sampling,'scope':'Baseline actual1k versus V2 actual completed1k; prefix counts confirmed by final training state'}
    lines=['# Dataset V2:screening controlado de1k','',
        'Novo run a partir do estado inicial real do Skip+Frequency1k. Única mudança:TRAIN recebe602 ilustrações adicionais e o sampler uniforme passa a2394 identidades. Pesos/optimizer/scaler/Python,Torch,GPU,NumPy,datasetRNG iniciais pareados; algoritmo do sampler original reproduzido no Torch/ROCm real.','',
        '32 canais/depth5/x2/identitySkip1;Frequency loss;codec.25;patch128;batch4×2;seed929;AdamW(.9,.99),wd1e-5;warmup300+cosine5000. Nenhuma combinação com Codec50/group/crop/Legacy/constant scheduler.','',
        f'Treino1000 steps/1000 updates,exit0,tentativa{training["selectedAttempt"]},tempo{train["trainingSeconds"]:.3f}s.8000 draws:anime{domains.get("anime",0)},illustration{domains.get("illustration",0)};fração realizada{fraction:.6%},expectativa uniforme{602/2394:.6%};identidades distintas{train["distinctTrainImagesSampled"]}. Codec draws{train["codecSamples"]}. Unique spatial crops:NOT MEASURED através de recoveries.','',
        'Validação permanece exclusivamente o protocolo anime fixo; não houve nova validação illustration nem acesso a heldouts V2/hidden. As identidades de validação/development/final-hidden foram mantidas no manifest.','']
    lines+=['Loss de treino não é diretamente comparável entre populações diferentes e não seleciona o dataset. A fração realizada vem das8000 identidades efetivamente consumidas pelo sampler, não de uma porcentagem presumida.','']
    lines += [f"O orçamento total é pareado, mas a exposição a anime cai de8000 para{domains['anime']} draws ({sampling['animeExposureChangePercent']:.4f}%). A união com amostragem uniforme testa essa alocação; não isola a qualidade intrínseca das ilustrações. Igualar a exposição a anime exigiria outro experimento declarado, sem alterar este run.",'']
    lines+=table(['Operation','State','Selected attempt'],[[k,v['state'],v['selectedAttempt']] for k,v in pipeline['results'].items()])
    if results['quality']:
        q=read(results['quality']['quality']);p=read(results['quality']['parity']);report.update(quality=q,parity=p);rows=[]
        old=read(ROOT/'diagnostics/zenkai-ablations/skip-loss-1k-quality/quality.json')
        baseline_pins=read(ROOT/'models/zenkai/experiments'/REFERENCE/'metadata.json')['checkpointsSha256']
        baseline_sha=baseline_pins['step-1000-master-fp32.pt']
        validate_artifact_pins({'fixedSkip':q['models'].get('skip-frequency-1k',{}),'historicalSkip':old['models'].get('skip-frequency-1k',{}),
            'fixedV3':q['models'].get('adore-lite-v3',{}),'historicalV3':old['models'].get('adore-lite-v3',{})},
            {'fixedSkip':baseline_sha,'historicalSkip':baseline_sha,'fixedV3':cfg['legacyV3Sha256'],'historicalV3':cfg['legacyV3Sha256']})
        for key in ('indices','validationManifestSha256','sourceManifestSha256','codecManifestSha256'):
            if q[key]!=old[key]:raise ValueError('Fixed protocol changed')
        if q['models']['v2-dataset-1k']['sha256']!=pins['step-1000-master-fp32.pt']:raise ValueError('Wrong fixed candidate')
        for domain,m in q['cases'].items():
            for metric,v3 in m['adore-lite-v3'].items():
                ref=m['skip-frequency-1k'][metric];new=m['v2-dataset-1k'][metric]
                rows.append([domain,metric,f'{v3:.9f}',f'{ref:.9f}',f'{new:.9f}',pct(new,ref),pct(new,v3)])
        lines+=['','## Mesmos160 samples','',
            'MAE/MSE/Edge/Laplacian/HF/DCT/LPIPS:menor é melhor. PSNR/SSIM/MS-SSIM:maior é melhor;PSNR emdB. Todas as métricas individuais são preservadas.','']
        lines+=table(['Domain','Metric','HistoricalV3','Anime1k','Anime+V2 1k','V2/Anime %','V2/V3 %'],rows)
        rows=[]
        for name,r in p['models'].items():
            for comparison in ('fusionFp32','deployFp16'):
                rows.append([name,comparison,max(c[comparison]['max_abs_error'] for c in r['cases']),max(c[comparison]['mean_abs_error'] for c in r['cases'])])
        lines+=['','## Paridade ROCm','']+table(['Model','Comparison','Worst max_abs_error','Worst mean_abs_error'],rows)
    else:lines+=['','Fixed/parity:NOT MEASURED.']
    if results['video']:
        ref=read(ROOT/'diagnostics/zenkai-real-video-skip-frequency-1k/report.json');new=read(results['video']['path'])
        validate_protocol(ref);validate_protocol(new);report['video']=new;rows=[]
        baseline_sha=read(ROOT/'models/zenkai/experiments'/REFERENCE/'metadata.json')['checkpointsSha256']['step-1000-master-fp32.pt']
        validate_artifact_pins({'videoSkip':ref['artifacts'].get(REFERENCE,{}),'historicalV3':ref['artifacts'].get('adore-lite-v3',{}),'videoV3':new['artifacts'].get('adore-lite-v3',{})},
            {'videoSkip':baseline_sha,'historicalV3':cfg['legacyV3Sha256'],'videoV3':cfg['legacyV3Sha256']})
        if ref['cutPolicy']!=new['cutPolicy'] or ref['backend']!=new['backend'] or ref['precision']!=new['precision']:raise ValueError('Video protocol changed')
        if new['artifacts'][RUN]['sha256']!=pins['step-1000-master-fp32.pt']:raise ValueError('Wrong V2 video artifact')
        for a,b in zip(ref['sequences'],new['sequences']):
            if a['manifestSha256']!=b['manifestSha256'] or a['suspectedCutTransitions']!=b['suspectedCutTransitions']:raise ValueError('Frozen video changed')
            for metric in ('mae','referenceCompensatedTemporalMae','allTransitionTemporalMae'):
                v3=a['models']['adore-lite-v3'][metric];r=a['models'][REFERENCE][metric];v=b['models'][RUN][metric]
                if abs(v3-b['models']['adore-lite-v3'][metric])>1e-9:raise ValueError('V3 CPU baseline changed')
                rows.append([Path(a['manifest']).parent.name,metric,f'{v3:.9f}',f'{r:.9f}',f'{v:.9f}',pct(v,r),pct(v,v3)])
        lines+=['','## Mesmos144 frames','',
            '3×48,360→720,CPU FP32,referência WebRip comprimida e LR H264. Cut policy original,all-transitions preservados. Não é certificação perceptual/blind/flicker/player.','']
        lines+=table(['Clip','Metric (lower better)','V3','Anime1k','Anime+V2 1k','V2/Anime %','V2/V3 %'],rows)
    else:lines+=['','Real video:NOT MEASURED.']
    if results['benchmark']:
        perf=read(results['benchmark']['path']);report['performance']=perf;rows=[]
        if not perf['complete'] or perf['precision']!='fp16' or perf['provider']!='ROCm':raise ValueError('Invalid completed performance')
        for case in perf['cases']:
            if sha(case['artifactPath'])!=case['artifactSha256'] or case['warmup']!=50 or case['iterations']!=500:raise ValueError('Incomplete performance')
            for component,m in case['components'].items():
                if 'medianMs' in m:rows.append([case['model'],f'{case["width"]}×{case["height"]}→{case["outputWidth"]}×{case["outputHeight"]}',component,
                    *[m[k] for k in ('meanMs','medianMs','p50Ms','p95Ms','p99Ms','minMs','maxMs','fps')],case['peakAllocatedMiB']])
        lines+=['','## Performance','',
            '50 warmups/500 entradas distintas,FP16 ROCm,componentes completos da execução selecionada. FPS equivalente não é FPS do player. Dataset não muda a geometria de inferência;variação entre runs não prova ganho de velocidade pelo dataset.','']
        lines+=table(['Model','Shape','Component','Mean','Median','P50','P95','P99','Min','Max','FPS equivalent','Peak MiB'],rows)
    else:lines+=['','Latency:NOT MEASURED.']
    if results['gradients']:
        report['gradients']=read(results['gradients']['path']);rows=[]
        for name,g in report['gradients']['runs'].items():
            for case in g['cases']:
                for group,fp32 in case['fp32']['groups'].items():
                    amp=case['amp']['groups'][group]
                    change=g['accumulatedWeightChanges'][group]
                    rows.append([name,case['case'],group,fp32['gradientL2'],amp['gradientL2'],
                        change['initialL2'],change['deltaL2'],change['deltaToInitialRatio'],amp['zeroElementsPct']])
        lines+=['','## Gradientes e mudança de pesos','',
            'Backward local sobre clean/heavy no checkpoint1k,sem update. AMP usa o scaler real salvo,com norma após unscale. Peso inicial e delta acumulado não são uma atualização isolada de optimizer.','']
        lines+=table(['Run','Case','Group','FP32 grad L2','AMP grad L2','Initial weight L2','Cumulative delta L2','Delta/initial','AMP zeros %'],rows)
    lines+=['','Normas de gradiente FP32/scaledAMP e deltas acumulados init→1k estão no diagnostic selecionado. Não foram feitos optimizer updates por essa análise;delta acumulado não é norma de atualização de um step. Export ONNX para esta ablation:NOT EXECUTED.','',
        'Decisão de dataset/continuação é separada e considera vídeo/métricas individuais,sem promoção automática ou15k.']
    (BASE/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (ROOT/'docs/zenkai-v2-dataset-results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save={**report,'trainingExitCode':0,'stepFinal':1000,'validationHistory':train['history'],
        'evaluationReport':str(BASE/'comparison.json'),'evaluationReportSha256':sha(BASE/'comparison.json')}
    (folder/'metadata.json').write_text(json.dumps(save,indent=2),encoding='utf-8')
    print('V2 complete selected results,individual tables,metadata and hashes saved')


if __name__=='__main__':main()
