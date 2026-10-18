"""Actual encode/decode degradation. Intra-only patches, not temporal footage."""
import sys,argparse,json,subprocess,random,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from PIL import Image
from lite_evaluation import ROOT,load_patches

def roundtrip(frames,codec='libx264',crf=24):
    exe=ROOT/'native/.tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
    if not exe.exists():raise RuntimeError('Install imageio-ffmpeg in native/.tools first')
    height,width=frames[0].shape[:2]
    flags=['-preset','veryfast'] if codec=='libx264' else ['-preset','ultrafast','-x265-params','pools=2:frame-threads=1:log-level=error'] if codec=='libx265' else ['-cpu-used','8','-row-mt','1']
    args=[str(exe),'-v','error','-threads','2','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r','24','-i','pipe:0',
          '-an','-c:v',codec,*flags,'-threads','2','-crf',str(crf),'-pix_fmt','yuv420p','-g','1','-f','ivf' if codec=='libaom-av1' else 'matroska','pipe:1']
    encoded=subprocess.run(args,input=np.stack(frames).tobytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=240).stdout
    decoded=subprocess.run([str(exe),'-v','error','-threads','2','-i','pipe:0','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],input=encoded,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=120).stdout
    result=np.frombuffer(decoded,np.uint8).reshape(-1,height,width,3)
    if len(result)!=len(frames):raise ValueError('Codec frame count mismatch')
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['train','validation'],default='validation')
    parser.add_argument('--dataset',type=Path,default=ROOT/'native/training-data/v3');parser.add_argument('--patches-per-image',type=int,default=4);args=parser.parse_args()
    patches=load_patches(args.split,per_image=args.patches_per_image if args.split=='train' else 1,seed=929 if args.split=='train' else 193,dataset=args.dataset)
    folder=args.dataset/f'codec-{args.split}';folder.mkdir(exist_ok=True)
    codecs=['libx264','libx265','libaom-av1'];rng=random.Random(194);records=[]
    for start in range(0,len(patches),32):
        codec=codecs[(start//32)%3];crf=rng.choices([18,24,30],[.3,.5,.2])[0]
        inputs=[np.array(Image.fromarray(p).resize((96,96),Image.Resampling.LANCZOS)) for p in patches[start:start+32]]
        outputs=roundtrip(inputs,codec,crf)
        for i,frame in enumerate(outputs):
            index=start+i;Image.fromarray(frame).save(folder/f'{index:04d}.png')
            records.append({'index':index,'codec':codec,'crf':crf,'targetSha256':hashlib.sha256(patches[index].tobytes()).hexdigest()})
        print(f'{args.split} codec cache {start+len(outputs)}/{len(patches)}',flush=True)
    (folder/'complete.json').write_text(json.dumps({'frames':records,'scope':'Actual intra-only codec roundtrip/YUV420 on image crops; not real video sequences'},indent=2))
if __name__=='__main__':main()
