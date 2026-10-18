"""Extract a larger real-anime LR/HR pair set from the supplied 4K episode."""
import argparse, hashlib, json, shutil, subprocess
from pathlib import Path

def sha256(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): d.update(chunk)
    return d.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--video',type=Path,required=True); p.add_argument('--ffmpeg',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--fps',type=float,default=.2); a=p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    hr=a.output/'hr'; lr=a.output/'lr'; hr.mkdir(parents=True); lr.mkdir()
    common=[str(a.ffmpeg),'-nostdin','-v','error','-threads','2','-i',str(a.video),'-map','0:v:0','-an','-sn','-dn']
    # Use separate invocations with explicit output patterns; filenames are aligned by index.
    subprocess.run(common+['-vf',f'fps={a.fps},scale=1280:720:flags=lanczos',str(hr/'%06d.png')],check=True)
    subprocess.run(common+['-vf',f'fps={a.fps},scale=640:360:flags=lanczos',str(lr/'%06d.png')],check=True)
    rows=[]
    for hp in sorted(hr.glob('*.png')):
        lp=lr/hp.name
        if not lp.is_file(): continue
        rows.append({'file':str(hp.resolve()),'pairedLr':str(lp.resolve()),'split':'train','sha256':sha256(hp),'pairedLrSha256':sha256(lp),'source':'full-episode'})
    if len(rows)<300: raise ValueError('Too few extracted pairs')
    a.output.joinpath('manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in rows)+'\n',encoding='utf-8')
    a.output.joinpath('metadata.json').write_text(json.dumps({'state':'complete','pairs':len(rows),'video':str(a.video.resolve()),'videoSha256':sha256(a.video),'fps':a.fps},indent=2),encoding='utf-8')
    print(json.dumps({'state':'complete','pairs':len(rows),'output':str(a.output)}))
if __name__=='__main__': main()
