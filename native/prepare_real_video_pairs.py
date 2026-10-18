"""Build a manifest of real anime HR frames paired with their 2x LR video frames."""
import argparse, ctypes, errno, hashlib, json, os, shutil, sys, tempfile
from pathlib import Path


def sha256(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): d.update(chunk)
    return d.hexdigest()


def relative_asset_path(path, output_root):
    try:
        return Path(os.path.relpath(Path(path).resolve(), Path(output_root).resolve())).as_posix()
    except ValueError as error:
        raise ValueError('Source media and output must share a volume for portable relative paths') from error


def partition_clip_rows(rows,holdout_every=4):
    if type(holdout_every) is not int or holdout_every<2:raise ValueError('Holdout interval must be at least two source clips')
    groups={}
    for row in rows:
        source=row.get('source')
        if not isinstance(source,str) or not source:raise ValueError('Missing source clip group')
        if 'split' in row:raise ValueError('Input rows must not be pre-split')
        groups.setdefault(source,[]).append(row)
    if len(groups)<2:raise ValueError('At least two source clips are required for a holdout split')
    train=[];holdout=[]
    for index,source in enumerate(sorted(groups)):
        target=holdout if index%holdout_every==0 else train
        split='holdout' if target is holdout else 'train'
        for row in sorted(groups[source],key=lambda item:(item.get('file',''),item['sha256'])):
            target.append({**row,'split':split,'splitGroup':source})
    if not train or not holdout:raise ValueError('Source clip split must contain train and holdout groups')
    return train,holdout


def _rename_directory_new(source, destination):
    """Atomically publish a directory without replacing any destination."""
    if sys.platform == 'win32':
        os.rename(source, destination)
        return
    if sys.platform.startswith('linux'):
        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, 'renameat2', None)
        if renameat2 is None:
            raise OSError(errno.ENOTSUP, 'renameat2 is unavailable; refusing unsafe publish')
        result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
        if result != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error), os.fspath(destination))
        return
    raise OSError(errno.ENOTSUP,
                  f'Atomic no-replace directory publish is unsupported on {sys.platform}')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--clips-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--holdout-every',type=int,default=4)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    rows=[]
    output_root=args.output.resolve()
    for clip in sorted(args.clips_root.glob('clip-*-360-to-720')):
        hr=clip/'hr'; lr=clip/'lr'
        for path in sorted(hr.glob('*.png')):
            low=lr/path.name
            if not low.is_file(): raise FileNotFoundError(low)
            row={'file':relative_asset_path(path,output_root),'pairedLr':relative_asset_path(low,output_root),
                         'sha256':sha256(path),'pairedLrSha256':sha256(low),'source':clip.name}
            rows.append(row)
    rows,holdout=partition_clip_rows(rows,args.holdout_every)
    if len(rows)<100 or not holdout: raise ValueError('Expected at least 100 training pairs and a holdout')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=f'.{args.output.name}.',suffix='.pending',dir=args.output.parent))
    try:
        manifest=staging/'manifest.jsonl'
        holdout_manifest=staging/'holdout.jsonl'
        metadata_path=staging/'metadata.json'
        manifest.write_text('\n'.join(json.dumps(row) for row in rows)+'\n',encoding='utf-8')
        holdout_manifest.write_text('\n'.join(json.dumps(row) for row in holdout)+'\n',encoding='utf-8')
        metadata_path.write_text(json.dumps({'state':'complete','trainPairs':len(rows),'holdoutPairs':len(holdout),'source':relative_asset_path(args.clips_root,output_root),'sourceSha256':sha256(manifest),'holdoutSha256':sha256(holdout_manifest),'splitStrategy':'whole source clips; frames from one clip never cross splits','holdoutEverySourceClips':args.holdout_every,'sourceGroups':len({row['source'] for row in rows+holdout}),'pathPolicy':'POSIX paths relative to the published dataset directory; source assets remain external.','limitations':['Source clip identity does not prove distinct production or scene; provide production metadata before claiming production-level independence.']},indent=2),encoding='utf-8')
        _rename_directory_new(staging,args.output)
    finally:
        if staging.exists():shutil.rmtree(staging)
    print(json.dumps({'state':'complete','trainPairs':len(rows),'holdoutPairs':len(holdout),'output':str(args.output)}))


if __name__=='__main__': main()
