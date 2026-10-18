"""Fail closed when the recorded real-video screening protocol has changed."""
import argparse
import hashlib
import json
from pathlib import Path


def validate_protocol(report):
    rows=report.get('sequences',[])
    if report.get('state')!='complete' or len(rows)!=3:
        raise ValueError('Three complete sequences required')
    if sum(row.get('frameCount',0) for row in rows)!=144:
        raise ValueError('Frozen protocol must contain144 frames')
    paths=[Path(row['manifest']).resolve() for row in rows]
    if len(set(paths))!=3:raise ValueError('Distinct manifests required')
    for row,path in zip(rows,paths):
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=row['manifestSha256']:
            raise ValueError(f'Frozen manifest hash changed:{path}')
        if json.loads(raw.decode('utf-8-sig')).get('frameCount')!=row['frameCount']:
            raise ValueError('Recorded/current frame counts differ')
    return paths


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    paths=validate_protocol(json.loads(args.report.read_text(encoding='utf-8-sig')))
    print(json.dumps({'state':'complete','frameCount':144,'manifests':list(map(str,paths))}))


if __name__=='__main__':main()
