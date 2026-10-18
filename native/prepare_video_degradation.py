"""CPU codec degradation of an explicitly supplied local, constant-rate video."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import subprocess
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
FFMPEG=ROOT/'native/.tools/imageio_ffmpeg/binaries/ffmpeg-win-x86_64-v7.1.exe'
ENCODERS={'h264':'libx264','h265':'libx265','av1':'libaom-av1'}
CONTAINERS={'.mp4','.mkv','.mov','.avi','.webm','.m4v','.ts','.mts','.m2ts'}
DEMUXERS='mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpegts'


def validate_options(codec,crf,gop,max_frames):
    if codec not in ENCODERS:raise ValueError('Unsupported codec')
    if not 0<=crf<=(63 if codec=='av1' else 51):raise ValueError('CRF outside codec range')
    if gop<1 or max_frames<2:raise ValueError('GOP must be positive and at least two frames are required')


def run(command):
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=600)
    if result.returncode:raise RuntimeError('FFmpeg failed: '+result.stderr[-4000:])
    return result.stderr


def encode_command(hr,output,codec,crf,gop,fps,count):
    flags=(['-preset','veryfast'] if codec=='h264' else
           ['-preset','ultrafast','-x265-params','pools=2:frame-threads=1:log-level=error'] if codec=='h265' else
           ['-cpu-used','8','-row-mt','1','-b:v','0'])
    return [str(FFMPEG),'-nostdin','-v','error','-threads','2','-framerate',fps,
            '-start_number','0','-i',str(hr/'%08d.png'),'-frames:v',str(count),'-an',
            '-vf','scale=iw/2:ih/2:flags=lanczos','-c:v',ENCODERS[codec],*flags,
            '-threads','2','-crf',str(crf),'-pix_fmt','yuv420p','-g',str(gop),str(output)]


def file_hash(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def frame_timing(log, count, expected_fps=None):
    """Read exact PTS ticks; showinfo's rounded pts_time loses long offsets."""
    configs=re.findall(r'config in time_base:\s*(\d+/\d+),\s*frame_rate:\s*(\d+/\d+)',log)
    if not configs:raise ValueError('Frame timing unavailable')
    timebase=Fraction(configs[0][0]);fps=Fraction(configs[0][1])
    if timebase<=0 or fps<=0:raise ValueError('Source frame rate unavailable')
    if any((Fraction(tb),Fraction(rate))!=(timebase,fps) for tb,rate in configs):
        raise ValueError('Variable frame timing configuration is unsupported')
    if expected_fps is not None and fps!=Fraction(expected_fps):
        raise ValueError('Codec roundtrip changed frame rate')
    matches=re.findall(r'\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\S+[^\r\n]*?\biskey:(\d)\s+type:([A-Z?])',log)
    rows=[{'index':int(n),'pts':int(pts),'keyframe':bool(int(key)),'frameType':kind}
          for n,pts,key,kind in matches[:count]]
    if len(rows)!=count or [row['index'] for row in rows]!=list(range(count)):
        raise ValueError('Decoded frame timestamps missing; alignment cannot be established')
    timestamps=[row['pts']*timebase for row in rows]
    interval=1/fps
    tolerance=max(timebase*Fraction(11,10),interval*Fraction(2,1000))
    if (any(b<=a or abs(b-a-interval)>tolerance for a,b in zip(timestamps,timestamps[1:]))
            or any(abs(value-timestamps[0]-index*interval)>tolerance
                   for index,value in enumerate(timestamps))):
        raise ValueError('Variable or discontinuous frame rate is unsupported; no silent frame duplication or dropping')
    return timebase,f'{fps.numerator}/{fps.denominator}',rows


def prepare_video(source,output,codec='h264',crf=24,gop=48,max_frames=240,synthetic=False):
    source=Path(source).resolve();output=Path(output).resolve()
    if not source.is_file() or source.suffix.lower() not in CONTAINERS:
        raise ValueError('Input must be one explicit local video container; image directories and playlists are not accepted')
    validate_options(codec,crf,gop,max_frames)
    if not output.is_relative_to(ROOT):raise ValueError('Output must stay inside the AnimeScaler workspace')
    if output.exists():raise ValueError('Output already exists; choose a new directory to avoid mixing sequences')
    if not FFMPEG.is_file():raise RuntimeError('Bundled ffmpeg is unavailable; no download will be attempted')
    source_hash=file_hash(source)
    output.mkdir(parents=True);hr=output/'hr';lr=output/'lr';hr.mkdir();lr.mkdir()
    # Software decoding only. No autorotation: dimensions and crop coordinates
    # refer to the source pixel raster, rather than display metadata.
    extraction=[str(FFMPEG),'-nostdin','-hide_banner','-v','info','-xerror','-copyts','-threads','2',
                '-protocol_whitelist','file,pipe','-format_whitelist',DEMUXERS,
                '-noautorotate','-i',str(source),'-map','0:v:0',
                '-an','-sn','-dn','-vf','showinfo','-frames:v',str(max_frames),
                '-fps_mode','passthrough','-threads','2','-start_number','0',str(hr/'%08d.png')]
    log=run(extraction)
    paths=sorted(hr.glob('*.png'))
    if len(paths)<2:raise ValueError('Input did not decode to a video sequence of at least two frames')
    timebase,fps,source_frames=frame_timing(log,len(paths))
    with Image.open(paths[0]) as first:original_size=first.size
    width,height=(v//4*4 for v in original_size)
    if min(width,height)<4:raise ValueError('Source is too small for 2x yuv420p degradation')
    # Identical top-left crop for every HR frame guarantees even LR dimensions.
    for path in paths:
        with Image.open(path) as image:
            if image.size!=original_size:raise ValueError('Variable frame dimensions are unsupported')
            pixels=image.convert('RGB').crop((0,0,width,height))
        pixels.save(path)
    encoded=output/'degraded.mkv'
    command=encode_command(hr,encoded,codec,crf,gop,fps,len(paths));run(command)
    decode=[str(FFMPEG),'-nostdin','-hide_banner','-v','info','-xerror','-copyts',
            '-threads','2','-i',str(encoded),'-map','0:v:0','-an','-vf','showinfo',
            '-fps_mode','passthrough','-threads','2','-start_number','0',str(lr/'%08d.png')]
    decoded_log=run(decode)
    low_paths=sorted(lr.glob('*.png'))
    if len(low_paths)!=len(paths):raise ValueError('Codec roundtrip changed frame count')
    encoded_timebase,encoded_fps,encoded_frames=frame_timing(decoded_log,len(low_paths),fps)
    tolerance=max(timebase,encoded_timebase)*Fraction(11,10)
    source_start=source_frames[0]['pts']*timebase
    encoded_start=encoded_frames[0]['pts']*encoded_timebase
    if any(abs((high['pts']*timebase-source_start)-(low['pts']*encoded_timebase-encoded_start))>tolerance
           for high,low in zip(source_frames,encoded_frames)):
        raise ValueError('Codec timestamps do not align with source presentation order')
    keyframes=[row['index'] for row in encoded_frames if row['keyframe']]
    if not keyframes or keyframes[0]!=0:raise ValueError('Encoded sequence does not start with a keyframe')
    max_observed_gop=max(b-a for a,b in zip(keyframes,keyframes[1:]+[len(paths)]))
    if max_observed_gop>gop:raise ValueError('Encoded keyframe interval exceeds requested GOP')
    frames=[]
    for index,(high,low) in enumerate(zip(paths,low_paths)):
        with Image.open(low) as image:
            if image.size!=(width//2,height//2):raise ValueError('LR dimensions do not align with HR')
        frames.append({'index':index,'sourcePts':source_frames[index]['pts'],
                       'sourcePtsSeconds':float(source_frames[index]['pts']*timebase),
                       'encodedPts':encoded_frames[index]['pts'],
                       'encodedPtsSeconds':float(encoded_frames[index]['pts']*encoded_timebase),
                       'encodedKeyframe':encoded_frames[index]['keyframe'],
                       'encodedFrameType':encoded_frames[index]['frameType'],
                       'hr':high.relative_to(output).as_posix(),'lr':low.relative_to(output).as_posix(),
                       'hrSha256':file_hash(high),'lrSha256':file_hash(low)})
    if file_hash(source)!=source_hash:raise ValueError('Source changed during preparation')
    report={'schemaVersion':2,'scope':'Synthetic integration fixture; NOT real dataset validation' if synthetic else 'Explicit local video sequence; codec degradation only; no model quality validation',
            'realVideoValidation':False,'syntheticSource':synthetic,'source':str(source),'sourceSha256':source_hash,
            'fps':fps,'sourceTimeBase':str(timebase),'frameCount':len(frames),'maxFrames':max_frames,
            'encodedFps':encoded_fps,'encodedTimeBase':str(encoded_timebase),
            'timestampPolicy':'Original source PTS preserved; encoded sequence starts at zero; presentation offsets verified',
            'selection':'First decoded frames in presentation order; bounded by maxFrames',
            'sourceSize':list(original_size),'hrSize':[width,height],'lrSize':[width//2,height//2],
            'crop':[0,0,width,height],'scale':2,'rotation':'Ignored; source pixel raster preserved',
            'codec':{'name':codec,'encoder':ENCODERS[codec],'crf':crf,'gop':gop,'pixelFormat':'yuv420p',
                     'resize':'lanczos','container':'matroska','intraOnlyRequested':gop==1,
                     'keyframeIndices':keyframes,'maxObservedGopFrames':max_observed_gop,
                     'interFrameObserved':any(not row['keyframe'] and row['frameType'] in ('P','B') for row in encoded_frames)},
            'encodedSha256':file_hash(encoded),'ffmpegVersion':subprocess.run([str(FFMPEG),'-version'],capture_output=True,text=True,check=True).stdout.splitlines()[0],
            'commands':{'extract':extraction,'encode':command,'decode':decode},'frames':frames}
    (output/'complete.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--codec',choices=ENCODERS,default='h264');parser.add_argument('--crf',type=int,default=24)
    parser.add_argument('--gop',type=int,default=48);parser.add_argument('--max-frames',type=int,default=240)
    parser.add_argument('--synthetic-source',action='store_true',help='Explicitly label synthetic fixtures; never counts as real validation')
    args=parser.parse_args()
    report=prepare_video(args.input,args.output,args.codec,args.crf,args.gop,args.max_frames,args.synthetic_source)
    print(json.dumps({'frameCount':report['frameCount'],'scope':report['scope'],'output':str(args.output)}))


if __name__=='__main__':main()
