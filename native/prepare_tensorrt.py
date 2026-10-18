"""Compile fixed input profiles before the viewer connects (builds can take minutes)."""
from pathlib import Path
import os
import sys
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parent))
from tensorrt_profiles import DEFAULT_INPUT_SHAPES,parse_custom_shape
os.environ['ANIMESCALER_BACKEND']='tensorrt'
os.environ['ANIMESCALER_TRT_BUILD']='1'
from server import Renderer

parser=argparse.ArgumentParser()
parser.add_argument('--shape',action='append',help='Extra input shape WIDTHxHEIGHT, e.g. 960x536')
args=parser.parse_args()
shapes=list(DEFAULT_INPUT_SHAPES)
for value in args.shape or []:
    try:
        shape=parse_custom_shape(value)
    except ValueError as error:parser.error(str(error))
    if shape not in shapes:shapes.append(shape)
renderer=Renderer()
for name in renderer.info['models']:
    renderer.select_model(name)
    for width,height in shapes:
        renderer.model.prepare((1,3,height,width))
print('TensorRT: perfis 16:9 preparados. Outras proporções exigem preparação específica.')
