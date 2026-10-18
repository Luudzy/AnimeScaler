"""Run the fixed image/codec validation without optional perceptual packages."""
import argparse
import json
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))
from model_registry import build_registry
from prepare_video_degradation import file_hash
from zenkai_evaluation import ZenkaiFixedEvaluator
from zenkai_realesr_lite import ZenkaiRealESRLite, ZenkaiAnimeSRLite


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=ROOT/'native/training-data/zenkai/validation-fixed.json')
    parser.add_argument('--codec', type=Path, default=ROOT/'native/training-data/zenkai/codec-validation-fixed/complete.json')
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = payload.get('model')
    config = payload.get('config', {})
    candidate = (ZenkaiAnimeSRLite if config.get('architecture') == 'AnimeSR-inspired multi-scale PReLU' else ZenkaiRealESRLite)(channels=32, blocks=6, scale=2).eval()
    candidate.load_state_dict(state, strict=True)
    registry = build_registry(ROOT)
    legacy = registry.load('adore-lite-v3', device='cpu', precision='fp32')
    evaluator = ZenkaiFixedEvaluator(args.manifest, per_case=32)
    evaluator.attach_codec(args.codec)
    try:
        candidate_id = 'zenkai-animesr-lite-v1-multiscale-500' if config.get('architecture') == 'AnimeSR-inspired multi-scale PReLU' else 'zenkai-realesr-lite-v1-smoke-500-retry2'
        result = evaluator.evaluate({'adore-lite-v3': legacy.infer, candidate_id: candidate}, device='cpu', perceptual=None)
    finally:
        registry.unload('adore-lite-v3')
    result.update({'state':'complete', 'releaseApproved':False, 'perceptual':'NOT MEASURED — optional LPIPS/MS-SSIM packages unavailable',
                   'candidateCheckpoint':str(args.checkpoint.resolve()), 'candidateCheckpointSha256':file_hash(args.checkpoint)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'state':'complete','output':str(args.output)}))


if __name__ == '__main__': main()
