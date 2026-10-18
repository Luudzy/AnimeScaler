"""Evaluate the isolated Zenkai-RealESR Lite checkpoint on the fixed real-video clips."""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "native"))

from evaluate_real_video import evaluate
from model_registry import build_registry
from prepare_video_degradation import file_hash
from zenkai_realesr_lite import ZenkaiRealESRLite, ZenkaiAnimeSRLite


def load_candidate(checkpoint: Path):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a model state")
    config = payload.get('config', {})
    model_class = ZenkaiAnimeSRLite if config.get('architecture') == 'AnimeSR-inspired multi-scale PReLU' else ZenkaiRealESRLite
    model = model_class(channels=32, blocks=6, scale=2)
    model.load_state_dict(state, strict=True)
    return model.eval(), payload, config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    torch.set_num_threads(2)
    candidate, payload, config = load_candidate(checkpoint)
    registry = build_registry(ROOT)
    legacy = registry.load("adore-lite-v3", device="cpu", precision="fp32")
    candidate_id = config.get('modelId') or config.get('run') or checkpoint.parent.name
    step = config.get('stepsExecuted') or config.get('steps')
    if step is None:
        name = checkpoint.name
        import re
        match = re.search(r'step-(\d+)', name)
        step = int(match.group(1)) if match else None
    artifacts = {
        "adore-lite-v3": legacy.artifact,
        candidate_id: {
            "path": str(checkpoint),
            "sha256": file_hash(checkpoint),
            "step": step,
            "fusion": "deploy_copy FP32",
        },
    }
    try:
        evaluate(
            args.manifest,
            {
                "adore-lite-v3": legacy.infer,
                candidate_id: candidate,
            },
            output,
            artifacts,
            save_all_frames=False,
        )
    finally:
        registry.unload("adore-lite-v3")


if __name__ == "__main__":
    main()
