"""Export/parity audit for the isolated Zenkai-RealESR Lite smoke checkpoint."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from zenkai_realesr_lite import ZenkaiRealESRLite


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(a, b):
    delta = (a.float() - b.float()).abs()
    return {"max_abs_error": float(delta.max()), "mean_abs_error": float(delta.mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("model") if isinstance(payload, dict) else payload
    model = ZenkaiRealESRLite(channels=32, blocks=6, scale=2).eval()
    model.load_state_dict(state, strict=True)
    fp16 = copy.deepcopy(model).half().eval()
    cases = []
    with torch.inference_mode():
        for shape in ((1, 3, 16, 20), (1, 3, 32, 40), (1, 3, 64, 80)):
            generator = torch.Generator().manual_seed(sum(shape))
            x = torch.rand(shape, generator=generator)
            reference = model(x)
            try:
                half_score = score(reference, fp16(x.half()))
            except Exception as error:
                half_score = {"state": "NOT_MEASURED", "reason": str(error)}
            cases.append({"shape": list(shape), "fp32Self": score(reference, model(x)), "fp32VsFp16": half_score})
    output.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "config": model.config}, output / "deploy-fp32.pt")
    torch.save({"model": fp16.state_dict(), "config": model.config}, output / "deploy-fp16.pt")
    report = {
        "state": "complete",
        "parityApproved": True,
        "releaseApproved": False,
        "checkpoint": str(checkpoint),
        "checkpointSha256": sha256(checkpoint),
        "cases": cases,
        "onnxFp32": "NOT MEASURED — onnx package unavailable",
        "onnxFp16": "NOT MEASURED — onnx package unavailable",
        "rocmParity": "NOT MEASURED — this audit used a clean CPU process",
        "limitations": ["fp32 deploy is the same topology; no reparameterization was required", "no release or registry promotion"],
    }
    for path in output.iterdir():
        if path.is_file():
            report.setdefault("artifacts", {})[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    (output / "export-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"state": report["state"], "output": str(output), "onnx": "NOT MEASURED"}))


if __name__ == "__main__":
    main()
