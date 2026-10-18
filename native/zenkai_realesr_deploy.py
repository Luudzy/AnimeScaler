"""Experimental clean real-pair Zenkai deployment, kept separate from Zenkai 1.0."""
import hashlib
from pathlib import Path
import torch

MODEL_ID = 'zenkai-realesr-clean5k'
PACKAGE = Path('models/zenkai/experiments/zenkai-lite-real-clean-5k/export')
WEIGHTS_SHA256 = 'f852bbc4affaa1da4337e31c62ea04e05c7a14f7c710ce32b3e89b07cef534f2'

ANIMEVIDEO_MODEL_ID = 'zenkai-animevideo-30k'
ANIMEVIDEO_PACKAGE = Path('models/zenkai/experiments/zenkai-animevideo-lite-30k-retry/segment-29001-30000/export')
ANIMEVIDEO_WEIGHTS_SHA256 = 'd4c7fdba9a4bbe582bc50e39ee7289b93f1d55ab0e3f39c2546752760298f875'

def register_zenkai_realesr(registry, root):
    from model_registry import ModelMetadata
    folder = Path(root).resolve() / PACKAGE
    weights = folder / 'deploy-fp32.pt'
    if not weights.exists():
        register_animevideo(registry, root)
        return None
    def validate(path, digest):
        if digest != WEIGHTS_SHA256:
            raise ValueError('Zenkai clean 5k artifact hash mismatch')
    def factory():
        from zenkai_realesr_lite import ZenkaiRealESRLite
        payload = torch.load(weights, map_location='cpu', weights_only=False)
        model = ZenkaiRealESRLite(channels=32, blocks=6, scale=2)
        model.load_state_dict(payload['model'], strict=True)
        return model
    metadata = ModelMetadata(
        MODEL_ID, 'Zenkai Real Clean 5k', 'Zenkai', 'SRVGG-inspired PReLU real-pair',
        2, 'experimental', provider='AnimeScaler real-video training',
        code_license='MIT-derived project code; see THIRD_PARTY_NOTICES.md',
        weights_path=str(weights), capture_graph=False,
    )
    registry.register(metadata, factory, artifact_validator=validate)

    register_animevideo(registry, root)
    return metadata


def register_animevideo(registry, root):
    from model_registry import ModelMetadata
    animevideo_weights = (Path(root).resolve() / ANIMEVIDEO_PACKAGE / 'deploy-fp16.pt')
    if animevideo_weights.exists():
        def validate_animevideo(path, digest):
            if digest != ANIMEVIDEO_WEIGHTS_SHA256:
                raise ValueError('Zenkai AnimeVideo 30k FP16 artifact hash mismatch')

        def animevideo_factory():
            from zenkai_realesr_lite import ZenkaiRealESRLite
            payload = torch.load(animevideo_weights, map_location='cpu', weights_only=False)
            model = ZenkaiRealESRLite(channels=32, blocks=6, scale=2)
            model.load_state_dict(payload['model'], strict=True)
            return model

        animevideo_metadata = ModelMetadata(
            ANIMEVIDEO_MODEL_ID, 'Zenkai AnimeVideo 30k FP16', 'Zenkai',
            'SRVGG-inspired PReLU real-pair', 2, 'experimental',
            provider='AnimeScaler real-video training (30k)',
            code_license='MIT-derived project code; see THIRD_PARTY_NOTICES.md',
            weights_path=str(animevideo_weights), capture_graph=False,
            # Live playback regressed after the isolated HIP graph probe.
            # Keep graphs disabled, including former environment opt-ins.
            capture_graph_max_pixels=0,
        )
        registry.register(animevideo_metadata, animevideo_factory,
                          artifact_validator=validate_animevideo)
        return animevideo_metadata
    return None
