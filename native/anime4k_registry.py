"""Experimental ROCm Anime4K pipelines; bind all original artifacts before load."""
import hashlib,json
from pathlib import Path
from model_registry import ModelMetadata

SPECS=(('anime4k-s','Anime4K S',2),('anime4k-m','Anime4K M',2),
       ('realesr-anime4k-s-360','RealESR 360p + Anime4K S',4))

def register_anime4k(registry,root):
    root=Path(root)
    for name,label,scale in SPECS:
        path=root/'models/anime4k'/f'{name}.json'
        if not path.exists():continue
        spec=json.loads(path.read_text(encoding='utf-8'))
        expected=[f"models/anime4k/Anime4K_Upscale_CNN_x2_{'M' if name=='anime4k-m' else 'S'}.glsl"]
        if name=='realesr-anime4k-s-360':expected.append('models/realesr-animevideov3.pth')
        if spec['id']!=name or [a['path'] for a in spec['artifacts']]!=expected:
            raise ValueError('Unsupported Anime4K manifest')
        # Capture immutable identity from registration, then re-check at loading.
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        def validate(artifact,actual,spec=spec,digest=digest):
            if actual!=digest:raise ValueError('Anime4K artifact changed: manifest')
            for row in spec['artifacts']:
                source=root/row['path']
                if hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:
                    raise ValueError('Anime4K artifact changed: '+row['path'])
        def factory(spec=spec,name=name):
            from anime4k_model import Anime4K,RealESRAnime4K
            ak=Anime4K.from_file(root/spec['artifacts'][0]['path'])
            if name!='realesr-anime4k-s-360':return ak
            from user_models import load_user_model
            from realesr_optimized import RealESRAnimeVideo
            raw,_=load_user_model(root,'realesr-anime')
            return RealESRAnime4K(RealESRAnimeVideo(raw),ak)
        registry.register(ModelMetadata(name,label,'Anime4K','Shader CNN equation port',scale,
            'experimental',provider='bloc97 / AnimeScaler port',license='MIT' if scale==2 else 'See component licenses',
            code_license='MIT' if scale==2 else 'MIT + BSD-3-Clause',source_url='https://github.com/bloc97/Anime4K',
            weights_path=str(path),capture_graph=False),factory,artifact_validator=validate)
