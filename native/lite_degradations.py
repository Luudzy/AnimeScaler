"""Reproducible moderate image degradations; codec outputs are a separate cache."""
import io
import random
import numpy as np
from PIL import Image,ImageFilter

def degrade(high,rng,kind='train',scale=2):
    size=(round(high.width/scale),round(high.height/scale))
    if kind=='clean':return high.resize(size,Image.Resampling.LANCZOS)
    low=high
    if rng.random()<.25:low=low.filter(ImageFilter.GaussianBlur(rng.uniform(.1,.6 if kind!='heavy' else .9)))
    if rng.random()<.25:low=low.resize((round(high.width*.75),round(high.height*.75)),Image.Resampling.BICUBIC)
    low=low.resize(size,rng.choice([Image.Resampling.BICUBIC,Image.Resampling.LANCZOS,Image.Resampling.BILINEAR,Image.Resampling.BOX]))
    if rng.random()<.6:
        y,cb,cr=low.convert('YCbCr').split()
        chroma=(max(1,size[0]//2),max(1,size[1]//2))
        cb=cb.resize(chroma,Image.Resampling.BOX).resize(size,Image.Resampling.BILINEAR)
        cr=cr.resize(chroma,Image.Resampling.BOX).resize(size,Image.Resampling.BILINEAR)
        low=Image.merge('YCbCr',(y,cb,cr)).convert('RGB')
    if rng.random()<.15:low=low.filter(ImageFilter.UnsharpMask(radius=.7,percent=70,threshold=2))
    if rng.random()<(.8 if kind=='heavy' else .35):
        buf=io.BytesIO();quality=rng.randint(55,80) if kind=='heavy' else rng.randint(75,98)
        low.save(buf,format='JPEG',quality=quality,subsampling=2);buf.seek(0);low=Image.open(buf).convert('RGB')
    if rng.random()<.15:
        arr=np.asarray(low,dtype=np.float32)
        noise=np.random.default_rng(rng.randrange(2**32)).normal(0,rng.uniform(.2,1.2),arr.shape)
        low=Image.fromarray(np.uint8(np.clip(np.round(arr+noise),0,255)))
    return low
