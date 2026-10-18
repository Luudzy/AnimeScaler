"""ROCm/PyTorch port of supplied Anime4K Upscale CNN S/M shader equations.

Original GLSL files and MIT notices remain the source of weights. This port
uses clamp-edge sampling and bilinear MAIN sampling, with an RGB-shared luma
residual. Equation parity is tested independently; mpv GPU parity is unmeasured.
"""
import hashlib,re
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


def numbers(text,count):
    values=text.split(',')
    if len(values)!=count or any(not re.fullmatch(r'\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*',v) for v in values):
        raise ValueError('Unsupported shader numeric constant')
    result=torch.tensor([float(v) for v in values],dtype=torch.float32)
    if not torch.isfinite(result).all():raise ValueError('Nonfinite shader weight')
    return result


class ShaderConv(nn.Module):
    def __init__(self,sources,weight,bias):
        super().__init__();self.sources=sources
        self.register_buffer('weight',weight);self.register_buffer('bias',bias)
    def forward(self,features):
        operands=[]
        for texture,split,negative in self.sources:
            value=features[texture]
            if negative:value=-value
            operands.append(F.relu(value) if split else value)
        value=operands[0] if len(operands)==1 else torch.cat(operands,dim=1)
        if self.weight.shape[-1]==3:value=F.pad(value,(1,1,1,1),mode='replicate')
        return F.conv2d(value,self.weight,self.bias)


class Anime4K(nn.Module):
    @classmethod
    def from_file(cls,path):
        path=Path(path);model=cls.from_text(path.read_text(encoding='utf-8'))
        model.shader_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        return model
    @classmethod
    def from_text(cls,text):
        sections=text.split('//!DESC')[1:]
        if len(sections) not in (5,9) or 'Depth-to-Space' not in sections[-1]:
            raise ValueError('Only supplied Upscale CNN S/M shaders supported')
        tail=sections[-1]
        expected_tail="""vec4 hook() {
            vec2 f0 = fract(conv2d_last_tf_pos * conv2d_last_tf_size);
            ivec2 i0 = ivec2(f0 * vec2(2.0));
            float c0 = conv2d_last_tf_tex((vec2(0.5) - f0) * conv2d_last_tf_pt + conv2d_last_tf_pos)[i0.y * 2 + i0.x];
            float c1 = c0; float c2 = c1; float c3 = c2;
            return vec4(c0, c1, c2, c3) + MAIN_tex(MAIN_pos);
        }"""
        normalize=lambda v:re.sub(r'\s+','',v)
        if normalize(tail[tail.index('vec4 hook()'):])!=normalize(expected_tail):
            raise ValueError('Unsupported output phase equation')
        for axis in ('WIDTH','HEIGHT'):
            dim='w' if axis=='WIDTH' else 'h'
            if re.search(r'//!'+axis+r' ([^\n]+)',tail)[1].strip()!=f'conv2d_last_tf.{dim} 2 *':
                raise ValueError('Unsupported output dimensions')
        known={'MAIN'};layers=[];saves=[]
        for section in sections[:-1]:
            desc=re.match(r'[^\n]*-Conv-4x([13])x\1x\d+',section)
            if not desc:raise ValueError('Unsupported shader pass')
            kernel=int(desc[1]);saved=re.search(r'//!SAVE (\w+)',section)
            if not saved:raise ValueError('Missing output binding')
            macros={}
            for name,expr in re.findall(r'#define (go_\d+\(x_off, y_off\)|g_\d+) ([^\n]+)',section):
                key=name.split('(')[0]
                texture=re.search(r'(\w+)_tex(?:Off)?\(',expr)
                if not texture or texture[1] not in known:raise ValueError('Unknown texture')
                texture=texture[1]
                sample=f'{texture}_texOff(vec2(x_off, y_off))' if key.startswith('go_') else f'{texture}_tex({texture}_pos)'
                allowed={f'({sample})':(False,False),f'(max(({sample}), 0.0))':(True,False),f'(max(-({sample}), 0.0))':(True,True)}
                if expr.strip() not in allowed:raise ValueError('Unsupported operand expression')
                split,negative=allowed[expr.strip()]
                macros[key]=(texture,split,negative)
            if not macros:raise ValueError('No shader operands')
            first=next(iter(macros.values()))[0]
            for axis in ('WIDTH','HEIGHT'):
                dim='w' if axis=='WIDTH' else 'h'
                directive=re.search(r'//!'+axis+r' ([^\n]+)',section)
                if not directive or directive[1].strip()!=f'{first}.{dim}':
                    raise ValueError('Unsupported pass dimensions')
            if re.search(r'//!COMPONENTS ([^\n]+)',section)[1].strip()!='4':raise ValueError('Unsupported components')
            keys=list(macros);weight=torch.zeros(4,len(keys)*4,kernel,kernel)
            body=re.search(r'vec4 hook\(\)\s*\{(.*?)\}',section,re.S)
            if not body:raise ValueError('Missing hook')
            bias=None;seen=set();term_count=0
            for statement in body[1].split(';'):
                statement=statement.strip()
                if not statement or statement=='return result':continue
                term=re.fullmatch(r'(?:vec4 result =|result \+=)\s*mat4\(([^)]+)\)\s*\*\s*(go_\d+|g_\d+)(?:\(([^)]+)\))?',statement)
                if term:
                    key=term[2]
                    if key not in macros:raise ValueError('Unknown operand')
                    offsets=numbers(term[3],2) if term[3] else torch.zeros(2)
                    if not torch.equal(offsets,offsets.round()):raise ValueError('Fractional sample offset')
                    dx,dy=map(int,offsets)
                    radius=kernel//2
                    if abs(dx)>radius or abs(dy)>radius or (key,dx,dy) in seen:raise ValueError('Invalid sample offset')
                    seen.add((key,dx,dy));term_count+=1
                    # GLSL mat4 initializer is column-major.
                    matrix=numbers(term[1],16).reshape(4,4).T
                    i=keys.index(key)*4;weight[:,i:i+4,dy+radius,dx+radius]=matrix
                    continue
                bias_term=re.fullmatch(r'result \+= vec4\(([^)]+)\)',statement)
                if bias_term and bias is None:bias=numbers(bias_term[1],4);continue
                raise ValueError('Unsupported executable shader statement')
            if bias is None or term_count!=len(keys)*kernel*kernel:raise ValueError('Incomplete convolution')
            layers.append(ShaderConv([macros[k] for k in keys],weight,bias));saves.append(saved[1]);known.add(saved[1])
        return cls(layers,saves)
    def __init__(self,layers,saves):
        super().__init__();self.layers=nn.ModuleList(layers);self.saves=saves
        self.requires_grad_(False);self.eval()
    def forward(self,x):
        features={'MAIN':torch.cat((x,x.new_ones((x.shape[0],1,*x.shape[-2:]))),dim=1)}
        for layer,save in zip(self.layers,self.saves):features[save]=layer(features)
        residual=F.pixel_shuffle(features['conv2d_last_tf'],2)
        output=F.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False)
        return output.add_(residual)
    def forward_to(self,x,output_size):
        size=tuple(output_size);output=x
        # Match the original WHEN: both dimensions must exceed 1.2x.
        for _ in range(4):
            if min(size[0]/output.shape[-2],size[1]/output.shape[-1])<=1.2:break
            output=self(output)
        return output if tuple(output.shape[-2:])==size else F.interpolate(output,size=size,mode='bilinear',align_corners=False)


class RealESRAnime4K(nn.Module):
    """Explicit quality/performance tradeoff: cap RealESR input at 640x360."""
    def __init__(self,sr,anime4k,max_width=640,max_height=360):
        super().__init__();self.sr=sr;self.anime4k=anime4k
        self.max_width=max_width;self.max_height=max_height
        self.requires_grad_(False);self.eval()
    def forward_to(self,x,output_size):
        h,w=x.shape[-2:];factor=min(1,self.max_height/h,self.max_width/w)
        ih,iw=max(2,int(h*factor)//2*2),max(2,int(w*factor)//2*2)
        value=x if (ih,iw)==(h,w) else F.interpolate(x,size=(ih,iw),mode='bilinear',align_corners=False)
        self.last_sr_input=(ih,iw)
        output=self.sr.forward_to(value,(ih*2,iw*2))
        # Quantized browser-visible bounds are not inserted between models.
        return self.anime4k.forward_to(output,output_size)
    def forward(self,x):return self.forward_to(x,(x.shape[-2]*4,x.shape[-1]*4))
