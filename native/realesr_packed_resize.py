"""Experimental packed PixelShuffle4 + residual + bilinear resize, no weight changes."""
import torch
import triton
import triton.language as tl


@triton.jit
def _sample(P, X, b, c, yy, xx, H: tl.constexpr, W: tl.constexpr, mask):
    yy = tl.minimum(tl.maximum(yy, 0), H * 4 - 1)
    xx = tl.minimum(tl.maximum(xx, 0), W * 4 - 1)
    cell = (yy // 4) * W + xx // 4
    phase = (yy % 4) * 4 + xx % 4
    p = tl.load(P + ((b * 3 + c) * 16 + phase) * H * W + cell, mask, 0)
    x = tl.load(X + (b * 3 + c) * H * W + cell, mask, 0)
    # Match the FP16 residual addition before interpolation, including rounding.
    return (p.to(tl.float32) + x.to(tl.float32)).to(P.dtype.element_ty).to(tl.float32)


@triton.jit
def _resize(P, X, O, H: tl.constexpr, W: tl.constexpr,
            OH: tl.constexpr, OW: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    mask = i < N * 3 * OH * OW
    xx = i % OW
    yy = (i // OW) % OH
    c = (i // (OH * OW)) % 3
    b = i // (3 * OH * OW)
    sy = tl.maximum((yy.to(tl.float32) + 0.5) * (4.0 * H / OH) - 0.5, 0.0)
    sx = tl.maximum((xx.to(tl.float32) + 0.5) * (4.0 * W / OW) - 0.5, 0.0)
    y0 = sy.to(tl.int32); x0 = sx.to(tl.int32)
    fy = sy - y0; fx = sx - x0
    a = _sample(P, X, b, c, y0, x0, H, W, mask)
    d = _sample(P, X, b, c, y0, x0+1, H, W, mask)
    e = _sample(P, X, b, c, y0+1, x0, H, W, mask)
    f = _sample(P, X, b, c, y0+1, x0+1, H, W, mask)
    value = (1-fy)*((1-fx)*a+fx*d) + fy*((1-fx)*e+fx*f)
    tl.store(O+i, value, mask)


def packed_resize(packed, source, size):
    if not (source.is_cuda and packed.is_cuda and source.is_contiguous() and packed.is_contiguous()):
        raise ValueError('Packed resize requires contiguous GPU tensors')
    n, c, h, w = source.shape
    if c != 3 or packed.shape != (n, 48, h, w) or source.dtype != packed.dtype:
        raise ValueError('Invalid PixelShuffle4 input')
    oh, ow = size
    if min(oh, ow) < 1:
        raise ValueError('Invalid output dimensions')
    output = source.new_empty((n, 3, oh, ow))
    _resize[(triton.cdiv(output.numel(), 256),)](packed, source, output, h, w, oh, ow, n, 256,
                                               enable_fp_fusion=False)
    return output
