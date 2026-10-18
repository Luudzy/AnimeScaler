"""Static 16:9 TensorRT shapes for the extension's preset/adaptive widths."""

DEFAULT_INPUT_SHAPES = (
    (1920,1080),(1280,720),(960,540),(640,360),
    (480,268),(320,180),(240,132),(160,88),
)


def parse_custom_shape(value):
    try:
        width,height=map(int,value.lower().split('x'))
    except (AttributeError,TypeError,ValueError):
        raise ValueError('Shape must be WIDTHxHEIGHT, multiples of four, within native input limits')
    if min(width,height)<40 or max(width,height)>1920 or width%4 or height%4 or width*height>1920*1080:
        raise ValueError('Shape must be WIDTHxHEIGHT, multiples of four, within native input limits')
    return width,height
