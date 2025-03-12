import compiler
from utils.index import Index, IndexList
from max.tensor import foreach, OutputTensor, InputTensor
from runtime.asyncrt import DeviceContextPtr

@compiler.register("pixellate")
struct Pixellate:
    """Pixellates an image into small squares."""

    @staticmethod
    fn execute[
        target: StringLiteral,
    ](
        out: OutputTensor,
        pixel_width: Int32,
        image: InputTensor[type=out.type, rank=out.rank],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        @always_inline
        fn pixellate[
            width: Int
        ](idx: IndexList[image.rank]) -> SIMD[image.type, width]:
            var pixel_idx = idx
            pixel_idx[0] = (pixel_idx[0] // Int(pixel_width)) * Int(pixel_width)
            pixel_idx[1] = (pixel_idx[1] // Int(pixel_width)) * Int(pixel_width)
            return image.load[width](pixel_idx)

        foreach[pixellate, target=target](out, ctx)
