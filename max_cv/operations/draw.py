from max.dtype import DType
from max.graph import ops, TensorType, TensorValue
from .common import assert_rgb
import numpy as np
from typing import Optional

def draw_circle(
        image: TensorValue,
        radius: int,
        color: tuple,
        width: int,
        center: Optional[tuple] = None
    ) -> TensorValue:
    assert_rgb(image)
    c = center or [image.shape.static_dims[0]//2, image.shape.static_dims[1]//2]
    return ops.custom(
        name='draw_circle',
        values=[
            image,
            ops.constant(radius, dtype=DType.float32),
            ops.constant(np.array(color)/255.0, dtype=DType.float32),
            ops.constant(width, dtype=DType.float32),
            ops.constant(np.array(c), dtype=DType.float32),
        ],
        out_types=[TensorType(dtype=image.dtype, shape=image.shape)],
    )[0].tensor