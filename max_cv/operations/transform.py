from enum import IntEnum
from max.graph import ops, TensorType, TensorValue, DeviceRef
from max.driver import Device


class FlipCode(IntEnum):
    VERTICAL = 0
    HORIZONTAL = 1
    BOTH = -1


"""Geometric transformations."""


def flip(device: Device, image: TensorValue, flip_code: FlipCode | int) -> TensorValue:
    """Flips an image.

    Args:
        image: A value representing an incoming image in a graph.
        flip_code: 0 for vertical, 1 for horizontal, -1 for both.

    Returns:
        A value representing the flipped image.
    """
    assert flip_code in [FlipCode.VERTICAL, FlipCode.HORIZONTAL, FlipCode.BOTH], (
        f"Invalid flip code: {flip_code}"
    )
    dref = DeviceRef.from_device(device)
    return ops.custom(
        name="flip",
        device=dref,
        values=[
            image,
        ],
        out_types=[TensorType(dtype=image.dtype, shape=image.shape, device=dref)],
        parameters={"flip_code": int(flip_code)},
    )[0].tensor
