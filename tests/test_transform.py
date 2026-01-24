import numpy as np
from max.driver import CPU, Tensor
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import TensorType, DeviceRef
import max_cv.operations as ops
from max_cv.operations import FlipCode
from .common import run_graph, make_graph


def test_flip(session: InferenceSession) -> None:
    device = CPU()

    width, height = 640, 400
    # Create 640x400 image with 3 channels
    c0 = np.arange(width * height).reshape(height, width, 1)
    c1 = c0 + 10
    c2 = c0 + 20
    input_data = np.concatenate([c0, c1, c2], axis=2).astype(np.float32)

    image_tensor = Tensor.from_numpy(input_data).to(device)

    graph_v = make_graph(
        "flip_v",
        forward=lambda x: ops.flip(device, x, FlipCode.VERTICAL),
        input_types=[
            TensorType(
                DType.float32,
                shape=image_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
    )
    result_v = run_graph(graph_v, image_tensor, session)
    output_v = result_v.to_numpy()

    expected_v = np.flip(input_data, axis=0)
    assert np.allclose(output_v, expected_v), (
        f"Vertical flip failed.\nExpected:\n{expected_v}\nGot:\n{output_v}"
    )

    graph_h = make_graph(
        "flip_h",
        forward=lambda x: ops.flip(device, x, FlipCode.HORIZONTAL),
        input_types=[
            TensorType(
                DType.float32,
                shape=image_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
    )
    result_h = run_graph(graph_h, image_tensor, session)
    output_h = result_h.to_numpy()

    expected_h = np.flip(input_data, axis=1)
    assert np.allclose(output_h, expected_h), (
        f"Horizontal flip failed.\nExpected:\n{expected_h}\nGot:\n{output_h}"
    )

    graph_b = make_graph(
        "flip_b",
        forward=lambda x: ops.flip(device, x, FlipCode.BOTH),
        input_types=[
            TensorType(
                DType.float32,
                shape=image_tensor.shape,
                device=DeviceRef.from_device(device),
            ),
        ],
    )
    result_b = run_graph(graph_b, image_tensor, session)
    output_b = result_b.to_numpy()

    expected_b = np.flip(np.flip(input_data, axis=0), axis=1)
    assert np.allclose(output_b, expected_b), (
        f"Both flip failed.\nExpected:\n{expected_b}\nGot:\n{output_b}"
    )
