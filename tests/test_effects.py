from max.driver import CPU
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import Graph, TensorType
import max_cv.operations as ops
from .common import generate_test_tensor, run_graph

def test_pixellate(session: InferenceSession) -> None:
    device = CPU()
    image_tensor = generate_test_tensor(device, dtype=DType.float32)
    graph = Graph(
        "pixellate",
        forward=lambda x: ops.pixellate(x, 10),
        input_types=[
            TensorType(image_tensor.dtype, shape=image_tensor.shape),
        ],
    )
    result = run_graph(graph, image_tensor, session)

    assert result.dtype == DType.float32
    assert result.shape == (4, 6, 3)
