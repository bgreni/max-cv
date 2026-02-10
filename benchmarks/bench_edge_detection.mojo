from benchmark import ThroughputMeasure, BenchId, BenchMetric, Bench, Bencher
from operations_mojo import SobelEdgeDetection
from .common import *
from tensor import (
    Input,
    Output,
)


fn run_edge_detection_benchmarks(mut bench: Bench) raises:
    sobel(bench)


fn sobel(mut bench: Bench) raises:
    var cpu = DeviceContext(api="cpu")
    var intensor = BenchTensor[Input, tspec_lum](cpu).rand()
    var outtensor = BenchTensor[Output, tspec_lum](cpu).rand()

    var els = intensor.size
    var elements = [ThroughputMeasure(BenchMetric.elements, els)]

    @parameter
    fn bench_cpu(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn run() raises:
            SobelEdgeDetection.execute["cpu"](
                outtensor.tensor, 0.5, intensor.tensor, cpu
            )

        b.iter[run]()

    bench.bench_function[bench_cpu](BenchId("sobel", "cpu"), elements)

    @parameter
    if has_accelerator():
        var gpu = DeviceContext()
        var gpu_intensor = BenchTensor[Input, tspec_lum](gpu).rand()
        var gpu_outtensor = BenchTensor[Output, tspec_lum](gpu).rand()

        @parameter
        fn bench_gpu(mut b: Bencher) raises:
            @parameter
            @always_inline
            fn run() raises:
                SobelEdgeDetection.execute["gpu"](
                    gpu_outtensor.tensor, 0.5, gpu_intensor.tensor, gpu
                )
                gpu.synchronize()

            b.iter[run]()

        bench.bench_function[bench_gpu](BenchId("sobel", "gpu"), elements)
