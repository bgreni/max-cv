from math import iota
from max.driver import cpu
from buffer.dimlist import DimList
from gpu.host import DeviceContext, DeviceBuffer
from tensor import ManagedTensorSlice, StaticTensorSpec, IOSpec, Input
from memory import AddressSpace
from memory import UnsafePointer
from random import rand
from utils import IndexList
from sys import size_of, has_accelerator, CompilationTarget

comptime dtype = DType.float32
comptime rank = 3
comptime tspec = _static_spec[dtype, rank](
    shape=DimList(600, 400, 3), strides=DimList(400, 3, 1)
)
comptime point_spec = _static_spec[dtype, 1](shape=DimList(2), strides=DimList(1))
comptime color_spec = _static_spec[dtype, 1](shape=DimList(3), strides=DimList(1))


fn gen_tensor[
    iospec: IOSpec
](ctx: DeviceContext) raises -> BenchTensor[iospec, tspec]:
    return BenchTensor[iospec, tspec](ctx).rand()


fn gen_color_tensor(
    ctx: DeviceContext,
) raises -> BenchTensor[Input, color_spec]:
    return BenchTensor[Input, color_spec](ctx).rand()


fn _static_spec[
    dtype: DType, rank: Int
](shape: DimList, strides: DimList, out spec: StaticTensorSpec[dtype, rank]):
    spec = type_of(spec)(
        shape=shape,
        alignment=size_of[dtype](),
        strides=strides,
        address_space=AddressSpace.GENERIC,
        exclusive=True,
        in_lambda=None,
        out_lambda=None,
        out_compute_lambda=None,
    )


@fieldwise_init
struct BenchTensor[
    dtype: DType,
    rank: Int, //,
    io_spec: IOSpec,
    static_spec: StaticTensorSpec[dtype, rank],
](Copyable, Movable):
    comptime tensor_type = ManagedTensorSlice[
        io_spec=Self.io_spec, static_spec=Self.static_spec
    ]
    comptime buffer_type = DeviceBuffer[Self.dtype]
    comptime ptr_type = UnsafePointer[Scalar[Self.dtype]]
    comptime size = Int(Self.static_spec.shape.product())

    var tensor: Self.tensor_type
    var buffer: Self.buffer_type

    fn __init__(out self, ctx: DeviceContext) raises:
        self.buffer = ctx.enqueue_create_buffer[Self.dtype](Self.size)
        ctx.synchronize()

        self.tensor = ManagedTensorSlice[
            io_spec=Self.io_spec, static_spec=Self.static_spec
        ](
            self.buffer.unsafe_ptr(),
            Self.static_spec.shape.into_index_list[Self.rank](),
            Self.static_spec.strides.into_index_list[Self.rank](),
        )

    fn unsafe_ptr(self) -> Self.ptr_type:
        return self.buffer.unsafe_ptr()

    fn rand(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            rand(host_buffer.unsafe_ptr(), Self.size)
            return self.copy()

    fn iota(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            iota(host_buffer.unsafe_ptr(), Self.size)
            return self.copy()
