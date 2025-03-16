from math import iota
from max.driver import cpu
from buffer.dimlist import DimList
from gpu.host import DeviceContext, DeviceBuffer
from max.tensor import (
    ManagedTensorSlice,
    StaticTensorSpec,
    IOSpec,
)
from memory import AddressSpace
from memory import UnsafePointer
from random import rand
from utils import IndexList
from sys import sizeof

alias dtype = DType.float32
alias rank = 3
alias tspec = _static_spec[dtype, rank](shape=(600, 400, 3), strides=(400, 3, 1))

fn gen_tensor[iospec: IOSpec, target: StringLiteral="cpu"](ctx: DeviceContext) raises -> BenchTensor[iospec, tspec]:
    return BenchTensor[iospec, tspec](ctx).rand()

fn _static_spec[
    dtype: DType, rank: Int
](shape: DimList, strides: DimList, out spec: StaticTensorSpec[dtype, rank]):
    spec = __type_of(spec)(
        shape=shape,
        alignment=sizeof[dtype](),
        strides=strides,
        address_space=AddressSpace.GENERIC,
        exclusive=True,
        in_lambda=None,
        out_lambda=None,
    )

@value
struct BenchTensor[
    dtype: DType,
    rank: Int, //,
    io_spec: IOSpec,
    static_spec: StaticTensorSpec[dtype, rank],
]:
    alias tensor_type = ManagedTensorSlice[
        io_spec=io_spec, static_spec=static_spec
    ]
    alias buffer_type = DeviceBuffer[dtype]
    alias ptr_type = UnsafePointer[Scalar[dtype]]
    alias size = Int(static_spec.shape.product())

    var tensor: Self.tensor_type
    var buffer: Self.buffer_type

    fn __init__(out self, ctx: DeviceContext) raises:
        self.buffer = ctx.enqueue_create_buffer[dtype](Self.size)

        self.tensor = ManagedTensorSlice[
            io_spec=io_spec, static_spec=static_spec
        ](
            self.buffer.unsafe_ptr(),
            Self.static_spec.shape.into_index_list[rank](),
            Self.static_spec.strides.into_index_list[rank](),
        )

    fn unsafe_ptr(self) -> Self.ptr_type:
        return self.buffer.unsafe_ptr()

    fn rand(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            rand(host_buffer.unsafe_ptr(), Self.size)
            return self

    fn iota(self) raises -> Self:
        with self.buffer.map_to_host() as host_buffer:
            iota(host_buffer.unsafe_ptr(), Self.size)
            return self