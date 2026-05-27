package backend.cpu.kernels.nn;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import operations.nn.pool.maxPool2d;
import tensor.DataType;

public final class CpuMaxPool2dKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        CpuStorageView output = call.output();
        switch (output.dtype()) {
            case FLOAT64 -> Pool2dDirectBackend.maxForwardF64(
                    require(call.operation()),
                    call.inputs().get(0),
                    output,
                    call.workspace().requireIntWorkspace());
            case FLOAT32 -> Pool2dDirectBackend.maxForwardF32(
                    require(call.operation()),
                    call.inputs().get(0),
                    output,
                    call.workspace().requireIntWorkspace());
            case BFLOAT16 -> Pool2dDirectBackend.maxForwardBF16(
                    require(call.operation()),
                    call.inputs().get(0),
                    output,
                    call.workspace().requireIntWorkspace());
            case INT32, INT64, BOOL -> unsupported(output.dtype());
        }
        return CpuKernelResult.completed();
    }

    private static maxPool2d require(Operation op) {
        if (!(op instanceof maxPool2d pool)) {
            throw new IllegalArgumentException("CpuMaxPool2dKernel requires maxPool2d operation");
        }
        return pool;
    }

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuMaxPool2dKernel does not support " + dtype);
    }
}
