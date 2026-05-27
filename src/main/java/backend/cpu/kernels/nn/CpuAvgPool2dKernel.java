package backend.cpu.kernels.nn;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import operations.nn.pool.avgPool2d;
import tensor.DataType;

public final class CpuAvgPool2dKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        CpuStorageView output = call.output();
        switch (output.dtype()) {
            case FLOAT64 -> Pool2dDirectBackend.avgForwardF64(
                    require(call.operation()), call.inputs().get(0), output);
            case FLOAT32 -> Pool2dDirectBackend.avgForwardF32(
                    require(call.operation()), call.inputs().get(0), output);
            case BFLOAT16 -> Pool2dDirectBackend.avgForwardBF16(
                    require(call.operation()), call.inputs().get(0), output);
            case INT32, INT64, BOOL -> unsupported(output.dtype());
        }
        return CpuKernelResult.completed();
    }

    private static avgPool2d require(Operation op) {
        if (!(op instanceof avgPool2d pool)) {
            throw new IllegalArgumentException("CpuAvgPool2dKernel requires avgPool2d operation");
        }
        return pool;
    }

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuAvgPool2dKernel does not support " + dtype);
    }
}
