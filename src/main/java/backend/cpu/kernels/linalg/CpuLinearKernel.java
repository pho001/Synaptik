package backend.cpu.kernels.linalg;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.linalg.linear;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

public final class CpuLinearKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        linear linear = require(call.operation());
        List<Tensor> inputs = call.inputTensors();
        Tensor node = call.outputTensor();
        CpuKernelContext context = call.context();
        switch (node.getDataType()) {
            case FLOAT64 -> LinearExecutor.forwardF64(linear, inputs.get(0), inputs.get(1),
                    linear.hasBias() ? inputs.get(2) : null, node, context);
            case FLOAT32 -> LinearExecutor.forwardF32(linear, inputs.get(0), inputs.get(1),
                    linear.hasBias() ? inputs.get(2) : null, node, context);
            case BFLOAT16 -> LinearExecutor.forwardBF16(linear, inputs.get(0), inputs.get(1),
                    linear.hasBias() ? inputs.get(2) : null, node, context);
            case INT32, INT64, BOOL -> unsupported(node.getDataType());
        }
        return CpuKernelResult.completed();
    }

    private static linear require(Operation op) {
        if (!(op instanceof linear linear)) {
            throw new IllegalArgumentException("CpuLinearKernel requires linear operation");
        }
        return linear;
    }

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuLinearKernel does not support " + dtype);
    }
}
