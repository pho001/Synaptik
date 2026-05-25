package backend.cpu.kernels;

import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public abstract class TypedCpuKernel implements CpuKernel {
    @Override
    public final CpuKernelResult execute(CpuKernelCall call) {
        Operation op = call.operation();
        List<Tensor> inputs = call.inputTensors();
        Tensor output = call.outputTensor();
        CpuKernelContext context = call.context();
        switch (output.getDataType()) {
            case FLOAT64 -> forwardF64(op, inputs, output, context);
            case FLOAT32 -> forwardF32(op, inputs, output, context);
            case BFLOAT16 -> forwardBF16(op, inputs, output, context);
            case INT32 -> forwardI32(op, inputs, output, context);
            case INT64 -> forwardI64(op, inputs, output, context);
            case BOOL -> forwardBOOL(op, inputs, output, context);
        }
        return CpuKernelResult.completed();
    }

    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support FLOAT64");
    }

    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support FLOAT32");
    }

    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support BFLOAT16");
    }

    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support BOOL");
    }

    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support INT32");
    }

    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        throw new UnsupportedOperationException(getClass().getSimpleName() + " does not support INT64");
    }
}
