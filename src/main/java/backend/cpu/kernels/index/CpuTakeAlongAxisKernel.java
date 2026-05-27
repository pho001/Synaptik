package backend.cpu.kernels.index;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.index.takeAlongAxis;
import tensor.Tensor;

import java.util.List;

public final class CpuTakeAlongAxisKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        if (!(call.operation() instanceof takeAlongAxis gatherOp)) {
            throw new IllegalArgumentException("CpuTakeAlongAxisKernel requires takeAlongAxis operation");
        }
        Tensor[] pair = requirePair(call.inputTensors());
        if (call.inputs().size() != 2) {
            throw new IllegalArgumentException("takeAlongAxis expects exactly two input storage views");
        }
        Tensor node = call.outputTensor();
        switch (node.getDataType()) {
            case FLOAT64 -> TakeAlongAxisLoops.takeAlongAxisF64(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getDimension());
            case FLOAT32 -> TakeAlongAxisLoops.takeAlongAxisF32(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getDimension());
            case BFLOAT16 -> TakeAlongAxisLoops.takeAlongAxisBF16(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getDimension());
            case BOOL -> TakeAlongAxisLoops.takeAlongAxisBOOL(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getDimension());
            case INT32 -> TakeAlongAxisLoops.takeAlongAxisI32(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getDimension());
            case INT64 -> TakeAlongAxisLoops.takeAlongAxisI64(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getDimension());
        }
        return CpuKernelResult.completed();
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("takeAlongAxis expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
