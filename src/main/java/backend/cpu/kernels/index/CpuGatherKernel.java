package backend.cpu.kernels.index;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.index.gather;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        if (!(call.operation() instanceof gather gatherOp)) {
            throw new IllegalArgumentException("CpuGatherKernel requires gather operation");
        }
        Tensor[] pair = requirePair(call.inputTensors());
        if (call.inputs().size() != 2) {
            throw new IllegalArgumentException("Gather expects exactly two input storage views");
        }
        Tensor node = call.outputTensor();
        switch (node.getDataType()) {
            case FLOAT64 -> GatherLoops.gatherF64(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getDimension());
            case FLOAT32 -> GatherLoops.gatherF32(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getDimension());
            case BFLOAT16 -> GatherLoops.gatherBF16(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getDimension());
            case BOOL -> GatherLoops.gatherBOOL(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getDimension());
            case INT32 -> GatherLoops.gatherI32(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getDimension());
            case INT64 -> GatherLoops.gatherI64(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getDimension());
        }
        return CpuKernelResult.completed();
    }

    private static Tensor[] requirePair(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("Gather expects exactly two inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
