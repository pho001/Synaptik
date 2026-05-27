package backend.cpu.kernels.index;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.index.gatherNd;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherNdKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        gatherNd gatherOp = requireOp(call.operation());
        Tensor[] pair = requirePair(call.operation(), call.inputTensors());
        if (call.inputs().size() != 2) {
            throw new IllegalArgumentException("gatherNd expects exactly two input storage views.");
        }
        Tensor node = call.outputTensor();
        switch (node.getDataType()) {
            case FLOAT64 -> GatherNdLoops.gatherNdF64(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getBatchDims());
            case FLOAT32 -> GatherNdLoops.gatherNdF32(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getBatchDims());
            case BFLOAT16 -> GatherNdLoops.gatherNdBF16(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getBatchDims());
            case BOOL -> GatherNdLoops.gatherNdBOOL(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getBatchDims());
            case INT32 -> GatherNdLoops.gatherNdI32(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getBatchDims());
            case INT64 -> GatherNdLoops.gatherNdI64(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), gatherOp.getBatchDims());
        }
        return CpuKernelResult.completed();
    }

    private static gatherNd requireOp(Operation op) {
        if (!(op instanceof gatherNd gatherOp)) {
            throw new IllegalArgumentException("CpuGatherNdKernel requires gatherNd operation.");
        }
        return gatherOp;
    }

    private static Tensor[] requirePair(Operation op, List<Tensor> inputs) {
        requireOp(op);
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("gatherNd expects exactly two inputs.");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
