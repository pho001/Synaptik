package backend.cpu.kernels.index;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.index.scatterNd;
import tensor.Tensor;

import java.util.List;

public final class CpuScatterNdKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        scatterNd scatterOp = requireOp(call.operation());
        Tensor[] triple = requireTriple(call.inputTensors());
        if (call.inputs().size() != 3) {
            throw new IllegalArgumentException("scatterNd expects exactly three input storage views.");
        }
        Tensor node = call.outputTensor();
        switch (node.getDataType()) {
            case FLOAT64 -> ScatterNdLoops.scatterNdF64(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getReduction(), scatterOp.getBatchDims());
            case FLOAT32 -> ScatterNdLoops.scatterNdF32(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getReduction(), scatterOp.getBatchDims());
            case BFLOAT16 -> ScatterNdLoops.scatterNdBF16(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getReduction(), scatterOp.getBatchDims());
            case BOOL -> ScatterNdLoops.scatterNdBOOL(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getReduction(), scatterOp.getBatchDims());
            case INT32 -> ScatterNdLoops.scatterNdI32(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getReduction(), scatterOp.getBatchDims());
            case INT64 -> ScatterNdLoops.scatterNdI64(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getReduction(), scatterOp.getBatchDims());
        }
        return CpuKernelResult.completed();
    }

    private static scatterNd requireOp(Operation op) {
        if (!(op instanceof scatterNd scatterOp)) {
            throw new IllegalArgumentException("CpuScatterNdKernel requires scatterNd operation");
        }
        return scatterOp;
    }

    private static Tensor[] requireTriple(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("scatterNd expects exactly three inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1), inputs.get(2)};
    }
}
