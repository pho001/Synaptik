package backend.cpu.kernels.index;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.index.scatterAdd;
import tensor.Tensor;

import java.util.List;

public final class CpuScatterAddKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        scatterAdd scatterAddOp = requireOp(call.operation());
        Tensor[] triple = requireTriple(call.inputTensors());
        if (call.inputs().size() != 3) {
            throw new IllegalArgumentException("scatterAdd expects exactly three input storage views.");
        }
        Tensor node = call.outputTensor();
        switch (node.getDataType()) {
            case FLOAT64 -> ScatterLoops.scatterAddF64(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterAddOp.getDimension());
            case FLOAT32 -> ScatterLoops.scatterAddF32(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterAddOp.getDimension());
            case BFLOAT16 -> ScatterLoops.scatterAddBF16(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterAddOp.getDimension());
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException(
                    "CpuScatterAddKernel does not support " + node.getDataType());
        }
        return CpuKernelResult.completed();
    }

    private static scatterAdd requireOp(Operation op) {
        if (!(op instanceof scatterAdd scatterAddOp)) {
            throw new IllegalArgumentException("CpuScatterAddKernel requires scatterAdd operation");
        }
        return scatterAddOp;
    }

    private static Tensor[] requireTriple(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("scatterAdd expects exactly three inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1), inputs.get(2)};
    }
}
