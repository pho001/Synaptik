package backend.cpu.kernels.index;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.index.scatterAxisAdd;
import tensor.Tensor;

import java.util.List;

public final class CpuScatterAxisAddKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        scatterAxisAdd scatterOp = requireOp(call.operation());
        Tensor[] triple = requireTriple(call.operation(), call.inputTensors());
        if (call.inputs().size() != 3) {
            throw new IllegalArgumentException("scatterAxisAdd expects exactly three input storage views.");
        }
        Tensor node = call.outputTensor();
        switch (node.getDataType()) {
            case FLOAT64 -> GatherAxisLoops.scatterAxisAddF64(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getAxis());
            case FLOAT32 -> GatherAxisLoops.scatterAxisAddF32(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getAxis());
            case BFLOAT16 -> GatherAxisLoops.scatterAxisAddBF16(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getAxis());
            case BOOL, INT32, INT64 -> throw new UnsupportedOperationException(
                    "CpuScatterAxisAddKernel does not support " + node.getDataType());
        }
        return CpuKernelResult.completed();
    }

    private static scatterAxisAdd requireOp(Operation op) {
        if (!(op instanceof scatterAxisAdd scatterOp)) {
            throw new IllegalArgumentException("CpuScatterAxisAddKernel requires scatterAxisAdd operation.");
        }
        return scatterOp;
    }

    private static Tensor[] requireTriple(Operation op, List<Tensor> inputs) {
        requireOp(op);
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("scatterAxisAdd expects exactly three inputs.");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1), inputs.get(2)};
    }
}
