package backend.cpu.kernels.index;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.index.scatterElements;
import tensor.Tensor;

import java.util.List;

public final class CpuScatterElementsKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        scatterElements scatterOp = requireOp(call.operation());
        Tensor[] triple = requireTriple(call.inputTensors());
        if (call.inputs().size() != 3) {
            throw new IllegalArgumentException("scatterElements expects exactly three input storage views.");
        }
        Tensor node = call.outputTensor();
        switch (node.getDataType()) {
            case FLOAT64 -> ScatterElementsLoops.scatterElementsF64(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getAxis(), scatterOp.getReduction());
            case FLOAT32 -> ScatterElementsLoops.scatterElementsF32(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getAxis(), scatterOp.getReduction());
            case BFLOAT16 -> ScatterElementsLoops.scatterElementsBF16(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getAxis(), scatterOp.getReduction());
            case BOOL -> ScatterElementsLoops.scatterElementsBOOL(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getAxis(), scatterOp.getReduction());
            case INT32 -> ScatterElementsLoops.scatterElementsI32(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getAxis(), scatterOp.getReduction());
            case INT64 -> ScatterElementsLoops.scatterElementsI64(triple[0], triple[1], triple[2], node,
                    call.inputs().get(0), call.inputs().get(1), call.inputs().get(2), call.output(),
                    scatterOp.getAxis(), scatterOp.getReduction());
        }
        return CpuKernelResult.completed();
    }

    private static scatterElements requireOp(Operation op) {
        if (!(op instanceof scatterElements scatterOp)) {
            throw new IllegalArgumentException("CpuScatterElementsKernel requires scatterElements operation");
        }
        return scatterOp;
    }

    private static Tensor[] requireTriple(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 3) {
            throw new IllegalArgumentException("scatterElements expects exactly three inputs");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1), inputs.get(2)};
    }
}
