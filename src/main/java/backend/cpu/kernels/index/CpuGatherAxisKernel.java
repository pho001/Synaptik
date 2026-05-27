package backend.cpu.kernels.index;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import operations.Operation;
import operations.index.gatherAxis;
import tensor.Tensor;

import java.util.List;

public final class CpuGatherAxisKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        Tensor[] pair = requirePair(call.operation(), call.inputTensors());
        if (call.inputs().size() != 2) {
            throw new IllegalArgumentException("gatherAxis expects exactly two input storage views.");
        }
        int axis = ((gatherAxis) call.operation()).getAxis();
        Tensor node = call.outputTensor();
        switch (node.getDataType()) {
            case FLOAT64 -> GatherAxisLoops.gatherAxisF64(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), axis);
            case FLOAT32 -> GatherAxisLoops.gatherAxisF32(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), axis);
            case BFLOAT16 -> GatherAxisLoops.gatherAxisBF16(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), axis);
            case BOOL -> GatherAxisLoops.gatherAxisBOOL(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), axis);
            case INT32 -> GatherAxisLoops.gatherAxisI32(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), axis);
            case INT64 -> GatherAxisLoops.gatherAxisI64(pair[0], pair[1], node,
                    call.inputs().get(0), call.inputs().get(1), call.output(), axis);
        }
        return CpuKernelResult.completed();
    }

    private static Tensor[] requirePair(Operation op, List<Tensor> inputs) {
        if (!(op instanceof gatherAxis)) {
            throw new IllegalArgumentException("CpuGatherAxisKernel requires gatherAxis operation.");
        }
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("gatherAxis expects exactly two inputs.");
        }
        return new Tensor[]{inputs.get(0), inputs.get(1)};
    }
}
