package backend.kernels.cpu;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuEqualToKernel implements CpuKernel {
    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (node.getDataType() != tensor.DataType.BOOL) {
            throw new IllegalArgumentException("equalTo kernel requires BOOL output.");
        }
        switch (inputs.get(0).getDataType()) {
            case FLOAT64 -> CompareKernelSupport.runF64(Operation.OpType.EQ, inputs.get(0).getFloat64Data(), inputs.get(1).getFloat64Data(), node.getBoolData(), context.broadcastPlan(), context.dispatchHints());
            case FLOAT32 -> CompareKernelSupport.runF32(Operation.OpType.EQ, inputs.get(0).getFloat32Data(), inputs.get(1).getFloat32Data(), node.getBoolData(), context.broadcastPlan(), context.dispatchHints());
            case BFLOAT16 -> CompareKernelSupport.runBF16(Operation.OpType.EQ, inputs.get(0).getBFloat16Data(), inputs.get(1).getBFloat16Data(), node.getBoolData(), context.broadcastPlan(), context.dispatchHints());
            case INT32, BOOL -> throw new IllegalArgumentException("equalTo does not support INT32/BOOL inputs.");
        }
    }
}
