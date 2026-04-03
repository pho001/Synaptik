package backend.kernels.cpu;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuGreaterThanKernel implements CpuKernel {
    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (node.getDataType() != tensor.DataType.BOOL) {
            throw new IllegalArgumentException("greaterThan kernel requires BOOL output.");
        }
        switch (inputs.get(0).getDataType()) {
            case FLOAT64 -> CompareKernelSupport.runF64(Operation.OpType.GT, inputs.get(0).getFloat64Data(), inputs.get(1).getFloat64Data(), node.getBoolData(), context.broadcastPlan(), context.dispatchHints());
            case FLOAT32 -> CompareKernelSupport.runF32(Operation.OpType.GT, inputs.get(0).getFloat32Data(), inputs.get(1).getFloat32Data(), node.getBoolData(), context.broadcastPlan(), context.dispatchHints());
            case FLOAT16 -> CompareKernelSupport.runF16(Operation.OpType.GT, inputs.get(0).getFloat16Data(), inputs.get(1).getFloat16Data(), node.getBoolData(), context.broadcastPlan(), context.dispatchHints());
            case BOOL -> throw new IllegalArgumentException("greaterThan does not support BOOL inputs.");
        }
    }
}
