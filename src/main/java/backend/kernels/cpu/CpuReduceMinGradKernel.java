package backend.kernels.cpu;

import operations.Operation;
import operations.reduceMinGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuReduceMinGradKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduceMinGrad gradOp = (reduceMinGrad) op;
        ReductionMinMaxGradKernelSupport.runF64(
                inputs.get(0).getFloat64Data(),
                inputs.get(0).getShapeUnsafe(),
                inputs.get(0).getStridesUnsafe(),
                inputs.get(1).getFloat64Data(),
                inputs.get(1).getShapeUnsafe(),
                inputs.get(2).getFloat64Data(),
                node.getFloat64Data(),
                gradOp.getDimension(),
                false
        );
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduceMinGrad gradOp = (reduceMinGrad) op;
        ReductionMinMaxGradKernelSupport.runF32(
                inputs.get(0).getFloat32Data(),
                inputs.get(0).getShapeUnsafe(),
                inputs.get(0).getStridesUnsafe(),
                inputs.get(1).getFloat32Data(),
                inputs.get(1).getShapeUnsafe(),
                inputs.get(2).getFloat32Data(),
                node.getFloat32Data(),
                gradOp.getDimension(),
                false
        );
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduceMinGrad gradOp = (reduceMinGrad) op;
        ReductionMinMaxGradKernelSupport.runF16(
                inputs.get(0).getFloat16Data(),
                inputs.get(0).getShapeUnsafe(),
                inputs.get(0).getStridesUnsafe(),
                inputs.get(1).getFloat16Data(),
                inputs.get(1).getShapeUnsafe(),
                inputs.get(2).getFloat16Data(),
                node.getFloat16Data(),
                gradOp.getDimension(),
                false
        );
    }
}
