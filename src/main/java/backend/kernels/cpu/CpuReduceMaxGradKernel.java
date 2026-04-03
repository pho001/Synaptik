package backend.kernels.cpu;

import operations.Operation;
import operations.reduceMaxGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuReduceMaxGradKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduceMaxGrad gradOp = (reduceMaxGrad) op;
        ReductionMinMaxGradKernelSupport.runF64(
                inputs.get(0).getFloat64Data(),
                inputs.get(0).getShapeUnsafe(),
                inputs.get(0).getStridesUnsafe(),
                inputs.get(0).getStorageOffsetUnsafe(),
                inputs.get(1).getFloat64Data(),
                inputs.get(1).getShapeUnsafe(),
                inputs.get(1).getStorageOffsetUnsafe(),
                inputs.get(2).getFloat64Data(),
                node.getFloat64Data(),
                node.getStorageOffsetUnsafe(),
                gradOp.getDimension(),
                true
        );
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduceMaxGrad gradOp = (reduceMaxGrad) op;
        ReductionMinMaxGradKernelSupport.runF32(
                inputs.get(0).getFloat32Data(),
                inputs.get(0).getShapeUnsafe(),
                inputs.get(0).getStridesUnsafe(),
                inputs.get(0).getStorageOffsetUnsafe(),
                inputs.get(1).getFloat32Data(),
                inputs.get(1).getShapeUnsafe(),
                inputs.get(1).getStorageOffsetUnsafe(),
                inputs.get(2).getFloat32Data(),
                node.getFloat32Data(),
                node.getStorageOffsetUnsafe(),
                gradOp.getDimension(),
                true
        );
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduceMaxGrad gradOp = (reduceMaxGrad) op;
        ReductionMinMaxGradKernelSupport.runF16(
                inputs.get(0).getFloat16Data(),
                inputs.get(0).getShapeUnsafe(),
                inputs.get(0).getStridesUnsafe(),
                inputs.get(0).getStorageOffsetUnsafe(),
                inputs.get(1).getFloat16Data(),
                inputs.get(1).getShapeUnsafe(),
                inputs.get(1).getStorageOffsetUnsafe(),
                inputs.get(2).getFloat16Data(),
                node.getFloat16Data(),
                node.getStorageOffsetUnsafe(),
                gradOp.getDimension(),
                true
        );
    }
}
