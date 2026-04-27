package backend.cpu.kernels.grad;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.reduction.reduceMinGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuReduceMinGradKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduceMinGrad gradOp = (reduceMinGrad) op;
        ReductionMinMaxGradExecutor.runF64(
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
                false
        );
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduceMinGrad gradOp = (reduceMinGrad) op;
        ReductionMinMaxGradExecutor.runF32(
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
                false
        );
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduceMinGrad gradOp = (reduceMinGrad) op;
        ReductionMinMaxGradExecutor.runBF16(
                inputs.get(0).getBFloat16Data(),
                inputs.get(0).getShapeUnsafe(),
                inputs.get(0).getStridesUnsafe(),
                inputs.get(0).getStorageOffsetUnsafe(),
                inputs.get(1).getBFloat16Data(),
                inputs.get(1).getShapeUnsafe(),
                inputs.get(1).getStorageOffsetUnsafe(),
                inputs.get(2).getBFloat16Data(),
                node.getBFloat16Data(),
                node.getStorageOffsetUnsafe(),
                gradOp.getDimension(),
                false
        );
    }
}
