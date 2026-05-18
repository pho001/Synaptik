package backend.cpu.kernels.grad;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.*;

import operations.Operation;
import operations.reduction.reduceMaxGrad;
import tensor.Tensor;

import java.util.List;

public final class CpuReduceMaxGradKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduceMaxGrad gradOp = (reduceMaxGrad) op;
        ReductionMinMaxGradExecutor.runF64(
                TensorInternalAccess.float64Data(inputs.get(0)),
                inputs.get(0).getShapeUnsafe(),
                inputs.get(0).getStridesUnsafe(),
                inputs.get(0).getStorageOffsetUnsafe(),
                TensorInternalAccess.float64Data(inputs.get(1)),
                inputs.get(1).getShapeUnsafe(),
                inputs.get(1).getStorageOffsetUnsafe(),
                TensorInternalAccess.float64Data(inputs.get(2)),
                TensorInternalAccess.float64Data(node),
                node.getStorageOffsetUnsafe(),
                gradOp.getDimension(),
                true
        );
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduceMaxGrad gradOp = (reduceMaxGrad) op;
        ReductionMinMaxGradExecutor.runF32(
                TensorInternalAccess.float32Data(inputs.get(0)),
                inputs.get(0).getShapeUnsafe(),
                inputs.get(0).getStridesUnsafe(),
                inputs.get(0).getStorageOffsetUnsafe(),
                TensorInternalAccess.float32Data(inputs.get(1)),
                inputs.get(1).getShapeUnsafe(),
                inputs.get(1).getStorageOffsetUnsafe(),
                TensorInternalAccess.float32Data(inputs.get(2)),
                TensorInternalAccess.float32Data(node),
                node.getStorageOffsetUnsafe(),
                gradOp.getDimension(),
                true
        );
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduceMaxGrad gradOp = (reduceMaxGrad) op;
        ReductionMinMaxGradExecutor.runBF16(
                TensorInternalAccess.bfloat16Data(inputs.get(0)),
                inputs.get(0).getShapeUnsafe(),
                inputs.get(0).getStridesUnsafe(),
                inputs.get(0).getStorageOffsetUnsafe(),
                TensorInternalAccess.bfloat16Data(inputs.get(1)),
                inputs.get(1).getShapeUnsafe(),
                inputs.get(1).getStorageOffsetUnsafe(),
                TensorInternalAccess.bfloat16Data(inputs.get(2)),
                TensorInternalAccess.bfloat16Data(node),
                node.getStorageOffsetUnsafe(),
                gradOp.getDimension(),
                true
        );
    }
}
