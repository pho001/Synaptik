package backend.cpu.kernels.reduction;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.reduction.ArgMaxTiePolicy;
import operations.reduction.argMax;
import tensor.Tensor;
import tensor.TensorMetadata;

import java.util.Arrays;
import java.util.List;

public final class CpuArgMaxKernel implements CpuKernel {
    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        argMax(op, inputs, node, TensorInternalAccess.int32Data(node), null);
    }

    @Override
    public void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        argMax(op, inputs, node, null, TensorInternalAccess.int64Data(node));
    }

    private static void argMax(Operation op, List<Tensor> inputs, Tensor node, int[] bestIndices32, long[] bestIndices64) {
        if (!(op instanceof argMax reduction)) {
            throw new IllegalArgumentException("CpuArgMaxKernel requires argMax operation.");
        }
        Tensor input = CpuSumKernel.requireSingleInput(inputs, "ArgMax");
        int axis = reduction.getDimension();
        int[] shape = input.getShapeUnsafe();
        if (axis < 0 || axis >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + axis);
        }
        double[] bestValues = new double[node.getFlatDataSize()];
        Arrays.fill(bestValues, Double.NEGATIVE_INFINITY);
        boolean[] seen = new boolean[node.getFlatDataSize()];
        if (bestIndices32 != null) {
            Arrays.fill(bestIndices32, 0);
        }
        if (bestIndices64 != null) {
            Arrays.fill(bestIndices64, 0L);
        }

        int[] inputDenseStrides = TensorMetadata.computeStrides(shape);
        int[] outShape = node.getShapeUnsafe();
        int[] outDenseStrides = TensorMetadata.computeStrides(outShape);
        boolean lastIndexWins = reduction.tiePolicy() == ArgMaxTiePolicy.LAST_INDEX;
        for (int logical = 0; logical < input.getFlatDataSize(); logical++) {
            int tmp = logical;
            int outLogical = 0;
            int axisCoord = 0;
            for (int d = 0, od = 0; d < shape.length; d++) {
                int coord = tmp / inputDenseStrides[d];
                tmp %= inputDenseStrides[d];
                if (d == axis) {
                    axisCoord = coord;
                    if (outShape.length == shape.length) {
                        od++;
                    }
                } else {
                    outLogical += coord * outDenseStrides[od++];
                }
            }
            double value = input.getByFlatIndex(logical);
            if (!seen[outLogical] || value > bestValues[outLogical]
                    || (lastIndexWins && Double.compare(value, bestValues[outLogical]) == 0)) {
                seen[outLogical] = true;
                bestValues[outLogical] = value;
                int offset = node.getStorageOffsetUnsafe() + outLogical;
                if (bestIndices32 != null) {
                    bestIndices32[offset] = axisCoord;
                } else {
                    bestIndices64[offset] = axisCoord;
                }
            }
        }
        TensorInternalAccess.markStorageModified(node);
    }
}
