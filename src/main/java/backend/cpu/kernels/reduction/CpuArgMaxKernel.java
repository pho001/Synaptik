package backend.cpu.kernels.reduction;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.reduction.argMax;
import tensor.Tensor;
import tensor.TensorMetadata;

import java.util.Arrays;
import java.util.List;

public final class CpuArgMaxKernel implements CpuKernel {
    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
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
        int[] bestIndices = node.getInt32Data();
        Arrays.fill(bestIndices, 0);

        int[] inputDenseStrides = TensorMetadata.computeStrides(shape);
        int[] outShape = node.getShapeUnsafe();
        int[] outDenseStrides = TensorMetadata.computeStrides(outShape);
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
            if (!seen[outLogical] || value > bestValues[outLogical]) {
                seen[outLogical] = true;
                bestValues[outLogical] = value;
                bestIndices[node.getStorageOffsetUnsafe() + outLogical] = axisCoord;
            }
        }
        node.markStorageModified();
    }
}
