package backend.cpu.kernels.reduction;

import tensor.TensorInternalAccess;

import tensor.dtype.TensorDTypeOps;
import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.reduction.reduceProd;
import tensor.Tensor;
import tensor.TensorMetadata;

import java.util.Arrays;
import java.util.List;

public final class CpuReduceProdKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduce(op, inputs, node);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduce(op, inputs, node);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        reduce(op, inputs, node);
    }

    private static void reduce(Operation op, List<Tensor> inputs, Tensor node) {
        if (!(op instanceof reduceProd reduction)) {
            throw new IllegalArgumentException("CpuReduceProdKernel requires reduceProd operation.");
        }
        Tensor input = CpuSumKernel.requireSingleInput(inputs, "ReduceProd");
        int dimension = reduction.getDimension();
        int[] shape = input.getShapeUnsafe();
        if (dimension < -1 || dimension >= shape.length) {
            throw new IllegalArgumentException("Dimension out of bounds: " + dimension);
        }
        fill(node, 1.0d);
        int[] inputDenseStrides = TensorMetadata.computeStrides(shape);
        int[] outShape = node.getShapeUnsafe();
        int[] outDenseStrides = TensorMetadata.computeStrides(outShape);
        for (int logical = 0; logical < input.getFlatDataSize(); logical++) {
            int outLogical = dimension == -1
                    ? 0
                    : outputLogicalIndex(logical, shape, inputDenseStrides, outShape.length, outDenseStrides, dimension);
            multiply(node, outLogical, input.getByFlatIndex(logical));
        }
        TensorInternalAccess.markStorageModified(node);
    }

    private static int outputLogicalIndex(
            int inputLogical,
            int[] inputShape,
            int[] inputDenseStrides,
            int outputRank,
            int[] outputDenseStrides,
            int reducedAxis
    ) {
        int tmp = inputLogical;
        int out = 0;
        for (int d = 0, od = 0; d < inputShape.length; d++) {
            int coord = tmp / inputDenseStrides[d];
            tmp %= inputDenseStrides[d];
            if (d == reducedAxis) {
                if (outputRank == inputShape.length) {
                    od++;
                }
                continue;
            }
            out += coord * outputDenseStrides[od++];
        }
        return out;
    }

    private static void fill(Tensor out, double value) {
        switch (out.getDataType()) {
            case FLOAT64 -> Arrays.fill(TensorInternalAccess.float64Data(out), value);
            case FLOAT32 -> Arrays.fill(TensorInternalAccess.float32Data(out), (float) value);
            case BFLOAT16 -> Arrays.fill(TensorInternalAccess.bfloat16Data(out), TensorDTypeOps.toBFloat16Bits((float) value));
            case INT32, BOOL -> throw new IllegalArgumentException("ReduceProd requires floating output.");
        }
    }

    private static void multiply(Tensor out, int logical, double value) {
        int index = out.getStorageOffsetUnsafe() + logical;
        switch (out.getDataType()) {
            case FLOAT64 -> TensorInternalAccess.float64Data(out)[index] *= value;
            case FLOAT32 -> TensorInternalAccess.float32Data(out)[index] *= (float) value;
            case BFLOAT16 -> {
                short[] data = TensorInternalAccess.bfloat16Data(out);
                data[index] = TensorDTypeOps.toBFloat16Bits(TensorDTypeOps.fromBFloat16Bits(data[index]) * (float) value);
            }
            case INT32, BOOL -> throw new IllegalArgumentException("ReduceProd requires floating output.");
        }
    }
}
