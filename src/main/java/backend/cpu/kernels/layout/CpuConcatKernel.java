package backend.cpu.kernels.layout;

import tensor.TensorInternalAccess;

import tensor.dtype.TensorDTypeOps;
import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.layout.concat;
import tensor.Tensor;
import tensor.TensorMetadata;

import java.util.List;

public final class CpuConcatKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        concat(op, inputs, node);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        concat(op, inputs, node);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        concat(op, inputs, node);
    }

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        concat(op, inputs, node);
    }

    @Override
    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        concat(op, inputs, node);
    }

    @Override
    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        concat(op, inputs, node);
    }

    private static void concat(Operation op, List<Tensor> inputs, Tensor node) {
        if (!(op instanceof concat concatOp)) {
            throw new IllegalArgumentException("CpuConcatKernel requires concat operation.");
        }
        if (inputs == null || inputs.isEmpty()) {
            throw new IllegalArgumentException("concat expects at least one input.");
        }
        int axis = concatOp.getAxis();
        int[] outShape = node.getShapeUnsafe();
        int[] outDenseStrides = TensorMetadata.computeStrides(outShape);
        int axisOffset = 0;
        for (Tensor input : inputs) {
            copyInput(input, node, axis, axisOffset, outDenseStrides);
            axisOffset += input.getShapeUnsafe()[axis];
        }
        TensorInternalAccess.markStorageModified(node);
    }

    private static void copyInput(Tensor input, Tensor out, int axis, int axisOffset, int[] outDenseStrides) {
        int[] inputShape = input.getShapeUnsafe();
        int[] inputDenseStrides = TensorMetadata.computeStrides(inputShape);
        int rank = inputShape.length;
        for (int logical = 0; logical < input.getFlatDataSize(); logical++) {
            int tmp = logical;
            int outLogical = 0;
            for (int d = 0; d < rank; d++) {
                int coord = tmp / inputDenseStrides[d];
                tmp %= inputDenseStrides[d];
                if (d == axis) {
                    coord += axisOffset;
                }
                outLogical += coord * outDenseStrides[d];
            }
            if (out.getDataType() == tensor.DataType.INT64) {
                TensorInternalAccess.int64Data(out)[outLogical] = input.getInt64ByFlatIndex(logical);
                continue;
            }
            write(out, outLogical, input.getByFlatIndex(logical));
        }
    }

    private static void write(Tensor out, int index, double value) {
        switch (out.getDataType()) {
            case FLOAT64 -> TensorInternalAccess.float64Data(out)[index] = value;
            case FLOAT32 -> TensorInternalAccess.float32Data(out)[index] = (float) value;
            case BFLOAT16 -> TensorInternalAccess.bfloat16Data(out)[index] = TensorDTypeOps.toBFloat16Bits((float) value);
            case INT32 -> TensorInternalAccess.int32Data(out)[index] = (int) value;
            case INT64 -> TensorInternalAccess.int64Data(out)[index] = (long) value;
            case BOOL -> TensorInternalAccess.boolData(out)[index] = value == 0.0d ? (byte) 0 : (byte) 1;
        }
    }
}
