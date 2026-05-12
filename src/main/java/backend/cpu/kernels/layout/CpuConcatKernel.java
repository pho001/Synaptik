package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.layout.concat;
import tensor.Tensor;
import tensor.TensorMetadata;

import java.util.List;

public final class CpuConcatKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        concat(op, inputs, node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        concat(op, inputs, node);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        concat(op, inputs, node);
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        concat(op, inputs, node);
    }

    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
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
        node.markStorageModified();
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
            write(out, outLogical, input.getByFlatIndex(logical));
        }
    }

    private static void write(Tensor out, int index, double value) {
        switch (out.getDataType()) {
            case FLOAT64 -> out.getFloat64Data()[index] = value;
            case FLOAT32 -> out.getFloat32Data()[index] = (float) value;
            case BFLOAT16 -> out.getBFloat16Data()[index] = CpuDTypeOps.toBFloat16Bits((float) value);
            case INT32 -> out.getInt32Data()[index] = (int) value;
            case BOOL -> out.getBoolData()[index] = value == 0.0d ? (byte) 0 : (byte) 1;
        }
    }
}
