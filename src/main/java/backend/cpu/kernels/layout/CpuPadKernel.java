package backend.cpu.kernels.layout;

import tensor.TensorInternalAccess;

import tensor.dtype.TensorDTypeOps;
import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import operations.layout.pad;
import tensor.Tensor;
import tensor.TensorMetadata;

import java.util.Arrays;
import java.util.List;

public final class CpuPadKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pad(op, inputs, node);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pad(op, inputs, node);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pad(op, inputs, node);
    }

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pad(op, inputs, node);
    }

    @Override
    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pad(op, inputs, node);
    }

    @Override
    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pad(op, inputs, node);
    }

    private static void pad(Operation op, List<Tensor> inputs, Tensor node) {
        if (!(op instanceof pad padOp)) {
            throw new IllegalArgumentException("CpuPadKernel requires pad operation.");
        }
        Tensor input = requireSingleInput(inputs);
        fill(node, padOp.getConstantValue());
        int[] before = padOp.getBefore();
        int[] inputShape = input.getShapeUnsafe();
        int[] inputDenseStrides = TensorMetadata.computeStrides(inputShape);
        int[] outDenseStrides = TensorMetadata.computeStrides(node.getShapeUnsafe());
        for (int logical = 0; logical < input.getFlatDataSize(); logical++) {
            int tmp = logical;
            int outLogical = 0;
            for (int d = 0; d < inputShape.length; d++) {
                int coord = tmp / inputDenseStrides[d];
                tmp %= inputDenseStrides[d];
                outLogical += (coord + before[d]) * outDenseStrides[d];
            }
            if (node.getDataType() == tensor.DataType.INT64) {
                TensorInternalAccess.int64Data(node)[outLogical] = input.getInt64ByFlatIndex(logical);
                continue;
            }
            write(node, outLogical, input.getByFlatIndex(logical));
        }
        TensorInternalAccess.markStorageModified(node);
    }

    private static Tensor requireSingleInput(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("pad expects exactly one input.");
        }
        return inputs.getFirst();
    }

    private static void fill(Tensor out, double value) {
        switch (out.getDataType()) {
            case FLOAT64 -> Arrays.fill(TensorInternalAccess.float64Data(out), value);
            case FLOAT32 -> Arrays.fill(TensorInternalAccess.float32Data(out), (float) value);
            case BFLOAT16 -> Arrays.fill(TensorInternalAccess.bfloat16Data(out), TensorDTypeOps.toBFloat16Bits((float) value));
            case INT32 -> Arrays.fill(TensorInternalAccess.int32Data(out), (int) value);
            case INT64 -> Arrays.fill(TensorInternalAccess.int64Data(out), (long) value);
            case BOOL -> Arrays.fill(TensorInternalAccess.boolData(out), value == 0.0d ? (byte) 0 : (byte) 1);
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
