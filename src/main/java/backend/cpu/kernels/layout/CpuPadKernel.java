package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import operations.layout.pad;
import tensor.Tensor;
import tensor.TensorMetadata;

import java.util.Arrays;
import java.util.List;

public final class CpuPadKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pad(op, inputs, node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pad(op, inputs, node);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pad(op, inputs, node);
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pad(op, inputs, node);
    }

    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pad(op, inputs, node);
    }

    @Override
    public void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
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
                node.getInt64Data()[outLogical] = input.getInt64ByFlatIndex(logical);
                continue;
            }
            write(node, outLogical, input.getByFlatIndex(logical));
        }
        node.markStorageModified();
    }

    private static Tensor requireSingleInput(List<Tensor> inputs) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("pad expects exactly one input.");
        }
        return inputs.getFirst();
    }

    private static void fill(Tensor out, double value) {
        switch (out.getDataType()) {
            case FLOAT64 -> Arrays.fill(out.getFloat64Data(), value);
            case FLOAT32 -> Arrays.fill(out.getFloat32Data(), (float) value);
            case BFLOAT16 -> Arrays.fill(out.getBFloat16Data(), CpuDTypeOps.toBFloat16Bits((float) value));
            case INT32 -> Arrays.fill(out.getInt32Data(), (int) value);
            case INT64 -> Arrays.fill(out.getInt64Data(), (long) value);
            case BOOL -> Arrays.fill(out.getBoolData(), value == 0.0d ? (byte) 0 : (byte) 1);
        }
    }

    private static void write(Tensor out, int index, double value) {
        switch (out.getDataType()) {
            case FLOAT64 -> out.getFloat64Data()[index] = value;
            case FLOAT32 -> out.getFloat32Data()[index] = (float) value;
            case BFLOAT16 -> out.getBFloat16Data()[index] = CpuDTypeOps.toBFloat16Bits((float) value);
            case INT32 -> out.getInt32Data()[index] = (int) value;
            case INT64 -> out.getInt64Data()[index] = (long) value;
            case BOOL -> out.getBoolData()[index] = value == 0.0d ? (byte) 0 : (byte) 1;
        }
    }
}
