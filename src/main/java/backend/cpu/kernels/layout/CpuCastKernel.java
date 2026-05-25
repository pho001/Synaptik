package backend.cpu.kernels.layout;

import tensor.TensorInternalAccess;

import tensor.dtype.TensorDTypeOps;
import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

public final class CpuCastKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        cast(op, inputs, node, context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        cast(op, inputs, node, context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        cast(op, inputs, node, context);
    }

    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        cast(op, inputs, node, context);
    }

    @Override
    protected void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        cast(op, inputs, node, context);
    }

    @Override
    protected void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        cast(op, inputs, node, context);
    }

    private static void cast(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("cast expects exactly one input.");
        }
        if (LayoutExecutor.tryRunNativeCast(inputs, node, context)) {
            return;
        }
        Tensor input = inputs.getFirst();
        int size = node.getFlatDataSize();
        if (input.getFlatDataSize() != size) {
            throw new IllegalArgumentException("cast requires input and output to have the same flat size.");
        }
        switch (node.getDataType()) {
            case FLOAT64 -> {
                double[] out = TensorInternalAccess.float64Data(node);
                for (int i = 0; i < size; i++) {
                    out[i] = input.getByFlatIndex(i);
                }
            }
            case FLOAT32 -> {
                float[] out = TensorInternalAccess.float32Data(node);
                for (int i = 0; i < size; i++) {
                    out[i] = (float) input.getByFlatIndex(i);
                }
            }
            case BFLOAT16 -> {
                short[] out = TensorInternalAccess.bfloat16Data(node);
                for (int i = 0; i < size; i++) {
                    out[i] = TensorDTypeOps.toBFloat16Bits((float) input.getByFlatIndex(i));
                }
            }
            case INT32 -> {
                int[] out = TensorInternalAccess.int32Data(node);
                for (int i = 0; i < size; i++) {
                    out[i] = (int) input.getByFlatIndex(i);
                }
            }
            case INT64 -> {
                long[] out = TensorInternalAccess.int64Data(node);
                for (int i = 0; i < size; i++) {
                    out[i] = (long) input.getByFlatIndex(i);
                }
            }
            case BOOL -> {
                byte[] out = TensorInternalAccess.boolData(node);
                for (int i = 0; i < size; i++) {
                    out[i] = input.getByFlatIndex(i) == 0.0d ? (byte) 0 : (byte) 1;
                }
            }
        }
        TensorInternalAccess.markStorageModified(node);
    }
}
