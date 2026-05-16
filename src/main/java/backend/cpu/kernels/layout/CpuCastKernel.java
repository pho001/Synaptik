package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.nativecpu.NativeCpuCastExecutor;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

public final class CpuCastKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        cast(op, inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        cast(op, inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        cast(op, inputs, node, context);
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        cast(op, inputs, node, context);
    }

    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        cast(op, inputs, node, context);
    }

    @Override
    public void forwardI64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        cast(op, inputs, node, context);
    }

    private static void cast(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("cast expects exactly one input.");
        }
        if (NativeCpuCastExecutor.tryRunCast(inputs, node, context)) {
            return;
        }
        Tensor input = inputs.getFirst();
        int size = node.getFlatDataSize();
        if (input.getFlatDataSize() != size) {
            throw new IllegalArgumentException("cast requires input and output to have the same flat size.");
        }
        switch (node.getDataType()) {
            case FLOAT64 -> {
                double[] out = node.getFloat64Data();
                for (int i = 0; i < size; i++) {
                    out[i] = input.getByFlatIndex(i);
                }
            }
            case FLOAT32 -> {
                float[] out = node.getFloat32Data();
                for (int i = 0; i < size; i++) {
                    out[i] = (float) input.getByFlatIndex(i);
                }
            }
            case BFLOAT16 -> {
                short[] out = node.getBFloat16Data();
                for (int i = 0; i < size; i++) {
                    out[i] = CpuDTypeOps.toBFloat16Bits((float) input.getByFlatIndex(i));
                }
            }
            case INT32 -> {
                int[] out = node.getInt32Data();
                for (int i = 0; i < size; i++) {
                    out[i] = (int) input.getByFlatIndex(i);
                }
            }
            case INT64 -> {
                long[] out = node.getInt64Data();
                for (int i = 0; i < size; i++) {
                    out[i] = (long) input.getByFlatIndex(i);
                }
            }
            case BOOL -> {
                byte[] out = node.getBoolData();
                for (int i = 0; i < size; i++) {
                    out[i] = input.getByFlatIndex(i) == 0.0d ? (byte) 0 : (byte) 1;
                }
            }
        }
        node.markStorageModified();
    }
}
