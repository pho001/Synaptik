package backend.cpu.kernels.layout;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.TypedCpuKernel;
import operations.Operation;
import operations.layout.fold2d;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;
import tensor.options.Window2dOptions;

import java.util.List;

public final class CpuFold2dKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        fold(op, inputs, node);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        fold(op, inputs, node);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        fold(op, inputs, node);
    }

    private static void fold(Operation op, List<Tensor> inputs, Tensor node) {
        if (!(op instanceof fold2d foldOp)) {
            throw new IllegalArgumentException("CpuFold2dKernel requires fold2d operation.");
        }
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("fold2d expects exactly one input.");
        }
        Tensor input = inputs.getFirst();
        Window2dOptions options = foldOp.getOptions();
        int[] outShape = node.getShapeUnsafe();
        int[] inputShape = input.getShapeUnsafe();
        int batch = outShape[0];
        int channels = outShape[1];
        int height = outShape[2];
        int width = outShape[3];
        int kernelH = options.kernelH();
        int kernelW = options.kernelW();
        int kernelArea = kernelH * kernelW;
        int outW = inferOutW(width, options);
        int outH = inputShape[2] / outW;
        int windowCount = outH * outW;
        double[] acc = new double[node.getFlatDataSize()];
        for (int n = 0; n < batch; n++) {
            for (int c = 0; c < channels; c++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        int columnChannel = c * kernelArea + kh * kernelW + kw;
                        for (int oy = 0; oy < outH; oy++) {
                            int iy = oy * options.strideH() - options.padH() + kh * options.dilationH();
                            for (int ox = 0; ox < outW; ox++) {
                                int ix = ox * options.strideW() - options.padW() + kw * options.dilationW();
                                if (iy < 0 || iy >= height || ix < 0 || ix >= width) {
                                    continue;
                                }
                                int inputIndex = (n * inputShape[1] + columnChannel) * windowCount + oy * outW + ox;
                                int outIndex = ((n * channels + c) * height + iy) * width + ix;
                                acc[outIndex] += input.getByFlatIndex(inputIndex);
                            }
                        }
                    }
                }
            }
        }
        writeAll(node, acc);
        TensorInternalAccess.markStorageModified(node);
    }

    private static int inferOutW(int width, Window2dOptions options) {
        int effectiveKernelW = options.dilationW() * (options.kernelW() - 1) + 1;
        int numerator = width + 2 * options.padW() - effectiveKernelW;
        return (options.ceilMode() ? (numerator + options.strideW() - 1) / options.strideW() : numerator / options.strideW()) + 1;
    }

    private static void writeAll(Tensor out, double[] values) {
        switch (out.getDataType()) {
            case FLOAT64 -> {
                double[] data = TensorInternalAccess.float64Data(out);
                System.arraycopy(values, 0, data, 0, values.length);
            }
            case FLOAT32 -> {
                float[] data = TensorInternalAccess.float32Data(out);
                for (int i = 0; i < values.length; i++) {
                    data[i] = (float) values[i];
                }
            }
            case BFLOAT16 -> {
                short[] data = TensorInternalAccess.bfloat16Data(out);
                for (int i = 0; i < values.length; i++) {
                    data[i] = TensorDTypeOps.toBFloat16Bits((float) values[i]);
                }
            }
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("fold2d supports floating dtypes only.");
        }
    }
}
