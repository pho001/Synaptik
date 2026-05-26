package backend.cpu.kernels.layout;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.kernels.TypedCpuKernel;
import operations.Operation;
import operations.layout.unfold2d;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.dtype.TensorDTypeOps;
import tensor.options.Window2dOptions;

import java.util.List;

public final class CpuUnfold2dKernel extends TypedCpuKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        unfold(op, inputs, node);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        unfold(op, inputs, node);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        unfold(op, inputs, node);
    }

    private static void unfold(Operation op, List<Tensor> inputs, Tensor node) {
        if (!(op instanceof unfold2d unfoldOp)) {
            throw new IllegalArgumentException("CpuUnfold2dKernel requires unfold2d operation.");
        }
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("unfold2d expects exactly one input.");
        }
        Tensor input = inputs.getFirst();
        Window2dOptions options = unfoldOp.getOptions();
        int[] inputShape = input.getShapeUnsafe();
        int[] outShape = node.getShapeUnsafe();
        int batch = inputShape[0];
        int channels = inputShape[1];
        int height = inputShape[2];
        int width = inputShape[3];
        int kernelH = options.kernelH();
        int kernelW = options.kernelW();
        int outH = outShape[2] / inferOutW(width, options);
        int outW = inferOutW(width, options);
        int kernelArea = kernelH * kernelW;
        int windowCount = outH * outW;
        for (int n = 0; n < batch; n++) {
            for (int c = 0; c < channels; c++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        int columnChannel = c * kernelArea + kh * kernelW + kw;
                        for (int oy = 0; oy < outH; oy++) {
                            int iy = oy * options.strideH() - options.padH() + kh * options.dilationH();
                            for (int ox = 0; ox < outW; ox++) {
                                int ix = ox * options.strideW() - options.padW() + kw * options.dilationW();
                                int outIndex = (n * outShape[1] + columnChannel) * windowCount + oy * outW + ox;
                                double value = 0.0d;
                                if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
                                    int inputIndex = ((n * channels + c) * height + iy) * width + ix;
                                    value = input.getByFlatIndex(inputIndex);
                                }
                                write(node, outIndex, value);
                            }
                        }
                    }
                }
            }
        }
        TensorInternalAccess.markStorageModified(node);
    }

    private static int inferOutW(int width, Window2dOptions options) {
        int effectiveKernelW = options.dilationW() * (options.kernelW() - 1) + 1;
        int numerator = width + 2 * options.padW() - effectiveKernelW;
        return (options.ceilMode() ? (numerator + options.strideW() - 1) / options.strideW() : numerator / options.strideW()) + 1;
    }

    private static void write(Tensor out, int index, double value) {
        switch (out.getDataType()) {
            case FLOAT64 -> TensorInternalAccess.float64Data(out)[index] = value;
            case FLOAT32 -> TensorInternalAccess.float32Data(out)[index] = (float) value;
            case BFLOAT16 -> TensorInternalAccess.bfloat16Data(out)[index] = TensorDTypeOps.toBFloat16Bits((float) value);
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("unfold2d supports floating dtypes only.");
        }
    }
}
