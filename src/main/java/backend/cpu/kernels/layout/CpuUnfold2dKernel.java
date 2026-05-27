package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.layout.unfold2d;
import tensor.DataType;
import tensor.options.Window2dOptions;

public final class CpuUnfold2dKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        if (!(call.operation() instanceof unfold2d unfoldOp)) {
            throw new IllegalArgumentException("CpuUnfold2dKernel requires unfold2d operation.");
        }
        LayoutStorageSupport.validateInputViews(1, call.inputs(), "unfold2d");
        CpuStorageView input = call.inputs().getFirst();
        CpuStorageView out = call.output();
        LayoutStorageSupport.validateView(input, out.dtype(), "input");
        LayoutStorageSupport.validateView(out, out.dtype(), "output");
        validateFloating(out.dtype(), "unfold2d");
        switch (out.dtype()) {
            case FLOAT64 -> unfoldF64(unfoldOp, input, out);
            case FLOAT32 -> unfoldF32(unfoldOp, input, out);
            case BFLOAT16 -> unfoldBF16(unfoldOp, input, out);
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("unfold2d supports floating dtypes only.");
        }
        return CpuKernelResult.completed();
    }

    private static void unfoldF64(unfold2d op, CpuStorageView input, CpuStorageView out) {
        WindowPlan plan = WindowPlan.create(op.getOptions(), input, out);
        for (int n = 0; n < plan.batch; n++) {
            for (int c = 0; c < plan.channels; c++) {
                for (int kh = 0; kh < plan.kernelH; kh++) {
                    for (int kw = 0; kw < plan.kernelW; kw++) {
                        int columnChannel = c * plan.kernelArea + kh * plan.kernelW + kw;
                        for (int oy = 0; oy < plan.outH; oy++) {
                            int iy = oy * plan.options.strideH() - plan.options.padH() + kh * plan.options.dilationH();
                            for (int ox = 0; ox < plan.outW; ox++) {
                                int ix = ox * plan.options.strideW() - plan.options.padW() + kw * plan.options.dilationW();
                                int outOffset = plan.outputOffset(n, columnChannel, oy * plan.outW + ox);
                                double value = 0.0d;
                                if (plan.inBounds(iy, ix)) {
                                    value = LayoutStorageSupport.readF64(input, plan.inputOffset(n, c, iy, ix));
                                }
                                LayoutStorageSupport.writeF64(out, outOffset, value);
                            }
                        }
                    }
                }
            }
        }
    }

    private static void unfoldF32(unfold2d op, CpuStorageView input, CpuStorageView out) {
        WindowPlan plan = WindowPlan.create(op.getOptions(), input, out);
        for (int n = 0; n < plan.batch; n++) {
            for (int c = 0; c < plan.channels; c++) {
                for (int kh = 0; kh < plan.kernelH; kh++) {
                    for (int kw = 0; kw < plan.kernelW; kw++) {
                        int columnChannel = c * plan.kernelArea + kh * plan.kernelW + kw;
                        for (int oy = 0; oy < plan.outH; oy++) {
                            int iy = oy * plan.options.strideH() - plan.options.padH() + kh * plan.options.dilationH();
                            for (int ox = 0; ox < plan.outW; ox++) {
                                int ix = ox * plan.options.strideW() - plan.options.padW() + kw * plan.options.dilationW();
                                int outOffset = plan.outputOffset(n, columnChannel, oy * plan.outW + ox);
                                float value = 0.0f;
                                if (plan.inBounds(iy, ix)) {
                                    value = LayoutStorageSupport.readF32(input, plan.inputOffset(n, c, iy, ix));
                                }
                                LayoutStorageSupport.writeF32(out, outOffset, value);
                            }
                        }
                    }
                }
            }
        }
    }

    private static void unfoldBF16(unfold2d op, CpuStorageView input, CpuStorageView out) {
        WindowPlan plan = WindowPlan.create(op.getOptions(), input, out);
        for (int n = 0; n < plan.batch; n++) {
            for (int c = 0; c < plan.channels; c++) {
                for (int kh = 0; kh < plan.kernelH; kh++) {
                    for (int kw = 0; kw < plan.kernelW; kw++) {
                        int columnChannel = c * plan.kernelArea + kh * plan.kernelW + kw;
                        for (int oy = 0; oy < plan.outH; oy++) {
                            int iy = oy * plan.options.strideH() - plan.options.padH() + kh * plan.options.dilationH();
                            for (int ox = 0; ox < plan.outW; ox++) {
                                int ix = ox * plan.options.strideW() - plan.options.padW() + kw * plan.options.dilationW();
                                int outOffset = plan.outputOffset(n, columnChannel, oy * plan.outW + ox);
                                float value = 0.0f;
                                if (plan.inBounds(iy, ix)) {
                                    value = LayoutStorageSupport.readBF16AsF32(input, plan.inputOffset(n, c, iy, ix));
                                }
                                LayoutStorageSupport.writeBF16(out, outOffset, value);
                            }
                        }
                    }
                }
            }
        }
    }

    private static void validateFloating(DataType dtype, String opName) {
        if (dtype == DataType.INT32 || dtype == DataType.INT64 || dtype == DataType.BOOL) {
            throw new UnsupportedOperationException(opName + " supports floating dtypes only.");
        }
    }

    private static int inferOutW(int width, Window2dOptions options) {
        int effectiveKernelW = options.dilationW() * (options.kernelW() - 1) + 1;
        int numerator = width + 2 * options.padW() - effectiveKernelW;
        return (options.ceilMode() ? (numerator + options.strideW() - 1) / options.strideW() : numerator / options.strideW()) + 1;
    }

    private static final class WindowPlan {
        private final Window2dOptions options;
        private final int batch;
        private final int channels;
        private final int height;
        private final int width;
        private final int kernelH;
        private final int kernelW;
        private final int kernelArea;
        private final int outH;
        private final int outW;
        private final int[] inputStrides;
        private final int inputBaseOffset;
        private final int[] outStrides;
        private final int outBaseOffset;

        private WindowPlan(
                Window2dOptions options,
                int batch,
                int channels,
                int height,
                int width,
                int kernelH,
                int kernelW,
                int outH,
                int outW,
                int[] inputStrides,
                int inputBaseOffset,
                int[] outStrides,
                int outBaseOffset
        ) {
            this.options = options;
            this.batch = batch;
            this.channels = channels;
            this.height = height;
            this.width = width;
            this.kernelH = kernelH;
            this.kernelW = kernelW;
            this.kernelArea = kernelH * kernelW;
            this.outH = outH;
            this.outW = outW;
            this.inputStrides = inputStrides;
            this.inputBaseOffset = inputBaseOffset;
            this.outStrides = outStrides;
            this.outBaseOffset = outBaseOffset;
        }

        static WindowPlan create(Window2dOptions options, CpuStorageView input, CpuStorageView out) {
            int[] inputShape = input.shape();
            int[] outShape = out.shape();
            if (inputShape.length != 4 || outShape.length != 3) {
                throw new IllegalArgumentException("unfold2d expects NCHW input and rank-3 output.");
            }
            int outW = inferOutW(inputShape[3], options);
            int outH = outShape[2] / outW;
            return new WindowPlan(
                    options,
                    inputShape[0],
                    inputShape[1],
                    inputShape[2],
                    inputShape[3],
                    options.kernelH(),
                    options.kernelW(),
                    outH,
                    outW,
                    input.strides(),
                    input.storageOffset(),
                    out.strides(),
                    out.storageOffset());
        }

        boolean inBounds(int y, int x) {
            return y >= 0 && y < height && x >= 0 && x < width;
        }

        int inputOffset(int n, int c, int y, int x) {
            return inputBaseOffset
                    + n * inputStrides[0]
                    + c * inputStrides[1]
                    + y * inputStrides[2]
                    + x * inputStrides[3];
        }

        int outputOffset(int n, int columnChannel, int windowIndex) {
            return outBaseOffset
                    + n * outStrides[0]
                    + columnChannel * outStrides[1]
                    + windowIndex * outStrides[2];
        }
    }
}
