package backend.cpu.kernels.layout;

import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import operations.layout.fold2d;
import tensor.DataType;
import tensor.options.Window2dOptions;

public final class CpuFold2dKernel implements CpuStorageAwareKernel {
    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        if (!(call.operation() instanceof fold2d foldOp)) {
            throw new IllegalArgumentException("CpuFold2dKernel requires fold2d operation.");
        }
        LayoutStorageSupport.validateInputViews(1, call.inputs(), "fold2d");
        CpuStorageView input = call.inputs().getFirst();
        CpuStorageView out = call.output();
        LayoutStorageSupport.validateView(input, out.dtype(), "input");
        LayoutStorageSupport.validateView(out, out.dtype(), "output");
        validateFloating(out.dtype(), "fold2d");
        switch (out.dtype()) {
            case FLOAT64 -> foldF64(foldOp, input, out);
            case FLOAT32 -> foldF32(foldOp, input, out);
            case BFLOAT16 -> foldBF16(foldOp, input, out);
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("fold2d supports floating dtypes only.");
        }
        return CpuKernelResult.completed();
    }

    private static void foldF64(fold2d op, CpuStorageView input, CpuStorageView out) {
        FoldPlan plan = FoldPlan.create(op.getOptions(), input, out);
        double[] acc = accumulateF64(plan, input);
        writeF64(out, plan, acc);
    }

    private static void foldF32(fold2d op, CpuStorageView input, CpuStorageView out) {
        FoldPlan plan = FoldPlan.create(op.getOptions(), input, out);
        double[] acc = accumulateF32(plan, input);
        writeF32(out, plan, acc);
    }

    private static void foldBF16(fold2d op, CpuStorageView input, CpuStorageView out) {
        FoldPlan plan = FoldPlan.create(op.getOptions(), input, out);
        double[] acc = accumulateBF16(plan, input);
        writeBF16(out, plan, acc);
    }

    private static double[] accumulateF64(FoldPlan plan, CpuStorageView input) {
        double[] acc = new double[plan.outputSize];
        for (int n = 0; n < plan.batch; n++) {
            for (int c = 0; c < plan.channels; c++) {
                for (int kh = 0; kh < plan.kernelH; kh++) {
                    for (int kw = 0; kw < plan.kernelW; kw++) {
                        int columnChannel = c * plan.kernelArea + kh * plan.kernelW + kw;
                        for (int oy = 0; oy < plan.outH; oy++) {
                            int iy = oy * plan.options.strideH() - plan.options.padH() + kh * plan.options.dilationH();
                            for (int ox = 0; ox < plan.outW; ox++) {
                                int ix = ox * plan.options.strideW() - plan.options.padW() + kw * plan.options.dilationW();
                                if (!plan.inBounds(iy, ix)) {
                                    continue;
                                }
                                acc[plan.outputLogical(n, c, iy, ix)] +=
                                        LayoutStorageSupport.readF64(input, plan.inputOffset(n, columnChannel, oy * plan.outW + ox));
                            }
                        }
                    }
                }
            }
        }
        return acc;
    }

    private static double[] accumulateF32(FoldPlan plan, CpuStorageView input) {
        double[] acc = new double[plan.outputSize];
        for (int n = 0; n < plan.batch; n++) {
            for (int c = 0; c < plan.channels; c++) {
                for (int kh = 0; kh < plan.kernelH; kh++) {
                    for (int kw = 0; kw < plan.kernelW; kw++) {
                        int columnChannel = c * plan.kernelArea + kh * plan.kernelW + kw;
                        for (int oy = 0; oy < plan.outH; oy++) {
                            int iy = oy * plan.options.strideH() - plan.options.padH() + kh * plan.options.dilationH();
                            for (int ox = 0; ox < plan.outW; ox++) {
                                int ix = ox * plan.options.strideW() - plan.options.padW() + kw * plan.options.dilationW();
                                if (!plan.inBounds(iy, ix)) {
                                    continue;
                                }
                                acc[plan.outputLogical(n, c, iy, ix)] +=
                                        LayoutStorageSupport.readF32(input, plan.inputOffset(n, columnChannel, oy * plan.outW + ox));
                            }
                        }
                    }
                }
            }
        }
        return acc;
    }

    private static double[] accumulateBF16(FoldPlan plan, CpuStorageView input) {
        double[] acc = new double[plan.outputSize];
        for (int n = 0; n < plan.batch; n++) {
            for (int c = 0; c < plan.channels; c++) {
                for (int kh = 0; kh < plan.kernelH; kh++) {
                    for (int kw = 0; kw < plan.kernelW; kw++) {
                        int columnChannel = c * plan.kernelArea + kh * plan.kernelW + kw;
                        for (int oy = 0; oy < plan.outH; oy++) {
                            int iy = oy * plan.options.strideH() - plan.options.padH() + kh * plan.options.dilationH();
                            for (int ox = 0; ox < plan.outW; ox++) {
                                int ix = ox * plan.options.strideW() - plan.options.padW() + kw * plan.options.dilationW();
                                if (!plan.inBounds(iy, ix)) {
                                    continue;
                                }
                                acc[plan.outputLogical(n, c, iy, ix)] +=
                                        LayoutStorageSupport.readBF16AsF32(input, plan.inputOffset(n, columnChannel, oy * plan.outW + ox));
                            }
                        }
                    }
                }
            }
        }
        return acc;
    }

    private static void writeF64(CpuStorageView out, FoldPlan plan, double[] acc) {
        for (int logical = 0; logical < acc.length; logical++) {
            LayoutStorageSupport.writeF64(out, plan.outputOffsetForLogical(logical), acc[logical]);
        }
    }

    private static void writeF32(CpuStorageView out, FoldPlan plan, double[] acc) {
        for (int logical = 0; logical < acc.length; logical++) {
            LayoutStorageSupport.writeF32(out, plan.outputOffsetForLogical(logical), (float) acc[logical]);
        }
    }

    private static void writeBF16(CpuStorageView out, FoldPlan plan, double[] acc) {
        for (int logical = 0; logical < acc.length; logical++) {
            LayoutStorageSupport.writeBF16(out, plan.outputOffsetForLogical(logical), (float) acc[logical]);
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

    private static final class FoldPlan {
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
        private final int outputSize;
        private final int[] inputStrides;
        private final int inputBaseOffset;
        private final int[] outShape;
        private final int[] outDense;
        private final int[] outStrides;
        private final int outBaseOffset;

        private FoldPlan(
                Window2dOptions options,
                int batch,
                int channels,
                int height,
                int width,
                int kernelH,
                int kernelW,
                int outH,
                int outW,
                int outputSize,
                int[] inputStrides,
                int inputBaseOffset,
                int[] outShape,
                int[] outDense,
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
            this.outputSize = outputSize;
            this.inputStrides = inputStrides;
            this.inputBaseOffset = inputBaseOffset;
            this.outShape = outShape;
            this.outDense = outDense;
            this.outStrides = outStrides;
            this.outBaseOffset = outBaseOffset;
        }

        static FoldPlan create(Window2dOptions options, CpuStorageView input, CpuStorageView out) {
            int[] inputShape = input.shape();
            int[] outShape = out.shape();
            if (inputShape.length != 3 || outShape.length != 4) {
                throw new IllegalArgumentException("fold2d expects rank-3 input and NCHW output.");
            }
            int outW = inferOutW(outShape[3], options);
            int outH = inputShape[2] / outW;
            return new FoldPlan(
                    options,
                    outShape[0],
                    outShape[1],
                    outShape[2],
                    outShape[3],
                    options.kernelH(),
                    options.kernelW(),
                    outH,
                    outW,
                    out.logicalSize(),
                    input.strides(),
                    input.storageOffset(),
                    outShape,
                    LayoutStorageSupport.denseStrides(outShape),
                    out.strides(),
                    out.storageOffset());
        }

        boolean inBounds(int y, int x) {
            return y >= 0 && y < height && x >= 0 && x < width;
        }

        int inputOffset(int n, int columnChannel, int windowIndex) {
            return inputBaseOffset
                    + n * inputStrides[0]
                    + columnChannel * inputStrides[1]
                    + windowIndex * inputStrides[2];
        }

        int outputLogical(int n, int c, int y, int x) {
            return ((n * channels + c) * height + y) * width + x;
        }

        int outputOffsetForLogical(int logical) {
            return LayoutStorageSupport.offsetForLogical(logical, outShape, outDense, outStrides, outBaseOffset);
        }
    }
}
