package backend.cpu.kernels.nn;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.*;
import backend.cpu.kernels.nn.conv2d.plan.ResolvedConv2dHints;

import backend.blas.BlasProvider;
import backend.blas.OpenBlasArrayGemm;
import backend.blas.OpenBlasRuntime;
import graph.execution.trace.ConvTraceMetadata;
import operations.nn.conv.conv2dGemm;
import operations.nn.conv.conv2dBackwardInput;
import operations.nn.conv.conv2dBackwardInputGemm;
import operations.nn.conv.conv2dBackwardWeight;
import operations.nn.conv.conv2dBackwardWeightGemm;
import tensor.options.Conv2dOptions;
import tensor.Tensor;

final class Conv2dGemmBackend {
    private static final String CONV_KIND_GEMM = "GEMM";

    private static final class ConvBlasStats {
        private int blasCalls;
        private int javaCalls;

        void recordBlas() {
            blasCalls++;
        }

        void recordJava() {
            javaCalls++;
        }
    }

    private Conv2dGemmBackend() {
    }

    static void backwardWeightF64(conv2dBackwardWeight op, Tensor input, Tensor outGrad, Tensor gradWeight) {
        runBackwardWeightF64(
                TensorInternalAccess.float64Data(input),
                input.getShapeUnsafe(),
                TensorInternalAccess.float64Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.float64Data(gradWeight),
                op.getWeightShape(),
                op.getOptions(),
                null,
                gradWeight
        );
    }

    static void backwardWeightF64(conv2dBackwardWeightGemm op, Tensor input, Tensor outGrad, Tensor gradWeight, CpuKernelContext context) {
        runBackwardWeightF64(
                TensorInternalAccess.float64Data(input),
                input.getShapeUnsafe(),
                TensorInternalAccess.float64Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.float64Data(gradWeight),
                op.getWeightShape(),
                op.getOptions(),
                context,
                gradWeight
        );
    }

    static void backwardWeightF32(conv2dBackwardWeight op, Tensor input, Tensor outGrad, Tensor gradWeight) {
        runBackwardWeightF32(
                TensorInternalAccess.float32Data(input),
                input.getShapeUnsafe(),
                TensorInternalAccess.float32Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.float32Data(gradWeight),
                op.getWeightShape(),
                op.getOptions(),
                null,
                gradWeight
        );
    }

    static void backwardWeightF32(conv2dBackwardWeightGemm op, Tensor input, Tensor outGrad, Tensor gradWeight, CpuKernelContext context) {
        runBackwardWeightF32(
                TensorInternalAccess.float32Data(input),
                input.getShapeUnsafe(),
                TensorInternalAccess.float32Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.float32Data(gradWeight),
                op.getWeightShape(),
                op.getOptions(),
                context,
                gradWeight
        );
    }

    static void backwardWeightBF16(conv2dBackwardWeight op, Tensor input, Tensor outGrad, Tensor gradWeight) {
        runBackwardWeightBF16(
                TensorInternalAccess.bfloat16Data(input),
                input.getShapeUnsafe(),
                TensorInternalAccess.bfloat16Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.bfloat16Data(gradWeight),
                op.getWeightShape(),
                op.getOptions(),
                null,
                gradWeight
        );
    }

    static void backwardWeightBF16(conv2dBackwardWeightGemm op, Tensor input, Tensor outGrad, Tensor gradWeight, CpuKernelContext context) {
        runBackwardWeightBF16(
                TensorInternalAccess.bfloat16Data(input),
                input.getShapeUnsafe(),
                TensorInternalAccess.bfloat16Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.bfloat16Data(gradWeight),
                op.getWeightShape(),
                op.getOptions(),
                context,
                gradWeight
        );
    }

    static void backwardInputF64(conv2dBackwardInput op, Tensor weight, Tensor outGrad, Tensor gradInput) {
        runBackwardInputF64(
                TensorInternalAccess.float64Data(weight),
                weight.getShapeUnsafe(),
                TensorInternalAccess.float64Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.float64Data(gradInput),
                op.getInputShape(),
                op.getOptions(),
                null,
                gradInput
        );
    }

    static void backwardInputF64(conv2dBackwardInputGemm op, Tensor weight, Tensor outGrad, Tensor gradInput, CpuKernelContext context) {
        runBackwardInputF64(
                TensorInternalAccess.float64Data(weight),
                weight.getShapeUnsafe(),
                TensorInternalAccess.float64Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.float64Data(gradInput),
                op.getInputShape(),
                op.getOptions(),
                context,
                gradInput
        );
    }

    static void backwardInputF32(conv2dBackwardInput op, Tensor weight, Tensor outGrad, Tensor gradInput) {
        runBackwardInputF32(
                TensorInternalAccess.float32Data(weight),
                weight.getShapeUnsafe(),
                TensorInternalAccess.float32Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.float32Data(gradInput),
                op.getInputShape(),
                op.getOptions(),
                null,
                gradInput
        );
    }

    static void backwardInputF32(conv2dBackwardInputGemm op, Tensor weight, Tensor outGrad, Tensor gradInput, CpuKernelContext context) {
        runBackwardInputF32(
                TensorInternalAccess.float32Data(weight),
                weight.getShapeUnsafe(),
                TensorInternalAccess.float32Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.float32Data(gradInput),
                op.getInputShape(),
                op.getOptions(),
                context,
                gradInput
        );
    }

    static void backwardInputBF16(conv2dBackwardInput op, Tensor weight, Tensor outGrad, Tensor gradInput) {
        runBackwardInputBF16(
                TensorInternalAccess.bfloat16Data(weight),
                weight.getShapeUnsafe(),
                TensorInternalAccess.bfloat16Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.bfloat16Data(gradInput),
                op.getInputShape(),
                op.getOptions(),
                null,
                gradInput
        );
    }

    static void backwardInputBF16(conv2dBackwardInputGemm op, Tensor weight, Tensor outGrad, Tensor gradInput, CpuKernelContext context) {
        runBackwardInputBF16(
                TensorInternalAccess.bfloat16Data(weight),
                weight.getShapeUnsafe(),
                TensorInternalAccess.bfloat16Data(outGrad),
                outGrad.getShapeUnsafe(),
                TensorInternalAccess.bfloat16Data(gradInput),
                op.getInputShape(),
                op.getOptions(),
                context,
                gradInput
        );
    }

    static void forwardF64(conv2dGemm op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        runF64(TensorInternalAccess.float64Data(input), input.getShapeUnsafe(), TensorInternalAccess.float64Data(weight), weight.getShapeUnsafe(),
                bias == null ? null : TensorInternalAccess.float64Data(bias), TensorInternalAccess.float64Data(out), out.getShapeUnsafe(), op.getOptions(), context, out);
    }

    static void forwardF32(conv2dGemm op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        runF32(TensorInternalAccess.float32Data(input), input.getShapeUnsafe(), TensorInternalAccess.float32Data(weight), weight.getShapeUnsafe(),
                bias == null ? null : TensorInternalAccess.float32Data(bias), TensorInternalAccess.float32Data(out), out.getShapeUnsafe(), op.getOptions(), context, out);
    }

    static void forwardBF16(conv2dGemm op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        runBF16(TensorInternalAccess.bfloat16Data(input), input.getShapeUnsafe(), TensorInternalAccess.bfloat16Data(weight), weight.getShapeUnsafe(),
                bias == null ? null : TensorInternalAccess.bfloat16Data(bias), TensorInternalAccess.bfloat16Data(out),
                out.getShapeUnsafe(),
                context == null || context.cpuWorkspace() == null ? null : context.cpuWorkspace().requireFloatWorkspace(),
                op.getOptions(),
                context,
                out);
    }

    private static void runF64(
            double[] input, int[] inputShape,
            double[] weight, int[] weightShape,
            double[] bias,
            double[] out, int[] outShape,
            Conv2dOptions options,
            CpuKernelContext context,
            Tensor node
    ) {
        int batch = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int outChannelsPerGroup = outChannels / options.groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        int outSpatial = outH * outW;
        ResolvedConv2dHints hints = requireHints(context, node, outSpatial, outChannelsPerGroup, kSize);
        boolean useBlas = hints.useBlas();
        ConvBlasStats stats = new ConvBlasStats();

        double[] packedWeight = new double[kSize * outChannelsPerGroup];
        double[] im2col = new double[outSpatial * kSize];
        double[] gemmOut = new double[outSpatial * outChannelsPerGroup];

        for (int b = 0; b < batch; b++) {
            for (int g = 0; g < options.groups(); g++) {
                packWeightF64(weight, weightShape, g, outChannelsPerGroup, channelsPerGroup, kernelH, kernelW, packedWeight);
                fillIm2colF64(input, inputShape, b, g, outH, outW, channelsPerGroup, kernelH, kernelW, options, im2col);
                if (!(useBlas && tryBlasF64(im2col, packedWeight, gemmOut, outSpatial, outChannelsPerGroup, kSize))) {
                    runJavaGemmF64(im2col, packedWeight, gemmOut, outSpatial, outChannelsPerGroup, kSize);
                    stats.recordJava();
                } else {
                    stats.recordBlas();
                }
                scatterOutputF64(gemmOut, out, bias, b, g, outChannelsPerGroup, outH, outW, outChannels);
            }
        }
        publishGemmTrace(node, hints, stats, context);
    }

    private static void runF32(
            float[] input, int[] inputShape,
            float[] weight, int[] weightShape,
            float[] bias,
            float[] out, int[] outShape,
            Conv2dOptions options,
            CpuKernelContext context,
            Tensor node
    ) {
        int batch = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int outChannelsPerGroup = outChannels / options.groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        int outSpatial = outH * outW;
        ResolvedConv2dHints hints = requireHints(context, node, outSpatial, outChannelsPerGroup, kSize);
        boolean useBlas = hints.useBlas();
        ConvBlasStats stats = new ConvBlasStats();

        float[] packedWeight = new float[kSize * outChannelsPerGroup];
        float[] im2col = new float[outSpatial * kSize];
        float[] gemmOut = new float[outSpatial * outChannelsPerGroup];

        for (int b = 0; b < batch; b++) {
            for (int g = 0; g < options.groups(); g++) {
                packWeightF32(weight, weightShape, g, outChannelsPerGroup, channelsPerGroup, kernelH, kernelW, packedWeight);
                fillIm2colF32(input, inputShape, b, g, outH, outW, channelsPerGroup, kernelH, kernelW, options, im2col);
                if (!(useBlas && tryBlasF32(im2col, packedWeight, gemmOut, outSpatial, outChannelsPerGroup, kSize))) {
                    runJavaGemmF32(im2col, packedWeight, gemmOut, outSpatial, outChannelsPerGroup, kSize);
                    stats.recordJava();
                } else {
                    stats.recordBlas();
                }
                scatterOutputF32(gemmOut, out, bias, b, g, outChannelsPerGroup, outH, outW, outChannels);
            }
        }
        publishGemmTrace(node, hints, stats, context);
    }

    private static void runBF16(
            short[] input, int[] inputShape,
            short[] weight, int[] weightShape,
            short[] bias,
            short[] out, int[] outShape,
            float[] workspace,
            Conv2dOptions options,
            CpuKernelContext context,
            Tensor node
    ) {
        int batch = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outShape[2];
        int outW = outShape[3];
        int outChannelsPerGroup = outChannels / options.groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        int outSpatial = outH * outW;
        ResolvedConv2dHints hints = requireHints(context, node, outSpatial, outChannelsPerGroup, kSize);
        boolean useBlas = hints.useBlas();
        ConvBlasStats stats = new ConvBlasStats();

        short[] packedWeight = new short[kSize * outChannelsPerGroup];
        short[] im2col = new short[outSpatial * kSize];
        float[] gemmOut = workspace != null && workspace.length >= outSpatial * outChannelsPerGroup
                ? workspace
                : new float[outSpatial * outChannelsPerGroup];

        for (int b = 0; b < batch; b++) {
            for (int g = 0; g < options.groups(); g++) {
                packWeightBF16(weight, weightShape, g, outChannelsPerGroup, channelsPerGroup, kernelH, kernelW, packedWeight);
                fillIm2colBF16(input, inputShape, b, g, outH, outW, channelsPerGroup, kernelH, kernelW, options, im2col);
                if (!(useBlas && tryBlasBF16(im2col, packedWeight, gemmOut, outSpatial, outChannelsPerGroup, kSize))) {
                    float[] packedWeightF32 = new float[packedWeight.length];
                    float[] im2colF32 = new float[im2col.length];
                    for (int i = 0; i < packedWeight.length; i++) {
                        packedWeightF32[i] = CpuDTypeOps.fromBFloat16Bits(packedWeight[i]);
                    }
                    for (int i = 0; i < im2col.length; i++) {
                        im2colF32[i] = CpuDTypeOps.fromBFloat16Bits(im2col[i]);
                    }
                    runJavaGemmF32(im2colF32, packedWeightF32, gemmOut, outSpatial, outChannelsPerGroup, kSize);
                    stats.recordJava();
                } else {
                    stats.recordBlas();
                }
                scatterOutputBF16(gemmOut, out, bias, b, g, outChannelsPerGroup, outH, outW, outChannels);
            }
        }
        publishGemmTrace(node, hints, stats, context);
    }

    private static void runBackwardWeightF64(
            double[] input,
            int[] inputShape,
            double[] outGrad,
            int[] outGradShape,
            double[] gradWeight,
            int[] weightShape,
            Conv2dOptions options,
            CpuKernelContext context,
            Tensor node
    ) {
        java.util.Arrays.fill(gradWeight, 0.0d);
        int batch = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / options.groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        int outSpatial = outH * outW;
        ResolvedConv2dHints hints = requireHints(context, node, kSize, outChannelsPerGroup, outSpatial);
        boolean useBlas = hints.useBlas();
        ConvBlasStats stats = new ConvBlasStats();

        double[] im2col = new double[outSpatial * kSize];
        double[] im2colTransposed = new double[kSize * outSpatial];
        double[] outGradRows = new double[outSpatial * outChannelsPerGroup];
        double[] packedGradWeight = new double[kSize * outChannelsPerGroup];

        for (int g = 0; g < options.groups(); g++) {
            java.util.Arrays.fill(packedGradWeight, 0.0d);
            for (int b = 0; b < batch; b++) {
                fillIm2colF64(input, inputShape, b, g, outH, outW, channelsPerGroup, kernelH, kernelW, options, im2col);
                transposeRowsToColumnsF64(im2col, im2colTransposed, outSpatial, kSize);
                fillOutGradRowsF64(outGrad, outGradShape, b, g, outH, outW, outChannelsPerGroup, outGradRows);
                if (!(useBlas && accumulateBlasF64(im2colTransposed, outGradRows, packedGradWeight, kSize, outChannelsPerGroup, outSpatial))) {
                    accumulateJavaGemmF64(im2colTransposed, outGradRows, packedGradWeight, kSize, outChannelsPerGroup, outSpatial);
                    stats.recordJava();
                } else {
                    stats.recordBlas();
                }
            }
            unpackWeightF64(packedGradWeight, gradWeight, weightShape, g, outChannelsPerGroup, channelsPerGroup, kernelH, kernelW);
        }
        publishGemmTrace(node, hints, stats, context);
    }

    private static void runBackwardWeightF32(
            float[] input,
            int[] inputShape,
            float[] outGrad,
            int[] outGradShape,
            float[] gradWeight,
            int[] weightShape,
            Conv2dOptions options,
            CpuKernelContext context,
            Tensor node
    ) {
        java.util.Arrays.fill(gradWeight, 0.0f);
        int batch = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / options.groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        int outSpatial = outH * outW;
        ResolvedConv2dHints hints = requireHints(context, node, kSize, outChannelsPerGroup, outSpatial);
        boolean useBlas = hints.useBlas();
        ConvBlasStats stats = new ConvBlasStats();

        float[] im2col = new float[outSpatial * kSize];
        float[] im2colTransposed = new float[kSize * outSpatial];
        float[] outGradRows = new float[outSpatial * outChannelsPerGroup];
        float[] packedGradWeight = new float[kSize * outChannelsPerGroup];

        for (int g = 0; g < options.groups(); g++) {
            java.util.Arrays.fill(packedGradWeight, 0.0f);
            for (int b = 0; b < batch; b++) {
                fillIm2colF32(input, inputShape, b, g, outH, outW, channelsPerGroup, kernelH, kernelW, options, im2col);
                transposeRowsToColumnsF32(im2col, im2colTransposed, outSpatial, kSize);
                fillOutGradRowsF32(outGrad, outGradShape, b, g, outH, outW, outChannelsPerGroup, outGradRows);
                if (!(useBlas && accumulateBlasF32(im2colTransposed, outGradRows, packedGradWeight, kSize, outChannelsPerGroup, outSpatial))) {
                    accumulateJavaGemmF32(im2colTransposed, outGradRows, packedGradWeight, kSize, outChannelsPerGroup, outSpatial);
                    stats.recordJava();
                } else {
                    stats.recordBlas();
                }
            }
            unpackWeightF32(packedGradWeight, gradWeight, weightShape, g, outChannelsPerGroup, channelsPerGroup, kernelH, kernelW);
        }
        publishGemmTrace(node, hints, stats, context);
    }

    private static void runBackwardWeightBF16(
            short[] input,
            int[] inputShape,
            short[] outGrad,
            int[] outGradShape,
            short[] gradWeight,
            int[] weightShape,
            Conv2dOptions options,
            CpuKernelContext context,
            Tensor node
    ) {
        int batch = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / options.groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        int outSpatial = outH * outW;
        ResolvedConv2dHints hints = requireHints(context, node, kSize, outChannelsPerGroup, outSpatial);
        boolean useBlas = hints.useBlas();
        ConvBlasStats stats = new ConvBlasStats();

        short[] im2col = new short[outSpatial * kSize];
        short[] im2colTransposed = new short[kSize * outSpatial];
        short[] outGradRows = new short[outSpatial * outChannelsPerGroup];
        float[] packedGradWeight = new float[kSize * outChannelsPerGroup];

        for (int g = 0; g < options.groups(); g++) {
            java.util.Arrays.fill(packedGradWeight, 0.0f);
            for (int b = 0; b < batch; b++) {
                fillIm2colBF16(input, inputShape, b, g, outH, outW, channelsPerGroup, kernelH, kernelW, options, im2col);
                transposeRowsToColumnsBF16(im2col, im2colTransposed, outSpatial, kSize);
                fillOutGradRowsBF16(outGrad, outGradShape, b, g, outH, outW, outChannelsPerGroup, outGradRows);
                if (!(useBlas && accumulateBlasBF16(im2colTransposed, outGradRows, packedGradWeight, kSize, outChannelsPerGroup, outSpatial))) {
                    accumulateJavaGemmBF16(im2colTransposed, outGradRows, packedGradWeight, kSize, outChannelsPerGroup, outSpatial);
                    stats.recordJava();
                } else {
                    stats.recordBlas();
                }
            }
            unpackWeightBF16(packedGradWeight, gradWeight, weightShape, g, outChannelsPerGroup, channelsPerGroup, kernelH, kernelW);
        }
        publishGemmTrace(node, hints, stats, context);
    }

    private static void runBackwardInputF64(
            double[] weight,
            int[] weightShape,
            double[] outGrad,
            int[] outGradShape,
            double[] gradInput,
            int[] inputShape,
            Conv2dOptions options,
            CpuKernelContext context,
            Tensor node
    ) {
        java.util.Arrays.fill(gradInput, 0.0d);
        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / options.groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        int outSpatial = outH * outW;
        ResolvedConv2dHints hints = requireHints(context, node, outSpatial, kSize, outChannelsPerGroup);
        boolean useBlas = hints.useBlas();
        ConvBlasStats stats = new ConvBlasStats();

        double[] packedWeight = new double[kSize * outChannelsPerGroup];
        double[] packedWeightTransposed = new double[outChannelsPerGroup * kSize];
        double[] outGradRows = new double[outSpatial * outChannelsPerGroup];
        double[] col = new double[outSpatial * kSize];

        for (int b = 0; b < batch; b++) {
            for (int g = 0; g < options.groups(); g++) {
                packWeightF64(weight, weightShape, g, outChannelsPerGroup, channelsPerGroup, kernelH, kernelW, packedWeight);
                transposeRowsToColumnsF64(packedWeight, packedWeightTransposed, kSize, outChannelsPerGroup);
                fillOutGradRowsF64(outGrad, outGradShape, b, g, outH, outW, outChannelsPerGroup, outGradRows);
                if (!(useBlas && tryBlasF64(outGradRows, packedWeightTransposed, col, outSpatial, kSize, outChannelsPerGroup))) {
                    runJavaGemmF64(outGradRows, packedWeightTransposed, col, outSpatial, kSize, outChannelsPerGroup);
                    stats.recordJava();
                } else {
                    stats.recordBlas();
                }
                accumulateColToGradInputF64(col, gradInput, inputShape, b, g, outH, outW, channelsPerGroup, kernelH, kernelW, options);
            }
        }
        publishGemmTrace(node, hints, stats, context);
    }

    private static void runBackwardInputF32(
            float[] weight,
            int[] weightShape,
            float[] outGrad,
            int[] outGradShape,
            float[] gradInput,
            int[] inputShape,
            Conv2dOptions options,
            CpuKernelContext context,
            Tensor node
    ) {
        java.util.Arrays.fill(gradInput, 0.0f);
        int batch = inputShape[0];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / options.groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        int outSpatial = outH * outW;
        ResolvedConv2dHints hints = requireHints(context, node, outSpatial, kSize, outChannelsPerGroup);
        boolean useBlas = hints.useBlas();
        ConvBlasStats stats = new ConvBlasStats();

        float[] packedWeight = new float[kSize * outChannelsPerGroup];
        float[] packedWeightTransposed = new float[outChannelsPerGroup * kSize];
        float[] outGradRows = new float[outSpatial * outChannelsPerGroup];
        float[] col = new float[outSpatial * kSize];

        for (int b = 0; b < batch; b++) {
            for (int g = 0; g < options.groups(); g++) {
                packWeightF32(weight, weightShape, g, outChannelsPerGroup, channelsPerGroup, kernelH, kernelW, packedWeight);
                transposeRowsToColumnsF32(packedWeight, packedWeightTransposed, kSize, outChannelsPerGroup);
                fillOutGradRowsF32(outGrad, outGradShape, b, g, outH, outW, outChannelsPerGroup, outGradRows);
                if (!(useBlas && tryBlasF32(outGradRows, packedWeightTransposed, col, outSpatial, kSize, outChannelsPerGroup))) {
                    runJavaGemmF32(outGradRows, packedWeightTransposed, col, outSpatial, kSize, outChannelsPerGroup);
                    stats.recordJava();
                } else {
                    stats.recordBlas();
                }
                accumulateColToGradInputF32(col, gradInput, inputShape, b, g, outH, outW, channelsPerGroup, kernelH, kernelW, options);
            }
        }
        publishGemmTrace(node, hints, stats, context);
    }

    private static void runBackwardInputBF16(
            short[] weight,
            int[] weightShape,
            short[] outGrad,
            int[] outGradShape,
            short[] gradInput,
            int[] inputShape,
            Conv2dOptions options,
            CpuKernelContext context,
            Tensor node
    ) {
        int batch = inputShape[0];
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int outChannels = weightShape[0];
        int channelsPerGroup = weightShape[1];
        int kernelH = weightShape[2];
        int kernelW = weightShape[3];
        int outH = outGradShape[2];
        int outW = outGradShape[3];
        int outChannelsPerGroup = outChannels / options.groups();
        int kSize = channelsPerGroup * kernelH * kernelW;
        int outSpatial = outH * outW;
        ResolvedConv2dHints hints = requireHints(context, node, outSpatial, kSize, outChannelsPerGroup);
        boolean useBlas = hints.useBlas();
        ConvBlasStats stats = new ConvBlasStats();

        short[] packedWeight = new short[kSize * outChannelsPerGroup];
        short[] packedWeightTransposed = new short[outChannelsPerGroup * kSize];
        short[] outGradRows = new short[outSpatial * outChannelsPerGroup];
        float[] col = new float[outSpatial * kSize];
        float[] gradInputAccum = new float[batch * inChannels * inH * inW];

        for (int b = 0; b < batch; b++) {
            for (int g = 0; g < options.groups(); g++) {
                packWeightBF16(weight, weightShape, g, outChannelsPerGroup, channelsPerGroup, kernelH, kernelW, packedWeight);
                transposeRowsToColumnsBF16(packedWeight, packedWeightTransposed, kSize, outChannelsPerGroup);
                fillOutGradRowsBF16(outGrad, outGradShape, b, g, outH, outW, outChannelsPerGroup, outGradRows);
                if (!(useBlas && tryBlasBF16(outGradRows, packedWeightTransposed, col, outSpatial, kSize, outChannelsPerGroup))) {
                    accumulateJavaGemmBF16NoAccum(outGradRows, packedWeightTransposed, col, outSpatial, kSize, outChannelsPerGroup);
                    stats.recordJava();
                } else {
                    stats.recordBlas();
                }
                accumulateColToGradInputBF16(col, gradInputAccum, inputShape, b, g, outH, outW, channelsPerGroup, kernelH, kernelW, options);
            }
        }

        for (int i = 0; i < gradInput.length; i++) {
            gradInput[i] = CpuDTypeOps.toBFloat16Bits(gradInputAccum[i]);
        }
        publishGemmTrace(node, hints, stats, context);
    }

    private static ResolvedConv2dHints requireHints(CpuKernelContext context, Tensor node, int expectedM, int expectedN, int expectedK) {
        if (context == null || context.conv2dHints() == null) {
            throw new IllegalStateException("Missing ResolvedConv2dHints for node " + node.getLabel());
        }
        ResolvedConv2dHints hints = context.conv2dHints();
        if (hints.m() != expectedM || hints.n() != expectedN || hints.k() != expectedK) {
            throw new IllegalStateException(
                    "Prepared conv2d GEMM hints do not match runtime shape for node " + node.getLabel()
                            + ": expected m/n/k=" + expectedM + "/" + expectedN + "/" + expectedK
                            + ", prepared=" + hints.m() + "/" + hints.n() + "/" + hints.k()
            );
        }
        if (hints.useBlas() && hints.provider() == BlasProvider.OPENBLAS_FFM && !OpenBlasRuntime.isAvailable()) {
            throw new IllegalStateException("Prepared conv2d GEMM plan requires OPENBLAS_FFM, but the bridge is not available.");
        }
        return hints;
    }

    private static void publishGemmTrace(Tensor node, ResolvedConv2dHints hints, ConvBlasStats stats, CpuKernelContext context) {
        if (context == null) {
            return;
        }
        String provider = hints == null ? "NONE" : hints.provider().name();
        context.publishConvTrace(node, new ConvTraceMetadata(
                CONV_KIND_GEMM,
                true,
                stats != null && stats.blasCalls > 0,
                provider,
                hints == null ? 0 : hints.m(),
                hints == null ? 0 : hints.n(),
                hints == null ? 0 : hints.k(),
                stats == null ? 0 : stats.blasCalls,
                stats == null ? 0 : stats.javaCalls
        ));
    }

    private static void fillIm2colF64(double[] input, int[] inputShape, int batch, int group, int outH, int outW,
                                      int channelsPerGroup, int kernelH, int kernelW, Conv2dOptions options, double[] out) {
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int inChannelBase = group * channelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            int inOriginH = oh * options.strideH() - options.padH();
            for (int ow = 0; ow < outW; ow++) {
                int inOriginW = ow * options.strideW() - options.padW();
                int col = 0;
                for (int icg = 0; icg < channelsPerGroup; icg++) {
                    int ic = inChannelBase + icg;
                    for (int kh = 0; kh < kernelH; kh++) {
                        int ih = inOriginH + kh * options.dilationH();
                        for (int kw = 0; kw < kernelW; kw++) {
                            int iw = inOriginW + kw * options.dilationW();
                            out[row * (channelsPerGroup * kernelH * kernelW) + col++] =
                                    (ih < 0 || ih >= inH || iw < 0 || iw >= inW)
                                            ? 0.0d
                                            : input[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)];
                        }
                    }
                }
                row++;
            }
        }
    }

    private static void fillIm2colF32(float[] input, int[] inputShape, int batch, int group, int outH, int outW,
                                      int channelsPerGroup, int kernelH, int kernelW, Conv2dOptions options, float[] out) {
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int inChannelBase = group * channelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            int inOriginH = oh * options.strideH() - options.padH();
            for (int ow = 0; ow < outW; ow++) {
                int inOriginW = ow * options.strideW() - options.padW();
                int col = 0;
                for (int icg = 0; icg < channelsPerGroup; icg++) {
                    int ic = inChannelBase + icg;
                    for (int kh = 0; kh < kernelH; kh++) {
                        int ih = inOriginH + kh * options.dilationH();
                        for (int kw = 0; kw < kernelW; kw++) {
                            int iw = inOriginW + kw * options.dilationW();
                            out[row * (channelsPerGroup * kernelH * kernelW) + col++] =
                                    (ih < 0 || ih >= inH || iw < 0 || iw >= inW)
                                            ? 0.0f
                                            : input[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)];
                        }
                    }
                }
                row++;
            }
        }
    }

    private static void fillIm2colBF16(short[] input, int[] inputShape, int batch, int group, int outH, int outW,
                                      int channelsPerGroup, int kernelH, int kernelW, Conv2dOptions options, short[] out) {
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int inChannelBase = group * channelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            int inOriginH = oh * options.strideH() - options.padH();
            for (int ow = 0; ow < outW; ow++) {
                int inOriginW = ow * options.strideW() - options.padW();
                int col = 0;
                for (int icg = 0; icg < channelsPerGroup; icg++) {
                    int ic = inChannelBase + icg;
                    for (int kh = 0; kh < kernelH; kh++) {
                        int ih = inOriginH + kh * options.dilationH();
                        for (int kw = 0; kw < kernelW; kw++) {
                            int iw = inOriginW + kw * options.dilationW();
                            out[row * (channelsPerGroup * kernelH * kernelW) + col++] =
                                    (ih < 0 || ih >= inH || iw < 0 || iw >= inW)
                                            ? (short) 0
                                            : input[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)];
                        }
                    }
                }
                row++;
            }
        }
    }

    private static void packWeightF64(double[] weight, int[] weightShape, int group, int outChannelsPerGroup,
                                      int channelsPerGroup, int kernelH, int kernelW, double[] packed) {
        int outChannelBase = group * outChannelsPerGroup;
        for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
            int oc = outChannelBase + ocg;
            int kIndex = 0;
            for (int icg = 0; icg < channelsPerGroup; icg++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        packed[kIndex * outChannelsPerGroup + ocg] =
                                weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)];
                        kIndex++;
                    }
                }
            }
        }
    }

    private static void packWeightF32(float[] weight, int[] weightShape, int group, int outChannelsPerGroup,
                                      int channelsPerGroup, int kernelH, int kernelW, float[] packed) {
        int outChannelBase = group * outChannelsPerGroup;
        for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
            int oc = outChannelBase + ocg;
            int kIndex = 0;
            for (int icg = 0; icg < channelsPerGroup; icg++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        packed[kIndex * outChannelsPerGroup + ocg] =
                                weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)];
                        kIndex++;
                    }
                }
            }
        }
    }

    private static void packWeightBF16(short[] weight, int[] weightShape, int group, int outChannelsPerGroup,
                                      int channelsPerGroup, int kernelH, int kernelW, short[] packed) {
        int outChannelBase = group * outChannelsPerGroup;
        for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
            int oc = outChannelBase + ocg;
            int kIndex = 0;
            for (int icg = 0; icg < channelsPerGroup; icg++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        packed[kIndex * outChannelsPerGroup + ocg] =
                                weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)];
                        kIndex++;
                    }
                }
            }
        }
    }

    private static void fillOutGradRowsF64(
            double[] outGrad,
            int[] outGradShape,
            int batch,
            int group,
            int outH,
            int outW,
            int outChannelsPerGroup,
            double[] out
    ) {
        int outChannels = outGradShape[1];
        int outChannelBase = group * outChannelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    out[row * outChannelsPerGroup + ocg] = outGrad[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)];
                }
                row++;
            }
        }
    }

    private static void fillOutGradRowsF32(
            float[] outGrad,
            int[] outGradShape,
            int batch,
            int group,
            int outH,
            int outW,
            int outChannelsPerGroup,
            float[] out
    ) {
        int outChannels = outGradShape[1];
        int outChannelBase = group * outChannelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    out[row * outChannelsPerGroup + ocg] = outGrad[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)];
                }
                row++;
            }
        }
    }

    private static void fillOutGradRowsBF16(
            short[] outGrad,
            int[] outGradShape,
            int batch,
            int group,
            int outH,
            int outW,
            int outChannelsPerGroup,
            short[] out
    ) {
        int outChannels = outGradShape[1];
        int outChannelBase = group * outChannelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    out[row * outChannelsPerGroup + ocg] = outGrad[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)];
                }
                row++;
            }
        }
    }

    private static void transposeRowsToColumnsF64(double[] src, double[] dst, int rows, int cols) {
        for (int row = 0; row < rows; row++) {
            int srcBase = row * cols;
            for (int col = 0; col < cols; col++) {
                dst[col * rows + row] = src[srcBase + col];
            }
        }
    }

    private static void transposeRowsToColumnsF32(float[] src, float[] dst, int rows, int cols) {
        for (int row = 0; row < rows; row++) {
            int srcBase = row * cols;
            for (int col = 0; col < cols; col++) {
                dst[col * rows + row] = src[srcBase + col];
            }
        }
    }

    private static void transposeRowsToColumnsBF16(short[] src, short[] dst, int rows, int cols) {
        for (int row = 0; row < rows; row++) {
            int srcBase = row * cols;
            for (int col = 0; col < cols; col++) {
                dst[col * rows + row] = src[srcBase + col];
            }
        }
    }

    private static void accumulateColToGradInputF64(
            double[] col,
            double[] gradInput,
            int[] inputShape,
            int batch,
            int group,
            int outH,
            int outW,
            int channelsPerGroup,
            int kernelH,
            int kernelW,
            Conv2dOptions options
    ) {
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int inChannelBase = group * channelsPerGroup;
        int row = 0;
        int kSize = channelsPerGroup * kernelH * kernelW;
        for (int oh = 0; oh < outH; oh++) {
            int inOriginH = oh * options.strideH() - options.padH();
            for (int ow = 0; ow < outW; ow++) {
                int inOriginW = ow * options.strideW() - options.padW();
                int colIndex = 0;
                int rowBase = row * kSize;
                for (int icg = 0; icg < channelsPerGroup; icg++) {
                    int ic = inChannelBase + icg;
                    for (int kh = 0; kh < kernelH; kh++) {
                        int ih = inOriginH + kh * options.dilationH();
                        for (int kw = 0; kw < kernelW; kw++) {
                            int iw = inOriginW + kw * options.dilationW();
                            double value = col[rowBase + colIndex++];
                            if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) {
                                continue;
                            }
                            gradInput[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)] += value;
                        }
                    }
                }
                row++;
            }
        }
    }

    private static void accumulateColToGradInputF32(
            float[] col,
            float[] gradInput,
            int[] inputShape,
            int batch,
            int group,
            int outH,
            int outW,
            int channelsPerGroup,
            int kernelH,
            int kernelW,
            Conv2dOptions options
    ) {
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int inChannelBase = group * channelsPerGroup;
        int row = 0;
        int kSize = channelsPerGroup * kernelH * kernelW;
        for (int oh = 0; oh < outH; oh++) {
            int inOriginH = oh * options.strideH() - options.padH();
            for (int ow = 0; ow < outW; ow++) {
                int inOriginW = ow * options.strideW() - options.padW();
                int colIndex = 0;
                int rowBase = row * kSize;
                for (int icg = 0; icg < channelsPerGroup; icg++) {
                    int ic = inChannelBase + icg;
                    for (int kh = 0; kh < kernelH; kh++) {
                        int ih = inOriginH + kh * options.dilationH();
                        for (int kw = 0; kw < kernelW; kw++) {
                            int iw = inOriginW + kw * options.dilationW();
                            float value = col[rowBase + colIndex++];
                            if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) {
                                continue;
                            }
                            gradInput[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)] += value;
                        }
                    }
                }
                row++;
            }
        }
    }

    private static void accumulateColToGradInputBF16(
            float[] col,
            float[] gradInput,
            int[] inputShape,
            int batch,
            int group,
            int outH,
            int outW,
            int channelsPerGroup,
            int kernelH,
            int kernelW,
            Conv2dOptions options
    ) {
        int inChannels = inputShape[1];
        int inH = inputShape[2];
        int inW = inputShape[3];
        int inChannelBase = group * channelsPerGroup;
        int row = 0;
        int kSize = channelsPerGroup * kernelH * kernelW;
        for (int oh = 0; oh < outH; oh++) {
            int inOriginH = oh * options.strideH() - options.padH();
            for (int ow = 0; ow < outW; ow++) {
                int inOriginW = ow * options.strideW() - options.padW();
                int colIndex = 0;
                int rowBase = row * kSize;
                for (int icg = 0; icg < channelsPerGroup; icg++) {
                    int ic = inChannelBase + icg;
                    for (int kh = 0; kh < kernelH; kh++) {
                        int ih = inOriginH + kh * options.dilationH();
                        for (int kw = 0; kw < kernelW; kw++) {
                            int iw = inOriginW + kw * options.dilationW();
                            float value = col[rowBase + colIndex++];
                            if (ih < 0 || ih >= inH || iw < 0 || iw >= inW) {
                                continue;
                            }
                            gradInput[indexNCHW(batch, ic, ih, iw, inChannels, inH, inW)] += value;
                        }
                    }
                }
                row++;
            }
        }
    }

    private static boolean accumulateBlasF64(double[] a, double[] b, double[] c, int m, int n, int k) {
        if (!OpenBlasRuntime.isBFloat16ToFloatGemmAvailable()) {
            return false;
        }
        try {
            OpenBlasArrayGemm.dgemmRowMajorNoTrans(m, n, k, 1.0d, a, k, b, n, 1.0d, c, n);
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static boolean accumulateBlasF32(float[] a, float[] b, float[] c, int m, int n, int k) {
        if (!OpenBlasRuntime.isBFloat16ToFloatGemmAvailable()) {
            return false;
        }
        try {
            OpenBlasArrayGemm.sgemmRowMajorNoTrans(m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static boolean accumulateBlasBF16(short[] a, short[] b, float[] c, int m, int n, int k) {
        if (!OpenBlasRuntime.isAvailable()) {
            return false;
        }
        try {
            OpenBlasArrayGemm.sbgemmRowMajorNoTrans(m, n, k, 1.0f, a, k, b, n, 1.0f, c, n);
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static void accumulateJavaGemmF64(double[] a, double[] b, double[] c, int m, int n, int k) {
        for (int i = 0; i < m; i++) {
            int aRow = i * k;
            int cRow = i * n;
            for (int p = 0; p < k; p++) {
                double av = a[aRow + p];
                int bRow = p * n;
                for (int j = 0; j < n; j++) {
                    c[cRow + j] += av * b[bRow + j];
                }
            }
        }
    }

    private static void accumulateJavaGemmF32(float[] a, float[] b, float[] c, int m, int n, int k) {
        for (int i = 0; i < m; i++) {
            int aRow = i * k;
            int cRow = i * n;
            for (int p = 0; p < k; p++) {
                float av = a[aRow + p];
                int bRow = p * n;
                for (int j = 0; j < n; j++) {
                    c[cRow + j] += av * b[bRow + j];
                }
            }
        }
    }

    private static void accumulateJavaGemmBF16(short[] a, short[] b, float[] c, int m, int n, int k) {
        for (int i = 0; i < m; i++) {
            int aRow = i * k;
            int cRow = i * n;
            for (int p = 0; p < k; p++) {
                float av = CpuDTypeOps.fromBFloat16Bits(a[aRow + p]);
                int bRow = p * n;
                for (int j = 0; j < n; j++) {
                    c[cRow + j] += av * CpuDTypeOps.fromBFloat16Bits(b[bRow + j]);
                }
            }
        }
    }

    private static void accumulateJavaGemmBF16NoAccum(short[] a, short[] b, float[] c, int m, int n, int k) {
        java.util.Arrays.fill(c, 0.0f);
        accumulateJavaGemmBF16(a, b, c, m, n, k);
    }

    private static void unpackWeightF64(
            double[] packed,
            double[] weight,
            int[] weightShape,
            int group,
            int outChannelsPerGroup,
            int channelsPerGroup,
            int kernelH,
            int kernelW
    ) {
        int outChannelBase = group * outChannelsPerGroup;
        for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
            int oc = outChannelBase + ocg;
            int kIndex = 0;
            for (int icg = 0; icg < channelsPerGroup; icg++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)] =
                                packed[kIndex * outChannelsPerGroup + ocg];
                        kIndex++;
                    }
                }
            }
        }
    }

    private static void unpackWeightF32(
            float[] packed,
            float[] weight,
            int[] weightShape,
            int group,
            int outChannelsPerGroup,
            int channelsPerGroup,
            int kernelH,
            int kernelW
    ) {
        int outChannelBase = group * outChannelsPerGroup;
        for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
            int oc = outChannelBase + ocg;
            int kIndex = 0;
            for (int icg = 0; icg < channelsPerGroup; icg++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)] =
                                packed[kIndex * outChannelsPerGroup + ocg];
                        kIndex++;
                    }
                }
            }
        }
    }

    private static void unpackWeightBF16(
            float[] packed,
            short[] weight,
            int[] weightShape,
            int group,
            int outChannelsPerGroup,
            int channelsPerGroup,
            int kernelH,
            int kernelW
    ) {
        int outChannelBase = group * outChannelsPerGroup;
        for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
            int oc = outChannelBase + ocg;
            int kIndex = 0;
            for (int icg = 0; icg < channelsPerGroup; icg++) {
                for (int kh = 0; kh < kernelH; kh++) {
                    for (int kw = 0; kw < kernelW; kw++) {
                        weight[indexOIHW(oc, icg, kh, kw, channelsPerGroup, kernelH, kernelW)] =
                                CpuDTypeOps.toBFloat16Bits(packed[kIndex * outChannelsPerGroup + ocg]);
                        kIndex++;
                    }
                }
            }
        }
    }

    private static boolean tryBlasBF16(short[] a, short[] b, float[] c, int m, int n, int k) {
        if (!OpenBlasRuntime.isAvailable()) {
            return false;
        }
        try {
            OpenBlasArrayGemm.sbgemmRowMajorNoTrans(m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static boolean tryBlasF64(double[] a, double[] b, double[] c, int m, int n, int k) {
        if (!OpenBlasRuntime.isAvailable()) {
            return false;
        }
        try {
            OpenBlasArrayGemm.dgemmRowMajorNoTrans(m, n, k, 1.0d, a, k, b, n, 0.0d, c, n);
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static boolean tryBlasF32(float[] a, float[] b, float[] c, int m, int n, int k) {
        if (!OpenBlasRuntime.isAvailable()) {
            return false;
        }
        try {
            OpenBlasArrayGemm.sgemmRowMajorNoTrans(m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
            return true;
        } catch (Throwable ignored) {
            return false;
        }
    }

    private static void runJavaGemmF64(double[] a, double[] b, double[] c, int m, int n, int k) {
        java.util.Arrays.fill(c, 0.0d);
        for (int i = 0; i < m; i++) {
            int aRow = i * k;
            int cRow = i * n;
            for (int p = 0; p < k; p++) {
                double av = a[aRow + p];
                int bRow = p * n;
                for (int j = 0; j < n; j++) {
                    c[cRow + j] += av * b[bRow + j];
                }
            }
        }
    }

    private static void runJavaGemmF32(float[] a, float[] b, float[] c, int m, int n, int k) {
        java.util.Arrays.fill(c, 0.0f);
        for (int i = 0; i < m; i++) {
            int aRow = i * k;
            int cRow = i * n;
            for (int p = 0; p < k; p++) {
                float av = a[aRow + p];
                int bRow = p * n;
                for (int j = 0; j < n; j++) {
                    c[cRow + j] += av * b[bRow + j];
                }
            }
        }
    }

    private static void scatterOutputF64(double[] gemmOut, double[] out, double[] bias, int batch, int group,
                                         int outChannelsPerGroup, int outH, int outW, int outChannels) {
        int outChannelBase = group * outChannelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    double value = gemmOut[row * outChannelsPerGroup + ocg] + (bias == null ? 0.0d : bias[oc]);
                    out[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)] = value;
                }
                row++;
            }
        }
    }

    private static void scatterOutputF32(float[] gemmOut, float[] out, float[] bias, int batch, int group,
                                         int outChannelsPerGroup, int outH, int outW, int outChannels) {
        int outChannelBase = group * outChannelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    float value = gemmOut[row * outChannelsPerGroup + ocg] + (bias == null ? 0.0f : bias[oc]);
                    out[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)] = value;
                }
                row++;
            }
        }
    }

    private static void scatterOutputBF16(float[] gemmOut, short[] out, short[] bias, int batch, int group,
                                         int outChannelsPerGroup, int outH, int outW, int outChannels) {
        int outChannelBase = group * outChannelsPerGroup;
        int row = 0;
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                for (int ocg = 0; ocg < outChannelsPerGroup; ocg++) {
                    int oc = outChannelBase + ocg;
                    float value = gemmOut[row * outChannelsPerGroup + ocg] + (bias == null ? 0.0f : CpuDTypeOps.fromBFloat16Bits(bias[oc]));
                    out[indexNCHW(batch, oc, oh, ow, outChannels, outH, outW)] = CpuDTypeOps.toBFloat16Bits(value);
                }
                row++;
            }
        }
    }

    private static int indexNCHW(int batch, int channel, int h, int w, int channels, int height, int width) {
        return ((batch * channels + channel) * height + h) * width + w;
    }

    private static int indexOIHW(int outChannel, int inChannel, int kh, int kw, int channelsPerGroup, int kernelH, int kernelW) {
        return ((outChannel * channelsPerGroup + inChannel) * kernelH + kh) * kernelW + kw;
    }
}
