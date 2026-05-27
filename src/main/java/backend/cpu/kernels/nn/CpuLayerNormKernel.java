package backend.cpu.kernels.nn;

import backend.cpu.execution.CpuKernelContext;
import backend.cpu.execution.CpuThreadPool;
import backend.cpu.kernels.CpuKernelCall;
import backend.cpu.kernels.CpuKernelResult;
import backend.cpu.kernels.CpuStorageAwareKernel;
import backend.cpu.storage.CpuStorageView;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import operations.Operation;
import operations.normalization.layerNorm;
import tensor.DataType;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuLayerNormKernel implements CpuStorageAwareKernel {
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int MIN_VECTOR_AXIS_MULTIPLIER = 4;
    private static final int PARALLEL_MIN_WORK = 16_384;

    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        layerNorm norm = require(call.operation());
        CpuStorageView input = requireInput(call, 0, "input");
        CpuStorageView gamma = requireInput(call, 1, "gamma");
        CpuStorageView beta = requireInput(call, 2, "beta");
        CpuStorageView output = call.output();
        if (output == null) {
            throw new IllegalArgumentException("LayerNorm output storage view is missing.");
        }
        switch (output.dtype()) {
            case FLOAT64 -> {
                requireSameDType(input, gamma, beta, output);
                executeF64(norm, input, gamma, beta, output, call.context());
            }
            case FLOAT32 -> {
                requireSameDType(input, gamma, beta, output);
                executeF32(norm, input, gamma, beta, output, call.context());
            }
            case BFLOAT16 -> {
                requireSameDType(input, gamma, beta, output);
                executeBF16(norm, input, gamma, beta, output, call.context());
            }
            case INT32, INT64, BOOL -> unsupported(output.dtype());
        }
        return CpuKernelResult.completed();
    }

    private static void executeF64(
            layerNorm norm,
            CpuStorageView input,
            CpuStorageView gamma,
            CpuStorageView beta,
            CpuStorageView output,
            CpuKernelContext context
    ) {
        validateLayout(input, gamma, beta, output, norm.getNormalizedRank());
        NormShape shape = resolveNormShape(input, gamma, beta, output, norm.getNormalizedRank());
        if (NormalizationStorageAccess.allArrays(input, gamma, beta, output)) {
            double[] in = input.requireF64Array();
            double[] scale = gamma.requireF64Array();
            double[] shift = beta.requireF64Array();
            double[] out = output.requireF64Array();
            runGroups(shape, context, group -> applyGroupF64(
                    in,
                    scale,
                    shift,
                    out,
                    input.storageOffset() + group * shape.normalizedSize(),
                    gamma.storageOffset(),
                    beta.storageOffset(),
                    output.storageOffset() + group * shape.normalizedSize(),
                    shape.normalizedSize(),
                    norm.getEpsilon()
            ));
            return;
        }
        if (NormalizationStorageAccess.allSegments(input, gamma, beta, output)) {
            MemorySegment in = input.requireSegment();
            MemorySegment scale = gamma.requireSegment();
            MemorySegment shift = beta.requireSegment();
            MemorySegment out = output.requireSegment();
            runGroups(shape, context, group -> applyGroupF64Segment(
                    in,
                    scale,
                    shift,
                    out,
                    input.storageOffset() + group * shape.normalizedSize(),
                    gamma.storageOffset(),
                    beta.storageOffset(),
                    output.storageOffset() + group * shape.normalizedSize(),
                    shape.normalizedSize(),
                    norm.getEpsilon()
            ));
            return;
        }

        double[] inArray = NormalizationStorageAccess.f64Array(input);
        MemorySegment inSegment = NormalizationStorageAccess.f64Segment(input);
        double[] scaleArray = NormalizationStorageAccess.f64Array(gamma);
        MemorySegment scaleSegment = NormalizationStorageAccess.f64Segment(gamma);
        double[] shiftArray = NormalizationStorageAccess.f64Array(beta);
        MemorySegment shiftSegment = NormalizationStorageAccess.f64Segment(beta);
        double[] outArray = NormalizationStorageAccess.f64Array(output);
        MemorySegment outSegment = NormalizationStorageAccess.f64Segment(output);
        runGroups(shape, context, group -> applyGroupF64Storage(
                inArray,
                inSegment,
                scaleArray,
                scaleSegment,
                shiftArray,
                shiftSegment,
                outArray,
                outSegment,
                input.storageOffset() + group * shape.normalizedSize(),
                gamma.storageOffset(),
                beta.storageOffset(),
                output.storageOffset() + group * shape.normalizedSize(),
                shape.normalizedSize(),
                norm.getEpsilon()
        ));
    }

    private static void executeF32(
            layerNorm norm,
            CpuStorageView input,
            CpuStorageView gamma,
            CpuStorageView beta,
            CpuStorageView output,
            CpuKernelContext context
    ) {
        validateLayout(input, gamma, beta, output, norm.getNormalizedRank());
        NormShape shape = resolveNormShape(input, gamma, beta, output, norm.getNormalizedRank());
        if (NormalizationStorageAccess.allArrays(input, gamma, beta, output)) {
            float[] in = input.requireF32Array();
            float[] scale = gamma.requireF32Array();
            float[] shift = beta.requireF32Array();
            float[] out = output.requireF32Array();
            runGroups(shape, context, group -> applyGroupF32(
                    in,
                    scale,
                    shift,
                    out,
                    input.storageOffset() + group * shape.normalizedSize(),
                    gamma.storageOffset(),
                    beta.storageOffset(),
                    output.storageOffset() + group * shape.normalizedSize(),
                    shape.normalizedSize(),
                    (float) norm.getEpsilon()
            ));
            return;
        }
        if (NormalizationStorageAccess.allSegments(input, gamma, beta, output)) {
            MemorySegment in = input.requireSegment();
            MemorySegment scale = gamma.requireSegment();
            MemorySegment shift = beta.requireSegment();
            MemorySegment out = output.requireSegment();
            runGroups(shape, context, group -> applyGroupF32Segment(
                    in,
                    scale,
                    shift,
                    out,
                    input.storageOffset() + group * shape.normalizedSize(),
                    gamma.storageOffset(),
                    beta.storageOffset(),
                    output.storageOffset() + group * shape.normalizedSize(),
                    shape.normalizedSize(),
                    (float) norm.getEpsilon()
            ));
            return;
        }

        float[] inArray = NormalizationStorageAccess.f32Array(input);
        MemorySegment inSegment = NormalizationStorageAccess.f32Segment(input);
        float[] scaleArray = NormalizationStorageAccess.f32Array(gamma);
        MemorySegment scaleSegment = NormalizationStorageAccess.f32Segment(gamma);
        float[] shiftArray = NormalizationStorageAccess.f32Array(beta);
        MemorySegment shiftSegment = NormalizationStorageAccess.f32Segment(beta);
        float[] outArray = NormalizationStorageAccess.f32Array(output);
        MemorySegment outSegment = NormalizationStorageAccess.f32Segment(output);
        runGroups(shape, context, group -> applyGroupF32Storage(
                inArray,
                inSegment,
                scaleArray,
                scaleSegment,
                shiftArray,
                shiftSegment,
                outArray,
                outSegment,
                input.storageOffset() + group * shape.normalizedSize(),
                gamma.storageOffset(),
                beta.storageOffset(),
                output.storageOffset() + group * shape.normalizedSize(),
                shape.normalizedSize(),
                (float) norm.getEpsilon()
        ));
    }

    private static void executeBF16(
            layerNorm norm,
            CpuStorageView input,
            CpuStorageView gamma,
            CpuStorageView beta,
            CpuStorageView output,
            CpuKernelContext context
    ) {
        validateLayout(input, gamma, beta, output, norm.getNormalizedRank());
        NormShape shape = resolveNormShape(input, gamma, beta, output, norm.getNormalizedRank());
        float[] inputContinuation = context.inputFloatContinuation(0, input.logicalSize());
        float[] scale = NormalizationStorageAccess.decodeBFloat16(gamma, gamma.storageOffset(), shape.normalizedSize());
        float[] shift = NormalizationStorageAccess.decodeBFloat16(beta, beta.storageOffset(), shape.normalizedSize());

        if (context.publishFloatContinuation() && context.cpuWorkspace() != null) {
            float[] out = context.cpuWorkspace().requireFloatWorkspace();
            if (out == null || out.length < output.logicalSize()) {
                throw new IllegalArgumentException("LayerNorm float continuation output is missing or too small.");
            }
            if (inputContinuation != null) {
                runGroups(shape, context, group -> applyGroupF32(
                        inputContinuation,
                        scale,
                        shift,
                        out,
                        group * shape.normalizedSize(),
                        0,
                        0,
                        group * shape.normalizedSize(),
                        shape.normalizedSize(),
                        (float) norm.getEpsilon()
                ));
            } else if (input.isArray()) {
                short[] in = input.requireBF16Array();
                runGroups(shape, context, group -> applyGroupBF16ToF32(
                        in,
                        scale,
                        shift,
                        out,
                        input.storageOffset() + group * shape.normalizedSize(),
                        0,
                        0,
                        group * shape.normalizedSize(),
                        shape.normalizedSize(),
                        (float) norm.getEpsilon()
                ));
            } else {
                MemorySegment in = input.requireSegment();
                runGroups(shape, context, group -> applyGroupBF16SegmentToF32(
                        in,
                        scale,
                        shift,
                        out,
                        input.storageOffset() + group * shape.normalizedSize(),
                        0,
                        0,
                        group * shape.normalizedSize(),
                        shape.normalizedSize(),
                        (float) norm.getEpsilon()
                ));
            }
            context.cpuWorkspace().publishFloatContinuation(output.logicalSize());
            return;
        }

        short[] outArray = NormalizationStorageAccess.bf16Array(output);
        MemorySegment outSegment = NormalizationStorageAccess.bf16Segment(output);
        if (inputContinuation != null) {
            if (output.isArray()) {
                runGroups(shape, context, group -> applyGroupF32ToBF16(
                        inputContinuation,
                        scale,
                        shift,
                        output.requireBF16Array(),
                        group * shape.normalizedSize(),
                        0,
                        0,
                        output.storageOffset() + group * shape.normalizedSize(),
                        shape.normalizedSize(),
                        (float) norm.getEpsilon()
                ));
                return;
            }
            runGroups(shape, context, group -> applyGroupF32ToBF16Storage(
                    inputContinuation,
                    scale,
                    shift,
                    outArray,
                    outSegment,
                    group * shape.normalizedSize(),
                    0,
                    0,
                    output.storageOffset() + group * shape.normalizedSize(),
                    shape.normalizedSize(),
                    (float) norm.getEpsilon()
            ));
            return;
        }

        if (NormalizationStorageAccess.allArrays(input, gamma, beta, output)) {
            runGroups(shape, context, group -> applyGroupBF16(
                    input.requireBF16Array(),
                    scale,
                    shift,
                    output.requireBF16Array(),
                    input.storageOffset() + group * shape.normalizedSize(),
                    0,
                    0,
                    output.storageOffset() + group * shape.normalizedSize(),
                    shape.normalizedSize(),
                    (float) norm.getEpsilon()
            ));
            return;
        }

        short[] inArray = NormalizationStorageAccess.bf16Array(input);
        MemorySegment inSegment = NormalizationStorageAccess.bf16Segment(input);
        runGroups(shape, context, group -> applyGroupBF16Storage(
                inArray,
                inSegment,
                scale,
                shift,
                outArray,
                outSegment,
                input.storageOffset() + group * shape.normalizedSize(),
                0,
                0,
                output.storageOffset() + group * shape.normalizedSize(),
                shape.normalizedSize(),
                (float) norm.getEpsilon()
        ));
    }

    private static void runGroups(NormShape shape, CpuKernelContext context, java.util.function.IntConsumer groupBody) {
        long work = (long) shape.groupCount() * shape.normalizedSize();
        int workers = context.plannedWorkers();
        if (shape.groupCount() <= 1 || workers <= 1 || work < PARALLEL_MIN_WORK) {
            for (int group = 0; group < shape.groupCount(); group++) {
                groupBody.accept(group);
            }
            return;
        }

        int chunkSize = Math.max(1, context.computeChunkSize(shape.groupCount(), 1, 1, 1));
        int chunks = (shape.groupCount() + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, workers, chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, shape.groupCount());
            for (int group = start; group < end; group++) {
                groupBody.accept(group);
            }
        });
    }

    private static void applyGroupF64(
            double[] in,
            double[] gamma,
            double[] beta,
            double[] out,
            int inBase,
            int gammaBase,
            int betaBase,
            int outBase,
            int normalizedSize,
            double epsilon
    ) {
        StatsF64 stats = computeStatsF64(in, inBase, normalizedSize);
        double invStd = 1.0d / Math.sqrt(Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d) + epsilon);
        if (canUseVectorPath(normalizedSize, F64_SPECIES.length())) {
            DoubleVector mean = DoubleVector.broadcast(F64_SPECIES, stats.mean());
            DoubleVector inv = DoubleVector.broadcast(F64_SPECIES, invStd);
            int upper = F64_SPECIES.loopBound(normalizedSize);
            int i = 0;
            for (; i < upper; i += F64_SPECIES.length()) {
                DoubleVector.fromArray(F64_SPECIES, in, inBase + i)
                        .sub(mean)
                        .mul(inv)
                        .mul(DoubleVector.fromArray(F64_SPECIES, gamma, gammaBase + i))
                        .add(DoubleVector.fromArray(F64_SPECIES, beta, betaBase + i))
                        .intoArray(out, outBase + i);
            }
            for (; i < normalizedSize; i++) {
                out[outBase + i] = ((in[inBase + i] - stats.mean()) * invStd) * gamma[gammaBase + i] + beta[betaBase + i];
            }
            return;
        }
        for (int i = 0; i < normalizedSize; i++) {
            out[outBase + i] = ((in[inBase + i] - stats.mean()) * invStd) * gamma[gammaBase + i] + beta[betaBase + i];
        }
    }

    private static void applyGroupF64Segment(
            MemorySegment in,
            MemorySegment gamma,
            MemorySegment beta,
            MemorySegment out,
            int inBase,
            int gammaBase,
            int betaBase,
            int outBase,
            int normalizedSize,
            double epsilon
    ) {
        StatsF64 stats = computeStatsF64Segment(in, inBase, normalizedSize);
        double invStd = 1.0d / Math.sqrt(Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d) + epsilon);
        for (int i = 0; i < normalizedSize; i++) {
            double value = in.get(JAVA_DOUBLE, (long) (inBase + i) * Double.BYTES);
            double scale = gamma.get(JAVA_DOUBLE, (long) (gammaBase + i) * Double.BYTES);
            double shift = beta.get(JAVA_DOUBLE, (long) (betaBase + i) * Double.BYTES);
            out.set(JAVA_DOUBLE, (long) (outBase + i) * Double.BYTES, ((value - stats.mean()) * invStd) * scale + shift);
        }
    }

    private static void applyGroupF64Storage(
            double[] inArray,
            MemorySegment inSegment,
            double[] gammaArray,
            MemorySegment gammaSegment,
            double[] betaArray,
            MemorySegment betaSegment,
            double[] outArray,
            MemorySegment outSegment,
            int inBase,
            int gammaBase,
            int betaBase,
            int outBase,
            int normalizedSize,
            double epsilon
    ) {
        StatsF64 stats = computeStatsF64(inArray, inSegment, inBase, normalizedSize);
        double invStd = 1.0d / Math.sqrt(Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d) + epsilon);
        for (int i = 0; i < normalizedSize; i++) {
            double value = NormalizationStorageAccess.readF64(inArray, inSegment, inBase + i);
            double scale = NormalizationStorageAccess.readF64(gammaArray, gammaSegment, gammaBase + i);
            double shift = NormalizationStorageAccess.readF64(betaArray, betaSegment, betaBase + i);
            NormalizationStorageAccess.writeF64(outArray, outSegment, outBase + i, ((value - stats.mean()) * invStd) * scale + shift);
        }
    }

    private static void applyGroupF32(
            float[] in,
            float[] gamma,
            float[] beta,
            float[] out,
            int inBase,
            int gammaBase,
            int betaBase,
            int outBase,
            int normalizedSize,
            float epsilon
    ) {
        StatsF32 stats = computeStatsF32(in, inBase, normalizedSize);
        float variance = (float) Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
        float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
        if (canUseVectorPath(normalizedSize, F32_SPECIES.length())) {
            FloatVector mean = FloatVector.broadcast(F32_SPECIES, stats.mean());
            FloatVector inv = FloatVector.broadcast(F32_SPECIES, invStd);
            int upper = F32_SPECIES.loopBound(normalizedSize);
            int i = 0;
            for (; i < upper; i += F32_SPECIES.length()) {
                FloatVector.fromArray(F32_SPECIES, in, inBase + i)
                        .sub(mean)
                        .mul(inv)
                        .mul(FloatVector.fromArray(F32_SPECIES, gamma, gammaBase + i))
                        .add(FloatVector.fromArray(F32_SPECIES, beta, betaBase + i))
                        .intoArray(out, outBase + i);
            }
            for (; i < normalizedSize; i++) {
                out[outBase + i] = ((in[inBase + i] - stats.mean()) * invStd) * gamma[gammaBase + i] + beta[betaBase + i];
            }
            return;
        }
        for (int i = 0; i < normalizedSize; i++) {
            out[outBase + i] = ((in[inBase + i] - stats.mean()) * invStd) * gamma[gammaBase + i] + beta[betaBase + i];
        }
    }

    private static void applyGroupF32Segment(
            MemorySegment in,
            MemorySegment gamma,
            MemorySegment beta,
            MemorySegment out,
            int inBase,
            int gammaBase,
            int betaBase,
            int outBase,
            int normalizedSize,
            float epsilon
    ) {
        StatsF32 stats = computeStatsF32Segment(in, inBase, normalizedSize);
        float variance = (float) Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
        float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = in.get(JAVA_FLOAT, (long) (inBase + i) * Float.BYTES);
            float scale = gamma.get(JAVA_FLOAT, (long) (gammaBase + i) * Float.BYTES);
            float shift = beta.get(JAVA_FLOAT, (long) (betaBase + i) * Float.BYTES);
            out.set(JAVA_FLOAT, (long) (outBase + i) * Float.BYTES, ((value - stats.mean()) * invStd) * scale + shift);
        }
    }

    private static void applyGroupF32Storage(
            float[] inArray,
            MemorySegment inSegment,
            float[] gammaArray,
            MemorySegment gammaSegment,
            float[] betaArray,
            MemorySegment betaSegment,
            float[] outArray,
            MemorySegment outSegment,
            int inBase,
            int gammaBase,
            int betaBase,
            int outBase,
            int normalizedSize,
            float epsilon
    ) {
        StatsF32 stats = computeStatsF32(inArray, inSegment, inBase, normalizedSize);
        float variance = (float) Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
        float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = NormalizationStorageAccess.readF32(inArray, inSegment, inBase + i);
            float scale = NormalizationStorageAccess.readF32(gammaArray, gammaSegment, gammaBase + i);
            float shift = NormalizationStorageAccess.readF32(betaArray, betaSegment, betaBase + i);
            NormalizationStorageAccess.writeF32(outArray, outSegment, outBase + i, ((value - stats.mean()) * invStd) * scale + shift);
        }
    }

    private static void applyGroupBF16(
            short[] in,
            float[] gamma,
            float[] beta,
            short[] out,
            int inBase,
            int gammaBase,
            int betaBase,
            int outBase,
            int normalizedSize,
            float epsilon
    ) {
        StatsF32 stats = computeStatsBF16(in, inBase, normalizedSize);
        float variance = (float) Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
        float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(in[inBase + i]);
            float scale = gamma[gammaBase + i];
            float shift = beta[betaBase + i];
            out[outBase + i] = TensorDTypeOps.toBFloat16Bits(((value - stats.mean()) * invStd) * scale + shift);
        }
    }

    private static void applyGroupBF16Storage(
            short[] inArray,
            MemorySegment inSegment,
            float[] gamma,
            float[] beta,
            short[] outArray,
            MemorySegment outSegment,
            int inBase,
            int gammaBase,
            int betaBase,
            int outBase,
            int normalizedSize,
            float epsilon
    ) {
        StatsF32 stats = computeStatsBF16(inArray, inSegment, inBase, normalizedSize);
        float variance = (float) Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
        float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = NormalizationStorageAccess.readBF16(inArray, inSegment, inBase + i);
            float scale = gamma[gammaBase + i];
            float shift = beta[betaBase + i];
            NormalizationStorageAccess.writeBF16(outArray, outSegment, outBase + i, ((value - stats.mean()) * invStd) * scale + shift);
        }
    }

    private static void applyGroupF32ToBF16(
            float[] in,
            float[] gamma,
            float[] beta,
            short[] out,
            int inBase,
            int gammaBase,
            int betaBase,
            int outBase,
            int normalizedSize,
            float epsilon
    ) {
        StatsF32 stats = computeStatsF32(in, inBase, normalizedSize);
        float variance = (float) Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
        float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
        if (canUseVectorPath(normalizedSize, F32_SPECIES.length())) {
            FloatVector mean = FloatVector.broadcast(F32_SPECIES, stats.mean());
            FloatVector inv = FloatVector.broadcast(F32_SPECIES, invStd);
            float[] lanes = new float[F32_SPECIES.length()];
            int upper = F32_SPECIES.loopBound(normalizedSize);
            int i = 0;
            for (; i < upper; i += F32_SPECIES.length()) {
                FloatVector.fromArray(F32_SPECIES, in, inBase + i)
                        .sub(mean)
                        .mul(inv)
                        .mul(FloatVector.fromArray(F32_SPECIES, gamma, gammaBase + i))
                        .add(FloatVector.fromArray(F32_SPECIES, beta, betaBase + i))
                        .intoArray(lanes, 0);
                for (int lane = 0; lane < F32_SPECIES.length(); lane++) {
                    out[outBase + i + lane] = TensorDTypeOps.toBFloat16Bits(lanes[lane]);
                }
            }
            for (; i < normalizedSize; i++) {
                out[outBase + i] = TensorDTypeOps.toBFloat16Bits(((in[inBase + i] - stats.mean()) * invStd) * gamma[gammaBase + i] + beta[betaBase + i]);
            }
            return;
        }
        for (int i = 0; i < normalizedSize; i++) {
            out[outBase + i] = TensorDTypeOps.toBFloat16Bits(((in[inBase + i] - stats.mean()) * invStd) * gamma[gammaBase + i] + beta[betaBase + i]);
        }
    }

    private static void applyGroupF32ToBF16Storage(
            float[] in,
            float[] gamma,
            float[] beta,
            short[] outArray,
            MemorySegment outSegment,
            int inBase,
            int gammaBase,
            int betaBase,
            int outBase,
            int normalizedSize,
            float epsilon
    ) {
        StatsF32 stats = computeStatsF32(in, inBase, normalizedSize);
        float variance = (float) Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
        float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            NormalizationStorageAccess.writeBF16(
                    outArray,
                    outSegment,
                    outBase + i,
                    ((in[inBase + i] - stats.mean()) * invStd) * gamma[gammaBase + i] + beta[betaBase + i]
            );
        }
    }

    private static void applyGroupBF16ToF32(
            short[] in,
            float[] gamma,
            float[] beta,
            float[] out,
            int inBase,
            int gammaBase,
            int betaBase,
            int outBase,
            int normalizedSize,
            float epsilon
    ) {
        StatsF32 stats = computeStatsBF16(in, inBase, normalizedSize);
        float variance = (float) Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
        float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(in[inBase + i]);
            out[outBase + i] = ((value - stats.mean()) * invStd) * gamma[gammaBase + i] + beta[betaBase + i];
        }
    }

    private static void applyGroupBF16SegmentToF32(
            MemorySegment in,
            float[] gamma,
            float[] beta,
            float[] out,
            int inBase,
            int gammaBase,
            int betaBase,
            int outBase,
            int normalizedSize,
            float epsilon
    ) {
        StatsF32 stats = computeStatsBF16Segment(in, inBase, normalizedSize);
        float variance = (float) Math.max(stats.meanSquares() - stats.mean() * stats.mean(), 0.0d);
        float invStd = (float) (1.0d / Math.sqrt(variance + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(in.get(JAVA_SHORT, (long) (inBase + i) * Short.BYTES));
            out[outBase + i] = ((value - stats.mean()) * invStd) * gamma[gammaBase + i] + beta[betaBase + i];
        }
    }

    private static StatsF64 computeStatsF64(double[] in, int base, int length) {
        if (canUseVectorPath(length, F64_SPECIES.length())) {
            int upper = F64_SPECIES.loopBound(length);
            DoubleVector sum = DoubleVector.zero(F64_SPECIES);
            DoubleVector sumSquares = DoubleVector.zero(F64_SPECIES);
            int i = 0;
            for (; i < upper; i += F64_SPECIES.length()) {
                DoubleVector value = DoubleVector.fromArray(F64_SPECIES, in, base + i);
                sum = sum.add(value);
                sumSquares = sumSquares.add(value.mul(value));
            }
            double total = sum.reduceLanes(VectorOperators.ADD);
            double totalSquares = sumSquares.reduceLanes(VectorOperators.ADD);
            for (; i < length; i++) {
                double value = in[base + i];
                total += value;
                totalSquares += value * value;
            }
            return new StatsF64(total / length, totalSquares / length);
        }
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            double value = in[base + i];
            total += value;
            totalSquares += value * value;
        }
        return new StatsF64(total / length, totalSquares / length);
    }

    private static StatsF64 computeStatsF64Segment(MemorySegment in, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            double value = in.get(JAVA_DOUBLE, (long) (base + i) * Double.BYTES);
            total += value;
            totalSquares += value * value;
        }
        return new StatsF64(total / length, totalSquares / length);
    }

    private static StatsF64 computeStatsF64(double[] array, MemorySegment segment, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            double value = NormalizationStorageAccess.readF64(array, segment, base + i);
            total += value;
            totalSquares += value * value;
        }
        return new StatsF64(total / length, totalSquares / length);
    }

    private static StatsF32 computeStatsF32(float[] in, int base, int length) {
        if (canUseVectorPath(length, F32_SPECIES.length())) {
            int upper = F32_SPECIES.loopBound(length);
            FloatVector sum = FloatVector.zero(F32_SPECIES);
            FloatVector sumSquares = FloatVector.zero(F32_SPECIES);
            int i = 0;
            for (; i < upper; i += F32_SPECIES.length()) {
                FloatVector value = FloatVector.fromArray(F32_SPECIES, in, base + i);
                sum = sum.add(value);
                sumSquares = sumSquares.add(value.mul(value));
            }
            double total = sum.reduceLanes(VectorOperators.ADD);
            double totalSquares = sumSquares.reduceLanes(VectorOperators.ADD);
            for (; i < length; i++) {
                float value = in[base + i];
                total += value;
                totalSquares += value * value;
            }
            return new StatsF32((float) (total / length), totalSquares / length);
        }
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = in[base + i];
            total += value;
            totalSquares += value * value;
        }
        return new StatsF32((float) (total / length), totalSquares / length);
    }

    private static StatsF32 computeStatsF32Segment(MemorySegment in, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = in.get(JAVA_FLOAT, (long) (base + i) * Float.BYTES);
            total += value;
            totalSquares += value * value;
        }
        return new StatsF32((float) (total / length), totalSquares / length);
    }

    private static StatsF32 computeStatsF32(float[] array, MemorySegment segment, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = NormalizationStorageAccess.readF32(array, segment, base + i);
            total += value;
            totalSquares += value * value;
        }
        return new StatsF32((float) (total / length), totalSquares / length);
    }

    private static StatsF32 computeStatsBF16(short[] in, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(in[base + i]);
            total += value;
            totalSquares += value * value;
        }
        return new StatsF32((float) (total / length), totalSquares / length);
    }

    private static StatsF32 computeStatsBF16Segment(MemorySegment in, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(in.get(JAVA_SHORT, (long) (base + i) * Short.BYTES));
            total += value;
            totalSquares += value * value;
        }
        return new StatsF32((float) (total / length), totalSquares / length);
    }

    private static StatsF32 computeStatsBF16(short[] array, MemorySegment segment, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = NormalizationStorageAccess.readBF16(array, segment, base + i);
            total += value;
            totalSquares += value * value;
        }
        return new StatsF32((float) (total / length), totalSquares / length);
    }

    private static boolean canUseVectorPath(int normalizedSize, int speciesLength) {
        return speciesLength > 1 && normalizedSize >= speciesLength * MIN_VECTOR_AXIS_MULTIPLIER;
    }

    private static void validateLayout(
            CpuStorageView input,
            CpuStorageView gamma,
            CpuStorageView beta,
            CpuStorageView output,
            int normalizedRank
    ) {
        if (!NormalizationStorageAccess.isDenseContiguous(input)
                || !NormalizationStorageAccess.isDenseContiguous(gamma)
                || !NormalizationStorageAccess.isDenseContiguous(beta)
                || !NormalizationStorageAccess.isDenseContiguous(output)) {
            throw new IllegalArgumentException("CpuLayerNormKernel requires contiguous input, gamma, beta, and output tensors.");
        }
        if (normalizedRank < 1 || normalizedRank > input.shape().length) {
            throw new IllegalArgumentException("Invalid LayerNorm normalized rank: " + normalizedRank);
        }
    }

    private static NormShape resolveNormShape(
            CpuStorageView input,
            CpuStorageView gamma,
            CpuStorageView beta,
            CpuStorageView output,
            int normalizedRank
    ) {
        int[] inputShape = input.shape();
        int[] gammaShape = gamma.shape();
        int[] betaShape = beta.shape();
        if (gammaShape.length != normalizedRank || betaShape.length != normalizedRank) {
            throw new IllegalArgumentException("LayerNorm parameter ranks must equal normalized rank.");
        }
        int normalizedSize = 1;
        int start = inputShape.length - normalizedRank;
        for (int i = 0; i < normalizedRank; i++) {
            int expected = inputShape[start + i];
            if (gammaShape[i] != expected || betaShape[i] != expected) {
                throw new IllegalArgumentException("LayerNorm parameter shapes must match trailing input dimensions.");
            }
            normalizedSize = Math.multiplyExact(normalizedSize, expected);
        }
        if (gamma.logicalSize() != normalizedSize || beta.logicalSize() != normalizedSize) {
            throw new IllegalArgumentException("LayerNorm parameter storage size mismatch.");
        }
        if (output.logicalSize() != input.logicalSize()) {
            throw new IllegalArgumentException("LayerNorm output size must match input size.");
        }
        return new NormShape(input.logicalSize() / normalizedSize, normalizedSize);
    }

    private static layerNorm require(Operation op) {
        if (!(op instanceof layerNorm norm)) {
            throw new IllegalArgumentException("CpuLayerNormKernel requires layerNorm operation.");
        }
        return norm;
    }

    private static CpuStorageView requireInput(CpuKernelCall call, int index, String name) {
        if (call.inputs().size() <= index || call.inputs().get(index) == null) {
            throw new IllegalArgumentException("LayerNorm " + name + " storage view is missing.");
        }
        return call.inputs().get(index);
    }

    private static void requireSameDType(
            CpuStorageView input,
            CpuStorageView gamma,
            CpuStorageView beta,
            CpuStorageView output
    ) {
        DataType dtype = output.dtype();
        if (input.dtype() != dtype || gamma.dtype() != dtype || beta.dtype() != dtype) {
            throw new IllegalArgumentException("LayerNorm storage dtype mismatch. input=" + input.dtype()
                    + ", gamma=" + gamma.dtype() + ", beta=" + beta.dtype() + ", output=" + dtype);
        }
    }

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuLayerNormKernel does not support " + dtype);
    }

    private record NormShape(int groupCount, int normalizedSize) {
    }

    private record StatsF64(double mean, double meanSquares) {
    }

    private record StatsF32(float mean, double meanSquares) {
    }
}
