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
import operations.normalization.rmsNorm;
import tensor.DataType;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuRmsNormKernel implements CpuStorageAwareKernel {
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int MIN_VECTOR_AXIS_MULTIPLIER = 4;
    private static final int PARALLEL_MIN_WORK = 16_384;

    @Override
    public CpuKernelResult execute(CpuKernelCall call) {
        rmsNorm norm = require(call.operation());
        CpuStorageView input = requireInput(call, 0, "input");
        CpuStorageView gamma = requireInput(call, 1, "gamma");
        CpuStorageView output = call.output();
        if (output == null) {
            throw new IllegalArgumentException("RMSNorm output storage view is missing.");
        }
        switch (output.dtype()) {
            case FLOAT64 -> {
                requireSameDType(input, gamma, output);
                executeF64(norm, input, gamma, output, call.context());
            }
            case FLOAT32 -> {
                requireSameDType(input, gamma, output);
                executeF32(norm, input, gamma, output, call.context());
            }
            case BFLOAT16 -> {
                requireSameDType(input, gamma, output);
                executeBF16(norm, input, gamma, output, call.context());
            }
            case INT32, INT64, BOOL -> unsupported(output.dtype());
        }
        return CpuKernelResult.completed();
    }

    private static void executeF64(
            rmsNorm norm,
            CpuStorageView input,
            CpuStorageView gamma,
            CpuStorageView output,
            CpuKernelContext context
    ) {
        validateLayout(input, gamma, output, norm.getNormalizedRank());
        NormShape shape = resolveNormShape(input, gamma, output, norm.getNormalizedRank());
        if (NormalizationStorageAccess.allArrays(input, gamma, output)) {
            double[] in = input.requireF64Array();
            double[] weights = gamma.requireF64Array();
            double[] out = output.requireF64Array();
            runGroups(shape, context, group -> applyGroupF64(
                    in,
                    weights,
                    out,
                    input.storageOffset() + group * shape.normalizedSize(),
                    gamma.storageOffset(),
                    output.storageOffset() + group * shape.normalizedSize(),
                    shape.normalizedSize(),
                    norm.getEpsilon()
            ));
            return;
        }
        if (NormalizationStorageAccess.allSegments(input, gamma, output)) {
            MemorySegment in = input.requireSegment();
            MemorySegment weights = gamma.requireSegment();
            MemorySegment out = output.requireSegment();
            runGroups(shape, context, group -> applyGroupF64Segment(
                    in,
                    weights,
                    out,
                    input.storageOffset() + group * shape.normalizedSize(),
                    gamma.storageOffset(),
                    output.storageOffset() + group * shape.normalizedSize(),
                    shape.normalizedSize(),
                    norm.getEpsilon()
            ));
            return;
        }

        double[] inArray = NormalizationStorageAccess.f64Array(input);
        MemorySegment inSegment = NormalizationStorageAccess.f64Segment(input);
        double[] weightsArray = NormalizationStorageAccess.f64Array(gamma);
        MemorySegment weightsSegment = NormalizationStorageAccess.f64Segment(gamma);
        double[] outArray = NormalizationStorageAccess.f64Array(output);
        MemorySegment outSegment = NormalizationStorageAccess.f64Segment(output);
        runGroups(shape, context, group -> applyGroupF64Storage(
                inArray,
                inSegment,
                weightsArray,
                weightsSegment,
                outArray,
                outSegment,
                input.storageOffset() + group * shape.normalizedSize(),
                gamma.storageOffset(),
                output.storageOffset() + group * shape.normalizedSize(),
                shape.normalizedSize(),
                norm.getEpsilon()
        ));
    }

    private static void executeF32(
            rmsNorm norm,
            CpuStorageView input,
            CpuStorageView gamma,
            CpuStorageView output,
            CpuKernelContext context
    ) {
        validateLayout(input, gamma, output, norm.getNormalizedRank());
        NormShape shape = resolveNormShape(input, gamma, output, norm.getNormalizedRank());
        if (NormalizationStorageAccess.allArrays(input, gamma, output)) {
            float[] in = input.requireF32Array();
            float[] weights = gamma.requireF32Array();
            float[] out = output.requireF32Array();
            runGroups(shape, context, group -> applyGroupF32(
                    in,
                    weights,
                    out,
                    input.storageOffset() + group * shape.normalizedSize(),
                    gamma.storageOffset(),
                    output.storageOffset() + group * shape.normalizedSize(),
                    shape.normalizedSize(),
                    (float) norm.getEpsilon()
            ));
            return;
        }
        if (NormalizationStorageAccess.allSegments(input, gamma, output)) {
            MemorySegment in = input.requireSegment();
            MemorySegment weights = gamma.requireSegment();
            MemorySegment out = output.requireSegment();
            runGroups(shape, context, group -> applyGroupF32Segment(
                    in,
                    weights,
                    out,
                    input.storageOffset() + group * shape.normalizedSize(),
                    gamma.storageOffset(),
                    output.storageOffset() + group * shape.normalizedSize(),
                    shape.normalizedSize(),
                    (float) norm.getEpsilon()
            ));
            return;
        }

        float[] inArray = NormalizationStorageAccess.f32Array(input);
        MemorySegment inSegment = NormalizationStorageAccess.f32Segment(input);
        float[] weightsArray = NormalizationStorageAccess.f32Array(gamma);
        MemorySegment weightsSegment = NormalizationStorageAccess.f32Segment(gamma);
        float[] outArray = NormalizationStorageAccess.f32Array(output);
        MemorySegment outSegment = NormalizationStorageAccess.f32Segment(output);
        runGroups(shape, context, group -> applyGroupF32Storage(
                inArray,
                inSegment,
                weightsArray,
                weightsSegment,
                outArray,
                outSegment,
                input.storageOffset() + group * shape.normalizedSize(),
                gamma.storageOffset(),
                output.storageOffset() + group * shape.normalizedSize(),
                shape.normalizedSize(),
                (float) norm.getEpsilon()
        ));
    }

    private static void executeBF16(
            rmsNorm norm,
            CpuStorageView input,
            CpuStorageView gamma,
            CpuStorageView output,
            CpuKernelContext context
    ) {
        validateLayout(input, gamma, output, norm.getNormalizedRank());
        NormShape shape = resolveNormShape(input, gamma, output, norm.getNormalizedRank());
        float[] inputContinuation = context.inputFloatContinuation(0, input.logicalSize());
        float[] weights = NormalizationStorageAccess.decodeBFloat16(gamma, gamma.storageOffset(), shape.normalizedSize());

        if (context.publishFloatContinuation() && context.cpuWorkspace() != null) {
            float[] out = context.cpuWorkspace().requireFloatWorkspace();
            if (out == null || out.length < output.logicalSize()) {
                throw new IllegalArgumentException("RmsNorm float continuation output is missing or too small.");
            }
            if (inputContinuation != null) {
                runGroups(shape, context, group -> applyGroupF32(
                        inputContinuation,
                        weights,
                        out,
                        group * shape.normalizedSize(),
                        0,
                        group * shape.normalizedSize(),
                        shape.normalizedSize(),
                        (float) norm.getEpsilon()
                ));
            } else if (input.isArray()) {
                short[] in = input.requireBF16Array();
                runGroups(shape, context, group -> applyGroupBF16ToF32(
                        in,
                        weights,
                        out,
                        input.storageOffset() + group * shape.normalizedSize(),
                        0,
                        group * shape.normalizedSize(),
                        shape.normalizedSize(),
                        (float) norm.getEpsilon()
                ));
            } else {
                MemorySegment in = input.requireSegment();
                runGroups(shape, context, group -> applyGroupBF16SegmentToF32(
                        in,
                        weights,
                        out,
                        input.storageOffset() + group * shape.normalizedSize(),
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
                        weights,
                        output.requireBF16Array(),
                        group * shape.normalizedSize(),
                        0,
                        output.storageOffset() + group * shape.normalizedSize(),
                        shape.normalizedSize(),
                        (float) norm.getEpsilon()
                ));
                return;
            }
            runGroups(shape, context, group -> applyGroupF32ToBF16Storage(
                    inputContinuation,
                    weights,
                    outArray,
                    outSegment,
                    group * shape.normalizedSize(),
                    0,
                    output.storageOffset() + group * shape.normalizedSize(),
                    shape.normalizedSize(),
                    (float) norm.getEpsilon()
            ));
            return;
        }

        if (NormalizationStorageAccess.allArrays(input, gamma, output)) {
            runGroups(shape, context, group -> applyGroupBF16(
                    input.requireBF16Array(),
                    weights,
                    output.requireBF16Array(),
                    input.storageOffset() + group * shape.normalizedSize(),
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
                weights,
                outArray,
                outSegment,
                input.storageOffset() + group * shape.normalizedSize(),
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

    private static void applyGroupF64(double[] in, double[] gamma, double[] out, int inBase, int gammaBase, int outBase, int normalizedSize, double epsilon) {
        double invRms = 1.0d / Math.sqrt(sumSquaresF64(in, inBase, normalizedSize) / normalizedSize + epsilon);
        if (canUseVectorPath(normalizedSize, F64_SPECIES.length())) {
            DoubleVector inv = DoubleVector.broadcast(F64_SPECIES, invRms);
            int upper = F64_SPECIES.loopBound(normalizedSize);
            int i = 0;
            for (; i < upper; i += F64_SPECIES.length()) {
                DoubleVector.fromArray(F64_SPECIES, in, inBase + i)
                        .mul(DoubleVector.fromArray(F64_SPECIES, gamma, gammaBase + i))
                        .mul(inv)
                        .intoArray(out, outBase + i);
            }
            for (; i < normalizedSize; i++) {
                out[outBase + i] = in[inBase + i] * gamma[gammaBase + i] * invRms;
            }
            return;
        }
        for (int i = 0; i < normalizedSize; i++) {
            out[outBase + i] = in[inBase + i] * gamma[gammaBase + i] * invRms;
        }
    }

    private static void applyGroupF64Segment(MemorySegment in, MemorySegment gamma, MemorySegment out, int inBase, int gammaBase, int outBase, int normalizedSize, double epsilon) {
        double invRms = 1.0d / Math.sqrt(sumSquaresF64Segment(in, inBase, normalizedSize) / normalizedSize + epsilon);
        for (int i = 0; i < normalizedSize; i++) {
            double value = in.get(JAVA_DOUBLE, (long) (inBase + i) * Double.BYTES);
            double scale = gamma.get(JAVA_DOUBLE, (long) (gammaBase + i) * Double.BYTES);
            out.set(JAVA_DOUBLE, (long) (outBase + i) * Double.BYTES, value * scale * invRms);
        }
    }

    private static void applyGroupF64Storage(
            double[] inArray,
            MemorySegment inSegment,
            double[] gammaArray,
            MemorySegment gammaSegment,
            double[] outArray,
            MemorySegment outSegment,
            int inBase,
            int gammaBase,
            int outBase,
            int normalizedSize,
            double epsilon
    ) {
        double invRms = 1.0d / Math.sqrt(sumSquaresF64(inArray, inSegment, inBase, normalizedSize) / normalizedSize + epsilon);
        for (int i = 0; i < normalizedSize; i++) {
            double value = NormalizationStorageAccess.readF64(inArray, inSegment, inBase + i);
            double scale = NormalizationStorageAccess.readF64(gammaArray, gammaSegment, gammaBase + i);
            NormalizationStorageAccess.writeF64(outArray, outSegment, outBase + i, value * scale * invRms);
        }
    }

    private static void applyGroupF32(float[] in, float[] gamma, float[] out, int inBase, int gammaBase, int outBase, int normalizedSize, float epsilon) {
        float invRms = (float) (1.0d / Math.sqrt(sumSquaresF32(in, inBase, normalizedSize) / normalizedSize + epsilon));
        if (canUseVectorPath(normalizedSize, F32_SPECIES.length())) {
            FloatVector inv = FloatVector.broadcast(F32_SPECIES, invRms);
            int upper = F32_SPECIES.loopBound(normalizedSize);
            int i = 0;
            for (; i < upper; i += F32_SPECIES.length()) {
                FloatVector.fromArray(F32_SPECIES, in, inBase + i)
                        .mul(FloatVector.fromArray(F32_SPECIES, gamma, gammaBase + i))
                        .mul(inv)
                        .intoArray(out, outBase + i);
            }
            for (; i < normalizedSize; i++) {
                out[outBase + i] = in[inBase + i] * gamma[gammaBase + i] * invRms;
            }
            return;
        }
        for (int i = 0; i < normalizedSize; i++) {
            out[outBase + i] = in[inBase + i] * gamma[gammaBase + i] * invRms;
        }
    }

    private static void applyGroupF32Segment(MemorySegment in, MemorySegment gamma, MemorySegment out, int inBase, int gammaBase, int outBase, int normalizedSize, float epsilon) {
        float invRms = (float) (1.0d / Math.sqrt(sumSquaresF32Segment(in, inBase, normalizedSize) / normalizedSize + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = in.get(JAVA_FLOAT, (long) (inBase + i) * Float.BYTES);
            float scale = gamma.get(JAVA_FLOAT, (long) (gammaBase + i) * Float.BYTES);
            out.set(JAVA_FLOAT, (long) (outBase + i) * Float.BYTES, value * scale * invRms);
        }
    }

    private static void applyGroupF32Storage(
            float[] inArray,
            MemorySegment inSegment,
            float[] gammaArray,
            MemorySegment gammaSegment,
            float[] outArray,
            MemorySegment outSegment,
            int inBase,
            int gammaBase,
            int outBase,
            int normalizedSize,
            float epsilon
    ) {
        float invRms = (float) (1.0d / Math.sqrt(sumSquaresF32(inArray, inSegment, inBase, normalizedSize) / normalizedSize + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = NormalizationStorageAccess.readF32(inArray, inSegment, inBase + i);
            float scale = NormalizationStorageAccess.readF32(gammaArray, gammaSegment, gammaBase + i);
            NormalizationStorageAccess.writeF32(outArray, outSegment, outBase + i, value * scale * invRms);
        }
    }

    private static void applyGroupBF16(short[] in, float[] gamma, short[] out, int inBase, int gammaBase, int outBase, int normalizedSize, float epsilon) {
        float invRms = (float) (1.0d / Math.sqrt(sumSquaresBF16(in, inBase, normalizedSize) / normalizedSize + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(in[inBase + i]);
            out[outBase + i] = TensorDTypeOps.toBFloat16Bits(value * gamma[gammaBase + i] * invRms);
        }
    }

    private static void applyGroupBF16Storage(
            short[] inArray,
            MemorySegment inSegment,
            float[] gamma,
            short[] outArray,
            MemorySegment outSegment,
            int inBase,
            int gammaBase,
            int outBase,
            int normalizedSize,
            float epsilon
    ) {
        float invRms = (float) (1.0d / Math.sqrt(sumSquaresBF16(inArray, inSegment, inBase, normalizedSize) / normalizedSize + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = NormalizationStorageAccess.readBF16(inArray, inSegment, inBase + i);
            NormalizationStorageAccess.writeBF16(outArray, outSegment, outBase + i, value * gamma[gammaBase + i] * invRms);
        }
    }

    private static void applyGroupF32ToBF16(float[] in, float[] gamma, short[] out, int inBase, int gammaBase, int outBase, int normalizedSize, float epsilon) {
        float invRms = (float) (1.0d / Math.sqrt(sumSquaresF32(in, inBase, normalizedSize) / normalizedSize + epsilon));
        if (canUseVectorPath(normalizedSize, F32_SPECIES.length())) {
            FloatVector inv = FloatVector.broadcast(F32_SPECIES, invRms);
            float[] lanes = new float[F32_SPECIES.length()];
            int upper = F32_SPECIES.loopBound(normalizedSize);
            int i = 0;
            for (; i < upper; i += F32_SPECIES.length()) {
                FloatVector.fromArray(F32_SPECIES, in, inBase + i)
                        .mul(FloatVector.fromArray(F32_SPECIES, gamma, gammaBase + i))
                        .mul(inv)
                        .intoArray(lanes, 0);
                for (int lane = 0; lane < F32_SPECIES.length(); lane++) {
                    out[outBase + i + lane] = TensorDTypeOps.toBFloat16Bits(lanes[lane]);
                }
            }
            for (; i < normalizedSize; i++) {
                out[outBase + i] = TensorDTypeOps.toBFloat16Bits(in[inBase + i] * gamma[gammaBase + i] * invRms);
            }
            return;
        }
        for (int i = 0; i < normalizedSize; i++) {
            out[outBase + i] = TensorDTypeOps.toBFloat16Bits(in[inBase + i] * gamma[gammaBase + i] * invRms);
        }
    }

    private static void applyGroupF32ToBF16Storage(
            float[] in,
            float[] gamma,
            short[] outArray,
            MemorySegment outSegment,
            int inBase,
            int gammaBase,
            int outBase,
            int normalizedSize,
            float epsilon
    ) {
        float invRms = (float) (1.0d / Math.sqrt(sumSquaresF32(in, inBase, normalizedSize) / normalizedSize + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            NormalizationStorageAccess.writeBF16(outArray, outSegment, outBase + i, in[inBase + i] * gamma[gammaBase + i] * invRms);
        }
    }

    private static void applyGroupBF16ToF32(short[] in, float[] gamma, float[] out, int inBase, int gammaBase, int outBase, int normalizedSize, float epsilon) {
        float invRms = (float) (1.0d / Math.sqrt(sumSquaresBF16(in, inBase, normalizedSize) / normalizedSize + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(in[inBase + i]);
            out[outBase + i] = value * gamma[gammaBase + i] * invRms;
        }
    }

    private static void applyGroupBF16SegmentToF32(MemorySegment in, float[] gamma, float[] out, int inBase, int gammaBase, int outBase, int normalizedSize, float epsilon) {
        float invRms = (float) (1.0d / Math.sqrt(sumSquaresBF16Segment(in, inBase, normalizedSize) / normalizedSize + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(in.get(JAVA_SHORT, (long) (inBase + i) * Short.BYTES));
            out[outBase + i] = value * gamma[gammaBase + i] * invRms;
        }
    }

    private static double sumSquaresF64(double[] in, int base, int length) {
        if (canUseVectorPath(length, F64_SPECIES.length())) {
            int upper = F64_SPECIES.loopBound(length);
            DoubleVector sum = DoubleVector.zero(F64_SPECIES);
            int i = 0;
            for (; i < upper; i += F64_SPECIES.length()) {
                DoubleVector value = DoubleVector.fromArray(F64_SPECIES, in, base + i);
                sum = sum.add(value.mul(value));
            }
            double total = sum.reduceLanes(VectorOperators.ADD);
            for (; i < length; i++) {
                double value = in[base + i];
                total += value * value;
            }
            return total;
        }
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            double value = in[base + i];
            total += value * value;
        }
        return total;
    }

    private static double sumSquaresF64Segment(MemorySegment in, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            double value = in.get(JAVA_DOUBLE, (long) (base + i) * Double.BYTES);
            total += value * value;
        }
        return total;
    }

    private static double sumSquaresF64(double[] array, MemorySegment segment, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            double value = NormalizationStorageAccess.readF64(array, segment, base + i);
            total += value * value;
        }
        return total;
    }

    private static double sumSquaresF32(float[] in, int base, int length) {
        if (canUseVectorPath(length, F32_SPECIES.length())) {
            int upper = F32_SPECIES.loopBound(length);
            FloatVector sum = FloatVector.zero(F32_SPECIES);
            int i = 0;
            for (; i < upper; i += F32_SPECIES.length()) {
                FloatVector value = FloatVector.fromArray(F32_SPECIES, in, base + i);
                sum = sum.add(value.mul(value));
            }
            double total = sum.reduceLanes(VectorOperators.ADD);
            for (; i < length; i++) {
                float value = in[base + i];
                total += value * value;
            }
            return total;
        }
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = in[base + i];
            total += value * value;
        }
        return total;
    }

    private static double sumSquaresF32Segment(MemorySegment in, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = in.get(JAVA_FLOAT, (long) (base + i) * Float.BYTES);
            total += value * value;
        }
        return total;
    }

    private static double sumSquaresF32(float[] array, MemorySegment segment, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = NormalizationStorageAccess.readF32(array, segment, base + i);
            total += value * value;
        }
        return total;
    }

    private static double sumSquaresBF16(short[] in, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(in[base + i]);
            total += value * value;
        }
        return total;
    }

    private static double sumSquaresBF16Segment(MemorySegment in, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(in.get(JAVA_SHORT, (long) (base + i) * Short.BYTES));
            total += value * value;
        }
        return total;
    }

    private static double sumSquaresBF16(short[] array, MemorySegment segment, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = NormalizationStorageAccess.readBF16(array, segment, base + i);
            total += value * value;
        }
        return total;
    }

    private static boolean canUseVectorPath(int normalizedSize, int speciesLength) {
        return speciesLength > 1 && normalizedSize >= speciesLength * MIN_VECTOR_AXIS_MULTIPLIER;
    }

    private static void validateLayout(CpuStorageView input, CpuStorageView gamma, CpuStorageView output, int normalizedRank) {
        if (!NormalizationStorageAccess.isDenseContiguous(input)
                || !NormalizationStorageAccess.isDenseContiguous(gamma)
                || !NormalizationStorageAccess.isDenseContiguous(output)) {
            throw new IllegalArgumentException("CpuRmsNormKernel requires contiguous input, gamma, and output tensors.");
        }
        if (normalizedRank < 1 || normalizedRank > input.shape().length) {
            throw new IllegalArgumentException("Invalid RMSNorm normalized rank: " + normalizedRank);
        }
    }

    private static NormShape resolveNormShape(CpuStorageView input, CpuStorageView gamma, CpuStorageView output, int normalizedRank) {
        int[] inputShape = input.shape();
        int[] gammaShape = gamma.shape();
        if (gammaShape.length != normalizedRank) {
            throw new IllegalArgumentException("RMSNorm gamma rank must equal normalized rank.");
        }
        if (output.logicalSize() != input.logicalSize()) {
            throw new IllegalArgumentException("RMSNorm output size must match input size.");
        }

        int normalizedSize = 1;
        int start = inputShape.length - normalizedRank;
        for (int i = 0; i < normalizedRank; i++) {
            int expected = inputShape[start + i];
            if (gammaShape[i] != expected) {
                throw new IllegalArgumentException("RMSNorm gamma shape must match trailing input dimensions.");
            }
            normalizedSize = Math.multiplyExact(normalizedSize, expected);
        }
        if (gamma.logicalSize() != normalizedSize) {
            throw new IllegalArgumentException("RMSNorm gamma storage size mismatch.");
        }
        int groupCount = input.logicalSize() / normalizedSize;
        return new NormShape(groupCount, normalizedSize);
    }

    private static rmsNorm require(Operation op) {
        if (!(op instanceof rmsNorm norm)) {
            throw new IllegalArgumentException("CpuRmsNormKernel requires rmsNorm operation.");
        }
        return norm;
    }

    private static CpuStorageView requireInput(CpuKernelCall call, int index, String name) {
        if (call.inputs().size() <= index || call.inputs().get(index) == null) {
            throw new IllegalArgumentException("RMSNorm " + name + " storage view is missing.");
        }
        return call.inputs().get(index);
    }

    private static void requireSameDType(CpuStorageView input, CpuStorageView gamma, CpuStorageView output) {
        DataType dtype = output.dtype();
        if (input.dtype() != dtype || gamma.dtype() != dtype) {
            throw new IllegalArgumentException("RMSNorm storage dtype mismatch. input=" + input.dtype()
                    + ", gamma=" + gamma.dtype() + ", output=" + dtype);
        }
    }

    private static void unsupported(DataType dtype) {
        throw new UnsupportedOperationException("CpuRmsNormKernel does not support " + dtype);
    }

    private record NormShape(int groupCount, int normalizedSize) {
    }
}
