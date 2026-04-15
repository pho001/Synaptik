package backend.kernels.cpu.nn;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuThreadPool;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import operations.Operation;
import operations.layerNorm;
import tensor.Tensor;

import java.util.List;

public final class CpuLayerNormKernel implements CpuKernel {
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int MIN_VECTOR_AXIS_MULTIPLIER = 4;
    private static final int PARALLEL_MIN_WORK = 16_384;

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        layerNorm norm = require(op);
        Tensor input = requireInput(inputs, 0, "input");
        Tensor gamma = requireInput(inputs, 1, "gamma");
        Tensor beta = requireInput(inputs, 2, "beta");
        executeF64(norm, input, gamma, beta, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        layerNorm norm = require(op);
        Tensor input = requireInput(inputs, 0, "input");
        Tensor gamma = requireInput(inputs, 1, "gamma");
        Tensor beta = requireInput(inputs, 2, "beta");
        executeF32(norm, input, gamma, beta, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        layerNorm norm = require(op);
        Tensor input = requireInput(inputs, 0, "input");
        Tensor gamma = requireInput(inputs, 1, "gamma");
        Tensor beta = requireInput(inputs, 2, "beta");
        executeBF16(norm, input, gamma, beta, node, context);
    }

    private static void executeF64(layerNorm norm, Tensor input, Tensor gamma, Tensor beta, Tensor node, CpuKernelContext context) {
        validateLayout(input, gamma, beta, node, norm.getNormalizedRank());
        double[] in = input.getFloat64Data();
        double[] scale = gamma.getFloat64Data();
        double[] shift = beta.getFloat64Data();
        double[] out = node.getFloat64Data();
        NormShape shape = resolveNormShape(input, gamma, beta, node, norm.getNormalizedRank());
        runGroups(shape, context, group -> applyGroupF64(
                in,
                scale,
                shift,
                out,
                input.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                gamma.getStorageOffsetUnsafe(),
                beta.getStorageOffsetUnsafe(),
                node.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                shape.normalizedSize(),
                norm.getEpsilon()
        ));
    }

    private static void executeF32(layerNorm norm, Tensor input, Tensor gamma, Tensor beta, Tensor node, CpuKernelContext context) {
        validateLayout(input, gamma, beta, node, norm.getNormalizedRank());
        float[] in = input.getFloat32Data();
        float[] scale = gamma.getFloat32Data();
        float[] shift = beta.getFloat32Data();
        float[] out = node.getFloat32Data();
        NormShape shape = resolveNormShape(input, gamma, beta, node, norm.getNormalizedRank());
        runGroups(shape, context, group -> applyGroupF32(
                in,
                scale,
                shift,
                out,
                input.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                gamma.getStorageOffsetUnsafe(),
                beta.getStorageOffsetUnsafe(),
                node.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                shape.normalizedSize(),
                (float) norm.getEpsilon()
        ));
    }

    private static void executeBF16(layerNorm norm, Tensor input, Tensor gamma, Tensor beta, Tensor node, CpuKernelContext context) {
        validateLayout(input, gamma, beta, node, norm.getNormalizedRank());
        short[] in = input.getBFloat16Data();
        short[] scale = gamma.getBFloat16Data();
        short[] shift = beta.getBFloat16Data();
        short[] out = node.getBFloat16Data();
        NormShape shape = resolveNormShape(input, gamma, beta, node, norm.getNormalizedRank());
        runGroups(shape, context, group -> applyGroupBF16(
                in,
                scale,
                shift,
                out,
                input.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                gamma.getStorageOffsetUnsafe(),
                beta.getStorageOffsetUnsafe(),
                node.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                shape.normalizedSize(),
                (float) norm.getEpsilon()
        ));
    }

    private static void runGroups(NormShape shape, CpuKernelContext context, java.util.function.IntConsumer groupBody) {
        long work = (long) shape.groupCount() * shape.normalizedSize();
        int workers = context.planner().plannedWorkers();
        if (shape.groupCount() <= 1 || workers <= 1 || work < PARALLEL_MIN_WORK) {
            for (int group = 0; group < shape.groupCount(); group++) {
                groupBody.accept(group);
            }
            return;
        }

        int chunkSize = Math.max(1, context.planner().computeChunkSize(shape.groupCount(), 1, 1, 1));
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

    private static void applyGroupBF16(
            short[] in,
            short[] gamma,
            short[] beta,
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
            float value = CpuDTypeOps.fromBFloat16Bits(in[inBase + i]);
            float scale = CpuDTypeOps.fromBFloat16Bits(gamma[gammaBase + i]);
            float shift = CpuDTypeOps.fromBFloat16Bits(beta[betaBase + i]);
            out[outBase + i] = CpuDTypeOps.toBFloat16Bits(((value - stats.mean()) * invStd) * scale + shift);
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

    private static StatsF32 computeStatsBF16(short[] in, int base, int length) {
        double total = 0.0d;
        double totalSquares = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = CpuDTypeOps.fromBFloat16Bits(in[base + i]);
            total += value;
            totalSquares += value * value;
        }
        return new StatsF32((float) (total / length), totalSquares / length);
    }

    private static boolean canUseVectorPath(int normalizedSize, int speciesLength) {
        return speciesLength > 1 && normalizedSize >= speciesLength * MIN_VECTOR_AXIS_MULTIPLIER;
    }

    private static void validateLayout(Tensor input, Tensor gamma, Tensor beta, Tensor node, int normalizedRank) {
        if (!input.isContiguous() || !gamma.isContiguous() || !beta.isContiguous() || !node.isContiguous()) {
            throw new IllegalArgumentException("CpuLayerNormKernel requires contiguous input, gamma, beta, and output tensors.");
        }
        if (normalizedRank < 1 || normalizedRank > input.getShapeUnsafe().length) {
            throw new IllegalArgumentException("Invalid LayerNorm normalized rank: " + normalizedRank);
        }
    }

    private static NormShape resolveNormShape(Tensor input, Tensor gamma, Tensor beta, Tensor node, int normalizedRank) {
        int[] inputShape = input.getShapeUnsafe();
        int[] gammaShape = gamma.getShapeUnsafe();
        int[] betaShape = beta.getShapeUnsafe();
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
            normalizedSize *= expected;
        }
        if (gamma.getFlatDataSize() != normalizedSize || beta.getFlatDataSize() != normalizedSize) {
            throw new IllegalArgumentException("LayerNorm parameter storage size mismatch.");
        }
        if (node.getFlatDataSize() != input.getFlatDataSize()) {
            throw new IllegalArgumentException("LayerNorm output size must match input size.");
        }
        return new NormShape(input.getFlatDataSize() / normalizedSize, normalizedSize);
    }

    private static layerNorm require(Operation op) {
        if (!(op instanceof layerNorm norm)) {
            throw new IllegalArgumentException("CpuLayerNormKernel requires layerNorm operation.");
        }
        return norm;
    }

    private static Tensor requireInput(List<Tensor> inputs, int index, String name) {
        if (inputs == null || inputs.size() <= index || inputs.get(index) == null) {
            throw new IllegalArgumentException("LayerNorm " + name + " tensor is missing.");
        }
        return inputs.get(index);
    }

    private record NormShape(int groupCount, int normalizedSize) {
    }

    private record StatsF64(double mean, double meanSquares) {
    }

    private record StatsF32(float mean, double meanSquares) {
    }
}
