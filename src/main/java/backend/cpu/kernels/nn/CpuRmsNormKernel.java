package backend.cpu.kernels.nn;

import tensor.TensorInternalAccess;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.CpuThreadPool;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import operations.Operation;
import operations.normalization.rmsNorm;
import tensor.Tensor;

import java.util.List;

public final class CpuRmsNormKernel implements CpuKernel {
    private static final VectorSpecies<Float> F32_SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Double> F64_SPECIES = DoubleVector.SPECIES_PREFERRED;
    private static final int MIN_VECTOR_AXIS_MULTIPLIER = 4;
    private static final int PARALLEL_MIN_WORK = 16_384;

    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        rmsNorm norm = require(op);
        Tensor input = requireInput(inputs, 0, "input");
        Tensor gamma = requireInput(inputs, 1, "gamma");
        executeF64(norm, input, gamma, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        rmsNorm norm = require(op);
        Tensor input = requireInput(inputs, 0, "input");
        Tensor gamma = requireInput(inputs, 1, "gamma");
        executeF32(norm, input, gamma, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        rmsNorm norm = require(op);
        Tensor input = requireInput(inputs, 0, "input");
        Tensor gamma = requireInput(inputs, 1, "gamma");
        float[] inputContinuation = context.inputFloatContinuation(0, input.getFlatDataSize());
        if (context.publishFloatContinuation() && context.cpuWorkspace() != null) {
            float[] out = context.cpuWorkspace().requireFloatWorkspace();
            if (inputContinuation != null) {
                executeBF16ToFloat(norm, input, inputContinuation, gamma, node, out, context);
            } else {
                executeBF16ToFloat(norm, input, gamma, node, out, context);
            }
            context.cpuWorkspace().publishFloatContinuation(node.getFlatDataSize());
            return;
        }
        if (inputContinuation != null) {
            executeBF16(norm, input, inputContinuation, gamma, node, context);
            return;
        }
        executeBF16(norm, input, gamma, node, context);
    }

    private static void executeF64(rmsNorm norm, Tensor input, Tensor gamma, Tensor node, CpuKernelContext context) {
        validateLayout(input, gamma, node, norm.getNormalizedRank());
        double[] in = TensorInternalAccess.float64Data(input);
        double[] weights = TensorInternalAccess.float64Data(gamma);
        double[] out = TensorInternalAccess.float64Data(node);
        NormShape shape = resolveNormShape(input, gamma, node, norm.getNormalizedRank());
        runGroups(shape, context, group -> applyGroupF64(
                in,
                weights,
                out,
                input.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                gamma.getStorageOffsetUnsafe(),
                node.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                shape.normalizedSize(),
                norm.getEpsilon()
        ));
    }

    private static void executeF32(rmsNorm norm, Tensor input, Tensor gamma, Tensor node, CpuKernelContext context) {
        validateLayout(input, gamma, node, norm.getNormalizedRank());
        float[] in = TensorInternalAccess.float32Data(input);
        float[] weights = TensorInternalAccess.float32Data(gamma);
        float[] out = TensorInternalAccess.float32Data(node);
        NormShape shape = resolveNormShape(input, gamma, node, norm.getNormalizedRank());
        runGroups(shape, context, group -> applyGroupF32(
                in,
                weights,
                out,
                input.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                gamma.getStorageOffsetUnsafe(),
                node.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                shape.normalizedSize(),
                (float) norm.getEpsilon()
        ));
    }

    private static void executeBF16(rmsNorm norm, Tensor input, Tensor gamma, Tensor node, CpuKernelContext context) {
        validateLayout(input, gamma, node, norm.getNormalizedRank());
        short[] in = TensorInternalAccess.bfloat16Data(input);
        NormShape shape = resolveNormShape(input, gamma, node, norm.getNormalizedRank());
        float[] weights = decodeBFloat16(TensorInternalAccess.bfloat16Data(gamma), gamma.getStorageOffsetUnsafe(), shape.normalizedSize());
        short[] out = TensorInternalAccess.bfloat16Data(node);
        runGroups(shape, context, group -> applyGroupBF16(
                in,
                weights,
                out,
                input.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                0,
                node.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                shape.normalizedSize(),
                (float) norm.getEpsilon()
        ));
    }

    private static void executeBF16(
            rmsNorm norm,
            Tensor input,
            float[] inputContinuation,
            Tensor gamma,
            Tensor node,
            CpuKernelContext context
    ) {
        validateLayout(input, gamma, node, norm.getNormalizedRank());
        if (inputContinuation == null) {
            throw new IllegalArgumentException("RmsNorm BF16 continuation input cannot be null.");
        }
        NormShape shape = resolveNormShape(input, gamma, node, norm.getNormalizedRank());
        float[] weights = decodeBFloat16(TensorInternalAccess.bfloat16Data(gamma), gamma.getStorageOffsetUnsafe(), shape.normalizedSize());
        short[] out = TensorInternalAccess.bfloat16Data(node);
        runGroups(shape, context, group -> applyGroupF32ToBF16(
                inputContinuation,
                weights,
                out,
                group * shape.normalizedSize(),
                0,
                node.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                shape.normalizedSize(),
                (float) norm.getEpsilon()
        ));
    }

    private static void executeBF16ToFloat(
            rmsNorm norm,
            Tensor input,
            Tensor gamma,
            Tensor node,
            float[] out,
            CpuKernelContext context
    ) {
        validateLayout(input, gamma, node, norm.getNormalizedRank());
        if (out == null || out.length < node.getFlatDataSize()) {
            throw new IllegalArgumentException("RmsNorm float continuation output is missing or too small.");
        }
        short[] in = TensorInternalAccess.bfloat16Data(input);
        NormShape shape = resolveNormShape(input, gamma, node, norm.getNormalizedRank());
        float[] weights = decodeBFloat16(TensorInternalAccess.bfloat16Data(gamma), gamma.getStorageOffsetUnsafe(), shape.normalizedSize());
        runGroups(shape, context, group -> applyGroupBF16ToF32(
                in,
                weights,
                out,
                input.getStorageOffsetUnsafe() + group * shape.normalizedSize(),
                0,
                group * shape.normalizedSize(),
                shape.normalizedSize(),
                (float) norm.getEpsilon()
        ));
    }

    private static void executeBF16ToFloat(
            rmsNorm norm,
            Tensor input,
            float[] inputContinuation,
            Tensor gamma,
            Tensor node,
            float[] out,
            CpuKernelContext context
    ) {
        validateLayout(input, gamma, node, norm.getNormalizedRank());
        if (inputContinuation == null) {
            throw new IllegalArgumentException("RmsNorm BF16 continuation input cannot be null.");
        }
        if (out == null || out.length < node.getFlatDataSize()) {
            throw new IllegalArgumentException("RmsNorm float continuation output is missing or too small.");
        }
        NormShape shape = resolveNormShape(input, gamma, node, norm.getNormalizedRank());
        float[] weights = decodeBFloat16(TensorInternalAccess.bfloat16Data(gamma), gamma.getStorageOffsetUnsafe(), shape.normalizedSize());
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

    private static void applyGroupBF16(short[] in, float[] gamma, short[] out, int inBase, int gammaBase, int outBase, int normalizedSize, float epsilon) {
        float invRms = (float) (1.0d / Math.sqrt(sumSquaresBF16(in, inBase, normalizedSize) / normalizedSize + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = CpuDTypeOps.fromBFloat16Bits(in[inBase + i]);
            out[outBase + i] = CpuDTypeOps.toBFloat16Bits(value * gamma[gammaBase + i] * invRms);
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
                    out[outBase + i + lane] = CpuDTypeOps.toBFloat16Bits(lanes[lane]);
                }
            }
            for (; i < normalizedSize; i++) {
                out[outBase + i] = CpuDTypeOps.toBFloat16Bits(in[inBase + i] * gamma[gammaBase + i] * invRms);
            }
            return;
        }
        for (int i = 0; i < normalizedSize; i++) {
            out[outBase + i] = CpuDTypeOps.toBFloat16Bits(in[inBase + i] * gamma[gammaBase + i] * invRms);
        }
    }

    private static void applyGroupBF16ToF32(short[] in, float[] gamma, float[] out, int inBase, int gammaBase, int outBase, int normalizedSize, float epsilon) {
        float invRms = (float) (1.0d / Math.sqrt(sumSquaresBF16(in, inBase, normalizedSize) / normalizedSize + epsilon));
        for (int i = 0; i < normalizedSize; i++) {
            float value = CpuDTypeOps.fromBFloat16Bits(in[inBase + i]);
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

    private static double sumSquaresBF16(short[] in, int base, int length) {
        double total = 0.0d;
        for (int i = 0; i < length; i++) {
            float value = CpuDTypeOps.fromBFloat16Bits(in[base + i]);
            total += value * value;
        }
        return total;
    }

    private static float[] decodeBFloat16(short[] values, int base, int length) {
        float[] decoded = new float[length];
        for (int i = 0; i < length; i++) {
            decoded[i] = CpuDTypeOps.fromBFloat16Bits(values[base + i]);
        }
        return decoded;
    }

    private static boolean canUseVectorPath(int normalizedSize, int speciesLength) {
        return speciesLength > 1 && normalizedSize >= speciesLength * MIN_VECTOR_AXIS_MULTIPLIER;
    }

    private static void validateLayout(Tensor input, Tensor gamma, Tensor node, int normalizedRank) {
        if (!input.isContiguous() || !gamma.isContiguous() || !node.isContiguous()) {
            throw new IllegalArgumentException("CpuRmsNormKernel requires contiguous input, gamma, and output tensors.");
        }
        if (normalizedRank < 1 || normalizedRank > input.getShapeUnsafe().length) {
            throw new IllegalArgumentException("Invalid RMSNorm normalized rank: " + normalizedRank);
        }
    }

    private static NormShape resolveNormShape(Tensor input, Tensor gamma, Tensor node, int normalizedRank) {
        int[] inputShape = input.getShapeUnsafe();
        int[] gammaShape = gamma.getShapeUnsafe();
        if (gammaShape.length != normalizedRank) {
            throw new IllegalArgumentException("RMSNorm gamma rank must equal normalized rank.");
        }
        if (node.getFlatDataSize() != input.getFlatDataSize()) {
            throw new IllegalArgumentException("RMSNorm output size must match input size.");
        }

        int normalizedSize = 1;
        int start = inputShape.length - normalizedRank;
        for (int i = 0; i < normalizedRank; i++) {
            int expected = inputShape[start + i];
            if (gammaShape[i] != expected) {
                throw new IllegalArgumentException("RMSNorm gamma shape must match trailing input dimensions.");
            }
            normalizedSize *= expected;
        }
        if (gamma.getFlatDataSize() != normalizedSize) {
            throw new IllegalArgumentException("RMSNorm gamma storage size mismatch.");
        }
        int groupCount = input.getFlatDataSize() / normalizedSize;
        return new NormShape(groupCount, normalizedSize);
    }

    private static rmsNorm require(Operation op) {
        if (!(op instanceof rmsNorm norm)) {
            throw new IllegalArgumentException("CpuRmsNormKernel requires rmsNorm operation.");
        }
        return norm;
    }

    private static Tensor requireInput(List<Tensor> inputs, int index, String name) {
        if (inputs == null || inputs.size() <= index || inputs.get(index) == null) {
            throw new IllegalArgumentException("RMSNorm " + name + " tensor is missing.");
        }
        return inputs.get(index);
    }

    private record NormShape(int groupCount, int normalizedSize) {
    }
}
