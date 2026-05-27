package backend.cpu.kernels.linalg;

import tensor.dtype.TensorDTypeOps;

import backend.cpu.execution.CpuKernelContext;

import backend.cpu.storage.CpuStorageView;
import tensor.TensorInternalAccess;

import backend.cpu.kernels.*;
import backend.cpu.kernels.linalg.matmul.bf16.BF16MatMulJavaBackend;
import backend.cpu.kernels.linalg.matmul.common.PackedLinearWeightCache;
import backend.cpu.provider.linalg.matmul.PreparedMatMulExecutable;
import backend.cpu.kernels.linalg.matmul.f32.F32MatMulJavaBackend;
import backend.cpu.kernels.linalg.matmul.f64.F64MatMulJavaBackend;
import backend.cpu.plan.linalg.matmul.ResolvedMatMulHints;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import operations.linalg.linear;
import tensor.Tensor;

import java.util.Arrays;

/**
 * CPU implementation for the semantic linear projection {@code [..., in] x [in, out]}.
 *
 * <p>The executor treats every leading input dimension as a flattened row prefix.
 * That means the same kernels cover both rank-2 matrices such as
 * {@code [batch, inFeatures]} and sequence-shaped tensors such as
 * {@code [batch, time, inFeatures]}. Bias handling is likewise last-dimension
 * based: {@code [outFeatures]} and {@code [1, outFeatures]} both add one value
 * per output feature to every flattened row.</p>
 */
final class LinearExecutor {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    private LinearExecutor() {
    }

    static void forwardStorage(linear op, CpuStorageView input, CpuStorageView weight, CpuStorageView bias, CpuStorageView out) {
        CpuLinearStorageLoops.execute(input, weight, op.hasBias() ? bias : null, out);
    }

    static void forwardDenseArrayF64(linear op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        if (!tryPackedLinearF64(input, weight, out, context)) {
            requireExecutable(context).execute(input, weight, out, context);
        }
        if (op.hasBias()) {
            addBiasF64(out, bias);
        }
    }

    static void forwardDenseArrayF32(linear op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        if (!tryPackedLinearF32(input, weight, out, context)) {
            requireExecutable(context).execute(input, weight, out, context);
        }
        if (op.hasBias()) {
            addBiasF32(out, bias);
        }
    }

    static void forwardDenseArrayBF16(linear op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        if (context.publishFloatContinuation() && tryPublishFloatContinuationBF16(input, weight, bias, out, context)) {
            return;
        }
        if (op.hasBias() && tryBlasForwardBF16WithFloatContinuation(input, weight, bias, out, context)) {
            return;
        }
        if (tryPackedLinearBF16(input, weight, out, context)) {
            if (op.hasBias()) {
                addBiasBF16(out, bias);
            }
            return;
        }
        requireExecutable(context).execute(input, weight, out, context);
        if (op.hasBias()) {
            addBiasBF16(out, bias);
        }
    }

    private static boolean tryBlasForwardBF16WithFloatContinuation(
            Tensor input,
            Tensor weight,
            Tensor bias,
            Tensor out,
            CpuKernelContext context
    ) {
        if (context == null || context.cpuWorkspace() == null) {
            return false;
        }
        if (!requireExecutable(context).executeToFloatWorkspace(input, weight, out, context, false)) {
            return false;
        }

        float[] tmp = context.cpuWorkspace().requireFloatWorkspace();
        short[] od = TensorInternalAccess.bfloat16Data(out);
        addBiasAndMaterializeBF16(tmp, od, TensorInternalAccess.bfloat16Data(bias), biasFeatureCount(bias), od.length);
        return true;
    }

    private static boolean tryPublishFloatContinuationBF16(
            Tensor input,
            Tensor weight,
            Tensor bias,
            Tensor out,
            CpuKernelContext context
    ) {
        if (context == null || context.cpuWorkspace() == null) {
            return false;
        }
        if (!requireExecutable(context).executeToFloatWorkspace(input, weight, out, context, true)) {
            return false;
        }

        float[] tmp = context.cpuWorkspace().requireFloatWorkspace();
        if (bias != null) {
            addBiasInPlace(tmp, TensorInternalAccess.bfloat16Data(bias), biasFeatureCount(bias), out.getFlatDataSize());
        }
        context.cpuWorkspace().publishFloatContinuation(out.getFlatDataSize());
        return true;
    }

    private static boolean tryPackedLinearF64(Tensor input, Tensor weight, Tensor out, CpuKernelContext context) {
        if (context == null || context.cpuWorkspace() == null || context.cpuWorkspace().packedLinearWeightCache() == null) {
            return false;
        }
        ResolvedMatMulHints hints = requireHints(context);
        if (hints.useBlas() || hints.useBatchedBlas()) {
            return false;
        }
        PackedLinearWeightCache.F64PackedWeights packed = context.cpuWorkspace().packedLinearWeightCache().requireF64(weight, hints);
        if (packed == null) {
            return false;
        }
        double[] outData = TensorInternalAccess.float64Data(out);
        Arrays.fill(outData, 0.0d);
        F64MatMulJavaBackend.runPacked(TensorInternalAccess.float64Data(input), input.getShapeUnsafe(), packed, outData, out.getShapeUnsafe(), hints);
        return true;
    }

    private static boolean tryPackedLinearF32(Tensor input, Tensor weight, Tensor out, CpuKernelContext context) {
        if (context == null || context.cpuWorkspace() == null || context.cpuWorkspace().packedLinearWeightCache() == null) {
            return false;
        }
        ResolvedMatMulHints hints = requireHints(context);
        if (hints.useBlas() || hints.useBatchedBlas()) {
            return false;
        }
        PackedLinearWeightCache.F32PackedWeights packed = context.cpuWorkspace().packedLinearWeightCache().requireF32(weight, hints);
        if (packed == null) {
            return false;
        }
        float[] outData = TensorInternalAccess.float32Data(out);
        Arrays.fill(outData, 0.0f);
        F32MatMulJavaBackend.runPacked(TensorInternalAccess.float32Data(input), input.getShapeUnsafe(), packed, outData, out.getShapeUnsafe(), hints);
        return true;
    }

    private static boolean tryPackedLinearBF16(Tensor input, Tensor weight, Tensor out, CpuKernelContext context) {
        if (context == null || context.cpuWorkspace() == null || context.cpuWorkspace().packedLinearWeightCache() == null) {
            return false;
        }
        ResolvedMatMulHints hints = requireHints(context);
        if (hints.useBlas() || hints.useBatchedBlas()) {
            return false;
        }
        PackedLinearWeightCache.BF16PackedWeights packed = context.cpuWorkspace().packedLinearWeightCache().requireBF16(weight, hints);
        if (packed == null) {
            return false;
        }
        BF16MatMulJavaBackend.runPacked(TensorInternalAccess.bfloat16Data(input), input.getShapeUnsafe(), packed, TensorInternalAccess.bfloat16Data(out), out.getShapeUnsafe(), hints);
        return true;
    }

    private static void addBiasF64(Tensor out, Tensor bias) {
        double[] outData = TensorInternalAccess.float64Data(out);
        double[] biasData = TensorInternalAccess.float64Data(bias);
        int outFeatures = biasFeatureCount(bias);
        int width = F64.length();
        int vectorLimit = outFeatures - (outFeatures % width);
        for (int base = 0; base < outData.length; base += outFeatures) {
            int j = 0;
            for (; j < vectorLimit; j += width) {
                DoubleVector.fromArray(F64, outData, base + j)
                        .add(DoubleVector.fromArray(F64, biasData, j))
                        .intoArray(outData, base + j);
            }
            for (; j < outFeatures; j++) {
                outData[base + j] += biasData[j];
            }
        }
    }

    private static void addBiasF32(Tensor out, Tensor bias) {
        float[] outData = TensorInternalAccess.float32Data(out);
        float[] biasData = TensorInternalAccess.float32Data(bias);
        int outFeatures = biasFeatureCount(bias);
        int width = F32.length();
        int vectorLimit = outFeatures - (outFeatures % width);
        for (int base = 0; base < outData.length; base += outFeatures) {
            int j = 0;
            for (; j < vectorLimit; j += width) {
                FloatVector.fromArray(F32, outData, base + j)
                        .add(FloatVector.fromArray(F32, biasData, j))
                        .intoArray(outData, base + j);
            }
            for (; j < outFeatures; j++) {
                outData[base + j] += biasData[j];
            }
        }
    }

    private static void addBiasBF16(Tensor out, Tensor bias) {
        short[] outData = TensorInternalAccess.bfloat16Data(out);
        short[] biasData = TensorInternalAccess.bfloat16Data(bias);
        int outFeatures = biasFeatureCount(bias);
        for (int i = 0; i < outData.length; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(outData[i]) + TensorDTypeOps.fromBFloat16Bits(biasData[i % outFeatures]);
            outData[i] = TensorDTypeOps.toBFloat16Bits(value);
        }
    }

    private static void addBiasAndMaterializeBF16(
            float[] src,
            short[] out,
            short[] bias,
            int outFeatures,
            int length
    ) {
        int limit = Math.min(length, Math.min(src.length, out.length));
        for (int i = 0; i < limit; i++) {
            float value = src[i] + TensorDTypeOps.fromBFloat16Bits(bias[i % outFeatures]);
            out[i] = TensorDTypeOps.toBFloat16Bits(value);
        }
    }

    private static void addBiasInPlace(float[] src, short[] bias, int outFeatures, int length) {
        int limit = Math.min(length, src.length);
        for (int i = 0; i < limit; i++) {
            src[i] += TensorDTypeOps.fromBFloat16Bits(bias[i % outFeatures]);
        }
    }

    /**
     * Returns the feature width used for row-wise bias broadcasting.
     *
     * <p>Validation in {@code LinearSpec} restricts bias to {@code [outFeatures]}
     * or {@code [1, outFeatures]}, so the last dimension is the only value the
     * executor needs when it walks the flattened output buffer.</p>
     */
    private static int biasFeatureCount(Tensor bias) {
        int[] shape = bias.getShapeUnsafe();
        return shape[shape.length - 1];
    }

    private static ResolvedMatMulHints requireHints(CpuKernelContext context) {
        ResolvedMatMulHints hints = context.matMulHints();
        if (hints == null) {
            throw new IllegalStateException("Missing ResolvedMatMulHints for linear execution.");
        }
        return hints;
    }

    private static PreparedMatMulExecutable requireExecutable(CpuKernelContext context) {
        PreparedMatMulExecutable executable = context.matMulExecutable();
        if (executable == null) {
            throw new IllegalStateException("Missing PreparedMatMulExecutable for linear execution.");
        }
        return executable;
    }
}
