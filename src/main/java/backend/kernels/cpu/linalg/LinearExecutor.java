package backend.kernels.cpu.linalg;

import backend.kernels.cpu.*;
import backend.kernels.cpu.linalg.matmul.plan.ResolvedMatMulHints;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import operations.linalg.linear;
import tensor.Tensor;

import java.util.Arrays;

final class LinearExecutor {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    private LinearExecutor() {
    }

    static void forwardF64(linear op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        if (!tryPackedLinearF64(input, weight, out, context)) {
            MatMulExecutor.forwardF64(input, weight, out, context);
        }
        if (op.hasBias()) {
            addBiasF64(out, bias);
        }
    }

    static void forwardF32(linear op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        if (!tryPackedLinearF32(input, weight, out, context)) {
            MatMulExecutor.forwardF32(input, weight, out, context);
        }
        if (op.hasBias()) {
            addBiasF32(out, bias);
        }
    }

    static void forwardBF16(linear op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        if (context.publishFloatContinuation() && tryPublishFloatContinuationBF16(input, weight, bias, out, context)) {
            return;
        }
        if (op.hasBias() && tryBlasForwardBF16WithFloatContinuation(input, weight, bias, out, context)) {
            return;
        }
        MatMulExecutor.forwardBF16(input, weight, out, context);
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
        ResolvedMatMulHints hints = requireHints(context);
        if (!hints.useBlas() && !hints.useBatchedBlas()) {
            return false;
        }

        int[] as = input.getShapeUnsafe();
        int[] bs = weight.getShapeUnsafe();
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];
        short[] ad = input.getBFloat16Data();
        short[] bd = weight.getBFloat16Data();
        short[] od = out.getBFloat16Data();
        float[] tmp = context.cpuWorkspace().requireFloatWorkspace();

        boolean executed = (as.length == 2 && bs.length == 2 && hints.useBlas()
                && MatMulBlasBackend.tryBlasBF16ToFloat(ad, bd, tmp, m, n, k))
                || (hints.useBatchedBlas()
                && MatMulBlasBackend.tryBatchedBlasBF16ToFloat(ad, as, bd, bs, tmp, out.getShapeUnsafe(), m, n, k));
        if (!executed) {
            return false;
        }

        addBiasAndMaterializeBF16(tmp, od, bias.getBFloat16Data(), bias.getShapeUnsafe()[0], od.length);
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
        ResolvedMatMulHints hints = requireHints(context);
        float[] tmp = context.cpuWorkspace().requireFloatWorkspace();
        if (!tryMatMulToFloatBF16(input, weight, out, hints, tmp)) {
            return false;
        }

        if (bias != null) {
            addBiasInPlace(tmp, bias.getBFloat16Data(), bias.getShapeUnsafe()[0], out.getFlatDataSize());
        }
        context.cpuWorkspace().publishFloatContinuation(out.getFlatDataSize());
        return true;
    }

    private static boolean tryMatMulToFloatBF16(
            Tensor input,
            Tensor weight,
            Tensor out,
            ResolvedMatMulHints hints,
            float[] tmp
    ) {
        int[] as = input.getShapeUnsafe();
        int[] bs = weight.getShapeUnsafe();
        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];
        short[] ad = input.getBFloat16Data();
        short[] bd = weight.getBFloat16Data();

        boolean executed = (as.length == 2 && bs.length == 2 && hints.useBlas()
                && MatMulBlasBackend.tryBlasBF16ToFloat(ad, bd, tmp, m, n, k))
                || (hints.useBatchedBlas()
                && MatMulBlasBackend.tryBatchedBlasBF16ToFloat(ad, as, bd, bs, tmp, out.getShapeUnsafe(), m, n, k));
        if (executed) {
            return true;
        }
        MatMulJavaBackend.runBF16ToFloat(ad, as, bd, bs, tmp, out.getShapeUnsafe(), hints);
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
        double[] outData = out.getFloat64Data();
        Arrays.fill(outData, 0.0d);
        MatMulJavaBackend.runPackedF64(input.getFloat64Data(), input.getShapeUnsafe(), packed, outData, out.getShapeUnsafe(), hints);
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
        float[] outData = out.getFloat32Data();
        Arrays.fill(outData, 0.0f);
        MatMulJavaBackend.runPackedF32(input.getFloat32Data(), input.getShapeUnsafe(), packed, outData, out.getShapeUnsafe(), hints);
        return true;
    }

    private static void addBiasF64(Tensor out, Tensor bias) {
        double[] outData = out.getFloat64Data();
        double[] biasData = bias.getFloat64Data();
        int outFeatures = bias.getShapeUnsafe()[0];
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
        float[] outData = out.getFloat32Data();
        float[] biasData = bias.getFloat32Data();
        int outFeatures = bias.getShapeUnsafe()[0];
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
        short[] outData = out.getBFloat16Data();
        short[] biasData = bias.getBFloat16Data();
        int outFeatures = bias.getShapeUnsafe()[0];
        for (int i = 0; i < outData.length; i++) {
            float value = CpuDTypeOps.fromBFloat16Bits(outData[i]) + CpuDTypeOps.fromBFloat16Bits(biasData[i % outFeatures]);
            outData[i] = CpuDTypeOps.toBFloat16Bits(value);
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
            float value = src[i] + CpuDTypeOps.fromBFloat16Bits(bias[i % outFeatures]);
            out[i] = CpuDTypeOps.toBFloat16Bits(value);
        }
    }

    private static void addBiasInPlace(float[] src, short[] bias, int outFeatures, int length) {
        int limit = Math.min(length, src.length);
        for (int i = 0; i < limit; i++) {
            src[i] += CpuDTypeOps.fromBFloat16Bits(bias[i % outFeatures]);
        }
    }

    private static ResolvedMatMulHints requireHints(CpuKernelContext context) {
        return MatMulExecutor.requireHints(context);
    }
}
