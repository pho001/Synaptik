package backend.kernels.cpu.linalg;

import backend.kernels.cpu.*;

import operations.linear;
import tensor.Tensor;

final class LinearExecutor {
    private LinearExecutor() {
    }

    static void forwardF64(linear op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        MatMulExecutor.forwardF64(input, weight, out, context);
        if (op.hasBias()) {
            addBiasF64(out, bias);
        }
    }

    static void forwardF32(linear op, Tensor input, Tensor weight, Tensor bias, Tensor out, CpuKernelContext context) {
        MatMulExecutor.forwardF32(input, weight, out, context);
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
        float[] tmp = context.cpuWorkspace().requireFloatWorkspace();

        boolean executed = (as.length == 2 && bs.length == 2 && hints.useBlas()
                && MatMulBlasBackend.tryBlasBF16ToFloat(ad, bd, tmp, m, n, k))
                || (hints.useBatchedBlas()
                && MatMulBlasBackend.tryBatchedBlasBF16ToFloat(ad, as, bd, bs, tmp, out.getShapeUnsafe(), m, n, k));
        if (!executed) {
            return false;
        }

        if (bias != null) {
            addBiasInPlace(tmp, bias.getBFloat16Data(), bias.getShapeUnsafe()[0], out.getFlatDataSize());
        }
        context.cpuWorkspace().publishFloatContinuation(out.getFlatDataSize());
        return true;
    }

    private static void addBiasF64(Tensor out, Tensor bias) {
        double[] outData = out.getFloat64Data();
        double[] biasData = bias.getFloat64Data();
        int outFeatures = bias.getShapeUnsafe()[0];
        for (int i = 0; i < outData.length; i++) {
            outData[i] += biasData[i % outFeatures];
        }
    }

    private static void addBiasF32(Tensor out, Tensor bias) {
        float[] outData = out.getFloat32Data();
        float[] biasData = bias.getFloat32Data();
        int outFeatures = bias.getShapeUnsafe()[0];
        for (int i = 0; i < outData.length; i++) {
            outData[i] += biasData[i % outFeatures];
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
