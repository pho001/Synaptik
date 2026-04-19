package backend.kernels.cpu.elementwise.unary.bf16;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import backend.kernels.cpu.elementwise.unary.support.CpuPowSupport;

public final class PowBF16 {
    private PowBF16() {}

    public static void run(short[] in, double exponent, short[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalar(in, exponent, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(in, exponent, out, hints);
        }
    }

    public static void run(float[] in, float exponent, short[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalar(in, exponent, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(in, exponent, out, hints);
        }
    }

    private static void scalar(short[] in, double exponent, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            float x = CpuDTypeOps.fromBFloat16Bits(in[i]);
            out[i] = CpuDTypeOps.toBFloat16Bits(CpuPowSupport.applyF32(x, (float) exponent));
        }
    }

    private static void scalar(float[] in, float exponent, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits(CpuPowSupport.applyF32(in[i], exponent));
        }
    }

    private static void parallel(short[] in, double exponent, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, exponent, out, start, end);
        });
    }

    private static void parallel(float[] in, float exponent, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, exponent, out, start, end);
        });
    }
}
