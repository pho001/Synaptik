package backend.kernels.cpu.elementwise.unary.bf16;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedDispatchHints;

public final class MulScalarBF16 {
    private MulScalarBF16() {}

    public static void run(short[] in, double scalar, short[] out, ResolvedDispatchHints hints) {
        float s = (float) scalar;
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalar(in, s, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(in, s, out, hints);
        }
    }

    public static void run(float[] in, float scalar, short[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalar(in, scalar, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(in, scalar, out, hints);
        }
    }

    private static void scalar(short[] in, float scalar, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(in[i]) * scalar);
        }
    }

    private static void scalar(float[] in, float scalar, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits(in[i] * scalar);
        }
    }

    private static void parallel(short[] in, float scalar, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, scalar, out, start, end);
        });
    }

    private static void parallel(float[] in, float scalar, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, scalar, out, start, end);
        });
    }
}
