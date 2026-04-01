package backend.kernels.cpu.f16;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedDispatchHints;

public final class PowF16 {
    private PowF16() {}

    public static void run(short[] in, double exponent, short[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalar(in, exponent, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(in, exponent, out, hints);
        }
    }

    private static void scalar(short[] in, double exponent, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            float x = CpuDTypeOps.fromHalfBits(in[i]);
            float y = (float) Math.pow(x, exponent);
            out[i] = CpuDTypeOps.toHalfBits(y);
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
}
