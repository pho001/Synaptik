package backend.kernels.cpu.bf16;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedDispatchHints;

public final class MinBF16 {
    private MinBF16() {}

    public static void run(short[] a, short[] b, short[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalar(a, b, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(a, b, out, hints);
        }
    }

    private static void scalar(short[] a, short[] b, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits(Math.min(CpuDTypeOps.fromBFloat16Bits(a[i]), CpuDTypeOps.fromBFloat16Bits(b[i])));
        }
    }

    private static void parallel(short[] a, short[] b, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(a, b, out, start, end);
        });
    }
}
