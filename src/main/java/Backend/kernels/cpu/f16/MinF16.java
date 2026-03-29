package Backend.kernels.cpu.f16;

import Backend.kernels.cpu.CpuDTypeOps;
import Backend.kernels.cpu.CpuExecutionConfig;
import Backend.kernels.cpu.CpuExecutionMode;
import Backend.kernels.cpu.CpuThreadPool;

public final class MinF16 {
    private MinF16() {}

    public static void run(short[] a, short[] b, short[] out, CpuExecutionMode mode, CpuExecutionConfig config) {
        switch (mode) {
            case VECTOR, SCALAR -> scalar(a, b, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(a, b, out, config);
        }
    }

    private static void scalar(short[] a, short[] b, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuDTypeOps.toHalfBits(Math.min(CpuDTypeOps.fromHalfBits(a[i]), CpuDTypeOps.fromHalfBits(b[i])));
        }
    }

    private static void parallel(short[] a, short[] b, short[] out, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(a, b, out, start, end);
        });
    }
}
