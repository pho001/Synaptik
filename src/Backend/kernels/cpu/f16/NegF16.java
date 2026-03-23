package Backend.kernels.cpu.f16;

import Backend.kernels.cpu.CpuDTypeOps;
import Backend.kernels.cpu.CpuExecutionConfig;
import Backend.kernels.cpu.CpuExecutionMode;
import Backend.kernels.cpu.CpuThreadPool;

public final class NegF16 {
    private NegF16() {}

    public static void run(short[] in, short[] out, CpuExecutionMode mode, CpuExecutionConfig config) {
        switch (mode) {
            case VECTOR, SCALAR -> scalar(in, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(in, out, config);
        }
    }

    private static void scalar(short[] in, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuDTypeOps.toHalfBits(-CpuDTypeOps.fromHalfBits(in[i]));
        }
    }

    private static void parallel(short[] in, short[] out, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, out, start, end);
        });
    }
}
