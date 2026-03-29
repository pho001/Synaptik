package Backend.kernels.cpu.f16;

import Backend.kernels.cpu.CpuDTypeOps;
import Backend.kernels.cpu.CpuExecutionConfig;
import Backend.kernels.cpu.CpuExecutionMode;
import Backend.kernels.cpu.CpuThreadPool;

public final class PowF16 {
    private PowF16() {}

    public static void run(short[] in, double exponent, short[] out, CpuExecutionMode mode, CpuExecutionConfig config) {
        switch (mode) {
            case VECTOR, SCALAR -> scalar(in, exponent, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(in, exponent, out, config);
        }
    }

    private static void scalar(short[] in, double exponent, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            float x = CpuDTypeOps.fromHalfBits(in[i]);
            float y = (float) Math.pow(x, exponent);
            out[i] = CpuDTypeOps.toHalfBits(y);
        }
    }

    private static void parallel(short[] in, double exponent, short[] out, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, exponent, out, start, end);
        });
    }
}
