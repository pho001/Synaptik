package backend.kernels.cpu.elementwise.unary.bf16;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;

public final class NegBF16 {
    private NegBF16() {}

    public static void run(short[] in, short[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalar(in, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(in, out, hints);
        }
    }

    public static void run(float[] in, short[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalar(in, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(in, out, hints);
        }
    }

    private static void scalar(short[] in, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits(-CpuDTypeOps.fromBFloat16Bits(in[i]));
        }
    }

    private static void scalar(float[] in, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits(-in[i]);
        }
    }

    private static void parallel(short[] in, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, out, start, end);
        });
    }

    private static void parallel(float[] in, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, out, start, end);
        });
    }
}
