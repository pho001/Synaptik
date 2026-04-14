package backend.kernels.cpu.elementwise.binary.bf16;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedDispatchHints;

public final class MaxBF16 {
    private MaxBF16() {}

    public static void run(short[] a, short[] b, short[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalar(a, b, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(a, b, out, hints);
        }
    }

    public static void run(float[] a, short[] b, short[] out, ResolvedDispatchHints hints) { runMixed(a, b, out, hints, true); }
    public static void run(short[] a, float[] b, short[] out, ResolvedDispatchHints hints) { runMixed(a, b, out, hints, false); }
    public static void run(float[] a, float[] b, short[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalar(a, b, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> {
                int chunkSize = hints.scalarChunkSize();
                int chunks = (out.length + chunkSize - 1) / chunkSize;
                CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                    int start = chunk * chunkSize;
                    int end = Math.min(start + chunkSize, out.length);
                    scalar(a, b, out, start, end);
                });
            }
        }
    }

    private static void scalar(short[] a, short[] b, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits(Math.max(CpuDTypeOps.fromBFloat16Bits(a[i]), CpuDTypeOps.fromBFloat16Bits(b[i])));
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

    private static void scalar(float[] a, float[] b, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits(Math.max(a[i], b[i]));
        }
    }

    private static void runMixed(Object first, Object second, short[] out, ResolvedDispatchHints hints, boolean firstIsFloat) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalarMixed(first, second, out, 0, out.length, firstIsFloat);
            case PARALLEL, PARALLEL_VECTOR -> {
                int chunkSize = hints.scalarChunkSize();
                int chunks = (out.length + chunkSize - 1) / chunkSize;
                CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
                    int start = chunk * chunkSize;
                    int end = Math.min(start + chunkSize, out.length);
                    scalarMixed(first, second, out, start, end, firstIsFloat);
                });
            }
        }
    }

    private static void scalarMixed(Object first, Object second, short[] out, int start, int end, boolean firstIsFloat) {
        float[] fa = firstIsFloat ? (float[]) first : null;
        short[] sa = firstIsFloat ? null : (short[]) first;
        short[] sb = firstIsFloat ? (short[]) second : null;
        float[] fb = firstIsFloat ? null : (float[]) second;
        for (int i = start; i < end; i++) {
            float left = firstIsFloat ? fa[i] : CpuDTypeOps.fromBFloat16Bits(sa[i]);
            float right = firstIsFloat ? CpuDTypeOps.fromBFloat16Bits(sb[i]) : fb[i];
            out[i] = CpuDTypeOps.toBFloat16Bits(Math.max(left, right));
        }
    }
}
