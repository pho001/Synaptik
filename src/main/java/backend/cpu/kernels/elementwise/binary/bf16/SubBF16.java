package backend.cpu.kernels.elementwise.binary.bf16;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuExecutionMode;
import backend.cpu.kernels.CpuThreadPool;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public final class SubBF16 {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private SubBF16() {}

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
            case VECTOR -> vector(a, b, out, 0, out.length);
            case PARALLEL -> parallel(a, b, out, hints);
            case PARALLEL_VECTOR -> parallelVector(a, b, out, hints);
            case SCALAR -> scalar(a, b, out, 0, out.length);
        }
    }

    private static void scalar(short[] a, short[] b, short[] out, int start, int end) {
        int i = start;
        int upper = end - ((end - start) & 3);
        for (; i < upper; i += 4) {
            out[i] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(a[i]) - CpuDTypeOps.fromBFloat16Bits(b[i]));
            out[i + 1] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(a[i + 1]) - CpuDTypeOps.fromBFloat16Bits(b[i + 1]));
            out[i + 2] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(a[i + 2]) - CpuDTypeOps.fromBFloat16Bits(b[i + 2]));
            out[i + 3] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(a[i + 3]) - CpuDTypeOps.fromBFloat16Bits(b[i + 3]));
        }
        for (; i < end; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(a[i]) - CpuDTypeOps.fromBFloat16Bits(b[i]));
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
            out[i] = CpuDTypeOps.toBFloat16Bits(a[i] - b[i]);
        }
    }

    private static void vector(float[] a, float[] b, short[] out, int start, int end) {
        float[] lanes = new float[SPECIES.length()];
        int width = SPECIES.length();
        int i = start;
        int upper = start + SPECIES.loopBound(end - start);
        for (; i < upper; i += width) {
            FloatVector.fromArray(SPECIES, a, i)
                    .sub(FloatVector.fromArray(SPECIES, b, i))
                    .intoArray(lanes, 0);
            for (int lane = 0; lane < width; lane++) {
                out[i + lane] = CpuDTypeOps.toBFloat16Bits(lanes[lane]);
            }
        }
        scalar(a, b, out, i, end);
    }

    private static void parallel(float[] a, float[] b, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(a, b, out, start, end);
        });
    }

    private static void parallelVector(float[] a, float[] b, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.vectorChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            vector(a, b, out, start, end);
        });
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
        int i = start;
        int upper = end - ((end - start) & 3);
        for (; i < upper; i += 4) {
            float left0 = firstIsFloat ? fa[i] : CpuDTypeOps.fromBFloat16Bits(sa[i]);
            float right0 = firstIsFloat ? CpuDTypeOps.fromBFloat16Bits(sb[i]) : fb[i];
            out[i] = CpuDTypeOps.toBFloat16Bits(left0 - right0);
            float left1 = firstIsFloat ? fa[i + 1] : CpuDTypeOps.fromBFloat16Bits(sa[i + 1]);
            float right1 = firstIsFloat ? CpuDTypeOps.fromBFloat16Bits(sb[i + 1]) : fb[i + 1];
            out[i + 1] = CpuDTypeOps.toBFloat16Bits(left1 - right1);
            float left2 = firstIsFloat ? fa[i + 2] : CpuDTypeOps.fromBFloat16Bits(sa[i + 2]);
            float right2 = firstIsFloat ? CpuDTypeOps.fromBFloat16Bits(sb[i + 2]) : fb[i + 2];
            out[i + 2] = CpuDTypeOps.toBFloat16Bits(left2 - right2);
            float left3 = firstIsFloat ? fa[i + 3] : CpuDTypeOps.fromBFloat16Bits(sa[i + 3]);
            float right3 = firstIsFloat ? CpuDTypeOps.fromBFloat16Bits(sb[i + 3]) : fb[i + 3];
            out[i + 3] = CpuDTypeOps.toBFloat16Bits(left3 - right3);
        }
        for (; i < end; i++) {
            float left = firstIsFloat ? fa[i] : CpuDTypeOps.fromBFloat16Bits(sa[i]);
            float right = firstIsFloat ? CpuDTypeOps.fromBFloat16Bits(sb[i]) : fb[i];
            out[i] = CpuDTypeOps.toBFloat16Bits(left - right);
        }
    }
}
