package backend.cpu.kernels.elementwise.binary.arrayloops;

import backend.cpu.kernels.CpuExecutionMode;
import backend.cpu.kernels.CpuThreadPool;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public final class MaxF32 {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private MaxF32() {}

    public static void run(float[] a, float[] b, float[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR -> vector(a, b, out);
            case PARALLEL -> parallel(a, b, out, hints);
            case PARALLEL_VECTOR -> parallelVector(a, b, out, hints);
            case SCALAR -> scalar(a, b, out, 0, out.length);
        }
    }

    private static void scalar(float[] a, float[] b, float[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = Math.max(a[i], b[i]);
    }

    private static void vector(float[] a, float[] b, float[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, a, i).max(FloatVector.fromArray(SPECIES, b, i)).intoArray(out, i);
        }
        scalar(a, b, out, i, out.length);
    }

    private static void parallel(float[] a, float[] b, float[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(a, b, out, start, end);
        });
    }

    private static void parallelVector(float[] a, float[] b, float[] out, ResolvedDispatchHints hints) {
        int width = SPECIES.length();
        int chunkSize = hints.vectorChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            int i = start;
            int upper = end - ((end - start) % width);
            for (; i < upper; i += width) {
                FloatVector.fromArray(SPECIES, a, i).max(FloatVector.fromArray(SPECIES, b, i)).intoArray(out, i);
            }
            scalar(a, b, out, i, end);
        });
    }
}
