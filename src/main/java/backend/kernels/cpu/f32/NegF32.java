package backend.kernels.cpu.f32;

import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedDispatchHints;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public final class NegF32 {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private NegF32() {}

    public static void run(float[] in, float[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR -> vector(in, out);
            case PARALLEL -> parallel(in, out, hints);
            case PARALLEL_VECTOR -> parallelVector(in, out, hints);
            case SCALAR -> scalar(in, out, 0, out.length);
        }
    }

    private static void scalar(float[] in, float[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = -in[i];
    }

    private static void vector(float[] in, float[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, in, i).neg().intoArray(out, i);
        }
        scalar(in, out, i, out.length);
    }

    private static void parallel(float[] in, float[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, out, start, end);
        });
    }

    private static void parallelVector(float[] in, float[] out, ResolvedDispatchHints hints) {
        int width = SPECIES.length();
        int chunkSize = hints.vectorChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            int i = start;
            int upper = end - ((end - start) % width);
            for (; i < upper; i += width) {
                FloatVector.fromArray(SPECIES, in, i).neg().intoArray(out, i);
            }
            scalar(in, out, i, end);
        });
    }
}
