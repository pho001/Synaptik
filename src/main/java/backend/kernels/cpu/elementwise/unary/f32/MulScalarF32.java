package backend.kernels.cpu.elementwise.unary.f32;

import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public final class MulScalarF32 {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private MulScalarF32() {}

    public static void run(float[] in, float scalar, float[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR -> vector(in, scalar, out);
            case PARALLEL -> parallel(in, scalar, out, hints);
            case PARALLEL_VECTOR -> parallelVector(in, scalar, out, hints);
            case SCALAR -> scalar(in, scalar, out, 0, out.length);
        }
    }

    private static void scalar(float[] in, float scalar, float[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = in[i] * scalar;
    }

    private static void vector(float[] in, float scalar, float[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        FloatVector s = FloatVector.broadcast(SPECIES, scalar);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector.fromArray(SPECIES, in, i).mul(s).intoArray(out, i);
        }
        scalar(in, scalar, out, i, out.length);
    }

    private static void parallel(float[] in, float scalar, float[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, scalar, out, start, end);
        });
    }

    private static void parallelVector(float[] in, float scalar, float[] out, ResolvedDispatchHints hints) {
        int width = SPECIES.length();
        int chunkSize = hints.vectorChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            int i = start;
            int upper = end - ((end - start) % width);
            FloatVector s = FloatVector.broadcast(SPECIES, scalar);
            for (; i < upper; i += width) {
                FloatVector.fromArray(SPECIES, in, i).mul(s).intoArray(out, i);
            }
            scalar(in, scalar, out, i, end);
        });
    }
}
