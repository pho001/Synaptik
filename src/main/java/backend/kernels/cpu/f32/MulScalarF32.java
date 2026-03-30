package backend.kernels.cpu.f32;

import backend.kernels.cpu.CpuExecutionConfig;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public final class MulScalarF32 {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private MulScalarF32() {}

    public static void run(float[] in, float scalar, float[] out, CpuExecutionMode mode, CpuExecutionConfig config) {
        switch (mode) {
            case VECTOR -> vector(in, scalar, out);
            case PARALLEL -> parallel(in, scalar, out, config);
            case PARALLEL_VECTOR -> parallelVector(in, scalar, out, config);
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

    private static void parallel(float[] in, float scalar, float[] out, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, scalar, out, start, end);
        });
    }

    private static void parallelVector(float[] in, float scalar, float[] out, CpuExecutionConfig config) {
        int width = SPECIES.length();
        int chunkSize = config.computeChunkSize(out.length, width);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
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
