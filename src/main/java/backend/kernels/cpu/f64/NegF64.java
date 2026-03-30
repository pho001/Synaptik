package backend.kernels.cpu.f64;

import backend.kernels.cpu.CpuExecutionConfig;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public final class NegF64 {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private NegF64() {}

    public static void run(double[] in, double[] out, CpuExecutionMode mode, CpuExecutionConfig config) {
        switch (mode) {
            case VECTOR -> vector(in, out);
            case PARALLEL -> parallel(in, out, config);
            case PARALLEL_VECTOR -> parallelVector(in, out, config);
            case SCALAR -> scalar(in, out, 0, out.length);
        }
    }

    private static void scalar(double[] in, double[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = -in[i];
    }

    private static void vector(double[] in, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, in, i).neg().intoArray(out, i);
        }
        scalar(in, out, i, out.length);
    }

    private static void parallel(double[] in, double[] out, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, out, start, end);
        });
    }

    private static void parallelVector(double[] in, double[] out, CpuExecutionConfig config) {
        int width = SPECIES.length();
        int chunkSize = config.computeChunkSize(out.length, width);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            int i = start;
            int upper = end - ((end - start) % width);
            for (; i < upper; i += width) {
                DoubleVector.fromArray(SPECIES, in, i).neg().intoArray(out, i);
            }
            scalar(in, out, i, end);
        });
    }
}
