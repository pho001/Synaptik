package backend.kernels.cpu.f64;

import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.ResolvedDispatchHints;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public final class MulScalarF64 {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private MulScalarF64() {}

    public static void run(double[] in, double scalar, double[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR -> vector(in, scalar, out);
            case PARALLEL -> parallel(in, scalar, out, hints);
            case PARALLEL_VECTOR -> parallelVector(in, scalar, out, hints);
            case SCALAR -> scalar(in, scalar, out, 0, out.length);
        }
    }

    private static void scalar(double[] in, double scalar, double[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = in[i] * scalar;
    }

    private static void vector(double[] in, double scalar, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        DoubleVector s = DoubleVector.broadcast(SPECIES, scalar);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, in, i).mul(s).intoArray(out, i);
        }
        scalar(in, scalar, out, i, out.length);
    }

    private static void parallel(double[] in, double scalar, double[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, scalar, out, start, end);
        });
    }

    private static void parallelVector(double[] in, double scalar, double[] out, ResolvedDispatchHints hints) {
        int width = SPECIES.length();
        int chunkSize = hints.vectorChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            int i = start;
            int upper = end - ((end - start) % width);
            DoubleVector s = DoubleVector.broadcast(SPECIES, scalar);
            for (; i < upper; i += width) {
                DoubleVector.fromArray(SPECIES, in, i).mul(s).intoArray(out, i);
            }
            scalar(in, scalar, out, i, end);
        });
    }
}
