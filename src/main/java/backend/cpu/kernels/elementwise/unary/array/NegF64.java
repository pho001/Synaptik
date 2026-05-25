package backend.cpu.kernels.elementwise.unary.array;

import backend.cpu.kernels.CpuExecutionMode;
import backend.cpu.kernels.CpuThreadPool;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

public final class NegF64 {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private NegF64() {}

    public static void run(double[] in, double[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR -> vector(in, out);
            case PARALLEL -> parallel(in, out, hints);
            case PARALLEL_VECTOR -> parallelVector(in, out, hints);
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

    private static void parallel(double[] in, double[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, out, start, end);
        });
    }

    private static void parallelVector(double[] in, double[] out, ResolvedDispatchHints hints) {
        int width = SPECIES.length();
        int chunkSize = hints.vectorChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
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
