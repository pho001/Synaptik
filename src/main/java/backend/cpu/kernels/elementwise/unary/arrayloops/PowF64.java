package backend.cpu.kernels.elementwise.unary.arrayloops;

import backend.cpu.kernels.CpuExecutionMode;
import backend.cpu.kernels.CpuThreadPool;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.elementwise.unary.support.CpuPowSupport;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class PowF64 {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private PowF64() {}

    public static void run(double[] in, double exponent, double[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR -> vector(in, exponent, out);
            case PARALLEL -> parallel(in, exponent, out, hints);
            case PARALLEL_VECTOR -> parallelVector(in, exponent, out, hints);
            case SCALAR -> scalar(in, exponent, out, 0, out.length);
        }
    }

    private static void scalar(double[] in, double exponent, double[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuPowSupport.applyF64(in[i], exponent);
        }
    }

    private static void vector(double[] in, double exponent, double[] out) {
        if (exponent != 0.0d && exponent != 1.0d && exponent != 2.0d && exponent != 0.5d && exponent != -1.0d) {
            scalar(in, exponent, out, 0, out.length);
            return;
        }
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        DoubleVector ones = DoubleVector.broadcast(SPECIES, 1.0d);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector vi = DoubleVector.fromArray(SPECIES, in, i);
            DoubleVector vo;
            if (exponent == 0.0d) vo = ones;
            else if (exponent == 1.0d) vo = vi;
            else if (exponent == 2.0d) vo = vi.mul(vi);
            else if (exponent == 0.5d) vo = vi.lanewise(VectorOperators.SQRT);
            else vo = ones.div(vi);
            vo.intoArray(out, i);
        }
        scalar(in, exponent, out, i, out.length);
    }

    private static void parallel(double[] in, double exponent, double[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, exponent, out, start, end);
        });
    }

    private static void parallelVector(double[] in, double exponent, double[] out, ResolvedDispatchHints hints) {
        if (exponent != 0.0d && exponent != 1.0d && exponent != 2.0d && exponent != 0.5d && exponent != -1.0d) {
            parallel(in, exponent, out, hints);
            return;
        }
        int width = SPECIES.length();
        int chunkSize = hints.vectorChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            int i = start;
            int upper = end - ((end - start) % width);
            DoubleVector ones = DoubleVector.broadcast(SPECIES, 1.0d);
            for (; i < upper; i += width) {
                DoubleVector vi = DoubleVector.fromArray(SPECIES, in, i);
                DoubleVector vo;
                if (exponent == 0.0d) vo = ones;
                else if (exponent == 1.0d) vo = vi;
                else if (exponent == 2.0d) vo = vi.mul(vi);
                else if (exponent == 0.5d) vo = vi.lanewise(VectorOperators.SQRT);
                else vo = ones.div(vi);
                vo.intoArray(out, i);
            }
            scalar(in, exponent, out, i, end);
        });
    }
}
