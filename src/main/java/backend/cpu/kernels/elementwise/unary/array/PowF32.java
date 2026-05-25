package backend.cpu.kernels.elementwise.unary.array;

import backend.cpu.kernels.CpuExecutionMode;
import backend.cpu.kernels.CpuThreadPool;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.elementwise.unary.support.CpuPowSupport;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class PowF32 {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private PowF32() {}

    public static void run(float[] in, float exponent, float[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR -> vector(in, exponent, out);
            case PARALLEL -> parallel(in, exponent, out, hints);
            case PARALLEL_VECTOR -> parallelVector(in, exponent, out, hints);
            case SCALAR -> scalar(in, exponent, out, 0, out.length);
        }
    }

    private static void scalar(float[] in, float exponent, float[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuPowSupport.applyF32(in[i], exponent);
        }
    }

    private static void vector(float[] in, float exponent, float[] out) {
        if (exponent != 0.0f && exponent != 1.0f && exponent != 2.0f && exponent != 0.5f && exponent != -1.0f) {
            scalar(in, exponent, out, 0, out.length);
            return;
        }
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        FloatVector ones = FloatVector.broadcast(SPECIES, 1.0f);
        for (; i < upper; i += SPECIES.length()) {
            FloatVector vi = FloatVector.fromArray(SPECIES, in, i);
            FloatVector vo;
            if (exponent == 0.0f) vo = ones;
            else if (exponent == 1.0f) vo = vi;
            else if (exponent == 2.0f) vo = vi.mul(vi);
            else if (exponent == 0.5f) vo = vi.lanewise(VectorOperators.SQRT);
            else vo = ones.div(vi);
            vo.intoArray(out, i);
        }
        scalar(in, exponent, out, i, out.length);
    }

    private static void parallel(float[] in, float exponent, float[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, exponent, out, start, end);
        });
    }

    private static void parallelVector(float[] in, float exponent, float[] out, ResolvedDispatchHints hints) {
        if (exponent != 0.0f && exponent != 1.0f && exponent != 2.0f && exponent != 0.5f && exponent != -1.0f) {
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
            FloatVector ones = FloatVector.broadcast(SPECIES, 1.0f);
            for (; i < upper; i += width) {
                FloatVector vi = FloatVector.fromArray(SPECIES, in, i);
                FloatVector vo;
                if (exponent == 0.0f) vo = ones;
                else if (exponent == 1.0f) vo = vi;
                else if (exponent == 2.0f) vo = vi.mul(vi);
                else if (exponent == 0.5f) vo = vi.lanewise(VectorOperators.SQRT);
                else vo = ones.div(vi);
                vo.intoArray(out, i);
            }
            scalar(in, exponent, out, i, end);
        });
    }
}
