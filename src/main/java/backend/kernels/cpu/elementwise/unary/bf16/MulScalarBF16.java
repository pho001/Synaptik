package backend.kernels.cpu.elementwise.unary.bf16;

import backend.kernels.cpu.CpuDTypeOps;
import backend.kernels.cpu.CpuExecutionMode;
import backend.kernels.cpu.CpuThreadPool;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public final class MulScalarBF16 {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private MulScalarBF16() {}

    public static void run(short[] in, double scalar, short[] out, ResolvedDispatchHints hints) {
        float s = (float) scalar;
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalar(in, s, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(in, s, out, hints);
        }
    }

    public static void run(float[] in, float scalar, short[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR -> vector(in, scalar, out, 0, out.length);
            case PARALLEL -> parallel(in, scalar, out, hints);
            case PARALLEL_VECTOR -> parallelVector(in, scalar, out, hints);
            case SCALAR -> scalar(in, scalar, out, 0, out.length);
        }
    }

    private static void scalar(short[] in, float scalar, short[] out, int start, int end) {
        int i = start;
        int upper = end - ((end - start) & 3);
        for (; i < upper; i += 4) {
            out[i] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(in[i]) * scalar);
            out[i + 1] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(in[i + 1]) * scalar);
            out[i + 2] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(in[i + 2]) * scalar);
            out[i + 3] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(in[i + 3]) * scalar);
        }
        for (; i < end; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits(CpuDTypeOps.fromBFloat16Bits(in[i]) * scalar);
        }
    }

    private static void scalar(float[] in, float scalar, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits(in[i] * scalar);
        }
    }

    private static void vector(float[] in, float scalar, short[] out, int start, int end) {
        float[] lanes = new float[SPECIES.length()];
        FloatVector scalarVector = FloatVector.broadcast(SPECIES, scalar);
        int width = SPECIES.length();
        int i = start;
        int upper = start + SPECIES.loopBound(end - start);
        for (; i < upper; i += width) {
            FloatVector.fromArray(SPECIES, in, i)
                    .mul(scalarVector)
                    .intoArray(lanes, 0);
            for (int lane = 0; lane < width; lane++) {
                out[i + lane] = CpuDTypeOps.toBFloat16Bits(lanes[lane]);
            }
        }
        scalar(in, scalar, out, i, end);
    }

    private static void parallel(short[] in, float scalar, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, scalar, out, start, end);
        });
    }

    private static void parallel(float[] in, float scalar, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, scalar, out, start, end);
        });
    }

    private static void parallelVector(float[] in, float scalar, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.vectorChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            vector(in, scalar, out, start, end);
        });
    }
}
