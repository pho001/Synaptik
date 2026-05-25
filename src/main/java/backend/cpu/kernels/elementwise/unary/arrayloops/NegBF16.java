package backend.cpu.kernels.elementwise.unary.arrayloops;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.kernels.CpuExecutionMode;
import backend.cpu.kernels.CpuThreadPool;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

public final class NegBF16 {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    private NegBF16() {}

    public static void run(short[] in, short[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR, SCALAR -> scalar(in, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallel(in, out, hints);
        }
    }

    public static void run(float[] in, short[] out, ResolvedDispatchHints hints) {
        CpuExecutionMode mode = hints.mode();
        switch (mode) {
            case VECTOR -> vector(in, out, 0, out.length);
            case PARALLEL -> parallel(in, out, hints);
            case PARALLEL_VECTOR -> parallelVector(in, out, hints);
            case SCALAR -> scalar(in, out, 0, out.length);
        }
    }

    private static void scalar(short[] in, short[] out, int start, int end) {
        int i = start;
        int upper = end - ((end - start) & 3);
        for (; i < upper; i += 4) {
            out[i] = CpuDTypeOps.toBFloat16Bits(-CpuDTypeOps.fromBFloat16Bits(in[i]));
            out[i + 1] = CpuDTypeOps.toBFloat16Bits(-CpuDTypeOps.fromBFloat16Bits(in[i + 1]));
            out[i + 2] = CpuDTypeOps.toBFloat16Bits(-CpuDTypeOps.fromBFloat16Bits(in[i + 2]));
            out[i + 3] = CpuDTypeOps.toBFloat16Bits(-CpuDTypeOps.fromBFloat16Bits(in[i + 3]));
        }
        for (; i < end; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits(-CpuDTypeOps.fromBFloat16Bits(in[i]));
        }
    }

    private static void scalar(float[] in, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuDTypeOps.toBFloat16Bits(-in[i]);
        }
    }

    private static void vector(float[] in, short[] out, int start, int end) {
        float[] lanes = new float[SPECIES.length()];
        int width = SPECIES.length();
        int i = start;
        int upper = start + SPECIES.loopBound(end - start);
        for (; i < upper; i += width) {
            FloatVector.fromArray(SPECIES, in, i)
                    .neg()
                    .intoArray(lanes, 0);
            for (int lane = 0; lane < width; lane++) {
                out[i + lane] = CpuDTypeOps.toBFloat16Bits(lanes[lane]);
            }
        }
        scalar(in, out, i, end);
    }

    private static void parallel(short[] in, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, out, start, end);
        });
    }

    private static void parallel(float[] in, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.scalarChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalar(in, out, start, end);
        });
    }

    private static void parallelVector(float[] in, short[] out, ResolvedDispatchHints hints) {
        int chunkSize = hints.vectorChunkSize();
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, hints.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            vector(in, out, start, end);
        });
    }
}
