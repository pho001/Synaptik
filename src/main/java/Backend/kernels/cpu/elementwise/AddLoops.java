package Backend.kernels.cpu.elementwise;

import Backend.kernels.cpu.CpuDTypeOps;
import Backend.kernels.cpu.CpuExecutionConfig;
import Backend.kernels.cpu.CpuThreadPool;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

final class AddLoops {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    private AddLoops() {}

    static void scalar(double[] a, double[] b, double[] out, int precisionMode) {
        for (int i = 0; i < out.length; i++) {
            out[i] = CpuDTypeOps.add(a[i], b[i], precisionMode);
        }
    }

    static void vector(double[] a, double[] b, double[] out, int precisionMode) {
        if (!CpuDTypeOps.isF64(precisionMode)) {
            scalar(a, b, out, precisionMode);
            return;
        }
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector va = DoubleVector.fromArray(SPECIES, a, i);
            DoubleVector vb = DoubleVector.fromArray(SPECIES, b, i);
            va.add(vb).intoArray(out, i);
        }
        for (; i < out.length; i++) {
            out[i] = a[i] + b[i];
        }
    }

    static void parallel(double[] a, double[] b, double[] out, CpuExecutionConfig config, int precisionMode) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            for (int i = start; i < end; i++) {
                out[i] = CpuDTypeOps.add(a[i], b[i], precisionMode);
            }
        });
    }

    static void parallelVector(double[] a, double[] b, double[] out, CpuExecutionConfig config, int precisionMode) {
        if (!CpuDTypeOps.isF64(precisionMode)) {
            parallel(a, b, out, config, precisionMode);
            return;
        }
        int width = SPECIES.length();
        int chunkSize = config.computeChunkSize(out.length, width);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);

            int i = start;
            int upper = end - ((end - start) % width);
            for (; i < upper; i += width) {
                DoubleVector va = DoubleVector.fromArray(SPECIES, a, i);
                DoubleVector vb = DoubleVector.fromArray(SPECIES, b, i);
                va.add(vb).intoArray(out, i);
            }
            for (; i < end; i++) {
                out[i] = a[i] + b[i];
            }
        });
    }
}
