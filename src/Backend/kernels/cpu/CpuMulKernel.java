package Backend.kernels.cpu;

import Operations.Operation;
import Tensor.Tensor;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.List;

public class CpuMulKernel implements CpuKernel {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forward(op, inputs, node, CpuExecutionConfig.defaults());
    }

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        double[] a = inputs.get(0).getData();
        double[] b = inputs.get(1).getData();
        double[] out = node.getData();
        CpuExecutionMode mode = config.modeFor(op, node);
        switch (mode) {
            case VECTOR -> vectorMul(a, b, out);
            case PARALLEL -> parallelMul(a, b, out, config);
            case PARALLEL_VECTOR -> parallelVectorMul(a, b, out, config);
            case SCALAR -> scalarMul(a, b, out);
        }
    }

    private static void scalarMul(double[] a, double[] b, double[] out) {
        for (int i = 0; i < out.length; i++) {
            out[i] = a[i] * b[i];
        }
    }

    private static void vectorMul(double[] a, double[] b, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector va = DoubleVector.fromArray(SPECIES, a, i);
            DoubleVector vb = DoubleVector.fromArray(SPECIES, b, i);
            va.mul(vb).intoArray(out, i);
        }
        for (; i < out.length; i++) {
            out[i] = a[i] * b[i];
        }
    }

    private static void parallelMul(double[] a, double[] b, double[] out, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            for (int i = start; i < end; i++) {
                out[i] = a[i] * b[i];
            }
        });
    }

    private static void parallelVectorMul(double[] a, double[] b, double[] out, CpuExecutionConfig config) {
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
                va.mul(vb).intoArray(out, i);
            }
            for (; i < end; i++) {
                out[i] = a[i] * b[i];
            }
        });
    }
}
