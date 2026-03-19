package Backend.kernels.cpu;

import Operations.Operation;
import Tensor.Tensor;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.List;

public class CpuReluKernel implements CpuKernel {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forward(op, inputs, node, CpuExecutionConfig.defaults());
    }

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        double[] in = inputs.get(0).getData();
        double[] out = node.getData();
        CpuExecutionMode mode = config.modeFor(op, node);
        switch (mode) {
            case VECTOR -> vectorRelu(in, out);
            case PARALLEL -> parallelRelu(in, out, config);
            case PARALLEL_VECTOR -> parallelVectorRelu(in, out, config);
            case SCALAR -> scalarRelu(in, out);
        }
    }

    private static void scalarRelu(double[] in, double[] out) {
        for (int i = 0; i < out.length; i++) {
            out[i] = Math.max(0, in[i]);
        }
    }

    private static void vectorRelu(double[] in, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        DoubleVector zero = DoubleVector.broadcast(SPECIES, 0.0);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector vi = DoubleVector.fromArray(SPECIES, in, i);
            vi.max(zero).intoArray(out, i);
        }
        for (; i < out.length; i++) {
            out[i] = Math.max(0, in[i]);
        }
    }

    private static void parallelRelu(double[] in, double[] out, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            for (int i = start; i < end; i++) {
                out[i] = Math.max(0, in[i]);
            }
        });
    }

    private static void parallelVectorRelu(double[] in, double[] out, CpuExecutionConfig config) {
        int width = SPECIES.length();
        int chunkSize = config.computeChunkSize(out.length, width);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            int i = start;
            int upper = end - ((end - start) % width);
            DoubleVector zero = DoubleVector.broadcast(SPECIES, 0.0);
            for (; i < upper; i += width) {
                DoubleVector vi = DoubleVector.fromArray(SPECIES, in, i);
                vi.max(zero).intoArray(out, i);
            }
            for (; i < end; i++) {
                out[i] = Math.max(0, in[i]);
            }
        });
    }
}
