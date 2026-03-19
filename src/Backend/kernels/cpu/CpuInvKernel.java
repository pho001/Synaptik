package Backend.kernels.cpu;

import Operations.Operation;
import Tensor.Tensor;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.List;

public class CpuInvKernel implements CpuKernel {
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
            case VECTOR -> vectorInv(in, out);
            case PARALLEL -> parallelInv(in, out, config);
            case PARALLEL_VECTOR -> parallelVectorInv(in, out, config);
            case SCALAR -> scalarInv(in, out);
        }
    }

    private static void scalarInv(double[] in, double[] out) {
        for (int i = 0; i < out.length; i++) {
            out[i] = 1.0 / in[i];
        }
    }

    private static void vectorInv(double[] in, double[] out) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        DoubleVector ones = DoubleVector.broadcast(SPECIES, 1.0);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector vi = DoubleVector.fromArray(SPECIES, in, i);
            ones.div(vi).intoArray(out, i);
        }
        for (; i < out.length; i++) {
            out[i] = 1.0 / in[i];
        }
    }

    private static void parallelInv(double[] in, double[] out, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            for (int i = start; i < end; i++) {
                out[i] = 1.0 / in[i];
            }
        });
    }

    private static void parallelVectorInv(double[] in, double[] out, CpuExecutionConfig config) {
        int width = SPECIES.length();
        int chunkSize = config.computeChunkSize(out.length, width);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            int i = start;
            int upper = end - ((end - start) % width);
            DoubleVector ones = DoubleVector.broadcast(SPECIES, 1.0);
            for (; i < upper; i += width) {
                DoubleVector vi = DoubleVector.fromArray(SPECIES, in, i);
                ones.div(vi).intoArray(out, i);
            }
            for (; i < end; i++) {
                out[i] = 1.0 / in[i];
            }
        });
    }
}
