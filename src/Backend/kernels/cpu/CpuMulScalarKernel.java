package Backend.kernels.cpu;

import Operations.Operation;
import Operations.mulScalar;
import Tensor.Tensor;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;

import java.util.List;

public class CpuMulScalarKernel implements CpuKernel {
    private static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forward(op, inputs, node, CpuExecutionConfig.defaults());
    }

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        double scalar = ((mulScalar) op).getScalar();
        double[] in = inputs.get(0).getData();
        double[] out = node.getData();
        CpuExecutionMode mode = config.modeFor(op, node);
        switch (mode) {
            case VECTOR -> vectorMulScalar(in, out, scalar);
            case PARALLEL -> parallelMulScalar(in, out, scalar, config);
            case PARALLEL_VECTOR -> parallelVectorMulScalar(in, out, scalar, config);
            case SCALAR -> scalarMulScalar(in, out, scalar);
        }
    }

    private static void scalarMulScalar(double[] in, double[] out, double scalar) {
        for (int i = 0; i < out.length; i++) {
            out[i] = in[i] * scalar;
        }
    }

    private static void vectorMulScalar(double[] in, double[] out, double scalar) {
        int i = 0;
        int upper = SPECIES.loopBound(out.length);
        DoubleVector s = DoubleVector.broadcast(SPECIES, scalar);
        for (; i < upper; i += SPECIES.length()) {
            DoubleVector.fromArray(SPECIES, in, i).mul(s).intoArray(out, i);
        }
        for (; i < out.length; i++) {
            out[i] = in[i] * scalar;
        }
    }

    private static void parallelMulScalar(double[] in, double[] out, double scalar, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            for (int i = start; i < end; i++) {
                out[i] = in[i] * scalar;
            }
        });
    }

    private static void parallelVectorMulScalar(double[] in, double[] out, double scalar, CpuExecutionConfig config) {
        int width = SPECIES.length();
        int chunkSize = config.computeChunkSize(out.length, width);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            int i = start;
            int upper = end - ((end - start) % width);
            DoubleVector s = DoubleVector.broadcast(SPECIES, scalar);
            for (; i < upper; i += width) {
                DoubleVector.fromArray(SPECIES, in, i).mul(s).intoArray(out, i);
            }
            for (; i < end; i++) {
                out[i] = in[i] * scalar;
            }
        });
    }
}
