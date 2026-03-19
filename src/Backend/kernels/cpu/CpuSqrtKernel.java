package Backend.kernels.cpu;

import Operations.Operation;
import Tensor.Tensor;

import java.util.List;

public class CpuSqrtKernel implements CpuKernel {
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
            case VECTOR, SCALAR -> scalarSqrt(in, out, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallelSqrt(in, out, config);
        }
    }

    private static void parallelSqrt(double[] in, double[] out, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalarSqrt(in, out, start, end);
        });
    }

    private static void scalarSqrt(double[] in, double[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = Math.sqrt(in[i]);
        }
    }
}
