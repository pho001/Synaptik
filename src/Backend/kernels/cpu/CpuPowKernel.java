package Backend.kernels.cpu;

import Operations.Operation;
import Operations.pow;
import Tensor.Tensor;

import java.util.List;

public class CpuPowKernel implements CpuKernel {
    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        forward(op, inputs, node, CpuExecutionConfig.defaults());
    }

    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node, CpuExecutionConfig config) {
        double exponent = ((pow) op).getExponent();
        double[] in = inputs.get(0).getData();
        double[] out = node.getData();
        CpuExecutionMode mode = config.modeFor(op, node);
        switch (mode) {
            case VECTOR, SCALAR -> scalarPow(in, out, exponent, 0, out.length);
            case PARALLEL, PARALLEL_VECTOR -> parallelPow(in, out, exponent, config);
        }
    }

    private static void parallelPow(double[] in, double[] out, double exponent, CpuExecutionConfig config) {
        int chunkSize = config.computeChunkSize(out.length, 1);
        int chunks = (out.length + chunkSize - 1) / chunkSize;
        CpuThreadPool.runChunks(chunks, config.plannedWorkers(), chunk -> {
            int start = chunk * chunkSize;
            int end = Math.min(start + chunkSize, out.length);
            scalarPow(in, out, exponent, start, end);
        });
    }

    private static void scalarPow(double[] in, double[] out, double exponent, int start, int end) {
        for (int i = start; i < end; i++) {
            if (exponent == 0.0) out[i] = 1.0;
            else if (exponent == 1.0) out[i] = in[i];
            else if (exponent == 2.0) out[i] = in[i] * in[i];
            else out[i] = Math.pow(in[i], exponent);
        }
    }
}
