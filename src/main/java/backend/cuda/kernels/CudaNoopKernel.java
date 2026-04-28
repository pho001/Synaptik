package backend.cuda.kernels;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

/**
 * CUDA fallback registry kernel for NOOP copy semantics.
 */
public class CudaNoopKernel implements CudaKernel {
    /**
     * Copies the first input into the output tensor when both expose double-array storage.
     */
    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        if (inputs == null || inputs.isEmpty()) return;
        double[] in = inputs.get(0).getData();
        double[] out = node.getData();
        if (in == null || out == null) return;
        System.arraycopy(in, 0, out, 0, Math.min(in.length, out.length));
    }
}
