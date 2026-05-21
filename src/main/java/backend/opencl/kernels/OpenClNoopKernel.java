package backend.opencl.kernels;

import operations.Operation;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.List;

/**
 * OpenCL fallback registry kernel for NOOP copy semantics.
 */
public class OpenClNoopKernel implements OpenClKernel {
    /**
     * Copies the first input into the output tensor when both expose FLOAT64 storage.
     */
    @Override
    public void forward(Operation op, List<Tensor> inputs, Tensor node) {
        if (inputs == null || inputs.isEmpty()) return;
        double[] in = TensorInternalAccess.float64Data(inputs.get(0));
        double[] out = TensorInternalAccess.float64Data(node);
        if (in == null || out == null) return;
        System.arraycopy(in, 0, out, 0, Math.min(in.length, out.length));
        TensorInternalAccess.markStorageModified(node);
    }
}
