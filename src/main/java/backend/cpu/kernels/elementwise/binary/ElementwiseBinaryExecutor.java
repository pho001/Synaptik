package backend.cpu.kernels.elementwise.binary;

import backend.cpu.kernels.CpuKernelContext;
import tensor.Tensor;

import java.util.List;

public final class ElementwiseBinaryExecutor {
    private ElementwiseBinaryExecutor() {}

    public static void execute(BinaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("Binary elementwise executor requires exactly 2 inputs.");
        }
        if (kernel instanceof CpuAddKernel addKernel) {
            AddStorageLoops.execute(addKernel, inputs, node, context);
            return;
        }
        BinaryStorageLoops.execute(kernel, inputs, node, context);
    }
}
