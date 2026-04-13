package backend.kernels.cpu.elementwise.binary;

import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.elementwise.ElementwiseLoops;
import tensor.Tensor;

import java.util.List;

public final class ElementwiseBinaryExecutor {
    private ElementwiseBinaryExecutor() {}

    public static void execute(BinaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.size() != 2) {
            throw new IllegalArgumentException("Binary elementwise executor requires exactly 2 inputs.");
        }
        ElementwiseLoops.runBinary(kernel, inputs.get(0), inputs.get(1), node, context);
    }
}
