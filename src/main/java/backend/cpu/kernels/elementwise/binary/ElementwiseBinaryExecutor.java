package backend.cpu.kernels.elementwise.binary;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import backend.cpu.nativecpu.NativeCpuElementwiseExecutor;
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
        if (NativeCpuElementwiseExecutor.tryRunBinary(kernel, inputs, node, context)) {
            return;
        }
        ElementwiseLoops.runBinary(kernel, inputs.get(0), inputs.get(1), node, context);
    }
}
