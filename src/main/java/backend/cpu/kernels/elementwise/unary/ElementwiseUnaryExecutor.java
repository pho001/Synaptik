package backend.cpu.kernels.elementwise.unary;

import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.elementwise.ElementwiseLoops;
import tensor.Tensor;

import java.util.List;

public final class ElementwiseUnaryExecutor {
    private ElementwiseUnaryExecutor() {}

    public static void execute(UnaryElementwiseKernel kernel, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Unary elementwise executor requires exactly 1 input.");
        }
        ElementwiseLoops.runUnary(kernel, inputs.get(0), node, context);
    }

    public static void execute(
            ScalarUnaryElementwiseKernel kernel,
            double parameterF64,
            float parameterF32,
            List<Tensor> inputs,
            Tensor node,
            CpuKernelContext context
    ) {
        if (inputs == null || inputs.size() != 1) {
            throw new IllegalArgumentException("Scalar unary elementwise executor requires exactly 1 input.");
        }
        ElementwiseLoops.runScalarUnary(kernel, parameterF64, parameterF32, inputs.get(0), node, context);
    }
}
