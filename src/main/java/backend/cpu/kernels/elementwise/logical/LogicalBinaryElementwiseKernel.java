package backend.cpu.kernels.elementwise.logical;

public interface LogicalBinaryElementwiseKernel {
    byte apply(byte left, byte right);
}
