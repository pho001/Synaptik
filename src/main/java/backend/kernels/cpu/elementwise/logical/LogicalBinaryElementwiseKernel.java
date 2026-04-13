package backend.kernels.cpu.elementwise.logical;

public interface LogicalBinaryElementwiseKernel {
    byte apply(byte left, byte right);
}
