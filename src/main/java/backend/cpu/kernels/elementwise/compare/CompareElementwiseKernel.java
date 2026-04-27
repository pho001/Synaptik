package backend.cpu.kernels.elementwise.compare;

public interface CompareElementwiseKernel {
    boolean testF64(double left, double right);

    boolean testF32(float left, float right);

    boolean testBF16(float left, float right);
}
