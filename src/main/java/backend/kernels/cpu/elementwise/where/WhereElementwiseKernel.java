package backend.kernels.cpu.elementwise.where;

public interface WhereElementwiseKernel {
    double applyF64(byte condition, double ifTrue, double ifFalse);

    float applyF32(byte condition, float ifTrue, float ifFalse);

    float applyBF16(byte condition, float ifTrue, float ifFalse);
}
