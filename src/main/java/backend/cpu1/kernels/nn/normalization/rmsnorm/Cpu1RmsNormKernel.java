package backend.cpu1.kernels.nn.normalization.rmsnorm;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedRmsNormUnit;

/**
 * Prepared cpu1 RMSNorm loop entry point.
 */
@FunctionalInterface
public interface Cpu1RmsNormKernel {
    void run(
            Cpu1PreparedRmsNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView output
    );
}
