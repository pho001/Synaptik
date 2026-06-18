package backend.cpu1.kernels.nn.normalization.layernorm;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedLayerNormUnit;

/**
 * Prepared cpu1 LayerNorm loop entry point.
 */
@FunctionalInterface
public interface Cpu1LayerNormKernel {
    void run(
            Cpu1PreparedLayerNormUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView gamma,
            Cpu1TensorView beta,
            Cpu1TensorView output
    );
}
