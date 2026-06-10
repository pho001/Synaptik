package backend.cpu1.kernels.dtype;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedDTypeUnit;

/**
 * Runtime kernel for one prepared cpu1 dtype conversion variant.
 */
public interface Cpu1DTypeKernel {
    void run(Cpu1PreparedDTypeUnit unit, Cpu1TensorView input, Cpu1TensorView output);
}
