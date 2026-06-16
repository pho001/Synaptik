package backend.cpu1.kernels.index;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedIndexUnit;

/**
 * Prepared index loop entry point.
 */
@FunctionalInterface
public interface Cpu1IndexKernel {
    void run(Cpu1PreparedIndexUnit unit, Cpu1TensorView input, Cpu1TensorView indices, Cpu1TensorView output);
}
