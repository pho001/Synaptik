package backend.cpu1.kernels.nn.pool.maxpool;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedMaxPool2dUnit;

/**
 * Prepared cpu1 MAX_POOL2D loop entry point.
 */
@FunctionalInterface
public interface Cpu1MaxPool2dKernel {
    void run(
            Cpu1PreparedMaxPool2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView output
    );
}
