package backend.cpu1.kernels.nn.pool.avgpool;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedAvgPool2dUnit;

/**
 * Prepared cpu1 AVG_POOL2D loop entry point.
 */
@FunctionalInterface
public interface Cpu1AvgPool2dKernel {
    void run(
            Cpu1PreparedAvgPool2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView output
    );
}
