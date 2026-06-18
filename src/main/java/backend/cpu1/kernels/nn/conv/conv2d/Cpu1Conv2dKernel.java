package backend.cpu1.kernels.nn.conv.conv2d;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedConv2dUnit;

/**
 * Prepared cpu1 CONV2D loop entry point.
 */
@FunctionalInterface
public interface Cpu1Conv2dKernel {
    void run(
            Cpu1PreparedConv2dUnit unit,
            Cpu1TensorView input,
            Cpu1TensorView weight,
            Cpu1TensorView bias,
            Cpu1TensorView output
    );
}
