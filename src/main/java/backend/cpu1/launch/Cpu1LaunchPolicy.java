package backend.cpu1.launch;

import backend.cpu1.exec.Cpu1KernelArgs;
import backend.cpu1.kernels.elementwise.Cpu1ElementwiseRangeRunner;

/**
 * Launch policy for a prepared cpu1 kernel.
 */
public interface Cpu1LaunchPolicy {
    /**
     * Runs a kernel over its prepared logical element range.
     *
     * @param kernelRunner concrete prepared kernel range runner
     * @param args run-bound kernel arguments
     */
    void launch(Cpu1ElementwiseRangeRunner kernelRunner, Cpu1KernelArgs args);
}
