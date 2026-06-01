package backend.cpu1.launch;

import backend.cpu1.exec.Cpu1KernelArgs;
import backend.cpu1.kernels.Cpu1KernelRangeRunner;

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
    void launch(Cpu1KernelRangeRunner kernelRunner, Cpu1KernelArgs args);
}
