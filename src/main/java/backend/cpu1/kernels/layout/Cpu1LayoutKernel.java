package backend.cpu1.kernels.layout;

import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.runtime.ExecutionContext;

/**
 * Runtime kernel for one prepared cpu1 layout/view variant.
 */
public interface Cpu1LayoutKernel {
    void run(Cpu1PreparedLayoutUnit unit, ExecutionContext context);
}
