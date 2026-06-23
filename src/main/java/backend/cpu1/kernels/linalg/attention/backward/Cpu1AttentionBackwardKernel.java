package backend.cpu1.kernels.linalg.attention.backward;

import backend.cpu1.prepare.Cpu1PreparedAttentionBackwardUnit;
import backend.runtime.ExecutionContext;

/**
 * Prepared cpu1 attention backward kernel entry point.
 */
@FunctionalInterface
public interface Cpu1AttentionBackwardKernel {
    void run(Cpu1PreparedAttentionBackwardUnit unit, ExecutionContext context);
}
