package backend.cpu1.kernels.linalg.attention;

import backend.cpu1.prepare.Cpu1PreparedAttentionUnit;
import runtime.execution.ExecutionContext;

/**
 * Prepared cpu1 attention kernel entry point.
 */
@FunctionalInterface
public interface Cpu1AttentionKernel {
    void run(Cpu1PreparedAttentionUnit unit, ExecutionContext context);
}
