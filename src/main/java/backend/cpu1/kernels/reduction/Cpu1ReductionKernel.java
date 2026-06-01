package backend.cpu1.kernels.reduction;

import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import backend.runtime.ExecutionContext;

/**
 * Prepared reduction loop entry point.
 */
@FunctionalInterface
public interface Cpu1ReductionKernel {
    void run(Cpu1PreparedReductionUnit unit, ExecutionContext context);
}
