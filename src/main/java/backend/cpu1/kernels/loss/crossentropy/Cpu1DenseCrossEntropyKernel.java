package backend.cpu1.kernels.loss.crossentropy;

import backend.cpu1.prepare.Cpu1PreparedDenseCrossEntropyLossUnit;
import runtime.execution.ExecutionContext;

@FunctionalInterface
public interface Cpu1DenseCrossEntropyKernel {
    void run(Cpu1PreparedDenseCrossEntropyLossUnit unit, ExecutionContext context);
}
