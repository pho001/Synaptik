package backend.cpu1.kernels.loss.crossentropy;

import backend.cpu1.prepare.Cpu1PreparedCrossEntropyLossUnit;
import backend.runtime.ExecutionContext;

@FunctionalInterface
public interface Cpu1CrossEntropyKernel {
    void run(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context);
}
