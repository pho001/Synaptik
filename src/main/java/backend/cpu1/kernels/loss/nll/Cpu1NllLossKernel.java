package backend.cpu1.kernels.loss.nll;

import backend.cpu1.prepare.Cpu1PreparedNllLossUnit;
import runtime.execution.ExecutionContext;

@FunctionalInterface
public interface Cpu1NllLossKernel {
    void run(Cpu1PreparedNllLossUnit unit, ExecutionContext context);
}
