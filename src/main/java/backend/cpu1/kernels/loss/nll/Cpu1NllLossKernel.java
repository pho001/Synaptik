package backend.cpu1.kernels.loss.nll;

import backend.cpu1.prepare.Cpu1PreparedNllLossUnit;
import backend.runtime.ExecutionContext;

@FunctionalInterface
public interface Cpu1NllLossKernel {
    void run(Cpu1PreparedNllLossUnit unit, ExecutionContext context);
}
