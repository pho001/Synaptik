package backend.cpu1.kernels.loss.mse;

import backend.cpu1.prepare.Cpu1PreparedMseLossUnit;
import runtime.execution.ExecutionContext;

@FunctionalInterface
public interface Cpu1MseLossKernel {
    void run(Cpu1PreparedMseLossUnit unit, ExecutionContext context);
}
