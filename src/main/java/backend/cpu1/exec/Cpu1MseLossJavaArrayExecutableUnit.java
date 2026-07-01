package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedMseLossUnit;
import runtime.execution.ExecutionContext;

/**
 * Runtime wrapper for prepared cpu1 MSE loss using Java array storage.
 */
public final class Cpu1MseLossJavaArrayExecutableUnit extends Cpu1MseLossExecutableUnit {
    public Cpu1MseLossJavaArrayExecutableUnit(Cpu1PreparedMseLossUnit preparedUnit) {
        super(preparedUnit);
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        runKernel(context);
    }
}
