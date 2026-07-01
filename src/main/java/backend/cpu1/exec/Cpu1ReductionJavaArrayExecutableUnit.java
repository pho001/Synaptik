package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import runtime.execution.ExecutionContext;

/**
 * Runtime wrapper for prepared cpu1 reductions using Java array storage.
 */
public final class Cpu1ReductionJavaArrayExecutableUnit extends Cpu1ReductionExecutableUnit {
    public Cpu1ReductionJavaArrayExecutableUnit(Cpu1PreparedReductionUnit preparedUnit) {
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
