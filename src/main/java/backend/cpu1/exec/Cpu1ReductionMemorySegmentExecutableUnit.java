package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import runtime.execution.ExecutionContext;

/**
 * Runtime wrapper for prepared cpu1 reductions using native CPU segment storage.
 */
public final class Cpu1ReductionMemorySegmentExecutableUnit extends Cpu1ReductionExecutableUnit {
    public Cpu1ReductionMemorySegmentExecutableUnit(Cpu1PreparedReductionUnit preparedUnit) {
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
