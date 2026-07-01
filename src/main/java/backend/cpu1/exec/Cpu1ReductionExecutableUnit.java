package backend.cpu1.exec;

import backend.cpu1.kernels.reduction.Cpu1ReductionKernel;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import runtime.execution.ExecutionContext;

/**
 * Base runtime wrapper for a prepared cpu1 reduction node.
 */
public abstract class Cpu1ReductionExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedReductionUnit preparedUnit;
    private final Cpu1ReductionKernel kernel;

    protected Cpu1ReductionExecutableUnit(Cpu1PreparedReductionUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
        this.kernel = preparedUnit.kernel();
    }

    public Cpu1PreparedReductionUnit preparedUnit() {
        return preparedUnit;
    }

    @Override
    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return preparedUnit.scratchBufferSpec();
    }

    protected void runKernel(ExecutionContext context) {
        kernel.run(preparedUnit, context);
    }
}
