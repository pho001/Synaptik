package backend.cpu1.exec;

import backend.cpu1.kernels.reduction.Cpu1ReductionKernel;
import backend.cpu1.prepare.Cpu1PreparedReductionUnit;
import runtime.execution.ExecutionContext;

/**
 * Runtime wrapper for a prepared cpu1 reduction node.
 */
public final class Cpu1ReductionExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedReductionUnit preparedUnit;
    private final Cpu1ReductionKernel kernel;

    public Cpu1ReductionExecutableUnit(Cpu1PreparedReductionUnit preparedUnit) {
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

    @Override
    public void run(ExecutionContext context) {
        kernel.run(preparedUnit, context);
    }
}
