package backend.cpu1.exec;

import backend.cpu1.kernels.loss.mse.Cpu1MseLossKernel;
import backend.cpu1.prepare.Cpu1PreparedMseLossUnit;
import runtime.execution.ExecutionContext;

public final class Cpu1MseLossExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedMseLossUnit preparedUnit;
    private final Cpu1MseLossKernel kernel;

    public Cpu1MseLossExecutableUnit(Cpu1PreparedMseLossUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
        this.kernel = preparedUnit.kernel();
    }

    public Cpu1PreparedMseLossUnit preparedUnit() {
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
