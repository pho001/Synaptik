package backend.cpu1.exec;

import backend.cpu1.kernels.layout.Cpu1LayoutKernel;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import backend.runtime.ExecutionContext;

import java.util.Objects;

/**
 * Runtime wrapper for a prepared cpu1 layout/view node.
 */
public final class Cpu1LayoutExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedLayoutUnit preparedUnit;
    private final Cpu1LayoutKernel kernel;

    public Cpu1LayoutExecutableUnit(Cpu1PreparedLayoutUnit preparedUnit) {
        this.preparedUnit = Objects.requireNonNull(preparedUnit, "preparedUnit cannot be null");
        this.kernel = preparedUnit.kernel();
    }

    public Cpu1PreparedLayoutUnit preparedUnit() {
        return preparedUnit;
    }

    @Override
    public Cpu1WorkspaceSpec workspaceSpec() {
        return preparedUnit.workspaceSpec();
    }

    @Override
    public void run(ExecutionContext context) {
        kernel.run(preparedUnit, context);
    }
}
