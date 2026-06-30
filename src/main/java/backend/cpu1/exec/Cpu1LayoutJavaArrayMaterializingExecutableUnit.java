package backend.cpu1.exec;

import backend.cpu1.kernels.layout.Cpu1LayoutJavaArrayKernelSupport;
import backend.cpu1.prepare.Cpu1PreparedLayoutUnit;
import runtime.execution.ExecutionContext;

/**
 * Executable unit for prepared materializing layout kernels using Java array storage.
 */
public final class Cpu1LayoutJavaArrayMaterializingExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedLayoutUnit preparedUnit;

    public Cpu1LayoutJavaArrayMaterializingExecutableUnit(Cpu1PreparedLayoutUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedLayoutUnit preparedUnit() {
        return preparedUnit;
    }

    @Override
    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return preparedUnit.scratchBufferSpec();
    }

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        preparedUnit.kernel().run(new Cpu1LayoutJavaArrayKernelSupport(preparedUnit, context));
    }
}
