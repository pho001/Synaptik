package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedAttentionBackwardUnit;
import runtime.execution.ExecutionContext;

/**
 * Base runtime wrapper for a prepared cpu1 attention backward specialized partition.
 */
public abstract class Cpu1AttentionBackwardExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedAttentionBackwardUnit preparedUnit;

    protected Cpu1AttentionBackwardExecutableUnit(Cpu1PreparedAttentionBackwardUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedAttentionBackwardUnit preparedUnit() {
        return preparedUnit;
    }

    @Override
    public Cpu1ScratchBufferSpec scratchBufferSpec() {
        return preparedUnit.scratchBufferSpec();
    }

    protected void runKernel(ExecutionContext context) {
        preparedUnit.kernel().run(preparedUnit, context);
    }
}
