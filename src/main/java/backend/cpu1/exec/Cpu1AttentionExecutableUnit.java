package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedAttentionUnit;
import runtime.execution.ExecutionContext;

/**
 * Base runtime wrapper for a prepared cpu1 attention node.
 */
public abstract class Cpu1AttentionExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedAttentionUnit preparedUnit;

    protected Cpu1AttentionExecutableUnit(Cpu1PreparedAttentionUnit preparedUnit) {
        if (preparedUnit == null) {
            throw new IllegalArgumentException("preparedUnit cannot be null");
        }
        this.preparedUnit = preparedUnit;
    }

    public Cpu1PreparedAttentionUnit preparedUnit() {
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
