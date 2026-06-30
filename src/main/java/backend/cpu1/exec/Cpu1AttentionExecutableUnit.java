package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedAttentionUnit;
import runtime.execution.ExecutionContext;

/**
 * Runtime wrapper for a prepared cpu1 attention node.
 */
public final class Cpu1AttentionExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedAttentionUnit preparedUnit;

    public Cpu1AttentionExecutableUnit(Cpu1PreparedAttentionUnit preparedUnit) {
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

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        preparedUnit.kernel().run(preparedUnit, context);
    }
}
