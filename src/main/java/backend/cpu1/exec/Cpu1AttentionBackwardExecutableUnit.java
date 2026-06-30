package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedAttentionBackwardUnit;
import runtime.execution.ExecutionContext;

/**
 * Runtime wrapper for a prepared cpu1 attention backward specialized region.
 */
public final class Cpu1AttentionBackwardExecutableUnit implements Cpu1ExecutableUnit {
    private final Cpu1PreparedAttentionBackwardUnit preparedUnit;

    public Cpu1AttentionBackwardExecutableUnit(Cpu1PreparedAttentionBackwardUnit preparedUnit) {
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

    @Override
    public void run(ExecutionContext context) {
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        preparedUnit.kernel().run(preparedUnit, context);
    }
}
