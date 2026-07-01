package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedAttentionUnit;
import runtime.execution.ExecutionContext;

/**
 * Runtime wrapper for prepared cpu1 attention using native CPU segment storage.
 */
public final class Cpu1AttentionMemorySegmentExecutableUnit extends Cpu1AttentionExecutableUnit {
    public Cpu1AttentionMemorySegmentExecutableUnit(Cpu1PreparedAttentionUnit preparedUnit) {
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
