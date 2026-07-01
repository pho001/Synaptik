package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedAttentionBackwardUnit;
import runtime.execution.ExecutionContext;

/**
 * Runtime wrapper for prepared cpu1 attention backward using native CPU segment storage.
 */
public final class Cpu1AttentionBackwardMemorySegmentExecutableUnit extends Cpu1AttentionBackwardExecutableUnit {
    public Cpu1AttentionBackwardMemorySegmentExecutableUnit(Cpu1PreparedAttentionBackwardUnit preparedUnit) {
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
