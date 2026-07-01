package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedAttentionBackwardUnit;
import runtime.execution.ExecutionContext;

/**
 * Runtime wrapper for prepared cpu1 attention backward using Java array storage.
 */
public final class Cpu1AttentionBackwardJavaArrayExecutableUnit extends Cpu1AttentionBackwardExecutableUnit {
    public Cpu1AttentionBackwardJavaArrayExecutableUnit(Cpu1PreparedAttentionBackwardUnit preparedUnit) {
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
