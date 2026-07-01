package backend.cpu1.exec;

import backend.cpu1.prepare.Cpu1PreparedAttentionUnit;
import runtime.execution.ExecutionContext;

/**
 * Runtime wrapper for prepared cpu1 attention using Java array storage.
 */
public final class Cpu1AttentionJavaArrayExecutableUnit extends Cpu1AttentionExecutableUnit {
    public Cpu1AttentionJavaArrayExecutableUnit(Cpu1PreparedAttentionUnit preparedUnit) {
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
