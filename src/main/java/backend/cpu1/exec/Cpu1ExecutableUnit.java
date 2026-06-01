package backend.cpu1.exec;

import backend.runtime.ExecutionContext;

/**
 * Runtime executable prepared by cpu1.
 */
public interface Cpu1ExecutableUnit {
    /**
     * Runs this prepared unit against the run-scoped execution context.
     *
     * @param context run-scoped execution context
     */
    void run(ExecutionContext context);
}
