package backend.cpu1.exec;

import runtime.execution.ExecutionContext;

/**
 * Runtime executable prepared by cpu1.
 */
public interface Cpu1ExecutableUnit {
    /**
     * Returns exact run-scoped scratch requirements for this prepared executable.
     *
     * @return scratch buffer spec, or {@link Cpu1ScratchBufferSpec#none()} when no scratch buffer is needed
     */
    default Cpu1ScratchBufferSpec scratchBufferSpec() {
        return Cpu1ScratchBufferSpec.none();
    }

    /**
     * Runs this prepared unit against the run-scoped execution context.
     *
     * @param context run-scoped execution context
     */
    void run(ExecutionContext context);
}
