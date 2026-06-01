package backend.cpu1.exec;

import backend.runtime.ExecutionContext;

/**
 * Runtime executable prepared by cpu1.
 */
public interface Cpu1ExecutableUnit {
    /**
     * Returns exact run-scoped scratch requirements for this prepared executable.
     *
     * @return workspace spec, or {@link Cpu1WorkspaceSpec#none()} when no workspace is needed
     */
    default Cpu1WorkspaceSpec workspaceSpec() {
        return Cpu1WorkspaceSpec.none();
    }

    /**
     * Runs this prepared unit against the run-scoped execution context.
     *
     * @param context run-scoped execution context
     */
    void run(ExecutionContext context);
}
