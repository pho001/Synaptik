package runtime.memory;

/**
 * Native or backend-owned resource whose lifetime is scoped to one prepared execution run.
 *
 * <p>Execution resources are registered with {@code ExecutionState} as they are allocated and closed when
 * the run finishes. Implementations must be idempotent: closing an already closed resource should not release
 * the same native handle twice. The contract deliberately does not declare checked exceptions so cleanup can
 * run from {@code finally} blocks without forcing unrelated compute APIs to expose native cleanup details.</p>
 */
public interface ExecutionResource extends AutoCloseable {
    /**
     * Releases the resource owned by the current execution run.
     */
    @Override
    void close();
}
