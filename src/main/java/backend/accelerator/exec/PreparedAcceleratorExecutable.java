package backend.accelerator.exec;

import backend.ComputeBackend;
import backend.runtime.ExecutionContext;

/**
 * Prepared runtime artifact for an accelerator-backed partition.
 *
 * <p>Implementations own any native bridge context or compiled executable needed
 * for a lowered partition and must fall back to CPU execution when the bridge is
 * unavailable or rejects the prepared artifact.</p>
 */
public interface PreparedAcceleratorExecutable {
    /**
     * Returns the accelerator backend this executable targets.
     */
    ComputeBackend backend();

    /**
     * Executes the prepared partition against tensors resolved from the runtime context.
     *
     * @param context runtime tensor lookup and execution flags for the current graph run
     */
    void execute(ExecutionContext context);
}
