package runtime.execution;

import graph.model.CompiledNode;
import trace.backend.StepTraceContribution;

/**
 * Backend-owned payload attached to prepared node metadata.
 */
public interface PreparedStepExecutable {
    void execute(
            CompiledNode node,
            PreparedStepMetadata metadata,
            ExecutionContext context
    );

    /**
     * Allocates run-scoped workspace state required by this prepared artifact.
     *
     * @param nodeId compiled node id owning this artifact in the prepared plan
     * @param allocator run-scoped allocation sink
     */
    default void allocateRuntimeState(int nodeId, PreparedRuntimeStateAllocator allocator) {
    }

    /**
     * Returns backend-owned trace metadata for one executed step.
     *
     * @param node compiled node represented by the step
     * @param metadata prepared execution metadata
     * @param context execution context after the step ran
     * @return trace contribution
     */
    default StepTraceContribution traceContribution(
            CompiledNode node,
            PreparedStepMetadata metadata,
            ExecutionContext context
    ) {
        return StepTraceContribution.empty();
    }
}
