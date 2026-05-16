package graph.execution.trace;

import tensor.DataType;

/**
 * Optimizer route diagnostics captured for one trainable parameter update.
 *
 * @param optimizer optimizer implementation name
 * @param route backend/storage route used for the update
 * @param dataType parameter dtype
 * @param parameterNodeId compiled trainable parameter node id
 * @param gradientNodeId compiled gradient node id
 * @param elementCount updated element count
 * @param fallbackReason fallback or ineligibility reason, blank for native success
 * @param publicationPolicy active optimizer-step publication policy
 * @param gradientPublication whether gradients were published, skipped, or disabled
 * @param optimizerStateStorage storage residency used for optimizer-owned state
 * @param bf16TrainingPolicy BF16 training policy evidence, blank for non-BF16 parameters
 */
public record NativeOptimizerTrace(
        String optimizer,
        String route,
        DataType dataType,
        int parameterNodeId,
        int gradientNodeId,
        int elementCount,
        String fallbackReason,
        String publicationPolicy,
        String gradientPublication,
        String optimizerStateStorage,
        String bf16TrainingPolicy
) {
    public NativeOptimizerTrace(
            String optimizer,
            String route,
            DataType dataType,
            int parameterNodeId,
            int gradientNodeId,
            int elementCount,
            String fallbackReason
    ) {
        this(
                optimizer,
                route,
                dataType,
                parameterNodeId,
                gradientNodeId,
                elementCount,
                fallbackReason,
                "",
                "",
                "",
                ""
        );
    }

    public NativeOptimizerTrace {
        optimizer = optimizer == null ? "" : optimizer;
        route = route == null ? "" : route;
        elementCount = Math.max(0, elementCount);
        fallbackReason = fallbackReason == null ? "" : fallbackReason;
        publicationPolicy = publicationPolicy == null ? "" : publicationPolicy;
        gradientPublication = gradientPublication == null ? "" : gradientPublication;
        optimizerStateStorage = optimizerStateStorage == null ? "" : optimizerStateStorage;
        bf16TrainingPolicy = bf16TrainingPolicy == null ? "" : bf16TrainingPolicy;
    }
}
