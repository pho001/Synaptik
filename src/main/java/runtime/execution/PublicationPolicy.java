package runtime.execution;

/**
 * Controls which run-scoped values are copied back to user-visible tensors after execution.
 *
 * <p>Publication is an API visibility policy, not backend planning. Disabling publication keeps computed
 * values in the per-run execution state and avoids device-to-CPU copies that are needed only to update
 * public {@code Tensor} storage.</p>
 */
public enum PublicationPolicy {
    /**
     * Publish every forward value represented by the compiled graph, then publish gradients.
     */
    ALL(true, true, true),

    /**
     * Publish only the graph output value and computed gradients.
     */
    OUTPUT_AND_GRADIENTS(false, true, true),

    /**
     * Publish only the graph output value.
     */
    OUTPUT_ONLY(false, true, false),

    /**
     * Do not publish graph values or gradients.
     */
    NONE(false, false, false);

    private final boolean allForwardValues;
    private final boolean outputValue;
    private final boolean gradients;

    PublicationPolicy(boolean allForwardValues, boolean outputValue, boolean gradients) {
        this.allForwardValues = allForwardValues;
        this.outputValue = outputValue;
        this.gradients = gradients;
    }

    /**
     * Returns whether every forward value should be synchronized to its publication tensor.
     *
     * @return {@code true} for full graph value publication
     */
    public boolean publishesAllForwardValues() {
        return allForwardValues;
    }

    /**
     * Returns whether the root output value should be synchronized to the user-visible root tensor.
     *
     * @return {@code true} when output publication is enabled
     */
    public boolean publishesOutputValue() {
        return outputValue;
    }

    /**
     * Returns whether computed gradients should be synchronized and attached to user-visible tensors.
     *
     * @return {@code true} when gradient publication is enabled
     */
    public boolean publishesGradients() {
        return gradients;
    }

    /**
     * Default policy for ordinary {@code execute(...)} calls.
     *
     * @return current public execution semantics
     */
    public static PublicationPolicy defaultExecution() {
        return OUTPUT_AND_GRADIENTS;
    }

    /**
     * Default policy for optimizer-step execution.
     *
     * @return output-only publication, preserving the no-eager-gradient-publication contract
     */
    public static PublicationPolicy defaultOptimizerStep() {
        return OUTPUT_ONLY;
    }
}
