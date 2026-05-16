package tuning.measure;

/**
 * Selects the executable action measured for a prepared workload.
 */
public enum MeasurementExecutionMode {
    /**
     * Measure ordinary prepared graph execution.
     */
    GRAPH_EXECUTION,

    /**
     * Measure forward/backward followed by an SGD optimizer step.
     */
    OPTIMIZER_STEP_SGD,

    /**
     * Measure forward/backward followed by an Adam optimizer step.
     */
    OPTIMIZER_STEP_ADAM;

    public boolean optimizerStep() {
        return this == OPTIMIZER_STEP_SGD || this == OPTIMIZER_STEP_ADAM;
    }
}
