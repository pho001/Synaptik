package tensor;

/**
 * Compile intent for convenience tensor execution APIs.
 *
 * AUTO preserves the historical behavior: if the graph has trainable leaf inputs,
 * compile a joint forward/backward artifact; otherwise compile forward only.
 */
public enum CompileMode {
    /**
     * Infer compile intent from the tensor graph.
     *
     * <p>Graphs with trainable leaf tensors compile training artifacts;
     * graphs without trainable inputs compile inference-only artifacts.</p>
     */
    AUTO,
    /** Compile only the forward pass and skip backward graph generation. */
    INFERENCE_ONLY,
    /** Compile forward and backward artifacts for gradient-producing execution. */
    TRAINING
}
