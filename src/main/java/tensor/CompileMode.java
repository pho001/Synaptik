package tensor;

/**
 * Compile intent for convenience tensor execution APIs.
 *
 * AUTO preserves the historical behavior: if the graph has trainable leaf inputs,
 * compile a joint forward/backward artifact; otherwise compile forward only.
 */
public enum CompileMode {
    AUTO,
    INFERENCE_ONLY,
    TRAINING
}
