package onnx;

/**
 * Controls how operation-free Synaptik leaf tensors are represented during export.
 */
public enum OnnxLeafTensorPolicy {
    /**
     * Every leaf tensor becomes a graph input. Tensor storage is not serialized.
     */
    INPUTS,

    /**
     * Every leaf tensor becomes an initializer. This is useful for self-contained fixtures.
     */
    INITIALIZERS,

    /**
     * Trainable leaves become graph inputs; non-trainable leaves become initializers.
     */
    TRAINABLE_INPUTS
}
