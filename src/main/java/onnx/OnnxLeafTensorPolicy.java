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
     * Every leaf tensor becomes a Constant node. This is useful when the ONNX graph
     * should preserve literal constant producers instead of initializer storage.
     */
    CONSTANT_NODES,

    /**
     * Trainable leaves become graph inputs; non-trainable leaves become initializers.
     */
    TRAINABLE_INPUTS
}
