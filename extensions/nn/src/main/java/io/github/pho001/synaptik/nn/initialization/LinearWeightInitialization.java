package io.github.pho001.synaptik.nn.initialization;

/**
 * Closed weight-initialization policies for an input-width-inferring Linear layer.
 *
 * <p>Each policy selects one existing rank-two initializer. The Linear layer supplies its
 * inferred {@code [outFeatures, inFeatures]} Shape, configured floating data type, and a fresh
 * deterministic random generator when its first compatible forward call initializes the layer.
 * The values choose no output width, bias, type, random algorithm, seed, execution behavior, or
 * persistent checkpoint encoding.</p>
 */
public enum LinearWeightInitialization {
    /** Unit-gain Glorot normal initialization. */
    GLOROT_NORMAL,

    /** Unit-gain Glorot uniform initialization. */
    GLOROT_UNIFORM,

    /** Fan-in Kaiming normal initialization with fixed rectified-linear-unit gain. */
    KAIMING_RELU_NORMAL,

    /** Fan-in Kaiming uniform initialization with fixed rectified-linear-unit gain. */
    KAIMING_RELU_UNIFORM
}
