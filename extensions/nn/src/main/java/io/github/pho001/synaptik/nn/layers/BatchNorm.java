package io.github.pho001.synaptik.nn.layers;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.normalization.BatchNormTrainingAttrs;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.BatchNormTrainingResult;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.initialization.ParameterInitializers;
import io.github.pho001.synaptik.nn.module.Buffer;
import io.github.pho001.synaptik.nn.module.ForwardContext;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Parameter;
import java.util.Objects;
import java.util.Optional;

/**
 * Stateful affine batch normalization over one explicit logical channel axis.
 *
 * <p>The layer owns mandatory rank-one state with one positive, fully static feature extent and
 * one exact floating data type. Trainable parameters are declared under {@code scale} then
 * {@code bias}; persistent state is declared under {@code runningMean} then
 * {@code runningVariance}. Momentum is a finite new-batch weight in {@code [0, 1]}, and epsilon
 * is finite and strictly positive. Both scalars retain their exact supplied representation and
 * have the state data type.</p>
 *
 * <p>{@link #forward(Tensor, ForwardContext)} treats the immutable context snapshot as
 * authoritative. Evaluation delegates once to Model batch-normalization inference and leaves
 * both buffers unchanged. Training delegates once to Model's pure five-output training
 * expression, then installs the exact next-running-mean expression followed by the exact
 * next-running-variance expression before returning normalized output. Installation changes
 * stable NN buffer bindings; it does not evaluate or mutate a Tensor, access storage, publish a
 * runtime value, or execute backend work.</p>
 *
 * <p>Each call snapshots the four current Tensor bindings. Earlier expressions therefore retain
 * their exact earlier inputs, while previously discovered {@link Parameter} and {@link Buffer}
 * wrappers observe later successful replacements through {@code value()}. Parameter replacement,
 * mode changes, buffer transition, and forward construction are not thread-safe as a combined
 * activity. The two buffer installations are ordered individual operations, not a transaction;
 * there is no rollback, version, checkpoint, or cross-binding atomicity guarantee.</p>
 */
public final class BatchNorm extends Module {
    private final int channelAxis;
    private final Dimension featureDimension;
    private final DataType stateType;
    private final ScalarValue momentum;
    private final ScalarValue epsilon;
    private final Parameter scale;
    private final Parameter bias;
    private final Buffer runningMean;
    private final Buffer runningVariance;

    /**
     * Creates batch normalization from exact caller-supplied parameter and running-statistic
     * Tensors.
     *
     * <p>All four Tensors must share one floating type and one structurally equal, fully static,
     * positive rank-one Shape. Scale and bias require gradient eligibility; the two initially
     * supplied running statistics must not be gradient-eligible. Momentum and epsilon must satisfy
     * the Model intrinsic training-attribute contract and exactly match the state type. Complete
     * validation precedes every state declaration and creates no Tensor or producer. The exact
     * four Tensor references and two scalar references are retained without copying, mutation, or
     * evaluation.</p>
     *
     * @param channelAxis non-negative logical input channel axis, validated against each input
     *     rank during forward construction
     * @param scale non-null floating, gradient-eligible, positive fully static rank-one scale
     *     Tensor retained exactly
     * @param bias non-null floating, gradient-eligible rank-one bias with the exact scale type and
     *     structural Shape, retained exactly
     * @param runningMean non-null floating, non-gradient rank-one running mean with the exact
     *     scale type and structural Shape, retained exactly
     * @param runningVariance non-null floating, non-gradient rank-one running variance with the
     *     exact scale type and structural Shape, retained exactly
     * @param momentum non-null finite floating new-batch weight in {@code [0, 1]} with the exact
     *     state type, retained exactly
     * @param epsilon non-null finite strictly positive floating stabilizer with the exact state
     *     type, retained exactly
     * @throws NullPointerException if a Tensor or scalar is null, checked in parameter order after
     *     validating {@code channelAxis}
     * @throws IllegalArgumentException if the channel axis is negative; state floating type,
     *     gradient eligibility, rank, static/positive Shape, exact type, or structural Shape
     *     requirements fail; momentum or epsilon violates the Model intrinsic contract; or either
     *     scalar type differs from the common state type
     */
    public BatchNorm(
            int channelAxis,
            Tensor scale,
            Tensor bias,
            Tensor runningMean,
            Tensor runningVariance,
            ScalarValue momentum,
            ScalarValue epsilon) {
        validateChannelAxis(channelAxis);
        Tensor suppliedScale = Objects.requireNonNull(scale, "scale");
        Tensor suppliedBias = Objects.requireNonNull(bias, "bias");
        Tensor suppliedRunningMean = Objects.requireNonNull(runningMean, "runningMean");
        Tensor suppliedRunningVariance = Objects.requireNonNull(
                runningVariance, "runningVariance");
        ScalarValue suppliedMomentum = Objects.requireNonNull(momentum, "momentum");
        ScalarValue suppliedEpsilon = Objects.requireNonNull(epsilon, "epsilon");

        StateSchema schema = validateScale(suppliedScale);
        validateParameter(suppliedBias, "bias", schema);
        validateBuffer(suppliedRunningMean, "running mean", schema);
        validateBuffer(suppliedRunningVariance, "running variance", schema);
        validateScalars(channelAxis, suppliedMomentum, suppliedEpsilon, schema.dataType());

        this.channelAxis = channelAxis;
        this.featureDimension = schema.shape().dimension(0);
        this.stateType = schema.dataType();
        this.momentum = suppliedMomentum;
        this.epsilon = suppliedEpsilon;
        this.scale = parameter("scale", suppliedScale);
        this.bias = parameter("bias", suppliedBias);
        this.runningMean = buffer("runningMean", suppliedRunningMean);
        this.runningVariance = buffer("runningVariance", suppliedRunningVariance);
    }

    /**
     * Creates batch normalization with one scale, zero bias, zero running mean, and one running
     * variance.
     *
     * <p>Caller-controlled validation completes before Tensor creation or identifier allocation.
     * The exact rank-one Shape {@code [featureCount]} is then used to create scale through
     * {@link ParameterInitializers#ones(Shape, DataType)}, bias through
     * {@link ParameterInitializers#zeros(Shape, DataType)}, running mean through
     * {@link TensorFactory#zeros(Shape, DataType, Optional, boolean)}, and running variance through
     * {@link TensorFactory#ones(Shape, DataType, Optional, boolean)}, in that order. The parameters
     * are gradient-eligible; the initial buffers are not. No random source, default scalar,
     * conversion, configurable initializer, or retained source exists.</p>
     *
     * <p>A Model allocation or identifier failure returns no partially constructed layer. Leaves
     * and identifiers created before a later failure are not rolled back.</p>
     *
     * @param featureCount positive static number of channel features
     * @param channelAxis non-negative logical input channel axis, validated against each input
     *     rank during forward construction
     * @param dataType non-null floating state type: FLOAT64, FLOAT32, or BFLOAT16
     * @param momentum non-null finite floating new-batch weight in {@code [0, 1]} with the exact
     *     state type, retained exactly
     * @param epsilon non-null finite strictly positive floating stabilizer with the exact state
     *     type, retained exactly
     * @throws NullPointerException if {@code dataType}, {@code momentum}, or {@code epsilon} is
     *     null, checked in that order after validating {@code channelAxis}
     * @throws IllegalArgumentException if {@code channelAxis} is negative, {@code featureCount}
     *     is not positive, {@code dataType} is not floating, a scalar violates the Model intrinsic
     *     contract or has a different type, or the feature count exceeds the Model Java-array
     *     limit
     * @throws ArithmeticException if checked Model element-count or layout arithmetic overflows
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws OutOfMemoryError if Model source or destination allocation fails
     */
    public BatchNorm(
            long featureCount,
            int channelAxis,
            DataType dataType,
            ScalarValue momentum,
            ScalarValue epsilon) {
        validateChannelAxis(channelAxis);
        DataType suppliedDataType = Objects.requireNonNull(dataType, "dataType");
        ScalarValue suppliedMomentum = Objects.requireNonNull(momentum, "momentum");
        ScalarValue suppliedEpsilon = Objects.requireNonNull(epsilon, "epsilon");
        if (featureCount <= 0) {
            throw new IllegalArgumentException(
                    "batch normalization featureCount must be positive: " + featureCount);
        }
        if (!suppliedDataType.isFloating()) {
            throw new IllegalArgumentException(
                    "batch normalization initialization requires floating data type: "
                            + suppliedDataType);
        }
        validateScalars(
                channelAxis, suppliedMomentum, suppliedEpsilon, suppliedDataType);

        Shape stateShape = Shape.of(featureCount);
        Tensor initializedScale = ParameterInitializers.ones(stateShape, suppliedDataType);
        Tensor initializedBias = ParameterInitializers.zeros(stateShape, suppliedDataType);
        Tensor initializedRunningMean = TensorFactory.zeros(
                stateShape, suppliedDataType, Optional.empty(), false);
        Tensor initializedRunningVariance = TensorFactory.ones(
                stateShape, suppliedDataType, Optional.empty(), false);

        this.channelAxis = channelAxis;
        this.featureDimension = stateShape.dimension(0);
        this.stateType = suppliedDataType;
        this.momentum = suppliedMomentum;
        this.epsilon = suppliedEpsilon;
        this.scale = parameter("scale", initializedScale);
        this.bias = parameter("bias", initializedBias);
        this.runningMean = buffer("runningMean", initializedRunningMean);
        this.runningVariance = buffer("runningVariance", initializedRunningVariance);
    }

    /**
     * Returns the stable scale parameter wrapper.
     *
     * @return the exact non-null wrapper declared under local name {@code scale}; its current
     *     binding has the layer's fixed data type and rank-one Shape
     */
    public Parameter scale() {
        return scale;
    }

    /**
     * Returns the stable bias parameter wrapper.
     *
     * @return the exact non-null wrapper declared under local name {@code bias}; its current
     *     binding has the layer's fixed data type and rank-one Shape
     */
    public Parameter bias() {
        return bias;
    }

    /**
     * Returns the stable running-mean buffer wrapper.
     *
     * @return the exact non-null wrapper declared under local name {@code runningMean};
     *     {@link Buffer#value()} observes the current symbolic statistic binding
     */
    public Buffer runningMean() {
        return runningMean;
    }

    /**
     * Returns the stable running-variance buffer wrapper.
     *
     * @return the exact non-null wrapper declared under local name {@code runningVariance};
     *     {@link Buffer#value()} observes the current symbolic statistic binding
     */
    public Buffer runningVariance() {
        return runningVariance;
    }

    /**
     * Builds one context-selected batch-normalization expression and applies its symbolic state
     * transition when training.
     *
     * <p>The method rejects null arguments, then reads scale, bias, running mean, and running
     * variance exactly once in that order. The input must have rank at least two; the stored
     * non-negative channel axis must be valid for that rank, and its exact Dimension must equal
     * the layer's fixed static feature Dimension structurally. This local equality requirement
     * rejects unresolved or unequal input channels before Model construction.</p>
     *
     * <p>The supplied context mode is read once and is authoritative even if this module's current
     * mode differs. Evaluation delegates once to
     * {@link Tensor#batchNormInference(int, Tensor, Tensor, Tensor, Tensor, ScalarValue)} and
     * changes no buffer. Training delegates once to
     * {@link Tensor#batchNormTraining(int, Tensor, Tensor, Tensor, Tensor, ScalarValue,
     * ScalarValue)}. Only after the complete result exists does this method install exact result
     * slot one as running mean, then exact slot two as running variance, and return exact slot
     * zero. A Model failure, including partial output-identifier consumption, changes neither
     * buffer. The two later installations are individual direct-buffer operations with no generic
     * rollback or atomic transaction guarantee.</p>
     *
     * @param input non-null rank-at-least-two floating Tensor whose selected channel Dimension
     *     structurally equals the fixed feature Dimension and which satisfies the delegated Model
     *     inference or training contract
     * @param context non-null immutable mode snapshot; its mode alone selects the branch for this
     *     call and it need not originate from this module
     * @return evaluation's fresh inference output, or training producer slot zero after exact
     *     next-statistic expressions from the same producer have been installed; never
     *     {@code null}
     * @throws NullPointerException if {@code input} or {@code context} is null, checked in that
     *     order
     * @throws IllegalArgumentException if input rank is below two, the selected input channel
     *     Dimension is unresolved or unequal to the fixed feature Dimension, or delegated Model
     *     type, reduction-domain, promotion, scalar, or state validation fails
     * @throws IndexOutOfBoundsException if the stored channel axis is outside the input rank
     * @throws IllegalStateException if Tensor identifier space is exhausted; training identifiers
     *     allocated for earlier producer slots may remain consumed without changing either buffer
     */
    public Tensor forward(Tensor input, ForwardContext context) {
        Tensor suppliedInput = Objects.requireNonNull(input, "input");
        ForwardContext suppliedContext = Objects.requireNonNull(context, "context");
        Tensor currentScale = scale.value();
        Tensor currentBias = bias.value();
        Tensor currentRunningMean = runningMean.value();
        Tensor currentRunningVariance = runningVariance.value();

        Shape inputShape = suppliedInput.descriptor().shape();
        if (inputShape.rank() < 2) {
            throw new IllegalArgumentException(
                    "batch normalization input rank must be at least 2, but was "
                            + inputShape.rank());
        }
        int normalizedAxis = inputShape.normalizeAxis(channelAxis);
        Dimension inputChannelDimension = inputShape.dimensions().get(normalizedAxis);
        if (!featureDimension.equals(inputChannelDimension)) {
            throw new IllegalArgumentException(
                    "batch normalization input channel Dimension must equal feature Dimension: "
                            + "input=" + inputChannelDimension + ", feature=" + featureDimension);
        }

        return switch (suppliedContext.mode()) {
            case EVALUATION -> suppliedInput.batchNormInference(
                    channelAxis,
                    currentScale,
                    currentBias,
                    currentRunningMean,
                    currentRunningVariance,
                    epsilon);
            case TRAINING -> {
                BatchNormTrainingResult result = suppliedInput.batchNormTraining(
                        channelAxis,
                        currentScale,
                        currentBias,
                        currentRunningMean,
                        currentRunningVariance,
                        momentum,
                        epsilon);
                replaceBuffer("runningMean", result.nextRunningMean());
                replaceBuffer("runningVariance", result.nextRunningVariance());
                yield result.output();
            }
        };
    }

    private static void validateChannelAxis(int channelAxis) {
        if (channelAxis < 0) {
            throw new IllegalArgumentException(
                    "batch normalization channelAxis must be non-negative: " + channelAxis);
        }
    }

    private static StateSchema validateScale(Tensor scale) {
        DataType dataType = scale.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "batch normalization scale must have a floating data type: " + dataType);
        }
        if (!scale.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "batch normalization scale must have requiresGrad == true");
        }
        Shape shape = scale.descriptor().shape();
        validateRankOneStatic(shape, "scale");
        long featureCount = ((StaticDimension) shape.dimension(0)).size();
        if (featureCount == 0) {
            throw new IllegalArgumentException(
                    "batch normalization scale feature extent must be positive: " + featureCount);
        }
        return new StateSchema(dataType, shape);
    }

    private static void validateParameter(Tensor tensor, String role, StateSchema schema) {
        DataType dataType = tensor.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "batch normalization " + role + " must have a floating data type: " + dataType);
        }
        if (!tensor.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "batch normalization " + role + " must have requiresGrad == true");
        }
        Shape shape = tensor.descriptor().shape();
        validateRankOneStatic(shape, role);
        validateStateTypeAndShape(dataType, shape, role, schema);
    }

    private static void validateBuffer(Tensor tensor, String role, StateSchema schema) {
        DataType dataType = tensor.descriptor().dataType();
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    "batch normalization " + role + " must have a floating data type: " + dataType);
        }
        if (tensor.descriptor().requiresGrad()) {
            throw new IllegalArgumentException(
                    "batch normalization " + role + " must have requiresGrad == false");
        }
        Shape shape = tensor.descriptor().shape();
        validateRankOneStatic(shape, role);
        validateStateTypeAndShape(dataType, shape, role, schema);
    }

    private static void validateRankOneStatic(Shape shape, String role) {
        if (shape.rank() != 1) {
            throw new IllegalArgumentException(
                    "batch normalization " + role + " Shape must have rank one: " + shape);
        }
        if (!shape.isFullyStatic()) {
            throw new IllegalArgumentException(
                    "batch normalization " + role + " Shape must be fully static: " + shape);
        }
    }

    private static void validateStateTypeAndShape(
            DataType dataType, Shape shape, String role, StateSchema schema) {
        if (dataType != schema.dataType()) {
            throw new IllegalArgumentException(
                    "batch normalization " + role + " data type must equal scale data type: scale="
                            + schema.dataType() + ", " + role + "=" + dataType);
        }
        if (!shape.equals(schema.shape())) {
            throw new IllegalArgumentException(
                    "batch normalization " + role + " Shape must equal scale Shape: scale="
                            + schema.shape() + ", " + role + "=" + shape);
        }
    }

    private static void validateScalars(
            int channelAxis,
            ScalarValue momentum,
            ScalarValue epsilon,
            DataType stateType) {
        new BatchNormTrainingAttrs(channelAxis, momentum, epsilon);
        if (momentum.dataType() != stateType) {
            throw new IllegalArgumentException(
                    "batch normalization momentum data type must equal state data type: momentum="
                            + momentum.dataType() + ", state=" + stateType);
        }
        if (epsilon.dataType() != stateType) {
            throw new IllegalArgumentException(
                    "batch normalization epsilon data type must equal state data type: epsilon="
                            + epsilon.dataType() + ", state=" + stateType);
        }
    }

    private record StateSchema(DataType dataType, Shape shape) {
    }
}
