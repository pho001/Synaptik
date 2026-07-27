package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;

/**
 * Builds the closed elementwise portion of the first-order derivative matrix.
 *
 * <p>The selected rows include promoted floating binary {@code ADD}/{@code SUB}/{@code MUL}/
 * {@code DIV}, scalar {@code ADD}/{@code SUB}/{@code MUL}/{@code DIV}, branch-only
 * {@code WHERE}, floating {@code CAST}, direct-zero {@code FLOOR}/{@code CEIL}/{@code SIGN}, and
 * {@code NEG}/{@code EXP}/{@code EXPM1}/{@code SIGMOID}/{@code TANH}/{@code ERF}. Binary and
 * branch contributions reverse broadcasting with ordinary {@code sumToShape} and then use an
 * ordinary floating cast when the selected input type differs. For input {@code x} and output
 * cotangent {@code g}, ERF constructs
 * {@code g * exp(-(x * x)) * C}, where {@code C} is the exact typed representation of
 * {@code 2 / sqrt(pi)}: BFLOAT16 bits {@code 0x3F90}, FLOAT32 bits {@code 0x3F906EBB}, or
 * FLOAT64 bits {@code 0x3FF20DD750429B6D}. The coefficient is scalar operation metadata, not a
 * Tensor leaf or logical-splat binding, and is never computed with a host transcendental
 * function. Every formula is expressed only through public Tensor operations. Preflight owns
 * descriptor, attributes, role, and policy validation before this owner is called.</p>
 *
 * <p>This owner constructs expression metadata only. It performs no graph capture, numerical
 * evaluation, storage access, lowering, backend selection, or execution.</p>
 */
final class ElementwiseGradientRules {
    private ElementwiseGradientRules() {}

    /**
     * Builds ordered input cotangents for one preflight-approved elementwise occurrence.
     *
     * @param producer exact preflight-approved original producer occurrence
     * @param outputIndex zero-based selected canonical output position
     * @param gradient non-null accumulated cotangent for that exact output
     * @param selectedInputs non-null input-position-aligned selected-route flags; observed but not
     *     mutated
     * @param constants non-null request-local owner of explicit exact typed zero/one splats
     * @return a new input-position-aligned Tensor array; a {@code null} element denotes an
     *     unselected or non-differentiable role
     * @throws IllegalStateException if called for an operation kind that did not pass the closed
     *     preflight matrix, or if preflight admitted ERF with a non-floating type
     */
    static Tensor[] apply(
            TensorProducer producer,
            int outputIndex,
            Tensor gradient,
            boolean[] selectedInputs,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor output = producer.output(outputIndex);
        if (producer.operation().kind() instanceof BinaryArithmeticKind kind) {
            Tensor left = producer.inputs().get(0);
            Tensor right = producer.inputs().get(1);
            return switch (kind) {
                case ADD -> new Tensor[] {
                    selectedInputs[0] ? normalize(gradient, left) : null,
                    selectedInputs[1] ? normalize(gradient, right) : null
                };
                case SUB -> new Tensor[] {
                    selectedInputs[0] ? normalize(gradient, left) : null,
                    selectedInputs[1] ? normalize(gradient.neg(), right) : null
                };
                case MUL -> new Tensor[] {
                    selectedInputs[0] ? normalize(gradient.mul(right), left) : null,
                    selectedInputs[1] ? normalize(gradient.mul(left), right) : null
                };
                case DIV -> new Tensor[] {
                    selectedInputs[0] ? normalize(gradient.div(right), left) : null,
                    selectedInputs[1]
                            ? normalize(
                                    gradient.mul(left).neg().div(right.mul(right)),
                                    right)
                            : null
                };
                case MIN, MAX, POW ->
                        throw new IllegalStateException("binary kind was not preflight-approved: " + kind);
            };
        }
        if (producer.operation().kind() instanceof ScalarElementwiseKind kind) {
            Tensor input = producer.inputs().getFirst();
            ScalarValueAttrs attrs = (ScalarValueAttrs) producer.operation().attrs();
            return switch (kind) {
                case ADD, SUB -> new Tensor[] {gradient};
                case MUL -> new Tensor[] {gradient.mul(attrs.value())};
                case DIV -> new Tensor[] {gradient.div(attrs.value())};
                case MIN, MAX, POW, CLAMP ->
                        throw new IllegalStateException("scalar kind was not preflight-approved: " + kind);
            };
        }
        if (producer.operation().kind() == WhereSelectionKind.WHERE) {
            Tensor condition = producer.inputs().get(0);
            Tensor ifTrue = producer.inputs().get(1);
            Tensor ifFalse = producer.inputs().get(2);
            Tensor zero = constants.zeroLike(output);
            return new Tensor[] {
                null,
                selectedInputs[1]
                        ? normalize(Tensor.where(condition, gradient, zero), ifTrue)
                        : null,
                selectedInputs[2]
                        ? normalize(Tensor.where(condition, zero, gradient), ifFalse)
                        : null
            };
        }
        if (producer.operation().kind() == CastKind.CAST) {
            Tensor input = producer.inputs().getFirst();
            return new Tensor[] {
                gradient.descriptor().dataType() == input.descriptor().dataType()
                        ? gradient
                        : gradient.cast(input.descriptor().dataType())
            };
        }
        if (producer.operation().kind() instanceof UnaryElementwiseKind kind) {
            return switch (kind) {
                case NEG -> new Tensor[] {gradient.neg()};
                case EXP -> new Tensor[] {gradient.mul(output)};
                case EXPM1 -> new Tensor[] {gradient.mul(output.add(constants.oneLike(output)))};
                case SIGMOID -> new Tensor[] {
                    gradient.mul(output).mul(constants.oneLike(output).sub(output))
                };
                case TANH -> new Tensor[] {
                    gradient.mul(constants.oneLike(output).sub(output.mul(output)))
                };
                case ERF -> new Tensor[] {
                    gradient.mul(producer.inputs().getFirst()
                                    .mul(producer.inputs().getFirst())
                                    .neg()
                                    .exp())
                            .mul(erfCoefficient(output.descriptor().dataType()))
                };
                case FLOOR, CEIL, SIGN ->
                        new Tensor[] {constants.zeroLike(producer.inputs().getFirst())};
                case ABS, RECIPROCAL, LOG, LOG1P, SQRT, RSQRT, RELU,
                        GELU, GELU_TANH_APPROXIMATION, SILU ->
                        throw new IllegalStateException("unary kind was not preflight-approved: " + kind);
            };
        }
        throw new IllegalStateException(
                "elementwise operation was not preflight-approved: " + producer.operation());
    }

    /**
     * Reverses ordinary broadcasting and converts one contribution to its selected input type.
     *
     * @param contribution non-null floating contribution in the producer output algebra
     * @param input non-null selected floating input whose exact Shape and data type are required
     * @return a non-null ordinary Tensor expression with the input's exact descriptor Shape and
     *     data type
     */
    private static Tensor normalize(Tensor contribution, Tensor input) {
        Tensor normalized = contribution.descriptor().shape().equals(input.descriptor().shape())
                ? contribution
                : contribution.sumToShape(input.descriptor().shape());
        return normalized.descriptor().dataType() == input.descriptor().dataType()
                ? normalized
                : normalized.cast(input.descriptor().dataType());
    }

    /**
     * Returns the fixed correctly rounded scalar coefficient for the selected ERF type.
     *
     * @param dataType non-null preflight-approved floating ERF data type
     * @return a non-null immutable scalar value with the task-specified raw representation
     * @throws IllegalStateException if preflight admitted an integral or BOOL type
     */
    private static ScalarValue erfCoefficient(DataType dataType) {
        return switch (dataType) {
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0x3F90);
            case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(0x3F906EBB));
            case FLOAT64 -> ScalarValue.float64(
                    Double.longBitsToDouble(0x3FF20DD750429B6DL));
            case INT32, INT64, BOOL ->
                    throw new IllegalStateException(
                            "ERF preflight admitted non-floating type: " + dataType);
        };
    }
}
