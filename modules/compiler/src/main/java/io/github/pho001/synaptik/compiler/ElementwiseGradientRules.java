package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastValueConversions;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;

/**
 * Builds the closed current 48-kind elementwise and activation portion of the first-order
 * derivative matrix.
 *
 * <p>The differentiable rows are the seven promoted floating binary arithmetic kinds, the eight
 * exact-type floating scalar kinds, branch-only {@code WHERE}, floating-to-floating
 * {@code CAST}, and all nineteen floating unary kinds. Binary and branch contributions reverse
 * broadcasting with ordinary {@code sumToShape} and then use an ordinary floating cast when the
 * selected input type differs. Comparisons, Boolean logic, floating classifications, the
 * {@code WHERE} condition, scalar attributes and bounds, and non-floating casts receive no
 * cotangent through their occurrences.</p>
 *
 * <p>Binary and scalar extrema split exact numeric ties by one half. Piecewise comparisons make
 * unordered NaN positions exact positive zero. {@code CLAMP} applies the same tie rule to the
 * ordered composition {@code MIN(MAX(x, min), max)}, so a tie at both stages contributes one
 * quarter. {@code ABS} and {@code RELU} choose exact positive zero at signed zero and NaN;
 * {@code FLOOR}, {@code CEIL}, and {@code SIGN} always return a direct exact positive zero.
 * Exact GELU, fixed tanh-approximation GELU, and SiLU use their analytic formulas for finite and
 * NaN inputs, return the incoming cotangent at positive infinity, and return exact positive zero
 * at negative infinity. Every other formula deliberately uses ordinary Tensor operations in the
 * specified order without a compiler-inserted domain or exceptional-value mask.</p>
 *
 * <p>Every generated occurrence is an ordinary public Tensor operation in the same expression
 * graph as its forward occurrence. Exact masks, ties, saved outputs, and guards remain exact;
 * formula arithmetic recursively inherits Model's operation-family primitive and aggregate
 * semantics. This class creates no backward-only approximation, output envelope, or numerical
 * selector lookup.</p>
 *
 * <p>For input {@code x} and output cotangent {@code g}, ERF constructs
 * {@code g * exp(-(x * x)) * C}, where {@code C} is the exact typed representation of
 * {@code 2 / sqrt(pi)}: BFLOAT16 bits {@code 0x3F90}, FLOAT16 bits {@code 0x3C83},
 * FLOAT32 bits {@code 0x3F906EBB}, or FLOAT64 bits {@code 0x3FF20DD750429B6D}. The coefficient is
 * scalar operation metadata, not a Tensor leaf or logical-splat binding, and is never computed
 * with a host transcendental function. Other coefficients use fixed BFLOAT16, FLOAT16, FLOAT32,
 * and FLOAT64 bit patterns for {@code 0.5}, {@code -0.5}, {@code 2}, inverse square roots, and the
 * two tanh-approximation constants. Scalar {@code POW} derives exponent-minus-one with exactly
 * one subtraction in the represented type. Coefficients remain operation metadata; only values
 * needed as Tensor comparison operands use request-local logical splats. Every formula is
 * expressed only through public Tensor operations. Preflight owns descriptor, attributes, role,
 * and policy validation before this owner is called.</p>
 *
 * <p>This owner constructs expression metadata only. It performs no graph capture, numerical
 * evaluation, storage access, lowering, backend selection, or execution.</p>
 *
 * <p>Descriptor access preserves original source roles without resolving primal values.
 * Real construction uses Tensor references directly; checked metadata construction retains
 * the same eager operation sites, including temporaries not used by a returned cotangent.</p>
 *
 * @param <V> direct Tensor or checked metadata operand representation
 */
final class ElementwiseGradientRules<V> {
    private final OrdinaryConstruction<V> ops;

    /**
     * Binds this formula owner to one compile-local construction route.
     * @param ops non-null direct Tensor or checked metadata construction
     */
    ElementwiseGradientRules(OrdinaryConstruction<V> ops) {
        this.ops = java.util.Objects.requireNonNull(ops);
    }

    /**
     * Builds existing formulas using explicit lexical bindings for every primal operand.
     * @param producer exact original occurrence, whose identities remain contribution keys
     * @param outputIndex selected canonical output slot
     * @param gradient accumulated output cotangent
     * @param selectedInputs ordered selected operand positions
     * @param primal non-null total binding of referenced original primals to local operands
     * @return input-position-aligned cotangents, null for unselected roles
     */
    V[] apply(
            OrdinaryConstruction.Source<V> producer,
            int outputIndex,
            V gradient,
            boolean[] selectedInputs,
            java.util.function.Function<V, V> primal) {
        V output = primal.apply(producer.output(outputIndex));
        if (producer.operation().kind() instanceof BinaryArithmeticKind kind) {
            V left = primal.apply(producer.inputs().get(0));
            V right = primal.apply(producer.inputs().get(1));
            return switch (kind) {
                case ADD -> ops.result(selectedInputs[0] ? normalize(gradient, left) : null,
                    selectedInputs[1] ? normalize(gradient, right) : null);
                case SUB -> ops.result(selectedInputs[0] ? normalize(gradient, left) : null,
                    selectedInputs[1] ? normalize(ops.neg(gradient), right) : null);
                case MUL -> ops.result(selectedInputs[0] ? normalize(ops.mul(gradient, right), left) : null,
                    selectedInputs[1] ? normalize(ops.mul(gradient, left), right) : null);
                case DIV -> ops.result(selectedInputs[0] ? normalize(ops.div(gradient, right), left) : null,
                    selectedInputs[1]
                            ? normalize(
                                    ops.div(ops.neg(ops.mul(gradient, left)), ops.mul(right, right)),
                                    right)
                            : null);
                case MIN -> ops.result(selectedInputs[0]
                            ? normalize(extremaContribution(
                                    ops.lessThan(left, right),
                                    ops.equalTo(left, right),
                                    gradient,
                                    ops.zeroLike(output),
                                    half(ops.descriptor(output).dataType())), left)
                            : null,
                    selectedInputs[1]
                            ? normalize(extremaContribution(
                                    ops.lessThan(right, left),
                                    ops.equalTo(right, left),
                                    gradient,
                                    ops.zeroLike(output),
                                    half(ops.descriptor(output).dataType())), right)
                            : null);
                case MAX -> ops.result(selectedInputs[0]
                            ? normalize(extremaContribution(
                                    ops.greaterThan(left, right),
                                    ops.equalTo(left, right),
                                    gradient,
                                    ops.zeroLike(output),
                                    half(ops.descriptor(output).dataType())), left)
                            : null,
                    selectedInputs[1]
                            ? normalize(extremaContribution(
                                    ops.greaterThan(right, left),
                                    ops.equalTo(right, left),
                                    gradient,
                                    ops.zeroLike(output),
                                    half(ops.descriptor(output).dataType())), right)
                            : null);
                case POW -> ops.result(selectedInputs[0]
                            ? normalize(
                                    ops.mul(ops.mul(gradient, right), ops.pow(left, ops.sub(right, ops.oneLike(right)))),
                                    left)
                            : null,
                    selectedInputs[1]
                            ? normalize(ops.mul(ops.mul(gradient, output), ops.log(left)), right)
                            : null);
            };
        }
        if (producer.operation().kind() instanceof ScalarElementwiseKind kind) {
            V input = primal.apply(producer.inputs().getFirst());
            return switch (kind) {
                case ADD, SUB -> ops.result(gradient);
                case MUL -> ops.result(ops.mul(gradient, scalarAttrs(producer).value()));
                case DIV -> ops.result(ops.div(gradient, scalarAttrs(producer).value()));
                case MIN -> ops.result(scalarExtrema(
                        input, gradient, scalarAttrs(producer).value(), false));
                case MAX -> ops.result(scalarExtrema(
                        input, gradient, scalarAttrs(producer).value(), true));
                case POW -> {
                    ScalarValue exponent = scalarAttrs(producer).value();
                    yield ops.result(ops.mul(ops.mul(gradient, exponent), ops.pow(input, exponentMinusOne(exponent))));
                }
                case CLAMP -> ops.result(clamp(
                        input,
                        gradient,
                        (ClampRangeAttrs) producer.operation().attrs()));
            };
        }
        if (producer.operation().kind() == WhereSelectionKind.WHERE) {
            V condition = primal.apply(producer.inputs().get(0));
            V ifTrue = primal.apply(producer.inputs().get(1));
            V ifFalse = primal.apply(producer.inputs().get(2));
            V zero = ops.zeroLike(output);
            return ops.result(null,
                selectedInputs[1]
                        ? normalize(ops.where(condition, gradient, zero), ifTrue)
                        : null,
                selectedInputs[2]
                        ? normalize(ops.where(condition, zero, gradient), ifFalse)
                        : null);
        }
        if (producer.operation().kind() == CastKind.CAST) {
            V input = primal.apply(producer.inputs().getFirst());
            return ops.result(ops.descriptor(gradient).dataType() == ops.descriptor(input).dataType()
                        ? gradient
                        : ops.cast(gradient, ops.descriptor(input).dataType()));
        }
        if (producer.operation().kind() instanceof UnaryElementwiseKind kind) {
            V input = primal.apply(producer.inputs().getFirst());
            DataType dataType = ops.descriptor(output).dataType();
            return switch (kind) {
                case ABS -> ops.result(ops.where(ops.greaterThan(input, ops.zeroLike(input)), gradient, ops.where(ops.lessThan(input, ops.zeroLike(input)), ops.neg(gradient), ops.zeroLike(input))));
                case NEG -> ops.result(ops.neg(gradient));
                case RECIPROCAL -> ops.result(ops.div(ops.neg(gradient), ops.mul(input, input)));
                case LOG -> ops.result(ops.div(gradient, input));
                case LOG1P -> ops.result(ops.div(gradient, ops.add(input, ops.oneLike(input))));
                case EXP -> ops.result(ops.mul(gradient, output));
                case EXPM1 -> ops.result(ops.mul(gradient, ops.add(output, ops.oneLike(output))));
                case SQRT -> ops.result(ops.div(gradient, ops.mul(output, two(dataType))));
                case RSQRT -> ops.result(ops.mul(ops.mul(ops.mul(ops.mul(gradient, negativeHalf(dataType)), output), output), output));
                case SIGMOID -> ops.result(ops.mul(ops.mul(gradient, output), ops.sub(ops.oneLike(output), output)));
                case TANH -> ops.result(ops.mul(gradient, ops.sub(ops.oneLike(output), ops.mul(output, output))));
                case ERF -> ops.result(ops.mul(ops.mul(gradient, ops.exp(ops.neg(ops.mul(input, input)))), erfCoefficient(ops.descriptor(output).dataType())));
                case FLOOR, CEIL, SIGN ->
                        ops.result(ops.zeroLike(input));
                case RELU -> ops.result(ops.where(ops.greaterThan(input, ops.zeroLike(input)), gradient, ops.zeroLike(input)));
                case GELU -> ops.result(gelu(input, gradient));
                case GELU_TANH_APPROXIMATION ->
                        ops.result(geluTanhApproximation(input, gradient));
                case SILU -> ops.result(silu(input, gradient));
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
    private V normalize(V contribution, V input) {
        V normalized = ops.descriptor(contribution).shape().equals(ops.descriptor(input).shape())
                ? contribution
                : ops.sumToShape(contribution, ops.descriptor(input).shape());
        return ops.descriptor(normalized).dataType() == ops.descriptor(input).dataType()
                ? normalized
                : ops.cast(normalized, ops.descriptor(input).dataType());
    }

    private ScalarValueAttrs scalarAttrs(OrdinaryConstruction.Source<V> producer) {
        return (ScalarValueAttrs) producer.operation().attrs();
    }

    private V extremaContribution(V ordered, V equal, V gradient, V zero, ScalarValue half) {
        return ops.where(ordered, gradient, ops.where(equal, ops.mul(gradient, half), zero));
    }

    private V scalarExtrema(
            V input,
            V gradient,
            ScalarValue bound,
            boolean maximum) {
        V boundTensor = ops.valueLike(bound, input);
        V ordered = maximum
                ? ops.greaterThan(input, boundTensor)
                : ops.lessThan(input, boundTensor);
        return extremaContribution(
                ordered,
                ops.equalTo(input, boundTensor),
                gradient,
                ops.zeroLike(input),
                half(ops.descriptor(input).dataType()));
    }

    private V clamp(V input, V gradient, ClampRangeAttrs attrs) {
        V min = ops.valueLike(attrs.minValue(), input);
        V max = ops.valueLike(attrs.maxValue(), input);
        ScalarValue half = half(ops.descriptor(input).dataType());
        V lowered = ops.maximum(input, attrs.minValue());
        V upperRouted = ops.where(ops.lessThan(lowered, max), gradient, ops.where(ops.equalTo(lowered, max), ops.mul(gradient, half), ops.zeroLike(input)));
        return ops.where(ops.greaterThan(input, min), upperRouted, ops.where(ops.equalTo(input, min), ops.mul(upperRouted, half), ops.zeroLike(input)));
    }

    private V gelu(V input, V gradient) {
        DataType type = ops.descriptor(input).dataType();
        V phiCdf = ops.mul(ops.add(ops.oneLike(input), ops.erf(ops.mul(input, invSqrt2(type)))), half(type));
        V phiPdfTerm = ops.mul(ops.mul(input, ops.exp(ops.mul(ops.mul(input, input), negativeHalf(type)))), invSqrt2Pi(type));
        V regular = ops.mul(gradient, ops.add(phiCdf, phiPdfTerm));
        return activationInfinitySelection(input, gradient, regular);
    }

    private V geluTanhApproximation(V input, V gradient) {
        DataType type = ops.descriptor(input).dataType();
        V squared = ops.mul(input, input);
        V u = ops.mul(ops.add(input, ops.mul(ops.mul(squared, input), geluCubic(type))), sqrt2OverPi(type));
        V t = ops.tanh(u);
        V uPrime = ops.mul(ops.add(ops.oneLike(input), ops.mul(squared, geluCubicDerivative(type))), sqrt2OverPi(type));
        V regular = ops.mul(gradient, ops.add(ops.mul(ops.add(ops.oneLike(input), t), half(type)), ops.mul(ops.mul(ops.mul(input, ops.sub(ops.oneLike(input), ops.mul(t, t))), uPrime), half(type))));
        return activationInfinitySelection(input, gradient, regular);
    }

    private V silu(V input, V gradient) {
        V sigmoid = ops.sigmoid(input);
        V regular = ops.mul(ops.mul(gradient, sigmoid), ops.add(ops.oneLike(input), ops.mul(input, ops.sub(ops.oneLike(input), sigmoid))));
        return activationInfinitySelection(input, gradient, regular);
    }

    private V activationInfinitySelection(
            V input,
            V gradient,
            V regular) {
        return ops.where(ops.isInf(input), ops.where(ops.greaterThan(input, ops.zeroLike(input)), gradient, ops.zeroLike(input)), regular);
    }

    private static ScalarValue exponentMinusOne(ScalarValue exponent) {
        return switch (exponent.dataType()) {
            case BFLOAT16 -> ScalarValue.bfloat16Bits(BFloat16Bits.fromFloat(
                    BFloat16Bits.toFloat(exponent.bfloat16Bits()) - 1.0f));
            case FLOAT16 -> CastValueConversions.convert(
                    ScalarValue.float32(
                            Float16Bits.toFloat(exponent.float16Bits()) - 1.0f),
                    DataType.FLOAT16);
            case FLOAT32 -> ScalarValue.float32(exponent.float32Value() - 1.0f);
            case FLOAT64 -> ScalarValue.float64(exponent.float64Value() - 1.0d);
            case INT32, INT64, BOOL ->
                    throw new IllegalStateException(
                            "scalar POW preflight admitted non-floating exponent: "
                                    + exponent.dataType());
        };
    }

    private static ScalarValue half(DataType dataType) {
        return coefficient(dataType, Coefficient.HALF);
    }

    private static ScalarValue negativeHalf(DataType dataType) {
        return coefficient(dataType, Coefficient.NEGATIVE_HALF);
    }

    private static ScalarValue two(DataType dataType) {
        return coefficient(dataType, Coefficient.TWO);
    }

    private static ScalarValue invSqrt2(DataType dataType) {
        return coefficient(dataType, Coefficient.INV_SQRT_2);
    }

    private static ScalarValue invSqrt2Pi(DataType dataType) {
        return coefficient(dataType, Coefficient.INV_SQRT_2_PI);
    }

    private static ScalarValue sqrt2OverPi(DataType dataType) {
        return coefficient(dataType, Coefficient.SQRT_2_OVER_PI);
    }

    private static ScalarValue geluCubic(DataType dataType) {
        return coefficient(dataType, Coefficient.GELU_CUBIC);
    }

    private static ScalarValue geluCubicDerivative(DataType dataType) {
        return coefficient(dataType, Coefficient.GELU_CUBIC_DERIVATIVE);
    }

    private static ScalarValue coefficient(DataType dataType, Coefficient coefficient) {
        return switch (dataType) {
            case BFLOAT16 -> ScalarValue.bfloat16Bits(coefficient.bfloat16Bits);
            case FLOAT16 -> ScalarValue.float16Bits(coefficient.float16Bits);
            case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(coefficient.float32Bits));
            case FLOAT64 -> ScalarValue.float64(Double.longBitsToDouble(coefficient.float64Bits));
            case INT32, INT64, BOOL ->
                    throw new IllegalStateException(
                            "elementwise preflight admitted non-floating type: " + dataType);
        };
    }

    private enum Coefficient {
        HALF(0x3F00, 0x3800, 0x3F000000, 0x3FE0000000000000L),
        NEGATIVE_HALF(0xBF00, 0xB800, 0xBF000000, 0xBFE0000000000000L),
        TWO(0x4000, 0x4000, 0x40000000, 0x4000000000000000L),
        INV_SQRT_2(0x3F35, 0x39A8, 0x3F3504F3, 0x3FE6A09E667F3BCDL),
        INV_SQRT_2_PI(0x3ECC, 0x3662, 0x3ECC422A, 0x3FD9884533D43651L),
        SQRT_2_OVER_PI(0x3F4C, 0x3A62, 0x3F4C422A, 0x3FE9884533D43651L),
        GELU_CUBIC(0x3D37, 0x29B9, 0x3D372713, 0x3FA6E4E26D4801F7L),
        GELU_CUBIC_DERIVATIVE(0x3E09, 0x304B, 0x3E095D4F, 0x3FC12BA9D1F60179L);

        private final short bfloat16Bits;
        private final short float16Bits;
        private final int float32Bits;
        private final long float64Bits;

        Coefficient(
                int bfloat16Bits, int float16Bits, int float32Bits, long float64Bits) {
            this.bfloat16Bits = (short) bfloat16Bits;
            this.float16Bits = (short) float16Bits;
            this.float32Bits = float32Bits;
            this.float64Bits = float64Bits;
        }
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
            case FLOAT16 -> ScalarValue.float16Bits((short) 0x3C83);
            case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(0x3F906EBB));
            case FLOAT64 -> ScalarValue.float64(
                    Double.longBitsToDouble(0x3FF20DD750429B6DL));
            case INT32, INT64, BOOL ->
                    throw new IllegalStateException(
                            "ERF preflight admitted non-floating type: " + dataType);
        };
    }
}
