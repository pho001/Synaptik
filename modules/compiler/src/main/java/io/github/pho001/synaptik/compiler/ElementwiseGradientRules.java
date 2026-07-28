package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ClampRangeAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarValueAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;

/**
 * Builds the closed 48-kind elementwise and activation portion of the first-order derivative
 * matrix through Compiler 0005A.
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
 * <p>For input {@code x} and output cotangent {@code g}, ERF constructs
 * {@code g * exp(-(x * x)) * C}, where {@code C} is the exact typed representation of
 * {@code 2 / sqrt(pi)}: BFLOAT16 bits {@code 0x3F90}, FLOAT32 bits {@code 0x3F906EBB}, or
 * FLOAT64 bits {@code 0x3FF20DD750429B6D}. The coefficient is scalar operation metadata, not a
 * Tensor leaf or logical-splat binding, and is never computed with a host transcendental
 * function. Other coefficients use fixed BFLOAT16, FLOAT32, and FLOAT64 bit patterns for
 * {@code 0.5}, {@code -0.5}, {@code 2}, inverse square roots, and the two tanh-approximation
 * constants. Scalar {@code POW} derives exponent-minus-one with exactly one subtraction in the
 * represented type. Coefficients remain operation metadata; only values needed as Tensor
 * comparison operands use request-local logical splats. Every formula is expressed only through
 * public Tensor operations. Preflight owns descriptor, attributes, role, and policy validation
 * before this owner is called.</p>
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
     * @param constants non-null request-local owner of explicit exact typed scalar splats
     * @return a new input-position-aligned Tensor array; a {@code null} element denotes an
     *     unselected or non-differentiable role
     * @throws IllegalStateException if called for an operation kind that did not pass the closed
     *     preflight matrix, if preflight admitted a coefficient formula with a non-floating type,
     *     or if scalar POW receives a non-floating exponent
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
                case MIN -> new Tensor[] {
                    selectedInputs[0]
                            ? normalize(extremaContribution(
                                    left.lessThan(right),
                                    left.equalTo(right),
                                    gradient,
                                    constants.zeroLike(output),
                                    half(output.descriptor().dataType())), left)
                            : null,
                    selectedInputs[1]
                            ? normalize(extremaContribution(
                                    right.lessThan(left),
                                    right.equalTo(left),
                                    gradient,
                                    constants.zeroLike(output),
                                    half(output.descriptor().dataType())), right)
                            : null
                };
                case MAX -> new Tensor[] {
                    selectedInputs[0]
                            ? normalize(extremaContribution(
                                    left.greaterThan(right),
                                    left.equalTo(right),
                                    gradient,
                                    constants.zeroLike(output),
                                    half(output.descriptor().dataType())), left)
                            : null,
                    selectedInputs[1]
                            ? normalize(extremaContribution(
                                    right.greaterThan(left),
                                    right.equalTo(left),
                                    gradient,
                                    constants.zeroLike(output),
                                    half(output.descriptor().dataType())), right)
                            : null
                };
                case POW -> new Tensor[] {
                    selectedInputs[0]
                            ? normalize(
                                    gradient.mul(right)
                                            .mul(left.pow(right.sub(constants.oneLike(right)))),
                                    left)
                            : null,
                    selectedInputs[1]
                            ? normalize(gradient.mul(output).mul(left.log()), right)
                            : null
                };
            };
        }
        if (producer.operation().kind() instanceof ScalarElementwiseKind kind) {
            Tensor input = producer.inputs().getFirst();
            return switch (kind) {
                case ADD, SUB -> new Tensor[] {gradient};
                case MUL -> new Tensor[] {gradient.mul(scalarAttrs(producer).value())};
                case DIV -> new Tensor[] {gradient.div(scalarAttrs(producer).value())};
                case MIN -> new Tensor[] {scalarExtrema(
                        input, gradient, scalarAttrs(producer).value(), false, constants)};
                case MAX -> new Tensor[] {scalarExtrema(
                        input, gradient, scalarAttrs(producer).value(), true, constants)};
                case POW -> {
                    ScalarValue exponent = scalarAttrs(producer).value();
                    yield new Tensor[] {gradient.mul(exponent)
                            .mul(input.pow(exponentMinusOne(exponent)))};
                }
                case CLAMP -> new Tensor[] {clamp(
                        input,
                        gradient,
                        (ClampRangeAttrs) producer.operation().attrs(),
                        constants)};
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
            Tensor input = producer.inputs().getFirst();
            DataType dataType = output.descriptor().dataType();
            return switch (kind) {
                case ABS -> new Tensor[] {Tensor.where(
                        input.greaterThan(constants.zeroLike(input)),
                        gradient,
                        Tensor.where(
                                input.lessThan(constants.zeroLike(input)),
                                gradient.neg(),
                                constants.zeroLike(input)))};
                case NEG -> new Tensor[] {gradient.neg()};
                case RECIPROCAL -> new Tensor[] {gradient.neg().div(input.mul(input))};
                case LOG -> new Tensor[] {gradient.div(input)};
                case LOG1P -> new Tensor[] {gradient.div(input.add(constants.oneLike(input)))};
                case EXP -> new Tensor[] {gradient.mul(output)};
                case EXPM1 -> new Tensor[] {gradient.mul(output.add(constants.oneLike(output)))};
                case SQRT -> new Tensor[] {gradient.div(output.mul(two(dataType)))};
                case RSQRT -> new Tensor[] {
                    gradient.mul(negativeHalf(dataType)).mul(output).mul(output).mul(output)
                };
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
                        new Tensor[] {constants.zeroLike(input)};
                case RELU -> new Tensor[] {Tensor.where(
                        input.greaterThan(constants.zeroLike(input)),
                        gradient,
                        constants.zeroLike(input))};
                case GELU -> new Tensor[] {gelu(input, gradient, constants)};
                case GELU_TANH_APPROXIMATION ->
                        new Tensor[] {geluTanhApproximation(input, gradient, constants)};
                case SILU -> new Tensor[] {silu(input, gradient, constants)};
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

    private static ScalarValueAttrs scalarAttrs(TensorProducer producer) {
        return (ScalarValueAttrs) producer.operation().attrs();
    }

    private static Tensor extremaContribution(
            Tensor ordered,
            Tensor equal,
            Tensor gradient,
            Tensor zero,
            ScalarValue half) {
        return Tensor.where(
                ordered,
                gradient,
                Tensor.where(equal, gradient.mul(half), zero));
    }

    private static Tensor scalarExtrema(
            Tensor input,
            Tensor gradient,
            ScalarValue bound,
            boolean maximum,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor boundTensor = constants.valueLike(bound, input);
        Tensor ordered = maximum
                ? input.greaterThan(boundTensor)
                : input.lessThan(boundTensor);
        return extremaContribution(
                ordered,
                input.equalTo(boundTensor),
                gradient,
                constants.zeroLike(input),
                half(input.descriptor().dataType()));
    }

    private static Tensor clamp(
            Tensor input,
            Tensor gradient,
            ClampRangeAttrs attrs,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor min = constants.valueLike(attrs.minValue(), input);
        Tensor max = constants.valueLike(attrs.maxValue(), input);
        ScalarValue half = half(input.descriptor().dataType());
        Tensor lowered = input.maximum(attrs.minValue());
        Tensor upperRouted = Tensor.where(
                lowered.lessThan(max),
                gradient,
                Tensor.where(
                        lowered.equalTo(max),
                        gradient.mul(half),
                        constants.zeroLike(input)));
        return Tensor.where(
                input.greaterThan(min),
                upperRouted,
                Tensor.where(
                        input.equalTo(min),
                        upperRouted.mul(half),
                        constants.zeroLike(input)));
    }

    private static Tensor gelu(
            Tensor input,
            Tensor gradient,
            FirstOrderAutograd.DerivativeConstants constants) {
        DataType type = input.descriptor().dataType();
        Tensor phiCdf = constants.oneLike(input)
                .add(input.mul(invSqrt2(type)).erf())
                .mul(half(type));
        Tensor phiPdfTerm = input.mul(
                        input.mul(input).mul(negativeHalf(type)).exp())
                .mul(invSqrt2Pi(type));
        Tensor regular = gradient.mul(phiCdf.add(phiPdfTerm));
        return activationInfinitySelection(input, gradient, regular, constants);
    }

    private static Tensor geluTanhApproximation(
            Tensor input,
            Tensor gradient,
            FirstOrderAutograd.DerivativeConstants constants) {
        DataType type = input.descriptor().dataType();
        Tensor squared = input.mul(input);
        Tensor u = input.add(squared.mul(input).mul(geluCubic(type)))
                .mul(sqrt2OverPi(type));
        Tensor t = u.tanh();
        Tensor uPrime = constants.oneLike(input)
                .add(squared.mul(geluCubicDerivative(type)))
                .mul(sqrt2OverPi(type));
        Tensor regular = gradient.mul(
                constants.oneLike(input).add(t).mul(half(type))
                        .add(input.mul(
                                        constants.oneLike(input).sub(t.mul(t)))
                                .mul(uPrime)
                                .mul(half(type))));
        return activationInfinitySelection(input, gradient, regular, constants);
    }

    private static Tensor silu(
            Tensor input,
            Tensor gradient,
            FirstOrderAutograd.DerivativeConstants constants) {
        Tensor sigmoid = input.sigmoid();
        Tensor regular = gradient.mul(sigmoid).mul(
                constants.oneLike(input)
                        .add(input.mul(constants.oneLike(input).sub(sigmoid))));
        return activationInfinitySelection(input, gradient, regular, constants);
    }

    private static Tensor activationInfinitySelection(
            Tensor input,
            Tensor gradient,
            Tensor regular,
            FirstOrderAutograd.DerivativeConstants constants) {
        return Tensor.where(
                input.isInf(),
                Tensor.where(
                        input.greaterThan(constants.zeroLike(input)),
                        gradient,
                        constants.zeroLike(input)),
                regular);
    }

    private static ScalarValue exponentMinusOne(ScalarValue exponent) {
        return switch (exponent.dataType()) {
            case BFLOAT16 -> ScalarValue.bfloat16Bits(BFloat16Bits.fromFloat(
                    BFloat16Bits.toFloat(exponent.bfloat16Bits()) - 1.0f));
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
            case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(coefficient.float32Bits));
            case FLOAT64 -> ScalarValue.float64(Double.longBitsToDouble(coefficient.float64Bits));
            case INT32, INT64, BOOL ->
                    throw new IllegalStateException(
                            "elementwise preflight admitted non-floating type: " + dataType);
        };
    }

    private enum Coefficient {
        HALF(0x3F00, 0x3F000000, 0x3FE0000000000000L),
        NEGATIVE_HALF(0xBF00, 0xBF000000, 0xBFE0000000000000L),
        TWO(0x4000, 0x40000000, 0x4000000000000000L),
        INV_SQRT_2(0x3F35, 0x3F3504F3, 0x3FE6A09E667F3BCDL),
        INV_SQRT_2_PI(0x3ECC, 0x3ECC422A, 0x3FD9884533D43651L),
        SQRT_2_OVER_PI(0x3F4C, 0x3F4C422A, 0x3FE9884533D43651L),
        GELU_CUBIC(0x3D37, 0x3D372713, 0x3FA6E4E26D4801F7L),
        GELU_CUBIC_DERIVATIVE(0x3E09, 0x3E095D4F, 0x3FC12BA9D1F60179L);

        private final short bfloat16Bits;
        private final int float32Bits;
        private final long float64Bits;

        Coefficient(int bfloat16Bits, int float32Bits, long float64Bits) {
            this.bfloat16Bits = (short) bfloat16Bits;
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
            case FLOAT32 -> ScalarValue.float32(Float.intBitsToFloat(0x3F906EBB));
            case FLOAT64 -> ScalarValue.float64(
                    Double.longBitsToDouble(0x3FF20DD750429B6DL));
            case INT32, INT64, BOOL ->
                    throw new IllegalStateException(
                            "ERF preflight admitted non-floating type: " + dataType);
        };
    }
}
