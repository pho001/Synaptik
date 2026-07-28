package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutKind;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;

/**
 * Builds explicit-state dropout's input cotangent from its canonical same-occurrence keep mask.
 *
 * <p>The rule never samples, reconstructs a mask, infers keep decisions from values, or advances
 * graph RNG state. Kept positions route the output cotangent through the exact typed
 * inverted-dropout denominator; dropped positions select the request-local exact positive zero
 * through {@code WHERE}, including when unselected arithmetic values would be NaN or infinity.
 * At either signed probability zero the typed denominator is positive one and the canonical
 * all-kept mask routes the cotangent unchanged. State and mask roles remain
 * non-differentiable. The formula describes compile-time Tensor construction only; it does not
 * prescribe sampling, state execution, saved-buffer lifetime, lowering, or backend behavior.</p>
 */
final class StochasticGradientRules {
    private StochasticGradientRules() {}

    /**
     * Builds the sole floating input cotangent for one approved dropout values output.
     *
     * @param producer exact original three-output dropout producer
     * @param gradient non-null accumulated values-output cotangent
     * @param constants non-null request-local exact typed positive-zero owner
     * @return a two-element input-aligned array containing the data cotangent and a {@code null}
     *     graph-state role
     * @throws IllegalStateException if the producer is not the preflight-approved dropout kind
     */
    static Tensor[] apply(
            TensorProducer producer,
            Tensor gradient,
            FirstOrderAutograd.DerivativeConstants constants) {
        if (producer.operation().kind() != DropoutKind.DROPOUT) {
            throw new IllegalStateException(
                    "stochastic operation was not preflight-approved: " + producer.operation());
        }
        Tensor input = producer.inputs().getFirst();
        Tensor mask = producer.output(1);
        double probability = ((DropoutAttrs) producer.operation().attrs()).probability();
        Tensor scaled = gradient.div(complement(
                input.descriptor().dataType(), 1.0d - probability));
        return new Tensor[] {
            Tensor.where(mask, scaled, constants.zeroLike(input)),
            null
        };
    }

    private static ScalarValue complement(DataType dataType, double value) {
        return switch (dataType) {
            case FLOAT64 -> ScalarValue.float64(value);
            case FLOAT32 -> ScalarValue.float32((float) value);
            case BFLOAT16 -> ScalarValue.bfloat16((float) value);
            case INT32, INT64, BOOL -> throw new IllegalArgumentException(
                    "dropout derivative requires floating data type: " + dataType);
        };
    }
}
