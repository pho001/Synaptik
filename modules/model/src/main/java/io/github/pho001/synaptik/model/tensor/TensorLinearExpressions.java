package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.Objects;

/**
 * Constructs linear projections by composing the existing transpose, MATMUL, and ADD helpers.
 *
 * <p>This package-private boundary validates every caller-controlled local failure before it
 * creates an intermediate tensor. Successful construction preserves the primitive producer
 * chain. It calls the existing package-private transpose, MATMUL, and ADD helpers directly only
 * after prevalidation and introduces no shared primitive or LINEAR operation. The helper is
 * field-free and retains no input or result state.</p>
 *
 * <p>Caller-controlled null, promotion, rank, static contraction, and optional bias failures
 * consume no Tensor identifier. Successful no-bias construction allocates PERMUTE then MATMUL;
 * successful biased construction additionally allocates ADD. Identifier exhaustion after an
 * earlier intermediate is not rolled back. This boundary does not execute values or define
 * compiler capture, gradient, layer, parameter, backend, storage, or runtime behavior.</p>
 */
final class TensorLinearExpressions {
    /** Prevents instantiation because linear expression construction is stateless. */
    private TensorLinearExpressions() {
    }

    /**
     * Creates {@code input.matmul(weight.transpose())} after complete local validation.
     *
     * @param input non-null rank-one-or-higher input whose final axis contains input features
     * @param weight non-null rank-two weight in {@code [outFeatures, inFeatures]} orientation
     * @return the non-null fresh storage-free MATMUL result produced after one PERMUTE
     *     intermediate; its Shape preserves leading input Dimensions and exact weight
     *     out-features, and its provenance exposes both producers at output index zero
     * @throws NullPointerException if {@code input} or {@code weight} is null, checked in that
     *     order with the parameter name as message
     * @throws IllegalArgumentException if numeric promotion fails, input rank is below one, weight
     *     rank is not two, or static contraction Dimensions differ; all occur before allocation
     * @throws IllegalStateException if Tensor identifier space is exhausted; a successfully
     *     allocated PERMUTE ID is not rolled back when MATMUL allocation then fails
     */
    static Tensor apply(Tensor input, Tensor weight) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");

        DataType productType = DataTypePromotion.promoteNumeric(
                input.descriptor().dataType(), weight.descriptor().dataType());
        Shape inputShape = input.descriptor().shape();
        Shape weightShape = weight.descriptor().shape();
        int inputRank = inputShape.rank();
        int weightRank = weightShape.rank();
        if (inputRank < 1) {
            throw new IllegalArgumentException(
                    "input rank must be at least 1: " + inputRank);
        }
        if (weightRank != 2) {
            throw new IllegalArgumentException(
                    "weight rank must be exactly 2: " + weightRank);
        }
        Dimension inputFeatures = inputShape.dimension(inputRank - 1);
        Dimension weightInputFeatures = weightShape.dimension(1);
        if (inputFeatures instanceof StaticDimension inputStatic
                && weightInputFeatures instanceof StaticDimension weightStatic
                && inputStatic.size() != weightStatic.size()) {
            throw new IllegalArgumentException(
                    "linear input feature dimension must match weight in-features dimension: input="
                            + inputFeatures + ", weight=" + weightInputFeatures);
        }

        Tensor transposedWeight = TensorPermutationExpressions.transpose(weight);
        Tensor product = TensorMatmulExpressions.apply(input, transposedWeight);
        return product;
    }

    /**
     * Creates {@code input.matmul(weight.transpose()).add(bias)} after complete local validation.
     *
     * @param input non-null rank-one-or-higher input whose final axis contains input features
     * @param weight non-null rank-two weight in {@code [outFeatures, inFeatures]} orientation
     * @param bias non-null rank-one bias whose sole Dimension structurally equals weight
     *     out-features
     * @return the non-null fresh storage-free ADD result after PERMUTE and MATMUL intermediates;
     *     its Shape is structurally equal to the product Shape and reuses its exact ordered
     *     Dimension references, although the outer Shape object may differ
     * @throws NullPointerException if {@code input}, {@code weight}, or {@code bias} is null,
     *     checked in that order with the parameter name as message
     * @throws IllegalArgumentException if input/weight promotion, input rank, weight rank, static
     *     contraction, product/bias promotion, bias rank, or exact bias Dimension validation fails,
     *     in that order and before allocation
     * @throws IllegalStateException if Tensor identifier space is exhausted; previously allocated
     *     intermediate IDs are not rolled back
     */
    static Tensor apply(Tensor input, Tensor weight, Tensor bias) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(weight, "weight");
        Objects.requireNonNull(bias, "bias");

        DataType productType = DataTypePromotion.promoteNumeric(
                input.descriptor().dataType(), weight.descriptor().dataType());
        Shape inputShape = input.descriptor().shape();
        Shape weightShape = weight.descriptor().shape();
        int inputRank = inputShape.rank();
        int weightRank = weightShape.rank();
        if (inputRank < 1) {
            throw new IllegalArgumentException(
                    "input rank must be at least 1: " + inputRank);
        }
        if (weightRank != 2) {
            throw new IllegalArgumentException(
                    "weight rank must be exactly 2: " + weightRank);
        }
        Dimension inputFeatures = inputShape.dimension(inputRank - 1);
        Dimension weightInputFeatures = weightShape.dimension(1);
        if (inputFeatures instanceof StaticDimension inputStatic
                && weightInputFeatures instanceof StaticDimension weightStatic
                && inputStatic.size() != weightStatic.size()) {
            throw new IllegalArgumentException(
                    "linear input feature dimension must match weight in-features dimension: input="
                            + inputFeatures + ", weight=" + weightInputFeatures);
        }

        DataType finalType = DataTypePromotion.promoteNumeric(
                productType, bias.descriptor().dataType());
        Shape biasShape = bias.descriptor().shape();
        int biasRank = biasShape.rank();
        if (biasRank != 1) {
            throw new IllegalArgumentException(
                    "bias rank must be exactly 1: " + biasRank);
        }
        Dimension biasOutputFeatures = biasShape.dimension(0);
        Dimension weightOutputFeatures = weightShape.dimension(0);
        if (!biasOutputFeatures.equals(weightOutputFeatures)) {
            throw new IllegalArgumentException(
                    "linear bias dimension must match weight out-features dimension: bias="
                            + biasOutputFeatures + ", weight=" + weightOutputFeatures);
        }

        Tensor transposedWeight = TensorPermutationExpressions.transpose(weight);
        Tensor product = TensorMatmulExpressions.apply(input, transposedWeight);
        return TensorBinaryExpressions.apply(product, bias, BinaryArithmeticKind.ADD);
    }
}
