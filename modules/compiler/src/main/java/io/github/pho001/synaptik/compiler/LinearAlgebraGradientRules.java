package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorProducer;

/**
 * Builds the closed role-aware floating {@code MATMUL} first-order formulas.
 *
 * <p>The four vector/matrix rank pairings use only public Tensor rank edits, permutation,
 * multiplication, matrix multiplication, sum-to-Shape, and floating cast operations. Batch
 * broadcasting is reversed independently for each selected operand, then an ordinary cast
 * converts a promoted cotangent to the selected operand type when needed. Preflight owns
 * promotion, rank, contraction, batch, output-Shape, attribute, and policy validation.</p>
 *
 * <p>For output cotangent {@code g}, left {@code l}, right {@code r}, and
 * {@code T(v)} denoting a swap of the final two axes, the four cases are: vector/vector
 * {@code [g * r, g * l]}; vector/matrix
 * {@code [squeeze(expandDims(g) @ T(r)), expandDims(l) @ expandDims(g)]};
 * matrix/vector {@code [expandDims(g) @ expandDims(r), T(l) @ g]}; and matrix/matrix
 * {@code [g @ T(r), T(l) @ g]}. Each selected result is reduced with
 * {@code sumToShape(operand.shape())} when batch broadcasting occurred and is then cast once when
 * its promoted type differs from the selected operand. An unselected role remains {@code null}.
 *
 * <p>This owner constructs expression metadata only. It reads no values or storage, captures no
 * graph, selects no numerical implementation, and performs no lowering or execution.</p>
 */
final class LinearAlgebraGradientRules {
    /**
     * Prevents construction of this stateless formula owner.
     */
    private LinearAlgebraGradientRules() {}

    /**
     * Builds selected ordered operand cotangents for one preflight-approved {@code MATMUL}.
     *
     * @param producer exact original two-input {@code MATMUL} producer occurrence
     * @param gradient non-null accumulated cotangent for its sole output
     * @param selectedInputs non-null two-position selected-route flags; observed but not mutated
     * @return a new two-position array containing each selected operand cotangent and {@code null}
     *     for each unselected role
     */
    static Tensor[] apply(
            TensorProducer producer, Tensor gradient, boolean[] selectedInputs) {
        Tensor left = producer.inputs().get(0);
        Tensor right = producer.inputs().get(1);
        int leftRank = left.descriptor().shape().rank();
        int rightRank = right.descriptor().shape().rank();
        Tensor leftGradient = null;
        Tensor rightGradient = null;

        if (leftRank == 1 && rightRank == 1) {
            if (selectedInputs[0]) {
                leftGradient = gradient.mul(right);
            }
            if (selectedInputs[1]) {
                rightGradient = gradient.mul(left);
            }
        } else if (leftRank == 1) {
            if (selectedInputs[0]) {
                int insertionAxis = gradient.descriptor().shape().rank() - 1;
                leftGradient = gradient.expandDims(insertionAxis)
                        .matmul(swapLastTwo(right))
                        .squeeze(insertionAxis)
                        .sumToShape(left.descriptor().shape());
            }
            if (selectedInputs[1]) {
                int insertionAxis = gradient.descriptor().shape().rank() - 1;
                rightGradient = left.expandDims(1)
                        .matmul(gradient.expandDims(insertionAxis))
                        .sumToShape(right.descriptor().shape());
            }
        } else if (rightRank == 1) {
            if (selectedInputs[0]) {
                leftGradient = gradient.expandDims(gradient.descriptor().shape().rank())
                        .matmul(right.expandDims(0))
                        .sumToShape(left.descriptor().shape());
            }
            if (selectedInputs[1]) {
                rightGradient = swapLastTwo(left)
                        .matmul(gradient)
                        .sumToShape(right.descriptor().shape());
            }
        } else {
            if (selectedInputs[0]) {
                leftGradient = gradient.matmul(swapLastTwo(right))
                        .sumToShape(left.descriptor().shape());
            }
            if (selectedInputs[1]) {
                rightGradient = swapLastTwo(left)
                        .matmul(gradient)
                        .sumToShape(right.descriptor().shape());
            }
        }
        return new Tensor[] {
            normalize(leftGradient, left),
            normalize(rightGradient, right)
        };
    }

    /**
     * Converts one optional promoted MATMUL cotangent to its selected operand contract.
     *
     * @param gradient selected promoted cotangent, or {@code null} for an unselected operand
     * @param input non-null original floating operand whose exact Shape and data type are required
     * @return {@code null} for an unselected operand; otherwise a non-null ordinary Tensor
     *     expression with the operand's exact Shape and data type
     */
    private static Tensor normalize(Tensor gradient, Tensor input) {
        if (gradient == null) {
            return null;
        }
        Tensor normalized = gradient.descriptor().shape().equals(input.descriptor().shape())
                ? gradient
                : gradient.sumToShape(input.descriptor().shape());
        return normalized.descriptor().dataType() == input.descriptor().dataType()
                ? normalized
                : normalized.cast(input.descriptor().dataType());
    }

    /**
     * Builds a public PERMUTE expression that exchanges only the final two axes.
     *
     * @param tensor non-null preflight-approved Tensor with rank at least two
     * @return a new Tensor expression with the final two axes exchanged
     */
    private static Tensor swapLastTwo(Tensor tensor) {
        int rank = tensor.descriptor().shape().rank();
        int[] axes = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            axes[axis] = axis;
        }
        axes[rank - 2] = rank - 1;
        axes[rank - 1] = rank - 2;
        return tensor.permute(axes);
    }
}
