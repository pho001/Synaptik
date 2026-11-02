package io.github.pho001.synaptik.compiler;


/**
 * Builds the closed role-aware floating {@code MATMUL} first-order formulas.
 *
 * <p>The four vector/matrix rank pairings use only public Tensor multiplication, rank edits,
 * explicit contiguous materialization of rank edits, permutation, matrix multiplication,
 * sum-to-Shape, and floating cast operations. Batch broadcasting is reversed independently for
 * each selected operand, then an ordinary cast converts a promoted cotangent to the selected
 * operand type when needed.
 * Preflight owns promotion, rank, contraction, batch, output-Shape, attribute, and policy
 * validation.</p>
 *
 * <p>For output cotangent {@code g}, left {@code l}, right {@code r},
 * {@code T(v)} denoting a swap of the final two axes, and {@code C(v)} denoting explicit
 * contiguous materialization, the four cases are: vector/vector {@code [g * r, g * l]};
 * vector/matrix
 * {@code [C(squeeze(C(expandDims(g)) @ T(r))), C(expandDims(l)) @ C(expandDims(g))]};
 * matrix/vector
 * {@code [C(expandDims(g)) @ C(expandDims(r)), C(squeeze(T(l) @ C(expandDims(g))))]};
 * and matrix/matrix
 * {@code [g @ T(r), T(l) @ g]}. Each selected result is reduced with
 * {@code sumToShape(operand.shape())} only when its exact Shape differs, and is then cast once when
 * its promoted type differs from the selected operand. An unselected role remains {@code null}.
 * The matrix/vector right formula inserts and removes the final singleton column axis, so a
 * batched cotangent retains its batch prefix instead of being interpreted as a matrix operand.
 *
 * <p>This owner constructs expression metadata only. It reads no values or storage, captures no
 * graph, selects no numerical implementation, and performs no lowering or execution.</p>
 *
 * <p>Descriptor access preserves original source roles without resolving primal values.
 * Real construction uses Tensor references directly; checked metadata construction retains
 * the same eager operation sites, including temporaries not used by a returned cotangent.</p>
 *
 * @param <V> direct Tensor or checked metadata operand representation
 */
final class LinearAlgebraGradientRules<V> {
    private final OrdinaryConstruction<V> ops;

    /**
     * Binds this formula owner to one compile-local construction route.
     * @param ops non-null direct Tensor or checked metadata construction
     */
    LinearAlgebraGradientRules(OrdinaryConstruction<V> ops) {
        this.ops = java.util.Objects.requireNonNull(ops);
    }

    /**
     * Binds original matrix operands, constructs their positional formulas and normalizes once.
     *
     * <p>Rank-case construction does not append a shape reduction. The common normalization step
     * retains actual batch reductions but omits an otherwise materialized same-Shape copy.</p>
     *
     * @param producer exact original matrix product
     * @param gradient accumulated output cotangent
     * @param selectedInputs two ordered role flags
     * @param primal total lexical binding for original operands
     * @return positional cotangents, null for unselected roles
     */
    V[] apply(
            OrdinaryConstruction.Source<V> producer,
            V gradient,
            boolean[] selectedInputs,
            java.util.function.Function<V, V> primal) {
        V left = primal.apply(producer.inputs().get(0));
        V right = primal.apply(producer.inputs().get(1));
        int leftRank = ops.descriptor(left).shape().rank();
        int rightRank = ops.descriptor(right).shape().rank();
        V leftGradient = null;
        V rightGradient = null;
        
        if (leftRank == 1 && rightRank == 1) {
            if (selectedInputs[0]) {
                leftGradient = ops.mul(gradient, right);
            }
            if (selectedInputs[1]) {
                rightGradient = ops.mul(gradient, left);
            }
        } else if (leftRank == 1) {
            if (selectedInputs[0]) {
                int insertionAxis = ops.descriptor(gradient).shape().rank() - 1;
                leftGradient = ops.contiguous(ops.squeeze(ops.matmul(ops.contiguous(ops.expandDims(gradient, insertionAxis)), swapLastTwo(right)), insertionAxis));
            }
            if (selectedInputs[1]) {
                int insertionAxis = ops.descriptor(gradient).shape().rank() - 1;
                rightGradient = ops.matmul(ops.contiguous(ops.expandDims(left, 1)), ops.contiguous(ops.expandDims(gradient, insertionAxis)));
            }
        } else if (rightRank == 1) {
            if (selectedInputs[0]) {
                leftGradient = ops.matmul(ops.contiguous(ops.expandDims(gradient, ops.descriptor(gradient).shape().rank())), ops.contiguous(ops.expandDims(right, 0)));
            }
            if (selectedInputs[1]) {
                int columnAxis = ops.descriptor(gradient).shape().rank();
                rightGradient = ops.contiguous(ops.squeeze(ops.matmul(swapLastTwo(left), ops.contiguous(ops.expandDims(gradient, columnAxis))), columnAxis));
            }
        } else {
            if (selectedInputs[0]) {
                leftGradient = ops.matmul(gradient, swapLastTwo(right));
            }
            if (selectedInputs[1]) {
                rightGradient = ops.matmul(swapLastTwo(left), gradient);
            }
        }
        return ops.result(normalize(leftGradient, left),
            normalize(rightGradient, right));
    }

    /**
     * Converts one optional promoted MATMUL cotangent to its selected operand contract.
     *
     * <p>Exact Shape equality, including equal symbolic Dimensions, bypasses reduction rather
     * than constructing a copying {@code sumToShape}. Different Shapes retain the ordinary
     * broadcast-reversing reduction before any required dtype cast.</p>
     *
     * @param gradient selected promoted cotangent, or {@code null} for an unselected operand
     * @param input non-null original floating operand whose exact Shape and data type are required
     * @return {@code null} for an unselected operand; otherwise a non-null ordinary Tensor
     *     expression with the operand's exact Shape and data type
     */
    private V normalize(V gradient, V input) {
        if (gradient == null) {
            return null;
        }
        V normalized = ops.descriptor(gradient).shape().equals(ops.descriptor(input).shape())
                ? gradient
                : ops.sumToShape(gradient, ops.descriptor(input).shape());
        return ops.descriptor(normalized).dataType() == ops.descriptor(input).dataType()
                ? normalized
                : ops.cast(normalized, ops.descriptor(input).dataType());
    }

    /**
     * Builds a public PERMUTE expression that exchanges only the final two axes.
     *
     * @param tensor non-null preflight-approved Tensor with rank at least two
     * @return a new Tensor expression with the final two axes exchanged
     */
    private V swapLastTwo(V tensor) {
        int rank = ops.descriptor(tensor).shape().rank();
        int[] axes = new int[rank];
        for (int axis = 0; axis < rank; axis++) {
            axes[axis] = axis;
        }
        axes[rank - 2] = rank - 1;
        axes[rank - 1] = rank - 2;
        return ops.permute(tensor, axes);
    }
}
