package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated storage-free matrix-multiplication tensor expressions.
 *
 * <p>This package-private, field-free boundary owns MATMUL-specific rank, contraction, and batch
 * algebra without changing general shape broadcasting. It preserves exact retained dimension
 * references, records obligations that cannot be proved until compiler validation or binding,
 * and rejects unequal unresolved batch dimensions when no exact result extent can be selected.
 * It creates metadata only and does not inspect values, allocate storage, solve graph constraints,
 * define gradients, capture a graph, lower an operation, select a kernel, or execute.</p>
 */
final class TensorMatmulExpressions {
    /** Prevents instantiation because MATMUL expression construction is stateless. */
    private TensorMatmulExpressions() {
    }

    /**
     * Creates one fresh derived tensor for ordered {@code left @ right} semantics.
     *
     * <p>Validation occurs in this exact order: null-check {@code left}, then {@code right};
     * promote their descriptor types; require left rank and then right rank to be at least one;
     * reject a proven static contraction mismatch; process aligned batch dimensions from leading
     * result axis to trailing; build the result Shape, unresolved descriptor, and parameterless
     * operation; and delegate exactly once to the central factory with ordered exact inputs
     * {@code [left, right]}. Every failure before that final delegation consumes no Tensor ID.</p>
     *
     * <p>Rank-one operands omit their temporary matrix result axis. Leading batch axes broadcast
     * right-aligned. Equal, singleton-expanded, or unpaired dimensions preserve exact input
     * references. An unresolved dimension paired with a static non-singleton yields that exact
     * static extent and defers a singleton-or-equal obligation. Unequal unresolved batch extents
     * are rejected. Unresolved contraction equality is deferred because the exact input
     * descriptors remain in provenance.</p>
     *
     * @param left non-null ordered left operand retained by exact reference in provenance
     * @param right non-null ordered right operand retained by exact reference in provenance
     * @return the non-null fresh, unlabeled, storage-free factory result with promoted type, exact
     *     derived Shape, unresolved layout, gradient-request OR, and output index zero
     * @throws NullPointerException if {@code left} or {@code right} is null, checked in that order
     *     with the parameter name as the message
     * @throws IllegalArgumentException if type promotion fails; an operand has rank zero; static
     *     contraction extents differ; batch extents are unequal static non-singletons; or two
     *     unequal unresolved batch extents cannot produce an exact local result dimension
     * @throws IllegalStateException if Tensor identifier space is exhausted during final factory
     *     delegation
     */
    static Tensor apply(Tensor left, Tensor right) {
        Objects.requireNonNull(left, "left");
        Objects.requireNonNull(right, "right");

        DataType resultType = DataTypePromotion.promoteNumeric(
                left.descriptor().dataType(), right.descriptor().dataType());
        Shape leftShape = left.descriptor().shape();
        Shape rightShape = right.descriptor().shape();
        int leftRank = leftShape.rank();
        int rightRank = rightShape.rank();
        if (leftRank < 1) {
            throw new IllegalArgumentException("left rank must be at least 1: " + leftRank);
        }
        if (rightRank < 1) {
            throw new IllegalArgumentException("right rank must be at least 1: " + rightRank);
        }

        Dimension leftInner = leftShape.dimension(leftRank - 1);
        Dimension rightInner = rightShape.dimension(rightRank == 1 ? 0 : rightRank - 2);
        if (leftInner instanceof StaticDimension leftStatic
                && rightInner instanceof StaticDimension rightStatic
                && leftStatic.size() != rightStatic.size()) {
            throw new IllegalArgumentException(
                    "matmul inner dimensions must match: left=" + leftInner
                            + ", right=" + rightInner);
        }

        int leftBatchRank = Math.max(0, leftRank - 2);
        int rightBatchRank = Math.max(0, rightRank - 2);
        int resultBatchRank = Math.max(leftBatchRank, rightBatchRank);
        List<Dimension> resultDimensions = new ArrayList<>(
                resultBatchRank + (leftRank == 1 ? 0 : 1) + (rightRank == 1 ? 0 : 1));
        int leftBatchOffset = resultBatchRank - leftBatchRank;
        int rightBatchOffset = resultBatchRank - rightBatchRank;
        for (int resultAxis = 0; resultAxis < resultBatchRank; resultAxis++) {
            boolean hasLeft = resultAxis >= leftBatchOffset;
            boolean hasRight = resultAxis >= rightBatchOffset;
            if (!hasLeft) {
                resultDimensions.add(rightShape.dimension(resultAxis - rightBatchOffset));
            } else if (!hasRight) {
                resultDimensions.add(leftShape.dimension(resultAxis - leftBatchOffset));
            } else {
                Dimension leftBatch = leftShape.dimension(resultAxis - leftBatchOffset);
                Dimension rightBatch = rightShape.dimension(resultAxis - rightBatchOffset);
                resultDimensions.add(broadcastBatchDimension(
                        leftBatch, rightBatch, resultAxis));
            }
        }
        if (leftRank != 1) {
            resultDimensions.add(leftShape.dimension(leftRank - 2));
        }
        if (rightRank != 1) {
            resultDimensions.add(rightShape.dimension(rightRank - 1));
        }

        Shape resultShape = Shape.ofDimensions(resultDimensions.toArray(new Dimension[0]));
        TensorDescriptor descriptor = new TensorDescriptor(
                resultType,
                resultShape,
                Optional.empty(),
                left.descriptor().requiresGrad() || right.descriptor().requiresGrad());
        Operation operation = new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, List.of(left, right));
    }

    /**
     * Selects one exact result Dimension for an aligned MATMUL batch-axis pair.
     *
     * <p>Equality retains {@code left}; a static singleton retains the opposing reference; and an
     * unresolved/static-non-singleton pair retains the exact static reference under a deferred
     * singleton-or-equal obligation. Two unequal unresolved dimensions fail because this local
     * model cannot select one exact result reference.</p>
     *
     * @param left non-null aligned left batch Dimension
     * @param right non-null aligned right batch Dimension
     * @param resultAxis zero-based leading-to-trailing result batch axis used in diagnostics
     * @return the non-null exact input Dimension reference selected for the result batch axis
     * @throws IllegalArgumentException if both dimensions are unequal static non-singletons or
     *     both are unequal unresolved dimensions
     */
    private static Dimension broadcastBatchDimension(
            Dimension left, Dimension right, int resultAxis) {
        if (left.equals(right)) {
            return left;
        }
        if (left instanceof StaticDimension leftStatic && leftStatic.size() == 1) {
            return right;
        }
        if (right instanceof StaticDimension rightStatic && rightStatic.size() == 1) {
            return left;
        }
        if (left instanceof StaticDimension leftStatic
                && right instanceof StaticDimension rightStatic) {
            throw new IllegalArgumentException(
                    "cannot broadcast matmul batch dimensions at result batch axis " + resultAxis
                            + ": left=" + leftStatic + ", right=" + rightStatic);
        }
        if (left instanceof StaticDimension) {
            return left;
        }
        if (right instanceof StaticDimension) {
            return right;
        }
        throw new IllegalArgumentException(
                "cannot derive exact matmul batch dimension at result batch axis " + resultAxis
                        + ": left=" + left + ", right=" + right);
    }
}
