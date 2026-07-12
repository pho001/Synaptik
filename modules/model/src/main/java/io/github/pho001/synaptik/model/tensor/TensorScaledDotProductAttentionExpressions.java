package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs locally validated storage-free scaled dot-product attention expressions.
 *
 * <p>This package-private, field-free boundary owns attention-specific type, rank, contraction,
 * three-way batch broadcast, mask broadcast, scale, descriptor, and provenance construction. It
 * retains exact input Dimension references whenever an exact output is locally derivable and
 * leaves unresolved equality, singleton-or-equal, and positivity obligations for later compiler
 * validation or binding. It does not inspect values, create constraints or gradients, decompose
 * the operation, choose an algorithm or backend, allocate storage, or execute.</p>
 */
final class TensorScaledDotProductAttentionExpressions {
    /** Prevents instantiation because attention expression construction is stateless. */
    private TensorScaledDotProductAttentionExpressions() {
    }

    /**
     * Creates an unmasked attention expression with ordered inputs {@code [query, key, value]}.
     *
     * @param query non-null rank-two-or-higher floating query retained as input zero
     * @param key non-null rank-two-or-higher floating key retained as input one
     * @param value non-null rank-two-or-higher floating value retained as input two
     * @param attrs non-null exact semantic attributes retained by reference
     * @return non-null fresh output with promoted type, exact derived Shape, unresolved layout,
     *     query/key/value gradient-request OR, and output-index-zero provenance
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if local type, rank, Shape, or scale validation fails
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    static Tensor apply(
            Tensor query,
            Tensor key,
            Tensor value,
            ScaledDotProductAttentionAttrs attrs) {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(attrs, "attrs");
        return build(query, key, value, List.of(query, key, value), attrs);
    }

    /**
     * Creates a masked attention expression with ordered inputs
     * {@code [query, key, value, mask]}.
     *
     * @param query non-null rank-two-or-higher floating query retained as input zero
     * @param key non-null rank-two-or-higher floating key retained as input one
     * @param value non-null rank-two-or-higher floating value retained as input two
     * @param mask non-null BOOL mask that must broadcast exactly to the derived score Shape
     * @param attrs non-null exact semantic attributes retained by reference
     * @return non-null fresh output with promoted type, exact derived Shape, unresolved layout,
     *     query/key/value gradient-request OR, and exact four-input provenance
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if local type, rank, Shape, mask, or scale validation fails
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    static Tensor apply(
            Tensor query,
            Tensor key,
            Tensor value,
            Tensor mask,
            ScaledDotProductAttentionAttrs attrs) {
        Objects.requireNonNull(query, "query");
        Objects.requireNonNull(key, "key");
        Objects.requireNonNull(value, "value");
        Objects.requireNonNull(mask, "mask");
        Objects.requireNonNull(attrs, "attrs");
        return build(query, key, value, List.of(query, key, value, mask), attrs);
    }

    private static Tensor build(
            Tensor query,
            Tensor key,
            Tensor value,
            List<Tensor> inputs,
            ScaledDotProductAttentionAttrs attrs) {
        DataType queryType = requireFloating(query.descriptor().dataType(), "query");
        DataType keyType = requireFloating(key.descriptor().dataType(), "key");
        DataType valueType = requireFloating(value.descriptor().dataType(), "value");
        DataType queryKeyType = DataTypePromotion.promoteFloating(queryType, keyType);
        DataType resultType = DataTypePromotion.promoteFloating(queryKeyType, valueType);

        Shape queryShape = query.descriptor().shape();
        Shape keyShape = key.descriptor().shape();
        Shape valueShape = value.descriptor().shape();
        int queryRank = queryShape.rank();
        int keyRank = keyShape.rank();
        int valueRank = valueShape.rank();
        requireRank(queryRank, "query");
        requireRank(keyRank, "key");
        requireRank(valueRank, "value");

        Dimension queryEmbedding = queryShape.dimension(queryRank - 1);
        Dimension keyEmbedding = keyShape.dimension(keyRank - 1);
        Dimension keySequence = keyShape.dimension(keyRank - 2);
        Dimension valueSequence = valueShape.dimension(valueRank - 2);
        if (queryEmbedding instanceof StaticDimension queryStatic && queryStatic.size() == 0) {
            throw new IllegalArgumentException(
                    "attention embedding dimension must be positive: " + queryEmbedding);
        }
        rejectUnequalStatic(
                queryEmbedding,
                keyEmbedding,
                "attention query/key embedding dimensions must match: query=",
                ", key=");
        rejectUnequalStatic(
                keySequence,
                valueSequence,
                "attention key/value sequence dimensions must match: key=",
                ", value=");

        int queryBatchRank = queryRank - 2;
        int keyBatchRank = keyRank - 2;
        int valueBatchRank = valueRank - 2;
        int resultBatchRank = Math.max(queryBatchRank, Math.max(keyBatchRank, valueBatchRank));
        List<Dimension> batchDimensions = new ArrayList<>(resultBatchRank);
        for (int resultAxis = 0; resultAxis < resultBatchRank; resultAxis++) {
            Dimension queryBatch = alignedBatchDimension(
                    queryShape, queryBatchRank, resultBatchRank, resultAxis);
            Dimension keyBatch = alignedBatchDimension(
                    keyShape, keyBatchRank, resultBatchRank, resultAxis);
            Dimension valueBatch = alignedBatchDimension(
                    valueShape, valueBatchRank, resultBatchRank, resultAxis);
            batchDimensions.add(broadcastBatchDimension(
                    queryBatch, keyBatch, valueBatch, resultAxis));
        }

        Dimension querySequence = queryShape.dimension(queryRank - 2);
        Dimension valueEmbedding = valueShape.dimension(valueRank - 1);
        List<Dimension> scoreDimensions = new ArrayList<>(batchDimensions);
        scoreDimensions.add(querySequence);
        scoreDimensions.add(keySequence);
        Shape scoreShape = Shape.ofDimensions(scoreDimensions.toArray(new Dimension[0]));
        List<Dimension> outputDimensions = new ArrayList<>(batchDimensions);
        outputDimensions.add(querySequence);
        outputDimensions.add(valueEmbedding);
        Shape outputShape = Shape.ofDimensions(outputDimensions.toArray(new Dimension[0]));

        if (inputs.size() == 4) {
            validateMask(inputs.get(3), scoreShape);
        }
        Optional<ScalarValue> scale = attrs.scale();
        if (scale.isPresent() && scale.orElseThrow().dataType() != resultType) {
            throw new IllegalArgumentException(
                    "scale data type must match promoted attention data type: scale="
                            + scale.orElseThrow().dataType() + ", promoted=" + resultType);
        }

        TensorDescriptor descriptor = new TensorDescriptor(
                resultType,
                outputShape,
                Optional.empty(),
                query.descriptor().requiresGrad()
                        || key.descriptor().requiresGrad()
                        || value.descriptor().requiresGrad());
        Operation operation = new Operation(
                ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION, attrs);
        return TensorFactory.createDerived(
                descriptor, Optional.empty(), operation, inputs);
    }

    private static DataType requireFloating(DataType dataType, String role) {
        if (!dataType.isFloating()) {
            throw new IllegalArgumentException(
                    role + " must have a floating data type, but was " + dataType);
        }
        return dataType;
    }

    private static void requireRank(int rank, String role) {
        if (rank < 2) {
            throw new IllegalArgumentException(role + " rank must be at least 2: " + rank);
        }
    }

    private static void rejectUnequalStatic(
            Dimension left,
            Dimension right,
            String prefix,
            String separator) {
        if (left instanceof StaticDimension leftStatic
                && right instanceof StaticDimension rightStatic
                && leftStatic.size() != rightStatic.size()) {
            throw new IllegalArgumentException(prefix + left + separator + right);
        }
    }

    private static Dimension alignedBatchDimension(
            Shape shape, int batchRank, int resultBatchRank, int resultAxis) {
        int offset = resultBatchRank - batchRank;
        return resultAxis < offset ? null : shape.dimension(resultAxis - offset);
    }

    private static Dimension broadcastBatchDimension(
            Dimension query, Dimension key, Dimension value, int resultAxis) {
        Dimension[] ordered = {query, key, value};
        Long selectedStaticSize = null;
        Dimension selectedStatic = null;
        for (Dimension dimension : ordered) {
            if (dimension instanceof StaticDimension staticDimension
                    && staticDimension.size() != 1) {
                if (selectedStaticSize != null
                        && selectedStaticSize.longValue() != staticDimension.size()) {
                    throw batchFailure(
                            "cannot broadcast attention batch dimensions at result batch axis ",
                            query,
                            key,
                            value,
                            resultAxis);
                }
                if (selectedStatic == null) {
                    selectedStatic = dimension;
                    selectedStaticSize = staticDimension.size();
                }
            }
        }
        if (selectedStatic != null) {
            return selectedStatic;
        }

        Dimension selectedUnresolved = null;
        for (Dimension dimension : ordered) {
            if (dimension != null && !(dimension instanceof StaticDimension)) {
                if (selectedUnresolved != null && !selectedUnresolved.equals(dimension)) {
                    throw batchFailure(
                            "cannot derive exact attention batch dimension at result batch axis ",
                            query,
                            key,
                            value,
                            resultAxis);
                }
                if (selectedUnresolved == null) {
                    selectedUnresolved = dimension;
                }
            }
        }
        if (selectedUnresolved != null) {
            return selectedUnresolved;
        }
        for (Dimension dimension : ordered) {
            if (dimension != null) {
                return dimension;
            }
        }
        throw new AssertionError("attention batch axis has no input dimension");
    }

    private static IllegalArgumentException batchFailure(
            String prefix,
            Dimension query,
            Dimension key,
            Dimension value,
            int resultAxis) {
        return new IllegalArgumentException(
                prefix + resultAxis + ": query=" + diagnosticDimension(query)
                        + ", key=" + diagnosticDimension(key)
                        + ", value=" + diagnosticDimension(value));
    }

    private static String diagnosticDimension(Dimension dimension) {
        return dimension == null ? "StaticDimension[size=1]" : dimension.toString();
    }

    private static void validateMask(Tensor mask, Shape scoreShape) {
        DataType maskType = mask.descriptor().dataType();
        if (maskType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    "mask must have BOOL data type, but was " + maskType);
        }
        Shape maskShape = mask.descriptor().shape();
        int maskRank = maskShape.rank();
        int scoreRank = scoreShape.rank();
        if (maskRank > scoreRank) {
            throw new IllegalArgumentException(
                    "mask rank must not exceed attention score rank: mask=" + maskRank
                            + ", score=" + scoreRank);
        }
        int maskOffset = scoreRank - maskRank;
        for (int scoreAxis = 0; scoreAxis < scoreRank; scoreAxis++) {
            if (scoreAxis < maskOffset) {
                continue;
            }
            Dimension maskDimension = maskShape.dimension(scoreAxis - maskOffset);
            Dimension scoreDimension = scoreShape.dimension(scoreAxis);
            if (maskDimension.equals(scoreDimension)
                    || maskDimension instanceof StaticDimension maskStatic
                            && maskStatic.size() == 1
                    || !(maskDimension instanceof StaticDimension)
                            && scoreDimension instanceof StaticDimension scoreStatic
                            && scoreStatic.size() != 1) {
                continue;
            }
            throw new IllegalArgumentException(
                    "mask cannot broadcast exactly to attention score shape at axis " + scoreAxis
                            + ": mask=" + maskDimension + ", score=" + scoreDimension);
        }
    }
}
