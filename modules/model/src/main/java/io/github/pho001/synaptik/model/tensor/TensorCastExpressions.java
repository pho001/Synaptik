package io.github.pho001.synaptik.model.tensor;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.CastValueConversions;
import io.github.pho001.synaptik.model.shape.Shape;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Constructs the locally validated storage-free cast tensor expression.
 *
 * <p>This package-private boundary owns the deterministic construction order for the sole
 * {@link CastKind#CAST} expression. It accepts every current source and target data-type pair,
 * retains the input descriptor's exact immutable {@link Shape} reference, leaves result layout
 * unresolved, derives floating-only gradient eligibility, records exact typed cast attributes and
 * one-input provenance, and delegates identity allocation exactly once to the central Tensor
 * factory.</p>
 *
 * <p>Construction is eager only for expression metadata. Values follow the exact scalar contract
 * in {@link CastValueConversions}, but this helper does not inspect or convert values or storage,
 * preserve physical layout, define gradient rules, query backend compatibility, capture a graph,
 * or eliminate same-type and chained casts. A same-type request is a fresh explicit expression
 * because compiler optimization owns redundant-cast removal.</p>
 */
final class TensorCastExpressions {
    /** Prevents instantiation because cast expression construction is stateless. */
    private TensorCastExpressions() {
    }

    /**
     * Creates one fresh derived tensor for an explicit data-type conversion request.
     *
     * <p>Validation and construction occur in this exact order: null-check {@code input}, then
     * {@code targetDataType}; read the source data type and exact input shape; derive gradient
     * eligibility from the input request and both types' floating categories; create one
     * unresolved result descriptor; create one {@link CastAttrs} value from the exact target;
     * create one {@link Operation} from {@link CastKind#CAST} and those exact attributes; and
     * delegate exact ordered producer input {@code [input]} once to
     * {@link TensorFactory#createDerived(TensorDescriptor, Optional, Operation, List)} with no
     * label. The factory creates the producer and index-zero provenance.</p>
     *
     * <p>Failures before the final factory delegation allocate no Tensor identity. A valid call,
     * including a same-type request, returns the factory's exact fresh, unlabeled, storage-free
     * result. The input Tensor, descriptor, shape, label, provenance, storage association, and
     * storage contents remain unchanged.</p>
     *
     * @param input non-null source tensor retained by exact reference as the sole provenance input;
     *     it is not mutated or inspected for values or storage
     * @param targetDataType non-null requested result data type retained by exact enum reference in
     *     the cast attributes
     * @return the non-null exact fresh derived Tensor returned by the central factory
     * @throws NullPointerException if {@code input} or {@code targetDataType} is null, checked in
     *     that order with the parameter name as the message; neither failure consumes an identity
     * @throws IllegalStateException if tensor identifier space is exhausted after all local
     *     immutable model values have been constructed
     */
    static Tensor apply(Tensor input, DataType targetDataType) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(targetDataType, "targetDataType");

        DataType sourceDataType = input.descriptor().dataType();
        Shape shape = input.descriptor().shape();
        boolean requiresGrad = input.descriptor().requiresGrad()
                && sourceDataType.isFloating()
                && targetDataType.isFloating();
        TensorDescriptor descriptor =
                new TensorDescriptor(targetDataType, shape, Optional.empty(), requiresGrad);
        CastAttrs attrs = new CastAttrs(targetDataType);
        Operation operation = new Operation(CastKind.CAST, attrs);
        return TensorFactory.createDerived(descriptor, Optional.empty(), operation, List.of(input));
    }
}
