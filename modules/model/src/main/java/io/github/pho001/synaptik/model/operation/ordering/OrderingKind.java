package io.github.pho001.synaptik.model.operation.ordering;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies stable, axis-wise full-ordering requests.
 *
 * <p>{@link #SORT} produces values and {@link #ARGSORT} produces logical indices. Both are
 * single-input, single-output semantics parameterized by {@link SortAttrs}. Equal keys retain
 * increasing logical input-index order; NaNs form a final class in either direction, and
 * negative zero precedes positive zero in ascending order. These semantics select no algorithm,
 * backend route, storage behavior, gradient rule, or execution support.</p>
 */
public enum OrderingKind implements OperationKind {
    /** Requests the input values in stable axis order. */
    SORT,

    /** Requests stable logical input indices in axis order. */
    ARGSORT;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(SortAttrs.class, 1, 1));

    /**
     * Returns the shared exact sort-attributes occurrence signature.
     *
     * @return the stable immutable one-input, one-output signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
