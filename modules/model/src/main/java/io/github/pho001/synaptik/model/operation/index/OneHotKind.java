package io.github.pho001.synaptik.model.operation.index;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent one-hot encoding semantics.
 *
 * <p>{@link #ONE_HOT} consumes one signed-integral indices tensor and produces one dense BOOL
 * tensor whose final axis has the depth stored in {@link OneHotAttrs}. For an input coordinate
 * {@code p} with eventual value {@code i}, the exact logical formula is
 * {@code result[p..., j] = (i == j)} for {@code 0 <= j < depth}. Valid execution requires
 * {@code 0 <= i < depth}; an invalid value does not wrap, clamp, select a default, or produce an
 * all-false row.</p>
 *
 * <p>This kind and its fixed one-input, one-output signature define meaning and occurrence
 * cardinality only. They carry no Tensor, Shape, storage, graph, gradient, compiler, backend,
 * bounds-checking, or execution state and do not claim that any later layer currently supports
 * the operation.</p>
 *
 * <p>The Model-owned numerical-profile contract leaves one-hot encoding exact under
 * {@code STRICT_IEEE} and {@code ACCELERATOR}: depth, index bounds, equality predicates, output
 * mapping, and BOOL result bits remain exact. Encoding applies no DAZ, FTZ, comparison epsilon,
 * or arithmetic tolerance. See the
 * <a href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#numerical-profiles">normative
 * numerical-profile contract</a>.</p>
 */
public enum OneHotKind implements OperationKind {
    /** Requests one exact dense BOOL trailing-axis indicator encoding. */
    ONE_HOT;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(OneHotAttrs.class, 1, 1));

    /**
     * Returns the fixed one-input, one-output one-hot signature.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
