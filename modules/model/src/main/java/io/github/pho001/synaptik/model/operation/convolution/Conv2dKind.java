package io.github.pho001.synaptik.model.operation.convolution;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Backend-independent identity for grouped NCHW two-dimensional cross-correlation.
 *
 * <p>One occurrence consumes ordered {@code [input, weight]} or
 * {@code [input, weight, bias]} tensors and produces exactly one output. The kind defines the
 * selected numerical meaning, not a reversed mathematical kernel, algorithm, decomposition,
 * gradient rule, compiler support, backend capability, lowering, storage, or execution route.</p>
 *
 * <p>Groups, stride, dilation, padding, layout mapping, contributor membership, and output
 * placement remain exact. Every declared product contributes once to a rounded sum tree, followed
 * by the optional bias addition; corresponding multiply/add fusion requires an unobservable
 * intermediate. FLOAT32, BFLOAT16, and FLOAT16 may use DAZ/FTZ only at named floating arithmetic
 * primitive inputs/results, including low multiplication; FLOAT64 may not. Homogeneous low
 * convolution works and accumulates in FLOAT32, then narrows once at the declared output. No term
 * may be dropped or invented, and no whole-output tolerance is implied. See the <a
 * href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics">Model
 * numerical-semantics contract</a>.</p>
 */
public enum Conv2dKind implements OperationKind {
    /**
     * Grouped NCHW cross-correlation with optional per-output-channel bias.
     *
     * <p>FLOAT64 output accumulates in FLOAT64. FLOAT32, FLOAT16, and BFLOAT16 output accumulate
     * in FLOAT32, with one final low-type conversion when selected. Reassociation and fused
     * multiply-add are permitted. Conceptual padding is positive zero and participates in ordinary
     * IEEE-754
     * multiplication, including multiplication by infinity. An empty channel contraction begins
     * from positive zero before optional bias.</p>
     */
    CONV2D;

    private static final List<OperationSignature> SIGNATURES = List.of(
            OperationSignature.inputRange(Conv2dAttrs.class, 2, 3, 1));

    /**
     * Returns the exact two-to-three-input and one-output convolution signature.
     *
     * @return stable immutable singleton list accepting only {@link Conv2dAttrs}
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
