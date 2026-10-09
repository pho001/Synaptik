package io.github.pho001.synaptik.model.operation.scan;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent cumulative-scan arithmetic.
 *
 * <p>A cumulative scan has one logical input and produces one output position for every input
 * position, so it preserves the input shape. The kind selects addition or multiplication, while
 * {@link CumulativeScanAttrs} supplies the normalized scan axis, inclusion mode, and traversal
 * direction. Neither kind stores input state, result state, or graph-occurrence identity.</p>
 *
 * <p>Both kinds accept the same family-owned one-input, one-output signature with exact
 * {@link CumulativeScanAttrs}. This vocabulary describes requested mathematics only; it does not
 * define eligible input types, result descriptors, accumulation precision, gradients, value
 * execution, storage, compiler behavior, or backend availability.</p>
 *
 * <p>Direction, exclusivity, output placement, identities, and each logical prefix's contributor
 * membership remain exact. Every prefix contributor participates once in its rounded ADD or MUL
 * tree; no term may be dropped, duplicated, invented, or pretruncated. Named floating arithmetic
 * primitives may use DAZ/FTZ for FLOAT32, BFLOAT16, and FLOAT16, including low multiplication, but
 * never FLOAT64. Homogeneous low scans work and accumulate in FLOAT32 with one final ties-to-even
 * narrowing for each declared output. Integral scans retain fixed-width modular arithmetic. See the
 * <a
 * href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#profile-free-numerical-semantics">Model
 * numerical-semantics contract</a>.</p>
 */
public enum CumulativeScanKind implements OperationKind {
    /**
     * Requests cumulative addition along the normalized axis.
     *
     * <p>For logical input {@code [1, 2, 3]}, the inclusive-forward, exclusive-forward,
     * inclusive-reverse, and exclusive-reverse meanings are {@code [1, 3, 6]},
     * {@code [0, 1, 3]}, {@code [6, 5, 3]}, and {@code [5, 3, 0]}. An exclusive scan emits the
     * additive identity zero at its first traversed position.</p>
     */
    CUM_SUM,

    /**
     * Requests cumulative multiplication along the normalized axis.
     *
     * <p>For logical input {@code [2, 3, 4]}, the inclusive-forward, exclusive-forward,
     * inclusive-reverse, and exclusive-reverse meanings are {@code [2, 6, 24]},
     * {@code [1, 2, 6]}, {@code [24, 12, 4]}, and {@code [12, 4, 1]}. An exclusive scan emits
     * the multiplicative identity positive one at its first traversed position.</p>
     *
     * <p>A zero-length axis has no output position at which to emit that identity. Integral
     * products use the exact input width with two's-complement modular multiplication. Floating
     * products propagate NaN, make zero times infinity NaN, and determine zero and infinity signs
     * by multiplication parity. These rules select mathematical results without selecting an
     * accumulation algorithm, intermediate rounding, NaN payload, backend route, or gradient
     * rule.</p>
     */
    CUM_PROD;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(CumulativeScanAttrs.class, 1, 1));

    /**
     * Returns the cumulative-scan one-input, one-output structural signature.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
