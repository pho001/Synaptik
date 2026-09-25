package io.github.pho001.synaptik.model.operation.linalg;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent vector, matrix, and batched matrix multiplication semantics.
 *
 * <p>{@link #MATMUL} contracts the final axis of the left operand with the final axis of a
 * rank-one right operand or the penultimate axis of a higher-rank right operand. Rank-one
 * operands are interpreted by temporarily inserting a matrix axis, and leading matrix batches
 * broadcast right-aligned. The kind carries no operands, shapes, constraints, storage, graph,
 * compiler, execution, gradient, algorithm, or backend-support state.</p>
 *
 * <p>The only valid composition pairs {@link #MATMUL} with
 * {@link NoOperationAttrs#INSTANCE}. Its family-owned signature fixes two logical inputs and one
 * logical output. Enum identity is the semantic identity; inherited text is diagnostic only and
 * is not a registry, serialization, dispatch, route, or kernel contract.</p>
 *
 * <p>Under the Model-owned graph numerical-profile contract, {@code STRICT_IEEE} retains this
 * contraction. For {@code ACCELERATOR FLOAT32}, exact broadcast mapping, contraction extent,
 * contributor membership, empty identity, and output placement are unchanged. Every pairwise
 * product participates exactly once; evaluation may use any binary tree, per-step FLOAT32
 * rounding, DAZ/FTZ, and corresponding product/add fusion. It may not drop, duplicate, invent,
 * pretruncate, or replace a term. The existing qualifying nonempty final exact-zero publication
 * freedom remains local to the final MATMUL cell; an empty contraction stays positive zero. This
 * is recursive construction freedom, not a final-output tolerance, and non-FLOAT32 behavior
 * remains strict. See the
 * <a href="https://github.com/pho001/Synaptik/blob/main/docs/architecture/contracts/foundational-modules.md#numerical-profiles">normative
 * numerical-profile contract</a>.</p>
 */
public enum MatmulKind implements OperationKind {
    /**
     * Requests the sum of pairwise products across one shared contraction dimension.
     *
     * <p>Floating results use the promoted floating type, with FLOAT32 accumulation for BFLOAT16
     * and FLOAT32 results and FLOAT64 accumulation for FLOAT64 results. Reassociation and fused
     * multiply-add are permitted, so bitwise or cross-backend identical rounding is not promised.
     * Signed-integral results use the promoted width and exact modular arithmetic modulo
     * {@code 2^32} or {@code 2^64}. Empty contractions produce positive floating zero or integral
     * zero. These policies define mathematical meaning without selecting an implementation.</p>
     */
    MATMUL;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(NoOperationAttrs.class, 2, 1));

    /**
     * Returns the fixed two-input, one-output parameterless MATMUL signature.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
