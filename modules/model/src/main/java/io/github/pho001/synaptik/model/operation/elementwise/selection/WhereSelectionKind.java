package io.github.pho001.synaptik.model.operation.elementwise.selection;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent, parameterless elementwise conditional-selection semantics.
 *
 * <p>{@link #WHERE} describes three ordered logical input roles: condition, true branch, and
 * false branch. The family-owned signature declares three inputs and one output. The kind stores
 * no inputs, identifies no graph occurrence, and creates no Tensor provenance or result
 * descriptor. Conditional branch selection is distinct from
 * scalar-index {@code select} and other indexing operations.</p>
 *
 * <p>This family has no intrinsic parameters. An {@link
 * io.github.pho001.synaptik.model.operation.Operation Operation} therefore represents its kind
 * explicitly with {@link io.github.pho001.synaptik.model.operation.NoOperationAttrs#INSTANCE
 * NoOperationAttrs.INSTANCE}. Operation construction enforces this exact attributes pairing, and
 * a compiled-node occurrence enforces the signature's input and output counts.</p>
 *
 * <p>This vocabulary defines conditional choice meaning only. It does not define condition or
 * branch descriptor eligibility, branch promotion, three-way broadcasting, a result descriptor,
 * evaluation order, gradients, execution, ONNX mapping, or backend availability. Enum identity
 * supplies typed equality and hashing, so an equally named constant in another operation family
 * remains a different semantic value. The inherited {@link #name()} and {@link #toString()} text
 * is stable diagnostic vocabulary only; it is not a serialization, parsing, registry, reflection,
 * or dispatch contract.</p>
 */
public enum WhereSelectionKind implements OperationKind {
    /**
     * Chooses between corresponding true-branch and false-branch values according to a condition.
     *
     * <p>The exact ordered logical roles are condition, true branch, and false branch. A true
     * condition selects the corresponding true-branch value; otherwise the corresponding
     * false-branch value is selected. Its signature declares the three ordered input positions
     * and one output position.</p>
     *
     * <p>This kind does not prescribe eager or lazy branch evaluation. Condition and branch
     * eligibility, promotion, broadcasting, result descriptor construction, gradients,
     * execution, ONNX mapping, and backend availability belong to later owning contracts.</p>
     */
    WHERE;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(NoOperationAttrs.class, 3, 1));

    /**
     * Returns the parameterless three-input, one-output conditional-selection signature.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
