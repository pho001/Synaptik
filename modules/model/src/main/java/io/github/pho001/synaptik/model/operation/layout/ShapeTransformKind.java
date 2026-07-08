package io.github.pho001.synaptik.model.operation.layout;

import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.util.List;

/**
 * Identifies backend-independent, one-input transformations whose intrinsic parameter is an
 * exact target shape.
 *
 * <p>Both kinds compose explicitly with {@link TargetShapeAttrs}:</p>
 *
 * <pre>{@code
 * Operation reshape = new Operation(ShapeTransformKind.RESHAPE, attrs);
 * Operation expand = new Operation(ShapeTransformKind.EXPAND, attrs);
 * }</pre>
 *
 * <p>Family-owned signatures enforce these exact pairings and declare one input and one output.
 * Target-shape compatibility remains input-aware validation outside the signature. This enum
 * stores no
 * input, target shape, result descriptor, layout, materialization requirement, or graph-occurrence
 * state.</p>
 *
 * <p>These kinds define logical meaning only. Public request normalization, element-count and
 * broadcast validation, layout and provenance construction, gradient behavior, compiler and
 * planning decisions, materialization, backend support, and execution belong to later owning
 * contracts. Inherited enum names are diagnostic text, not serialization, parsing, registry,
 * dispatch, reflection, or kernel identifiers.</p>
 */
public enum ShapeTransformKind implements OperationKind {
    /**
     * Preserves the ordered logical element sequence while interpreting it through the target
     * coordinates in {@link TargetShapeAttrs}.
     *
     * <p>The eventual input and target must have compatible logical element counts, but this kind
     * neither stores those counts nor validates them. It also does not decide whether the result
     * is a view or copy, derive layout, define gradients, or execute the transformation.</p>
     */
    RESHAPE,

    /**
     * Logically repeats compatible singleton dimensions or introduces repeated leading axes to
     * produce the target shape in {@link TargetShapeAttrs}.
     *
     * <p>The eventual input and target must satisfy the expansion contract, but this kind neither
     * stores input dimensions nor validates broadcasting. It also does not choose zero-stride
     * geometry, aliasing or materialization, define gradients, or execute the transformation.</p>
     */
    EXPAND;

    private static final List<OperationSignature> SIGNATURES =
            List.of(OperationSignature.fixed(TargetShapeAttrs.class, 1, 1));

    /**
     * Returns the shared target-shape one-input, one-output structural signature.
     *
     * @return the stable immutable singleton signature list
     */
    @Override
    public List<OperationSignature> signatures() {
        return SIGNATURES;
    }
}
