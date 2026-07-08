package io.github.pho001.synaptik.model.operation;

import java.util.List;
import java.util.Objects;

/**
 * Identifies one backend-independent kind of operation by its semantic identity.
 *
 * <p>Implementations are immutable values with stable structural equality and hashing. A family
 * enum is the usual implementation: its inherited {@link Enum#name()} method satisfies the name
 * contract, while the family explicitly declares immutable accepted {@link #signatures()}
 * without global registration. Other implementations must provide equivalent stable value
 * semantics and fail closed by declaring their complete structural variants.</p>
 *
 * <p>The name is intended for diagnostics and deterministic model inspection. It is not a
 * serialization token, backend-dispatch key, execution route, kernel name, or reflective class
 * name. Kinds from different concrete types remain distinct even when {@link #name()} returns the
 * same text; callers compare the typed kind values rather than consulting a global string
 * registry.</p>
 */
public interface OperationKind {
    /**
     * Returns the stable semantic name of this operation kind.
     *
     * <p>Repeated calls for the same immutable kind return equal text. Implementations must return
     * a non-null, non-blank value suitable for diagnostics and deterministic model inspection.</p>
     *
     * @return the stable, non-null, non-blank semantic name
     */
    String name();

    /**
     * Returns the stable immutable structural variants accepted by this semantic kind.
     *
     * <p>The result must be non-null, non-empty, immutable, and stable across calls. Every element
     * must be non-null, and no two elements may declare the same exact attributes class. The list
     * is owned by the concrete operation family and is not a global registry, backend capability
     * table, compiler schema, or execution route inventory.</p>
     *
     * @return the non-null, non-empty immutable ordered list of accepted signatures
     */
    List<OperationSignature> signatures();

    /**
     * Resolves the unique structural variant accepting an attributes value.
     *
     * <p>Resolution performs exact runtime-class matching in stable signature order. It validates
     * the family declaration while scanning, fails closed for missing or malformed declarations,
     * and allocates no collection on a successful lookup. It does not inspect operands or perform
     * graph, compiler, backend, or execution validation.</p>
     *
     * @param attrs the non-null immutable attributes value to match without retaining
     * @return the unique non-null family-owned signature accepting {@code attrs}
     * @throws NullPointerException if {@code attrs} or the declared signature list is null
     * @throws IllegalStateException if the family declares no signatures, a null signature, or
     *     duplicate variants for one exact attributes class
     * @throws IllegalArgumentException if no declared signature accepts the exact attributes class
     */
    default OperationSignature signatureFor(OperationAttrs attrs) {
        Objects.requireNonNull(attrs, "attrs");
        List<OperationSignature> declared = Objects.requireNonNull(signatures(), "signatures");
        if (declared.isEmpty()) {
            throw new IllegalStateException(diagnosticName() + " declares no operation signatures");
        }

        OperationSignature match = null;
        for (int index = 0; index < declared.size(); index++) {
            OperationSignature candidate = Objects.requireNonNull(
                    declared.get(index), "signatures[" + index + "]");
            for (int earlierIndex = 0; earlierIndex < index; earlierIndex++) {
                OperationSignature earlier = declared.get(earlierIndex);
                if (earlier.attributesType() == candidate.attributesType()) {
                    throw new IllegalStateException(
                            diagnosticName() + " declares duplicate signature attributes type "
                                    + candidate.attributesType().getName());
                }
            }
            if (candidate.acceptsAttributes(attrs)) {
                match = candidate;
            }
        }

        if (match == null) {
            StringBuilder expected = new StringBuilder();
            for (int index = 0; index < declared.size(); index++) {
                if (index > 0) {
                    expected.append(", ");
                }
                expected.append(declared.get(index).attributesType().getName());
            }
            throw new IllegalArgumentException(
                    diagnosticName() + " does not accept attributes type "
                            + attrs.getClass().getName() + "; expected one of [" + expected + "]");
        }
        return match;
    }

    private String diagnosticName() {
        return getClass().getName() + "." + name();
    }
}
