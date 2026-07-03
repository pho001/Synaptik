package io.github.pho001.synaptik.model.operation;

/**
 * Identifies one backend-independent kind of operation by its semantic identity.
 *
 * <p>Implementations are immutable values with stable structural equality and hashing. A family
 * enum is the usual implementation: its inherited {@link Enum#name()} method satisfies this
 * contract without registration or an additional name field. Other implementations must provide
 * equivalent stable value semantics.</p>
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
}
