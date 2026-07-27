package io.github.pho001.synaptik.planning.capability;

import io.github.pho001.synaptik.backend.contract.BackendId;

/**
 * Answers whether one named backend can semantically own an operation occurrence.
 *
 * <p>This planning-owned interface is an explicitly supplied compile-time collaboration that a
 * concrete backend may implement. The current package-private compiler artifact entry supplies
 * providers to Planning once per final graph node and retains only the selected
 * {@link BackendId}; neither the compiler result nor a Planning output retains the provider. This
 * interface is not a registry, discovery mechanism, service locator, availability report,
 * hard-requirement evaluator, scoring policy, route selector, preparer, or execution service.</p>
 *
 * <p>An implementation's immutable configuration participates in its capability answer. For the
 * same immutable query and unchanged immutable provider configuration, repeated answers are
 * deterministic.</p>
 */
public interface BackendCapabilityProvider {
    /**
     * Returns the backend whose semantic ownership capability this provider describes.
     *
     * <p>The identity is stable for the lifetime of the provider. It names an ownership domain
     * but does not prove registration, availability, or executable readiness.</p>
     *
     * @return the stable non-null backend identity represented by this provider
     */
    BackendId backendId();

    /**
     * Reports whether this provider's backend can semantically own one immutable operation
     * occurrence.
     *
     * <p>The result is deterministic for an immutable query and unchanged immutable provider
     * configuration. It does not report registration or availability, evaluate a hard backend
     * requirement, score ownership candidates, select a kernel or route, inspect prepare or
     * runtime state, or promise successful preparation or execution. Current compiler
     * orchestration uses this answer only as one input to backend-neutral ownership selection.</p>
     *
     * @param query the non-null immutable operation occurrence to inspect
     * @return {@code true} only when this provider's backend supports semantic ownership of the
     *     described occurrence; {@code false} carries no diagnostic reason
     * @throws NullPointerException if {@code query} is {@code null}; implementations must use the
     *     exception message {@code query}
     */
    boolean supports(OperationCapabilityQuery query);
}
