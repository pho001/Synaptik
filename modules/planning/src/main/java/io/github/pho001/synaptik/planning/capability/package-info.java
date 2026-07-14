/**
 * Defines backend-neutral compile-time contracts for operation capability questions.
 *
 * <p>A capability question combines immutable backend-independent operation semantics with the
 * ordered logical input and output descriptors of one structurally valid occurrence. An
 * explicitly supplied provider answers whether its named backend can semantically own that
 * occurrence. The package also contains an internal per-query step that validates a complete
 * provider-to-availability-snapshot association, applies current availability and one optional
 * exact hard requirement, and retains the supported backend identities that remain eligible:</p>
 *
 * <pre>{@code
 * OperationCapabilityQuery -> BackendCapabilityProvider -> boolean ownership capability
 *                          -> internal hard eligibility -> ordered BackendId values
 * }</pre>
 *
 * <p>The internal eligibility result follows provider encounter order and retains only backend
 * identities. An exact-device or device-class requirement uses the matching immutable snapshot
 * only to prove current matching availability; the provider answer remains backend-level, and no
 * device is selected or retained. A valid no-match result is an immutable empty list.</p>
 *
 * <p>The public surface remains the query and provider contracts. Reusable or public capability
 * matrices, public planning orchestration, scoring, profiles, ownership selection, partitioning,
 * device-level capability or selection, compiler integration, preparation, kernel or route
 * selection, runtime state, execution, and provider implementations remain outside this package
 * or planned.</p>
 */
package io.github.pho001.synaptik.planning.capability;
