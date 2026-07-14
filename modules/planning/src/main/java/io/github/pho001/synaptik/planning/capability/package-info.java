/**
 * Defines backend-neutral compile-time contracts for operation capability questions.
 *
 * <p>A capability question combines immutable backend-independent operation semantics with the
 * ordered logical input and output descriptors of one structurally valid occurrence. An
 * explicitly supplied provider answers whether its named backend can semantically own that
 * occurrence. The package also contains an internal per-query step that validates a complete
 * provider-to-availability-snapshot association, applies current availability and one optional
 * exact hard requirement, and retains the supported backend identities that remain eligible. A
 * second internal step selects one owner from that complete ordered candidate set:</p>
 *
 * <pre>{@code
 * OperationCapabilityQuery -> BackendCapabilityProvider -> boolean ownership capability
 *                          -> internal hard eligibility -> ordered BackendId values
 *                          -> internal baseline selection -> one BackendId owner
 * }</pre>
 *
 * <p>The internal eligibility result follows provider encounter order and retains only backend
 * identities. An exact-device or device-class requirement uses the matching immutable snapshot
 * only to prove current matching availability; the provider answer remains backend-level. The
 * baseline selector optionally prefers the first eligible backend whose associated snapshot
 * reports the configured coarse device class, then falls back to the first eligible backend.
 * Provider order resolves ties. Empty hard eligibility fails internally before selection. No
 * device is selected or retained by either step.</p>
 *
 * <p>The public surface remains the query and provider contracts. Reusable or public capability
 * matrices, public planning orchestration or owner selection, numeric or cost scoring,
 * operation-family or workload classification, profiles, partitioning, logical memory, compiler
 * integration, device-level capability or selection, preparation, kernel or route selection,
 * runtime state, execution, and provider implementations remain outside this package or
 * planned.</p>
 */
package io.github.pho001.synaptik.planning.capability;
