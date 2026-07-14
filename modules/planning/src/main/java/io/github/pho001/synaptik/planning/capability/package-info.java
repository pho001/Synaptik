/**
 * Defines backend-neutral compile-time contracts for operation capability questions.
 *
 * <p>A capability question combines immutable backend-independent operation semantics with the
 * ordered logical input and output descriptors of one structurally valid occurrence. An
 * explicitly supplied provider answers whether its named backend can semantically own that
 * occurrence:</p>
 *
 * <pre>{@code
 * OperationCapabilityQuery -> BackendCapabilityProvider -> boolean ownership capability
 * }</pre>
 *
 * <p>Capability is distinct from backend registration and availability, hard-requirement
 * evaluation, candidate scoring, ownership selection, partitioning, device selection,
 * preparation, kernel or route selection, runtime state, and execution. Those behaviors and
 * provider implementations are not supplied by this package.</p>
 */
package io.github.pho001.synaptik.planning.capability;
