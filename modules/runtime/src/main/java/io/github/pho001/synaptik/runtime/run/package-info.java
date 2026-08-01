/**
 * Defines cold creation, ownership, structural residency, validity, binding, and cleanup for one
 * complete logical run, binds exact prepared publication coordinates, and owns the completed
 * result lease.
 *
 * <p>{@link io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding} associates an
 * exact physical buffer representation with
 * {@link io.github.pho001.synaptik.runtime.run.RunResourceOwnership#BORROWED borrowed} or
 * {@link io.github.pho001.synaptik.runtime.run.RunResourceOwnership#RUN_OWNED run-owned}
 * cleanup. {@link io.github.pho001.synaptik.runtime.run.RunState} snapshots those ordered
 * bindings plus one run-owned workspace representation per prepared workspace position into
 * private arrays and owns their one-run lifecycle. Every bound representation is structurally
 * resident until closure. Buffer copies also carry independent explicit validity bits: borrowed
 * inputs start valid and newly created run-owned buffers start invalid. Workspaces are run-owned
 * scratch and carry no logical validity.
 *
 * <p>The {@code runtime.execution} package may cold-bind selected representations from an open
 * state into a backend-owned typed invocation. That invocation retains the exact state and
 * rejects execution after the state closes, but owns neither the state nor its resources. This
 * Package-private cold setup invokes immutable backend-owned creators and performs deterministic
 * reverse rollback before a complete state exists. Concrete backends implement allocation,
 * physical access, transfer, and cleanup mechanics.
 *
 * <p>{@link io.github.pho001.synaptik.runtime.run.PreparedPublication} uses exact dense Runtime
 * coordinates to cold-bind one already-created buffer representation directly into a
 * {@link io.github.pho001.synaptik.runtime.run.BoundPublication}. Publishing requires that exact
 * copy to be valid at that moment and changes only one local one-shot flag. It performs no lookup,
 * fallback, transfer, conversion, or backend work. A completed ordered set creates a
 * {@link io.github.pho001.synaptik.runtime.run.RunResult}, which privately retains result aliases
 * and leases the whole state cleanup lifecycle while exposing no output value or representation.
 * Empty results are valid; partial publication transfers no cleanup responsibility.
 *
 * <p>{@link io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner} performs the complete
 * synchronous lifecycle. It creates one isolated state, cold-binds every remaining occurrence
 * before the first action, traverses direct bound actions in schedule order, applies declared
 * executable read/write validity, and either leases the state to a result or closes it after
 * failure. The stateless runner may serve concurrent calls whose mutable resources remain
 * isolated. This package does not inspect storage, choose a backend, infer physical aliasing or
 * coherence, pool resources, emit trace events, or expose public output values.
 */
package io.github.pho001.synaptik.runtime.run;
