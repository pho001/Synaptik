/**
 * Defines the current ownership, binding, and cleanup foundation for one complete logical run.
 *
 * <p>{@link io.github.pho001.synaptik.runtime.run.BufferRepresentationBinding} associates an
 * exact physical buffer representation with
 * {@link io.github.pho001.synaptik.runtime.run.RunResourceOwnership#BORROWED borrowed} or
 * {@link io.github.pho001.synaptik.runtime.run.RunResourceOwnership#RUN_OWNED run-owned}
 * cleanup. {@link io.github.pho001.synaptik.runtime.run.RunState} snapshots those ordered
 * bindings plus one run-owned workspace representation per prepared workspace position into
 * private arrays and owns their one-run lifecycle.
 *
 * <p>This package does not allocate or access storage, choose a representation, establish
 * validity or residency, transfer or publish data, bind executable invocations, execute a
 * schedule, pool resources, or define a result. Concrete backends implement physical
 * representations and cleanup mechanics. Later Runtime contracts add typed cold-bound
 * invocation, residency, transfer, scheduling, publication, and runner behavior without changing
 * the one-state-per-complete-run boundary.
 */
package io.github.pho001.synaptik.runtime.run;
