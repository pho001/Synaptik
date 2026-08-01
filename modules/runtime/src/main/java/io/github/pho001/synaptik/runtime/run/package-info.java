/**
 * Defines cold creation, ownership, structural residency, validity, binding, and cleanup for one
 * complete logical run.
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
 * physical access, transfer, and cleanup mechanics. This package does not inspect storage, choose
 * a backend, copy or transfer data, infer coherence, execute a schedule, publish data, pool
 * resources, or define a result. Later Runtime contracts add explicit transfer, execution,
 * publication, and runner transitions without changing the one-state-per-complete-run boundary.
 */
package io.github.pho001.synaptik.runtime.run;
