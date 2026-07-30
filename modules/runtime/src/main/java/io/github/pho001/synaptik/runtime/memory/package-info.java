/**
 * Defines Runtime-owned immutable slot identities and final prepared-memory geometry.
 *
 * <p>{@link io.github.pho001.synaptik.runtime.memory.BufferSlot} and
 * {@link io.github.pho001.synaptik.runtime.memory.WorkspaceSlot} are nominally distinct,
 * non-negative identities interpreted within one prepared-memory-plan context.
 * {@link io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan} snapshots ordered,
 * unique per-domain entries that associate each slot with an exact non-negative byte size and
 * positive power-of-two byte alignment.
 *
 * <p>These contracts contain final reusable geometry only. They retain no graph value, Prepare
 * requirement, source-to-slot association, physical buffer or workspace, storage handle,
 * address, allocation, resource ownership, device, residency, or per-run binding. Prepare-owned
 * assignment and finalization, Runtime slot access, physical allocation, resource lifetime, and
 * execution remain later work. This package allocates no memory and provides no storage or
 * execution access.
 */
package io.github.pho001.synaptik.runtime.memory;
