/**
 * Defines immutable Runtime-owned recipes that order already-prepared work.
 *
 * <p>A {@link io.github.pho001.synaptik.runtime.schedule.PreparedSchedule} retains one exact
 * prepared memory plan and an immutable ordered snapshot of step occurrences. Current schedules
 * may start with one optional representation-creation occurrence and may then contain executable
 * or buffer-transfer occurrences. Empty, executable-only, and transfer-only schedules remain
 * valid. The creation prefix makes one
 * immutable {@code PreparedRepresentationPlan} reachable but does not invoke its callbacks.
 * A buffer-transfer occurrence retains one immutable prepared recipe and invokes no work;
 * transfer to an equivalent already-created destination is materialization, not another step
 * kind. Schedules define neither per-run state nor binding, execution, resource allocation,
 * publication, or cleanup behavior.</p>
 */
package io.github.pho001.synaptik.runtime.schedule;
