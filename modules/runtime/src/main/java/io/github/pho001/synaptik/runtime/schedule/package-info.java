/**
 * Defines immutable Runtime-owned recipes that order already-prepared work.
 *
 * <p>A {@link io.github.pho001.synaptik.runtime.schedule.PreparedSchedule} retains one exact
 * prepared memory plan and an immutable ordered snapshot of step occurrences. Current schedules
 * may start with one optional representation-creation occurrence and may then contain executable
 * or buffer-transfer occurrences followed by an optional dense publication-only suffix. Empty,
 * executable-only, transfer-only, and zero-publication schedules remain valid. Publication
 * result indices equal suffix encounter order, and distinct result positions may intentionally
 * name the same exact representation. The creation prefix makes one
 * immutable {@code PreparedRepresentationPlan} reachable but does not invoke its callbacks.
 * A buffer-transfer occurrence retains one immutable prepared recipe and invokes no work;
 * transfer to an equivalent already-created destination is materialization, not another step
 * kind. A publication occurrence retains only an immutable exact-coordinate recipe and invokes
 * no binding, validity check, physical work, or ownership transition. Schedules define neither
 * per-run state nor binding, execution, resource allocation, output access, or cleanup
 * behavior.</p>
 */
package io.github.pho001.synaptik.runtime.schedule;
