/**
 * Defines immutable Runtime-owned recipes that order already-prepared work.
 *
 * <p>A {@link io.github.pho001.synaptik.runtime.schedule.PreparedSchedule} retains one exact
 * prepared memory plan and an immutable ordered snapshot of step occurrences. Current schedules
 * may start with one optional representation-creation occurrence and may then contain executable
 * occurrences. Empty and executable-only schedules remain valid. The creation prefix makes one
 * immutable {@code PreparedRepresentationPlan} reachable but does not invoke its callbacks.
 * Schedules define neither per-run state nor binding, execution, resource allocation, transfer,
 * materialization, publication, or cleanup behavior.</p>
 */
package io.github.pho001.synaptik.runtime.schedule;
