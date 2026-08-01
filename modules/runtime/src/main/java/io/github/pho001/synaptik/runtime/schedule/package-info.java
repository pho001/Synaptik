/**
 * Defines immutable Runtime-owned recipes that order already-prepared work.
 *
 * <p>A {@link io.github.pho001.synaptik.runtime.schedule.PreparedSchedule} retains one exact
 * prepared memory plan and an immutable ordered snapshot of step occurrences. Current schedules
 * contain executable occurrences only. They define neither per-run state nor binding, execution,
 * resource allocation, transfer, materialization, publication, or cleanup behavior.</p>
 */
package io.github.pho001.synaptik.runtime.schedule;
