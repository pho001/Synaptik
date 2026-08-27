/**
 * Defines the current analysis stage of the shared Prepare-to-backend handoff.
 *
 * <p>A {@link io.github.pho001.synaptik.prepare.analysis.PrepareContext} projects one fully static
 * planned partition without exposing Compiler-owned aggregates. Its immutable {@link
 * io.github.pho001.synaptik.prepare.analysis.PartitionDag} supplies ordered local producer,
 * consumer, edge, external-input-occurrence, and sink facts without selecting backend policy. A
 * concrete backend implements
 * {@link io.github.pho001.synaptik.prepare.analysis.BackendPartitionPreparer}, retains its selected
 * lowering and route in an opaque
 * {@link io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan}, and returns exact
 * {@link io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement} declarations.
 * Shared slot assignment and backend executable finalization remain later lifecycle stages and
 * are intentionally absent from this package. The current contracts also contain no dynamic
 * binding, physical resource, measurement, cache mutation, schedule, or Runtime execution
 * behavior.</p>
 */
package io.github.pho001.synaptik.prepare.analysis;
