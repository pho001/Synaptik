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
 * A {@link io.github.pho001.synaptik.prepare.analysis.BackendPartitionTuningHandoff} can separately
 * associate one exact partition with a backend-owned complete candidate batch and an optional
 * backend-owned selected decision. The two tuning roles are method-free, so shared Prepare may
 * transport those values but cannot interpret their private candidate or decision vocabulary.
 * Shared slot assignment and backend executable finalization remain later lifecycle stages and
 * are intentionally absent from this package. The current contracts also contain no dynamic
 * binding, physical resource, measurement, cache mutation, schedule, or Runtime execution
 * behavior.</p>
 *
 * <p>General scans remain single outer nodes with exact independently scoped compiled-body
 * attachments. Context validation checks every local descriptor, ordered formal/actual binding,
 * outer role descriptor and locally knowable demand obligation before analysis; complete-graph
 * projection also checks demand against all outer uses. Local body IDs never enter the outer
 * logical resource inventory. Length-aware forward and generated reverse projections preserve
 * ordered ROW(0), rank-two ROW(1) and SHARED roles, validating static boundary extents rather than
 * inferring roles from equal sizes. The final non-gradient INT64 lengths actual has no local
 * formal and contributes nothing to outer gradient eligibility. Actual layouts are retained
 * independently of concrete backend read predicates. Homogeneous FLOAT32/FLOAT64 local values,
 * including unused declarations, remain separate from that INT64 boundary. This transport reads
 * no length values, duplicates no body-algebra proof, and does not execute a scan, differentiate
 * its body or qualify a concrete backend route.</p>
 */
package io.github.pho001.synaptik.prepare.analysis;
