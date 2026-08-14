/**
 * Defines unsupported portable-route realization facts over common CPU lowering.
 *
 * <p>The route plan pairs route-independent canonical IR with the exact Class-File specialization
 * selected by common CPU analysis. The specialization contains exactly one scalar or
 * preferred-species vector body; parallel orchestration reuses that body and stays outside the
 * generated class. The plan may be consumed by artifact realization and finalization, but it does
 * not interpret graph semantics, own the scalar reference implementation, select a native
 * provider, schedule workers, or declare resource lifetime.
 * A functional-scatter specialization embeds its selected family, represented type, reduction,
 * carrier accesses, and output-domain contribution loop. It may additionally expose one direct
 * scratch-segment parameter when its already-lowered floating-product row requires the declared
 * per-range accumulator; the exact-product state machine is embedded in that generated entry.
 * Complete index and replacement-uniqueness validation remains outside generated work. Concrete
 * geometry, workspace identity, size, slice offset, and lifetime remain prepared invocation facts
 * rather than route or artifact-selection decisions.
 * A fold specialization remains a two-boundary workspace-free scalar artifact and embeds its
 * family, represented type, access form, mapping, and sequential addition body; parallel-scalar
 * orchestration reuses that same artifact over disjoint output-coordinate ranges.
 * An ordering specialization remains scalar and adds one direct scratch-segment parameter. SORT
 * and ARGSORT have two boundaries; TOP_K has three and writes both outputs from one artifact.
 * Concrete workspace identity, slice offset, axis, K, and slice ranges stay in the bound
 * invocation, while parallel-scalar orchestration assigns complete disjoint slices.
 * A random specialization remains scalar and workspace-free. It has one initializer output or
 * five ordered dropout boundaries; cold invocation geometry owns concrete layouts, while
 * explicit dropout state values remain direct input data.
 * A cumulative-scan specialization remains scalar, has exactly one input and one output, and
 * carries no scratch parameter. Parallel-scalar orchestration reuses the same artifact over
 * complete independent slices; concrete extents, layouts, carriers, and range count stay cold.
 * An ordinary aggregate specialization is likewise scalar, two-boundary, and scratch-free.
 * Its generated entry bridges to the CPU-owned static output-cell body; concrete selected-domain
 * geometry and parallel output ranges stay cold.
 *
 * <p>The plan is immutable cold-path state; route dispatch never occurs inside the generated loop.
 */
package io.github.pho001.synaptik.backend.cpu.internal.route.portable;
