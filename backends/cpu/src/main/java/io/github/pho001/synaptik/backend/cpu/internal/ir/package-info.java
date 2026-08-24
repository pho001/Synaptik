/**
 * Defines the unsupported, route-independent canonical CPU portable-kernel representations.
 *
 * <p>The pointwise representation records typed boundary and virtual values, one forty-eight-opcode
 * family-oriented pointwise vocabulary, exact typed scalar and two-bound clamp bits, the selected
 * scalar-power realization where applicable, all nineteen distinct same-typed floating unary
 * semantics, ordered computation, and the boundary-versus-virtual role of canonical BOOL values
 * used as stored bytes or unit-private generated masks,
 * normalized access forms, primitive half-open loop bounds, and ordered stores. Each access form
 * selects one of five scalar state-machine regimes while its binding retains cold extents, base
 * offset, effective strides, start coordinates/address, and exact half-open accessed span. It deliberately
 * excludes routes, slots, graph identities, concrete extents, segment instances, generator
 * versions, and invocation bindings so structurally compatible shapes can share one artifact.
 * A selected contiguous copy produces an adjusted canonical consumer access structure for the
 * generated artifact, while its concrete extent, source binding, workspace, and cost evidence
 * remain instance facts outside canonical identity.
 *
 * <p>The affine representation records one static one-source/one-result represented-bit copy,
 * its ordered structural mapping steps, exact read/write access forms, and whether iteration uses
 * logical elements or a deduplicated distinct-address write domain. Concrete composed addresses
 * remain cold plan facts. Internal view values remain graph and logical-memory values but do not
 * become affine IR instructions, stores, declarations, or Runtime slots.
 *
 * <p>The static movement representation records one PAD, TILE, CONCAT, STACK, window-extraction,
 * or functional SLICE_UPDATE family, all-six-type represented-bit identity, unique input access
 * forms in first-occurrence order, the ordered occurrence map, one injective output access form,
 * and exact padding bits when applicable. Slice-update structure retains only output rank and its
 * exact {@code [base, update]} occurrence map; signed placement geometry remains cold. Concrete
 * extents, strides, offsets, axes, repeats, segment prefixes, starts, lengths, and steps remain
 * instance geometry outside generated-artifact identity.</p>
 *
 * <p>The functional-scatter representation keeps SCATTER_ELEMENTS, Gather-compatible
 * SCATTER_ADD, and SCATTER_ND distinct. It records the represented reduction, exact three-role
 * occurrence map over deduplicated inputs, mixed index/data boundary types, structural accesses,
 * one injective data-shaped output, and whether the generated entry accepts exact-product
 * scratch. Axes, batch and tuple geometry, extents, layout magnitudes, ranges, carriers, scratch
 * sizes, values, and validation results remain cold facts.</p>

 * <p>The fold representation keeps FOLD_AXIS and FOLD2D distinct and records only their common
 * represented type, structural input/output access forms, and canonical sequential-addition
 * policy. Concrete axes, window geometry, extents, layout magnitudes, ranges, and carriers remain
 * cold prepared facts.</p>

 * <p>The ordering representation keeps SORT, ARGSORT, and TOP_K distinct. It records the
 * represented input/value type, direction, TOP_K output-order flag, ordered read/write boundary
 * access forms, one- or two-output structure, and explicit two-index-region scratch policy.
 * Concrete axis, K, extents, layout magnitudes, carriers, assigned workspace, and range slicing
 * remain cold prepared facts.</p>

 * <p>The indexing representation keeps GATHER, GATHER_ELEMENTS, GATHER_ND, and ONE_HOT distinct,
 * records semantic occurrence-to-unique-boundary mapping, permits mixed index/data/output types,
 * and retains one injective output store. Concrete extents, normalized axis, batch and tuple
 * counts, one-hot depth, layout magnitudes, carriers, values, and validation results remain cold
 * facts outside structural identity.</p>
 *
 * <p>The random representation distinguishes raw state initialization from explicit-state
 * dropout. Its identity fixes the CPU-private counter algorithm, uniform conversion, exact
 * probability bits, represented type, finite-precision policy, and one- or five-boundary roles;
 * layouts, carriers, ranges, and consumed dropout state values remain cold facts.</p>
 *
 * <p>The cumulative-scan representation distinguishes addition from multiplication and records
 * the five-type represented arithmetic, normalized axis role, inclusive/exclusive and forward/
 * reverse modes, sequential typed-rounding policy, and exact read-input/write-output access
 * structure. Extents, offsets, stride magnitudes, carriers, slots, workers, and slice ranges
 * remain cold facts.</p>
 *
 * <p>The aggregate representation records SUM, MEAN, PROD, MIN, MAX, ALL, or ANY; exact ordinary
 * or bound SUM-to-Shape form; increasing selected-axis membership; retention; structural two-
 * boundary access; numerical state/domain shape; extrema floating policy; and complete-output-cell
 * ranges. SUM-to-Shape additionally retains exact bound source and target extents because their
 * right-aligned mapping changes generated bytes. Floating numerical reductions use exact run-owned
 * state, while integral reductions and no-reduction represented copies are workspace-free.
 * Concrete layout magnitudes, carriers, slots, workers, and run identities remain cold.</p>
 *
 * <p>Lowering creates these cold immutable models. Portable code generation and the scalar
 * reference realization may consume their established semantics, but Runtime never receives or
 * interprets them.
 */
package io.github.pho001.synaptik.backend.cpu.internal.ir;
