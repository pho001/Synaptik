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

 * <p>The indexing representation keeps GATHER, GATHER_ELEMENTS, GATHER_ND, and ONE_HOT distinct,
 * records semantic occurrence-to-unique-boundary mapping, permits mixed index/data/output types,
 * and retains one injective output store. Concrete extents, normalized axis, batch and tuple
 * counts, one-hot depth, layout magnitudes, carriers, values, and validation results remain cold
 * facts outside structural identity.</p>
 *
 * <p>Lowering creates these cold immutable models. Portable code generation and the scalar
 * reference realization may consume their established semantics, but Runtime never receives or
 * interprets them.
 */
package io.github.pho001.synaptik.backend.cpu.internal.ir;
