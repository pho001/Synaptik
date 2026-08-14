/**
 * Owns unsupported structural identity, compatibility interning, and optional cold persistence.
 *
 * <p>Keys combine canonical lowering with the selected numerical mode, generated scalar/vector
 * compute form, exact preferred FLOAT64 species when applicable, ordered boundary data-type and
 * carrier pattern, typed opcode/scalar-immediate structure, explicit scalar-power realization,
 * and selected direct-versus-
 * materialized source position under the current generator schema.
 * Instance extents, offsets, strides, costs, chunk configuration, worker identity, and Runtime
 * resources remain outside artifact identity. Indexing identity includes its closed family,
 * occurrence map, mixed ordered boundary types, and structural access/rank form while excluding
 * axes, tuple/depth values, concrete geometry, carriers, and validation results. Functional
 * slice-update identity includes the movement family, result rank, structural access forms, and
 * exact {@code [base, update]} occurrence map while excluding starts, lengths, steps, extents,
 * and layout magnitudes. Functional-scatter identity additionally records its family, reduction,
 * occurrence map, structural accesses, and optional exact-product scratch signature while keeping
 * concrete axes, extents, ranges, and scratch sizes cold. Fold identity records its family,
 * represented type, boundary ranks/accesses, and explicit addition policy while leaving concrete
 * window and layout geometry cold. Ordering identity records SORT/ARGSORT/TOP_K, represented
 * type, direction/output-order flags, boundary roles, output count, and scratch-bearing entry
 * shape while leaving axis, K, extents, layout magnitudes, workspace identity, and ranges cold.
 * Schema 19 adds CPU-private explicit-state initializer/dropout algorithm identity, baked raw
 * initializer/probability bits, one- or five-boundary entry shape, and exact finite-precision
 * policy. Schema 20 adds cumulative sum/product, axis and mode roles, sequential typed rounding,
 * and a two-boundary workspace-free slice entry. Schema 21 adds ordinary extrema and Boolean
 * output-cell reduction identity plus its direct static-body bridge. Schema 22 adds embedded typed
 * scan/aggregate bodies and the proved dense heap-array integer-address pointwise loop category.
 * Schema 23 adds proved dense heap-array integer affine-copy and movement bodies with invariant
 * invocation geometry hoisted before their loops. Schema 23 is current-only.
 * Four complete candidate plans, one realized
 * artifact, zero fixed-shape variants, and zero unrolled variants are the current hard budget.
 * With no trusted root, realization remains entirely in memory. With a root, one bounded
 * current-schema envelope may supply class bytes after compatibility, integrity, size, class
 * shape, and entry-descriptor verification; every absence or failure falls back safely to
 * deterministic in-memory emission.
 *
 * <p>This package collaborates with canonical IR and Class-File generation. It owns neither graph
 * lowering nor Runtime lookup, JIT machine code, profiling state, workload tuning-cache state, or
 * run-resource lifetime. Ordinary prepare performs no measurement, and the default root is absent.
 */
package io.github.pho001.synaptik.backend.cpu.internal.cache;
