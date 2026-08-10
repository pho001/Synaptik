/**
 * Owns unsupported complete-partition unit formation, fusion, affine composition, and canonical
 * lowering.
 *
 * <p>Lowering consumes the semantic and logical-memory facts projected by shared Prepare. It
 * derives supported binary arithmetic results with Model {@code ShapeBroadcast}, preserves
 * Shape for scalar arithmetic, clamp, and power, preserves canonical BOOL logic, classifies exact
 * FLOAT32/FLOAT64 scalar exponents into one proved realization or the direct fallback, retains
 * Tensor/Tensor power as a direct instruction, maps all nineteen same-typed FLOAT32/FLOAT64 unary
 * kinds to one instruction each, keeps floating classification separate,
 * normalizes resolved layouts into five access regimes, proves output-write injectivity, and
 * decides fusion legality before resource declaration. Private intermediates, including eligible
 * comparison/classification BOOL results consumed inside the unit, remain virtual canonical-IR
 * values. It collaborates with {@code internal.ir} and route-neutral CPU
 * preparation, but never allocates Runtime resources, selects a physical slot, or delegates graph
 * interpretation to a route implementation.
 *
 * <p>The bounded affine family accepts only one-through-eight connected one-input/one-output
 * static resolved-layout view occurrences. It composes their coordinate mappings on the cold
 * path, keeps eligible same-unit intermediates virtual, and derives one deterministic source-to-
 * result address table. A zero-stride result uses one write per distinct address only when all
 * repeated logical coordinates select the same represented source value. Final boundary
 * materialization is always explicit because shared preparation provides no cross-value aliasing.
 *
 * <p>The bounded non-affine movement family accepts exactly one fully static resolved-layout
 * PAD, TILE, CONCAT, or STACK occurrence. It retains semantic input occurrence order while
 * declaring each distinct input once, requires one distinct injective output, and lowers exact
 * extents, offsets, strides, axes, padding widths, repeats, and composition prefixes into compact
 * cold geometry rather than a per-output-element table.</p>
 *
 * <p>The selected {@code CpuMaterializationPlan} is a separate route-independent copy fact. It
 * retains original source geometry and canonical dense consumer geometry without changing the
 * Model graph, backend-neutral logical memory, or boundary {@code ValueId}. At most one selected
 * input receives CPU-private workspace ID {@code 0}.
 *
 * <p>Lowering runs on the preparation cold path; no lowering object or Model operation reaches the
 * generated execution loop.
 */
package io.github.pho001.synaptik.backend.cpu.internal.lowering;
