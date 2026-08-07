/**
 * Owns unsupported complete-partition unit formation, fusion, and canonical lowering.
 *
 * <p>Lowering consumes the semantic and logical-memory facts projected by shared Prepare. It
 * derives supported binary arithmetic results with Model {@code ShapeBroadcast}, preserves
 * Shape for scalar arithmetic and power, classifies exact FLOAT32/FLOAT64 scalar exponents into
 * one proved realization or the direct fallback, requires exact GELU shape,
 * normalizes resolved layouts into five access regimes, proves output-write injectivity, and
 * decides fusion legality before resource declaration. Private intermediates remain virtual
 * canonical-IR values. It collaborates with {@code internal.ir} and route-neutral CPU
 * preparation, but never allocates Runtime resources, selects a physical slot, or delegates graph
 * interpretation to a route implementation.
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
