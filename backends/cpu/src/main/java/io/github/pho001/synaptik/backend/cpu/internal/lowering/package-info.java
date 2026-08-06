/**
 * Owns unsupported complete-partition unit formation, fusion, and canonical lowering.
 *
 * <p>Lowering consumes the semantic and logical-memory facts projected by shared Prepare. It
 * derives ADD and MUL results with Model {@code ShapeBroadcast}, requires exact GELU shape,
 * normalizes resolved layouts into five access regimes, proves output-write injectivity, and
 * decides fusion legality before resource declaration. Private intermediates remain virtual
 * canonical-IR values. It collaborates with {@code internal.ir} and route-neutral CPU
 * preparation, but never allocates Runtime resources, selects a physical slot, or delegates graph
 * interpretation to a route implementation.
 *
 * <p>Lowering runs on the preparation cold path; no lowering object or Model operation reaches the
 * generated execution loop.
 */
package io.github.pho001.synaptik.backend.cpu.internal.lowering;
