/**
 * Owns the unsupported CPU analysis and post-assignment finalization lifecycle.
 *
 * <p>Analysis consumes one complete CPU-owned planned partition, forms computation units, retains
 * normalized access bindings and the selected ordered carrier pattern, selects scalar or
 * preferred-species vector compute plus single-thread or bounded parallel orchestration, and
 * declares each boundary's exact referenced storage span before shared assignment. Finalization
 * verifies those assignments and any required borrowed worker group, realizes the already-selected
 * scalar or vector artifact, and constructs one partition executable. It must not reinterpret
 * graph semantics, change fusion, route, strategy, or species selection, or introduce an
 * undeclared resource after shared Prepare has assigned slots.
 *
 * <p>All work in this package is cold-path work. Runtime collaborates only through the resulting
 * prepared executable and never receives the canonical kernel intermediate representation.
 */
package io.github.pho001.synaptik.backend.cpu.internal.prepare;
