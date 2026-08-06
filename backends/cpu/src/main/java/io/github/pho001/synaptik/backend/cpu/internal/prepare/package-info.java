/**
 * Owns the unsupported CPU analysis and post-assignment finalization lifecycle.
 *
 * <p>Analysis consumes one complete CPU-owned planned partition, forms computation units, selects
 * the portable scalar strategy, and declares exact boundary resources before shared assignment.
 * Finalization verifies those assignments, realizes the already-selected artifact, and constructs
 * one partition executable. It must not reinterpret graph semantics, change fusion or route
 * selection, or introduce an undeclared resource after shared Prepare has assigned slots.
 *
 * <p>All work in this package is cold-path work. Runtime collaborates only through the resulting
 * prepared executable and never receives the canonical kernel intermediate representation.
 */
package io.github.pho001.synaptik.backend.cpu.internal.prepare;
