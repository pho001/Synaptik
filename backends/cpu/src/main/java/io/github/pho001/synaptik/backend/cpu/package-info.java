/**
 * Supplies the sole supported CPU capability provider and its stable {@code cpu} identity.
 * The provider truthfully reports the implemented fully static pointwise and affine occurrence
 * matrices plus the bounded one-node {@code PAD}, {@code TILE}, {@code CONCAT}, and {@code STACK}
 * represented-bit movement matrix. Movement covers all six Model data types, requires resolved
 * layouts and an injective output, and bounds composition to sixteen semantic input occurrences.
 * The provider exposes no route, carrier, preparation, or execution API.
 *
 * <p>The {@code internal} namespace contains unsupported implementation contracts for complete-
 * partition lowering, code generation, storage, and execution. No type in that namespace is a
 * supported public API.</p>
 */
package io.github.pho001.synaptik.backend.cpu;
