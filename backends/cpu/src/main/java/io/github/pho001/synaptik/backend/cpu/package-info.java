/**
 * Supplies the sole supported CPU capability provider and its stable {@code cpu} identity.
 * The provider truthfully reports the implemented fully static pointwise and affine occurrence
 * matrices plus bounded one-node movement, window-extraction, and indexing matrices. Movement
 * covers all six Model data types; two-dimensional unfold is floating-only. Indexing covers
 * {@code GATHER}, {@code GATHER_ELEMENTS}, {@code GATHER_ND}, and {@code ONE_HOT} with INT32 or
 * INT64 indices, resolved layouts, an injective distinct output, and strict bounds validation.
 * The provider exposes no route, carrier, preparation, or execution API.
 *
 * <p>The {@code internal} namespace contains unsupported implementation contracts for complete-
 * partition lowering, code generation, storage, and execution. No type in that namespace is a
 * supported public API.</p>
 */
package io.github.pho001.synaptik.backend.cpu;
