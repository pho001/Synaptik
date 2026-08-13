/**
 * Supplies the sole supported CPU capability provider and its stable {@code cpu} identity.
 * The provider truthfully reports the implemented fully static pointwise and affine occurrence
 * matrices plus bounded one-node movement, window-extraction, indexing, functional scatter,
 * slice-update, overlap-fold, and stable ordering/selection matrices. Movement covers all six Model data types; two-dimensional unfold is
 * floating-only. Slice update accepts both current signed finite-coordinate and target-relative
 * forms, retains the base Shape, and functionally replaces selected positions without input
 * mutation. Indexing covers
 * {@code GATHER}, {@code GATHER_ELEMENTS}, {@code GATHER_ND}, and {@code ONE_HOT} with INT32 or
 * INT64 indices, resolved layouts, an injective distinct output, and strict bounds validation.
 * Fold covers numeric general-axis overlap addition and floating canonical NCHW two-dimensional
 * overlap addition with represented positive-zero initialization and padding exclusion.
 * Ordering covers stable SORT and ARGSORT plus two-output TOP_K for all six represented types,
 * with fixed NaN-last and signed-zero order, logical INT64 indices, represented-bit value copies,
 * deterministic unsorted selection, and static resolved-layout output requirements.
 * The provider exposes no route, carrier, preparation, or execution API.
 *
 * <p>The {@code internal} namespace contains unsupported implementation contracts for complete-
 * partition lowering, code generation, storage, and execution. No type in that namespace is a
 * supported public API.</p>
 */
package io.github.pho001.synaptik.backend.cpu;
