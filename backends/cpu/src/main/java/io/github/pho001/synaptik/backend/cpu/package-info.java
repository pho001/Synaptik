/**
 * Supplies the sole supported CPU capability provider and its stable {@code cpu} identity.
 * The provider truthfully reports the implemented fully static pointwise and affine occurrence
 * matrices plus bounded one-node movement, window-extraction, indexing, functional scatter,
 * slice-update, overlap-fold, stable ordering/selection, explicit-state random, and cumulative-
 * scan matrices. Movement covers all six Model data types; two-dimensional unfold is
 * floating-only. Slice update accepts both current signed finite-coordinate and target-relative
 * forms, retains the base Shape, and functionally replaces selected positions without input
 * mutation. Indexing covers
 * {@code GATHER}, {@code GATHER_ELEMENTS}, {@code GATHER_ND}, and {@code ONE_HOT} with INT32 or
 * INT64 indices, resolved layouts, an injective distinct output, and strict bounds validation.
 * Fold covers numeric general-axis overlap addition and floating canonical NCHW two-dimensional
 * overlap addition with represented positive-zero initialization and padding exclusion.
 * Ordering covers stable SORT and ARGSORT plus two-output TOP_K for all six represented types,
 * with fixed NaN-last and signed-zero order, logical INT64 indices, represented-bit value copies,
 * deterministic unsorted selection, and static resolved-layout output requirements. Explicit
 * graph random coverage consists of a zero-input raw two-word initializer and FLOAT64/FLOAT32
 * inverted dropout with ordered value/state inputs and value/BOOL-mask/next-state outputs. The
 * generated scalar route uses the CPU-private V1 mapping, exact binary64 threshold/scaling,
 * modulo advancement, and no workspace; BFLOAT16 remains unsupported.
 * Cumulative scan covers one static resolved-layout {@code CUM_SUM} or {@code CUM_PROD}
 * occurrence across FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64 in every inclusive/exclusive
 * and forward/reverse mode. It keeps sequential typed accumulation within a logical slice,
 * parallelizes only across whole slices, and declares no workspace or materialization.
 * Ordinary aggregate coverage adds exact full, single-axis, and multi-axis SUM/PROD/MIN/MAX over
 * five numeric represented types, floating-only MEAN, and ALL/ANY over canonical BOOL. It
 * parallelizes only across whole output cells. Floating SUM/MEAN/PROD declare run-owned exact-state
 * workspace; all other rows are workspace-free, and no row uses partial/combine state or
 * materialization. Binding-aware SUM-to-Shape additionally accepts one fully bound static
 * right-aligned target over the five numeric types. It reduces leading and unequal target-one
 * axes, copies represented bits when no axis reduces, and retains the same output-cell ownership
 * and exact floating or modular integral SUM semantics.
 * Arg-extrema coverage accepts exactly one fully static resolved-layout {@code ARG_MIN} or
 * {@code ARG_MAX} occurrence over the five numeric types, producing logical selected-axis
 * coordinates as INT64 with first- or last-index tie selection, NaN preference, and signed-zero
 * ordering. It owns complete output cells and uses no workspace or materialization.
 * Masked-reduction coverage accepts exactly one fully static resolved-layout, axis-removing
 * {@code SUM} or {@code MEAN} over FLOAT64, FLOAT32, or BFLOAT16 data and a canonical BOOL mask.
 * The mask broadcasts directionally and right-aligned exactly to the data Shape; false positions
 * are excluded before data loading or classification. Scalar or parallel-scalar execution owns
 * complete output cells, keeps an exact selected count, and uses one run-owned exact-state slice
 * per simultaneously used range without mask materialization, partial state, or combination.
 * Advanced-reduction coverage accepts exactly one fully static resolved-layout LOG_SUM_EXP,
 * VARIANCE, STANDARD_DEVIATION, L1_NORM, or L2_NORM occurrence over FLOAT64, FLOAT32, or
 * BFLOAT16. It preserves ordered axes and complete output-cell ownership; only L1 and statistics
 * use the existing exact-state workspace, while log-sum-exp and L2 remain workspace-free.
 * Softmax coverage accepts exactly one first-class {@code SOFTMAX} or {@code LOG_SOFTMAX}
 * occurrence over FLOAT64, FLOAT32, or BFLOAT16 with a positive selected extent. It preserves
 * Shape and axis identity, validates the complete finite input and maximum-shift domain before
 * mutation or worker submission, and executes direct stable three-pass complete-slice ranges
 * without workspace or materialization.
 * Trailing Layer/RMS normalization and arbitrary-axis batch-normalization inference and training
 * are separate first-class static families. Batch training preserves five explicit inputs and
 * outputs, assigns complete channels to scalar or parallel-scalar ranges, uses one exact-state
 * slice per active range, and publishes next and saved statistics without hidden state.
 * Pointwise CAST covers all 36 ordered pairs among FLOAT64, FLOAT32, BFLOAT16, INT64, INT32, and
 * BOOL through generated scalar or caller-parallel scalar execution. It follows the Model's exact
 * represented-bit, rounding, NaN, saturation, narrowing, and canonical-BOOL rules over arrays,
 * native-order segments, and mixed carriers. A cross-type CAST keeps its explicit boundary inside
 * a bounded pointwise directed acyclic graph and disables vector compute for that unit; same-type
 * vector eligibility is unchanged. The provider exposes no route, carrier, preparation, or
 * execution API.
 *
 * <p>The {@code internal} namespace contains unsupported implementation contracts for complete-
 * partition lowering, code generation, storage, and execution. No type in that namespace is a
 * supported public API.</p>
 */
package io.github.pho001.synaptik.backend.cpu;
