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
 * PAD, TILE, CONCAT, STACK, window-extraction, or functional SLICE_UPDATE occurrence. It retains
 * semantic input occurrence order while declaring each distinct input once, requires one distinct
 * injective output, and lowers exact extents, offsets, strides, axes, padding widths, repeats,
 * composition prefixes, and signed slice placement into compact cold geometry rather than a
 * per-output-element table. Slice update normalizes both current attribute forms to one
 * rank-sized start/length/step mapping and leaves base and update inputs unchanged.</p>

 * <p>The indexing family accepts exactly one fully static resolved-layout GATHER,
 * GATHER_ELEMENTS, GATHER_ND, or ONE_HOT occurrence. It preserves semantic input occurrence
 * order while declaring each exact input value once, derives compact layout and family geometry,
 * and requires one distinct injective output. It creates no workspace or per-index/per-output
 * table; execution-time values remain unavailable until direct cold binding.</p>
 *
 * <p>The functional-scatter family accepts exactly one fully static resolved-layout
 * SCATTER_ELEMENTS, Gather-compatible SCATTER_ADD, or SCATTER_ND occurrence with ordered
 * {@code [data, indices, updates]} roles. It preserves those roles while deduplicating exact input
 * values, derives compact coordinate geometry, requires one distinct injective data-shaped
 * output, and declares no materialization. Only floating {@code MUL} with a non-empty possible
 * calculation derives one checked per-range exact-product scratch slice; every other row has no
 * workspace.</p>

 * <p>The overlap-fold family accepts exactly one fully static resolved-layout FOLD_AXIS or
 * FOLD2D occurrence. It derives checked compact general-axis or canonical NCHW geometry, retains
 * logical input repetition from zero strides, proves a distinct injective output, and selects no
 * materialization or workspace.</p>

 * <p>The stable ordering family accepts exactly one fully static resolved-layout SORT, ARGSORT,
 * or TOP_K occurrence. It revalidates Model Shape/output-slot roles, derives complete independent
 * logical-axis slices, proves one or two distinct injective outputs, and sizes two INT64 merge-
 * index regions for each selected execution range. Axis, K, offsets, strides, and slice geometry
 * remain cold; value comparison and scratch allocation occur later.</p>
 *
 * <p>The selected {@code CpuMaterializationPlan} is a separate route-independent copy fact. It
 * retains original source geometry and canonical dense consumer geometry without changing the
 * Model graph, backend-neutral logical memory, or boundary {@code ValueId}. At most one selected
 * input receives CPU-private workspace ID {@code 0}. Scatter product scratch uses the same
 * analysis-local ID only in a mutually exclusive scatter plan. Stable ordering uses that same
 * ID in its mutually exclusive exact run-owned merge-scratch plan.
 *
 * <p>Random lowering admits exactly one static resolved-layout initializer or FLOAT64/FLOAT32
 * dropout occurrence. It preserves initializer output or
 * {@code [value,state,output,keepMask,nextState]} boundary order, derives complete output spans
 * and logical draw geometry, and declares no workspace or mutable generator state.</p>
 *
 * <p>Cumulative-scan lowering admits exactly one static resolved-layout CUM_SUM or CUM_PROD
 * occurrence across FLOAT64, FLOAT32, BFLOAT16, INT32, and INT64. It derives the independent
 * non-axis slice domain, retains exact input/output layout geometry, requires a distinct
 * injective output, and declares exactly those two buffers with no workspace, materialization,
 * partial scan, or combine state.</p>
 *
 * <p>Lowering runs on the preparation cold path; no lowering object or Model operation reaches the
 * generated execution loop.
 */
package io.github.pho001.synaptik.backend.cpu.internal.lowering;
