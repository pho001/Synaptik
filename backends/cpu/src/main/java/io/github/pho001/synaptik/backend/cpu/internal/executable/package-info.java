/**
 * Owns unsupported immutable prepared-partition execution and direct cold binding.
 *
 * <p>The partition executable strongly owns its verified generated artifact and exact prepared
 * buffer selections. During cold binding it validates CPU representations, exact carrier pattern,
 * geometry, mutability, liveness, size, alignment, canonical BOOL bytes, and input/output
 * accessed-span overlap, then
 * captures direct array or segment references and primitive range/address state in one invocation.
 * Static movement recipes additionally retain compact immutable output/input layout geometry and
 * family mapping facts. Cold binding packs only the selected invocation range's coordinate and
 * address state; it retains no per-output-element selector or address table.
 * Indexing recipes retain similarly compact geometry plus one direct typed validator. Each run
 * validates the complete logical index domain on the invoking thread before every generated call
 * or worker submission. Invalid input therefore mutates no physical output bytes. Empty index
 * domains validate without a load; zero output still validates any non-empty index domain and,
 * after success, invokes no generated entry and submits no worker work.
 * Scatter recipes retain three semantic input roles over deduplicated input boundaries. Cold
 * binding validates all bounds before any replacement-target duplicate check and before generated
 * work, so failure leaves the output untouched. It then packs reusable coordinate state for
 * disjoint output ranges. Floating-product scatter also binds one exact declared workspace and
 * assigns a non-overlapping scratch slice to each selected range; other scatter rows bind no
 * workspace. Inputs remain borrowed and read-only, and output overlap with any input is rejected.
 * Fold recipes likewise reject physical input/output overlap before any generated call or worker
 * submission, then bind one invocation-private geometry array per disjoint output range. They
 * retain no shared accumulator, atomics, merge state, or hidden scratch.
 * Ordering recipes bind complete logical-axis slices, one exact run-owned workspace, and one
 * non-overlapping merge-scratch region per selected range. They reject every input/output and
 * TOP_K output/output overlap before scratch mutation, output writes, generated calls, or worker
 * submission. SORT copies represented values, ARGSORT writes logical-axis INT64 coordinates, and
 * one TOP_K invocation writes values and indices; unsorted TOP_K uses increasing original index.
 * For a selected materialization it validates the original source and exact contiguous workspace,
 * retains the original carrier for the copy, substitutes the workspace segment only in the
 * generated consumer pattern, and completes one canonical-order copy before any consumer chunk.
 * A parallel plan additionally borrows one caller-owned fixed platform-worker group. Cold binding
 * partitions a requested non-empty range into deterministic contiguous non-overlapping chunks;
 * zero chunks submit nothing, one chunk runs inline, and two or more chunks join synchronously.
 * Worker failure, interruption, racing close, nested submission, and group shutdown have explicit
 * deterministic contracts. Runtime invokes the bound object without graph semantics, route
 * selection, reflection, allocation, storage classification, or resource lookup in the generated
 * hot loop.
 *
 * <p>Random binding permits input/input overlap but rejects every input/output and output/output
 * overlap from complete spans before mutation or submission. It binds one generated
 * {@code [0,0)} state prologue followed by zero or more deterministic dropout element ranges;
 * initialization and empty dropout therefore still write their state result exactly once.</p>
 *
 * <p>Cumulative-scan binding validates the complete input and output carriers and rejects their
 * complete physical-span overlap before a generated call or worker submission. It packs one
 * invocation-private coordinate array per range and partitions only the independent slice
 * domain, so one slice and its sequential accumulator never cross worker boundaries.</p>
 *
 * <p>Aggregate binding validates complete input/output overlap, all logical canonical
 * Boolean input bytes, and any exact-state workspace's size, alignment, accessibility, and buffer
 * non-overlap before mutation. It packs invocation-private coordinates, assigns disjoint
 * workspace slices to floating numerical ranges, and partitions only complete output cells;
 * selected domains are never split or combined. Bound SUM-to-Shape uses the same lifecycle and
 * rejects overlap even when its no-reduction form performs a raw represented-bit copy.</p>
 *
 * <p>Arg-extrema binding independently validates complete numeric-input and INT64-output spans,
 * rejects physical overlap before mutation or submission, and packs invocation-private affine
 * geometry. Scalar and parallel-scalar execution partition only complete output cells; every
 * generated call traverses the full selected axis and stores one logical coordinate.</p>
 *
 * <p>Masked-reduction binding validates ordered data, canonical-BOOL mask, output, and exact-state
 * workspace spans before mutation or worker submission. Read-only data and mask may overlap, but
 * the output must not overlap either input and scratch must not overlap any buffer. Each scalar
 * or parallel-scalar range owns complete output cells and one private exact-state slice; selected
 * count remains invocation-local and no selected domain is split or combined.</p>
 *
 * <p>Runtime retains run-level lifecycle ownership; CPU memory representations retain physical
 * allocation and release ownership. Composition owns and closes the worker group; finalizers,
 * prepared executables, and bound invocations only borrow it.
 */
package io.github.pho001.synaptik.backend.cpu.internal.executable;
