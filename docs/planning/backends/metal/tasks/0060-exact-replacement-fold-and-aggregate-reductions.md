# Task 0060: Exact Replacement, Non-overlap Fold, and Aggregate Reductions

## Status

Complete. Planning landed at `dd94e492`, implementation at `d06db07e`, focused native/public
Engine proof at `eda09533`, and backend/native documentation at `62f18cd8`. The final independent
cumulative Class C review of `dd94e492..62f18cd8` returned `APPROVE` with zero P0/P1/P2.

## Change class

Class C. This is one serial capability, native custom-program, structural MPSGraph, preflight,
execution, test, package, and documentation cutover. It changes no Model semantics, public API,
native ABI, export set, schema encoding, or backend identity encoding.

## Goal

Implement every currently authorized exact replacement, non-overlapping fold, and elementary
aggregate occurrence without atomics, data races, fallback, retry, autotuning, or timing. Preserve
schema 15, every backend-local identity at 16, native ABI 5, and exactly thirteen exports because
the current image already encodes every affected wire and attribute.

Production starts at exactly `61 true / 54 false`. The bounded target is exactly
`69 true / 46 false`: eight newly true operation kinds and no widening of an existing arithmetic
result set. Native structural execution starts at `79 executable / 36 nonexecutable` and targets
`87 / 28` by adding the eight non-ordering aggregate recipes at wires `106..108` and `111..115`.
The route catalog classification remains `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`; the custom
catalog advances from `38 AVAILABLE / 77 PENDING / 0 UNAVAILABLE_WITH_PROOF` to
`46 / 69 / 0`.

## Exact inventory and decision

| Wire | Model kind | Current structure | Production decision in this task |
|---:|---|---|---|
| 70 | `SCATTER_ADD` | executable direct MPSGraph recipe | Remains false. It is intrinsically additive, includes the base and every update, and is never a replacement operation. Unique indices do not remove the addition proof obligation. |
| 72 | `SCATTER_ND` | executable direct MPSGraph recipe | Admit only `ScatterReduction.NONE` with globally unique destination scalars proven from all tuples and suffix positions before any write. All six data carriers; canonical `INT32` or `INT64` indices; static canonical no-gradient descriptors. ADD/MUL/MIN/MAX remain false. |
| 76 | `SLICE_UPDATE` | executable direct MPSGraph recipe | Admit exact functional replacement for all six carriers over static canonical no-gradient base/update/output descriptors. Signed nonzero `SliceAttrs` and exact static crop-to-shape forms must prove every destination in range. Update extents remain positive; zero-length SliceAttrs are capability-false. |
| 80 | `FOLD_AXIS` | executable composed MPSGraph recipe | Admit only structurally non-overlapping geometry for `FLOAT64`, `FLOAT32`, and `BFLOAT16`. Initialize uncovered outputs to represented zero, then copy every unique contributor bit-exactly. Integral and BOOL carriers and any overlap remain false. |
| 82 | `FOLD2D` | executable direct MPSGraph recipe | Same exact three-floating-carrier non-overlap/copy/zero-fill domain for static NCHW columns and target. Padding and ceiling grids skip conceptual out-of-range contributors. Any contributor collision remains false. |
| 84 | `FOLD3D` | executable composed MPSGraph recipe | Same exact three-floating-carrier non-overlap/copy/zero-fill domain for static NCDHW columns and target. Padding and ceiling grids skip conceptual out-of-range contributors. Any contributor collision remains false. |
| 106 | `PROD` | registered direct recipe, not executable | Add direct structural MPSGraph execution. Production admits only canonical static no-gradient `INT32`/`INT64` full, single-axis, and ordered multi-axis forms, including empty-axis identity mapping and keep-dimensions. Custom execution uses exact-width modular multiplication in deterministic logical contributor order. Floating PROD remains false. |
| 107 | `ALL` | registered direct recipe, not executable | Add direct structural MPSGraph execution and exact canonical static no-gradient BOOL full/single/multi-axis custom execution. Empty-axis form is an exact copy; ordinary selected domains use the exact true identity. |
| 108 | `ANY` | registered direct recipe, not executable | Add direct structural MPSGraph execution and exact canonical static no-gradient BOOL full/single/multi-axis custom execution. Empty-axis form is an exact copy; ordinary selected domains use the exact false identity. |
| 109 | `ARG_MAX` | registered composed recipe, not executable | Deferred unchanged to the dedicated ordering task. INT64 output, first/last tie policy, NaN, signed-zero, and axis-order proof remain mandatory. |
| 110 | `ARG_MIN` | registered composed recipe, not executable | Deferred unchanged with `ARG_MAX`. |
| 111 | `LOG_SUM_EXP` | registered composed recipe, not executable | Add only package-private forceable structural MPSGraph composition. Production remains false pending exponent/log primitive-site proof and complete contributor/result-set validation. |
| 112 | `VARIANCE` | registered direct recipe, not executable | Add only package-private forceable structural MPSGraph execution. Production remains false pending correction/divisor, contributor, rounding, signed-zero, NaN, DAZ/FTZ, and gradient proof. |
| 113 | `STANDARD_DEVIATION` | registered composed recipe, not executable | Add only forceable variance-plus-square-root structure. Production remains false pending variance and square-root proofs. |
| 114 | `L1_NORM` | registered composed recipe, not executable | Add only forceable abs-plus-sum structure. Production remains false pending complete floating contributor/result-set proof. |
| 115 | `L2_NORM` | registered composed recipe, not executable | Add only forceable square-plus-sum-plus-square-root structure. Production remains false pending multiplication, accumulation, and square-root proof. |

Existing wire 18 `SCATTER_ELEMENTS` replacement remains a regression control: only
`ScatterReduction.NONE` with runtime global destination uniqueness is admitted. Its reduction
variants stay false. Pool wires `97..100` are explicitly deferred to the later pooling task.
Losses, normalization, convolution, attention, RNG, recurrent, and unrelated elementary arithmetic
remain out of scope.

## Gate A: capability and static geometry

1. Capability is occurrence-exact and common to both numerical profiles only where the admitted
   operation is byte-exact or integer/Boolean exact.
2. Every admitted occurrence is fully static, canonical, no-gradient, rank bounded by schema limits,
   and has checked positive dimensions and byte geometry. Rank-zero is admitted only where the
   operation contract permits it.
3. `SCATTER_ND/NONE` validates tuple depth, batch prefix, suffix mapping, index type, all bounds, and
   global scalar destination uniqueness. No output copy or update write begins before the complete
   index pass succeeds.
4. `SLICE_UPDATE` validates the complete signed coordinate mapping before execution. Positive update
   dimensions make the functional mapping injective; zero-length SliceAttrs fail capability rather
   than introducing zero-extent schema values.
5. Fold admission proves no two in-bounds contributors address the same destination. For axis fold,
   the exact condition includes `step >= windowSize`; image folds require every stride to be at
   least its effective dilated kernel. Padding and ceiling-grid contributors outside the target are
   skipped rather than rejected.
6. Zero dimensions, dynamic/unresolved/view/aliasing/out-of-span/malformed/overlapping geometry,
   scatter reduction modes, unsupported types, and gradients remain false. Empty aggregate axes
   remain valid point identities on positive-dimensional or rank-zero tensors.

## Gate B: deterministic custom execution

1. Add fixed custom kernels inside the existing shared custom-program route; no route wire or image
   encoding changes.
2. Replacement/fold kernels use no atomics. They zero/copy the complete canonical target and write
   only prevalidated globally unique destinations. Base, indices, updates, and fold columns remain
   unchanged; input/output aliasing fails before work.
3. Integer PROD assigns one output thread per result and traverses selected contributors in stable
   row-major logical order, using unsigned raw-word multiplication for exact `2^32`/`2^64` modular
   behavior. No signed-overflow semantics are relied upon.
4. BOOL ALL/ANY assign one output thread per result and traverse contributors in stable row-major
   order. Every input BOOL byte is validated before dispatch; malformed input causes no target
   mutation.
5. Repeated and concurrent runs retain isolated run-owned outputs/workspaces and identical results.
   There is no fallback, retry, runtime route selection, atomics, nondeterministic tree, or timing.

## Gate C: structural MPSGraph execution

1. Mark and implement exact bounded structural recipes for `PROD`, `ALL`, `ANY`, `LOG_SUM_EXP`,
   `VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM` using the installed-SDK selectors and
   finite compositions already frozen by Task 0056.
2. Structural forcing remains package-private and result-set-only. It grants no production
   capability, route choice, numerical approval, fallback, or public API.
3. Java and native validators independently enforce attribute kind, type, rank, axes, correction,
   shape, cardinality, state, and target freshness before MPSGraph compilation.
4. `ARG_MAX` and `ARG_MIN` remain nonexecutable until the ordering task owns their INT64 cast and
   complete tie/special-value proof.

## Gate D: proof and regressions

1. Add capability truth-table tests proving exactly `69/46`, the exact eight newly true kinds, and
   every blocked reduction/window control above.
2. Add raw schema/native malformed tests for reduction forms, axes, correction, scatter reduction,
   tuple geometry, slice mapping, fold geometry, type mismatches, and old/stale schema/identity
   rejection. Schema remains 15 and identities remain 16.
3. Real native tests cover every admitted carrier/type, signed zeros/NaNs as inert copied payloads,
   integer extremes and modular overflow, BOOL identities and malformed bytes, uncovered fold zeros,
   negative slice steps, suffix scatter updates, duplicate/out-of-range rejection before writes,
   canaries, input preservation, repeated runs, and concurrency.
4. Force every newly executable structural recipe through create/run/release and compare only with
   its Model result set; keep capability false for wires `111..115`.
5. Add public no-skip Engine tests for each of the eight newly true kinds, lifecycle/close behavior,
   repeated sessions, concurrency, exact publication, and failure atomicity. Reuse existing transfer
   contracts; add no fallback or alternate scheduling convention.
6. Rebuild/sign/package the native dylib and verify ABI 5, schema 15, exact thirteen exports, package
   manifest/checksums, and Gradle package consumption. Run focused Metal/Engine/CPU-contract and
   changed-module Javadocs. Per request, run no final full repository build.
7. Run independent cumulative Class C review after implementation and proof. Remediate every
   P0/P1/P2, repeat affected proof, and obtain final zero-finding approval before marking Complete.

## Completion contract

Completion requires one clean commit chain from `f3ad5e12`, exact counts and blocker documentation,
all affected call sites/tests/docs updated, no compatibility shim or stale alias, clean diff, and a
final independent cumulative Class C `APPROVE` with zero P0/P1/P2. Formal blockers are reported as
blockers, never hidden behind MPSGraph availability or a narrowed test.

## Completion evidence

- The complete Metal module test task passed against the rebuilt real dylib, including the exact
  `69/46` capability ledger, `87/28` structural registry, `75/35/5` MPSGraph catalog, and
  `46/69/0` custom catalog.
- Native proofs exercised all six replacement carrier widths and all three floating fold carriers,
  INT32 and INT64 Scatter-ND indices, signed SliceAttrs and crop placement, uncovered fold zeros,
  padding/ceiling/wide non-overlap geometry, modular INT32/INT64 PROD, canonical BOOL ALL/ANY,
  empty-axis and rank-zero identities, malformed BOOL, duplicate/out-of-range atomic rejection,
  alias rejection.
- The public no-skip Engine proof passed under both numerical profiles across repeated independent
  sessions, same-session concurrency, duplicate/out-of-range failure recovery, and exact
  publications for every newly admitted operation kind.
- The dylib was rebuilt, ad-hoc signed with the fixed identifier, packaged, independently verified,
  and consumed by the public Engine proof from the verified package path. Metal Javadocs passed.
- Per request, no final full repository build was run. Independent cumulative Class C review
  returned `APPROVE` with zero P0/P1/P2.
