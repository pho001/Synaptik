# Task 0060: Exact Replacement, Non-overlap Fold, and Aggregate Reductions

## Reader context

Replacement scatter and slice update construct a new target from base/update inputs; they do not mutate the application's base storage. Complete bounds and uniqueness checks precede writes. Non-overlap fold can copy into distinct destinations, whereas overlap and ScatterAdd require arithmetic and a different proof.

Integer PROD and Boolean ALL/ANY have exact carrier rules rather than floating tolerances. The custom program owns the recipe; Runtime owns fresh targets. Recorded blocked arithmetic/fold domains and totals describe this checkpoint; later overlap and aggregate routes are separate qualifications.

- [Current indexing and failure-before-mutation rules](../../../../backend-guide/metal-backend.md#physical-authentication-and-fixed-routes) — Distinguishes replacement checks from arithmetic collisions.
- [Current fold and aggregate coverage](../../../../backend-guide/metal-backend.md#ordinary-numerical-and-exact-domains) — Locates later overlap support without changing old gates.

## Status

Complete. [Planning](../../../../glossary.md#term-planning) landed at `dd94e492`, implementation at `d06db07e`, focused native/public
[Engine](../../../../glossary.md#term-engine) proof at `eda09533`, [backend](../../../../glossary.md#backend)/native documentation at `62f18cd8`, and final fold-domain/count
remediation at `a4fe4754`. Independent cumulative re-review of `55cdebc3..a4fe4754` returned
`APPROVE` with zero P0/P1/P2.

## Change class

Class C. This is one serial capability, native custom-program, structural [MPSGraph](../../../../glossary.md#term-mpsgraph), [preflight](../../../../glossary.md#term-preflight),
execution, test, package, and documentation cutover. It changes no [Model](../../../../glossary.md#term-model) semantics, public [API](../../../../glossary.md#term-application-programming-interface-api),
native [ABI](../../../../glossary.md#term-application-binary-interface-abi), export set, [schema](../../../../glossary.md#term-schema) encoding, or [backend identity](../../../../glossary.md#backend-identity--backendid) encoding.

## Goal

Implement every currently authorized exact replacement, non-overlapping fold, and elementary
aggregate [occurrence](../../../../glossary.md#term-occurrence) without atomics, data races, fallback, retry, [autotuning](../../../../glossary.md#autotuning--model-autotuning), or timing. Preserve
[schema](../../../../glossary.md#term-schema) 15, every [backend](../../../../glossary.md#backend)-local [identity](../../../../glossary.md#term-identity-meanings) at 16, native [ABI](../../../../glossary.md#term-application-binary-interface-abi) 5, and exactly thirteen exports because
the current image already encodes every affected wire and attribute.

Production starts at exactly `61 true / 54 false`. The bounded [target](../../../../glossary.md#term-target) is exactly
`69 true / 46 false`: eight newly true [operation](../../../../glossary.md#operation) kinds and no [widening](../../../../glossary.md#term-widening) of an existing arithmetic
result set. Native structural execution starts at `79 executable / 36 nonexecutable` and targets
`87 / 28` by adding the eight non-ordering aggregate recipes at wires `106..108` and `111..115`.
The route catalog classification remains `75 DIRECT / 35 COMPOSED / 5 UNAVAILABLE`; the custom
catalog advances from `38 AVAILABLE / 77 PENDING / 0 UNAVAILABLE_WITH_PROOF` to
`46 / 69 / 0`.

## Exact inventory and decision

| Wire | [Model](../../../../glossary.md#term-model) kind | Current structure | Production decision in this task |
|---:|---|---|---|
| 70 | `SCATTER_ADD` | executable direct [MPSGraph](../../../../glossary.md#term-mpsgraph) recipe | Remains false. It is intrinsically additive, includes the base and every update, and is never a replacement [operation](../../../../glossary.md#operation). Unique indices do not remove the addition proof obligation. |
| 72 | `SCATTER_ND` | executable direct MPSGraph recipe | Admit only `ScatterReduction.NONE` with globally unique destination [scalars](../../../../glossary.md#term-scalar) proven from all tuples and suffix positions before any write. All six data carriers; [canonical](../../../../glossary.md#term-canonical-form) [`INT32`](../../../../glossary.md#data-type--datatype) or [`INT64`](../../../../glossary.md#data-type--datatype) indices; static canonical no-[gradient](../../../../glossary.md#term-gradient) [descriptors](../../../../glossary.md#term-descriptor). ADD/MUL/MIN/MAX remain false. |
| 76 | `SLICE_UPDATE` | executable direct MPSGraph recipe | Admit exact functional replacement for all six carriers over static canonical no-gradient base/update/output descriptors. Signed nonzero `SliceAttrs` and exact static crop-to-[shape](../../../../glossary.md#shape) forms must prove every destination in range. Update extents remain positive; zero-length SliceAttrs are capability-false. |
| 80 | `FOLD_AXIS` | executable composed MPSGraph recipe | Admit only structurally non-overlapping geometry for [`FLOAT64`](../../../../glossary.md#term-float64), [`FLOAT32`](../../../../glossary.md#term-float32), and [`BFLOAT16`](../../../../glossary.md#term-bfloat16). Initialize uncovered outputs to represented zero, then copy every unique contributor bit-exactly. Integral and [BOOL](../../../../glossary.md#data-type--datatype) carriers and any overlap remain false. |
| 82 | `FOLD2D` | executable direct MPSGraph recipe | Same exact three-floating-[carrier](../../../../glossary.md#term-storage-carrier) non-overlap/copy/zero-fill domain for static [NCHW](../../../../glossary.md#term-nchw) columns and [target](../../../../glossary.md#term-target). [Padding](../../../../glossary.md#padding) and ceiling grids skip conceptual out-of-range contributors. Any contributor collision remains false. |
| 84 | `FOLD3D` | executable composed MPSGraph recipe | Same exact three-floating-carrier non-overlap/copy/zero-fill domain for static [NCDHW](../../../../glossary.md#conv3d--ncdhw) columns and target. Padding and ceiling grids skip conceptual out-of-range contributors. Any contributor collision remains false. |
| 106 | `PROD` | registered direct recipe, not executable | Add direct structural MPSGraph execution. Production admits only canonical static no-gradient `INT32`/`INT64` full, single-[axis](../../../../glossary.md#term-axis), and ordered multi-axis forms, including empty-axis [identity](../../../../glossary.md#term-identity-meanings) mapping and keep-[dimensions](../../../../glossary.md#dimension). Custom execution uses exact-width modular multiplication in deterministic logical contributor order. Floating PROD remains false. |
| 107 | `ALL` | registered direct recipe, not executable | Add direct structural MPSGraph execution and exact canonical static no-gradient BOOL full/single/multi-axis custom execution. Empty-axis form is an exact copy; ordinary selected domains use the exact true identity. |
| 108 | `ANY` | registered direct recipe, not executable | Add direct structural MPSGraph execution and exact canonical static no-gradient BOOL full/single/multi-axis custom execution. Empty-axis form is an exact copy; ordinary selected domains use the exact false identity. |
| 109 | `ARG_MAX` | registered composed recipe, not executable | Deferred unchanged to the dedicated ordering task. INT64 output, first/last tie policy, [NaN](../../../../glossary.md#term-nan), [signed-zero](../../../../glossary.md#nan-infinity-and-signed-zero), and axis-order proof remain mandatory. |
| 110 | `ARG_MIN` | registered composed recipe, not executable | Deferred unchanged with `ARG_MAX`. |
| 111 | `LOG_SUM_EXP` | registered composed recipe, not executable | Add only package-private forceable structural MPSGraph composition. Production remains false pending [exponent](../../../../glossary.md#term-exponent)/[log](../../../../glossary.md#term-log) primitive-site proof and complete contributor/result-set [validation](../../../../glossary.md#term-validation). |
| 112 | [`VARIANCE`](../../../../glossary.md#term-variance) | registered direct recipe, not executable | Add only package-private forceable structural MPSGraph execution. Production remains false pending correction/divisor, contributor, [rounding](../../../../glossary.md#term-rounding), signed-zero, NaN, [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz), and gradient proof. |
| 113 | `STANDARD_DEVIATION` | registered composed recipe, not executable | Add only forceable variance-plus-square-root structure. Production remains false pending variance and square-root proofs. |
| 114 | `L1_NORM` | registered composed recipe, not executable | Add only forceable abs-plus-sum structure. Production remains false pending complete floating contributor/result-set proof. |
| 115 | `L2_NORM` | registered composed recipe, not executable | Add only forceable square-plus-sum-plus-square-root structure. Production remains false pending multiplication, accumulation, and square-root proof. |

Existing wire 18 `SCATTER_ELEMENTS` replacement remains a [regression](../../../../glossary.md#term-regression) control: only
`ScatterReduction.NONE` with [runtime](../../../../glossary.md#term-runtime) global destination uniqueness is admitted. Its reduction
variants stay false. Pool wires `97..100` are explicitly deferred to the later [pooling](../../../../glossary.md#term-pooling) task.
[Losses](../../../../glossary.md#loss), [normalization](../../../../glossary.md#term-normalization), [convolution](../../../../glossary.md#term-channels-first-convolution-layers), attention, [RNG](../../../../glossary.md#graph-rng-state--graphrngstate), recurrent, and unrelated elementary arithmetic
remain out of [scope](../../../../glossary.md#term-scope).

## Gate A: capability and static geometry

1. Capability is occurrence-exact and common to both numerical profiles only where the admitted
   operation is byte-exact or integer/Boolean exact.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [operation glossary](../../../../glossary.md#operation); [Boolean glossary](../../../../glossary.md#data-type--datatype))</span>
2. Every admitted occurrence is fully static, canonical, no-gradient, rank bounded by schema limits,
   and has checked positive dimensions and byte geometry. Rank-zero is admitted only where the
   operation contract permits it.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [gradient glossary](../../../../glossary.md#term-gradient); [rank glossary](../../../../glossary.md#term-rank); [schema glossary](../../../../glossary.md#term-schema); [dimensions glossary](../../../../glossary.md#dimension); [Rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
3. `SCATTER_ND/NONE` validates tuple depth, batch prefix, suffix mapping, index type, all bounds, and
   global scalar destination uniqueness. No output copy or update write begins before the complete
   index pass succeeds.<span data-glossary-navigation="true"> ([tuple depth glossary](../../../../glossary.md#term-tuple-depth); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [scalar glossary](../../../../glossary.md#term-scalar))</span>
4. `SLICE_UPDATE` validates the complete signed coordinate mapping before execution. Positive update
   dimensions make the functional mapping injective; zero-length SliceAttrs fail capability rather
   than introducing zero-extent schema values.<span data-glossary-navigation="true"> ([coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [injective glossary](../../../../glossary.md#term-injective); [extent glossary](../../../../glossary.md#term-extent))</span>
5. Fold admission proves no two in-bounds contributors address the same destination. For axis fold,
   the exact condition includes <span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [axis glossary](../../../../glossary.md#term-axis))</span>`step >= windowSize`; image folds require every stride to be at
   least its effective dilated kernel. Padding and ceiling-grid contributors outside the target are
   skipped rather than rejected.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [kernel glossary](../../../../glossary.md#kernel); [Padding glossary](../../../../glossary.md#padding); [target glossary](../../../../glossary.md#term-target))</span>
6. Zero dimensions, dynamic/unresolved/view/aliasing/out-of-span/malformed/overlapping geometry,
   scatter reduction modes, unsupported types, and gradients remain false. Empty aggregate axes
   remain valid point identities on positive-dimensional or rank-zero tensors.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([view glossary](../../../../glossary.md#term-view-meanings); [aliasing glossary](../../../../glossary.md#term-aliasing); [tensors glossary](../../../../glossary.md#tensor))</span>

## Gate B: deterministic custom execution

1. Add fixed custom kernels inside the existing shared custom-program route; no route wire or image
   encoding changes.<span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel))</span>
2. Replacement/fold kernels use no atomics. They zero/copy the complete canonical target and write
   only prevalidated globally unique destinations. Base, indices, updates, and fold columns remain
   unchanged; input/output aliasing fails before work.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [target glossary](../../../../glossary.md#term-target); [aliasing glossary](../../../../glossary.md#term-aliasing))</span>
3. Integer PROD assigns one output thread per result and traverses selected contributors in stable
   row-major logical order, using unsigned raw-word multiplication for exact <span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major))</span>`2^32`/`2^64` modular
   behavior. No signed-overflow semantics are relied upon.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow))</span>
4. BOOL ALL/ANY assign one output thread per result and traverse contributors in stable row-major
   order. Every input BOOL byte is validated before dispatch; malformed input causes no target
   mutation.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
5. Repeated and concurrent runs retain isolated run-owned outputs/workspaces and identical results.
   There is no fallback, retry, runtime route selection, atomics, nondeterministic tree, or timing.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## Gate C: structural [MPSGraph](../../../../glossary.md#term-mpsgraph) execution

1. Mark and implement exact bounded structural recipes for `PROD`, `ALL`, `ANY`, `LOG_SUM_EXP`,
   [`VARIANCE`](../../../../glossary.md#term-variance), `STANDARD_DEVIATION`, `L1_NORM`, and `L2_NORM` using the installed-SDK selectors and
   finite compositions already frozen by Task 0056.<span data-glossary-navigation="true"> ([SDK glossary](../../../../glossary.md#term-sdk); [finite glossary](../../../../glossary.md#term-finite))</span>
2. Structural forcing remains package-private and result-set-only. It grants no production
   capability, route choice, numerical approval, fallback, or public API.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
3. Java and native validators independently enforce attribute kind, type, rank, axes, correction,
   shape, cardinality, state, and target freshness before MPSGraph compilation.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [axes glossary](../../../../glossary.md#term-axis); [shape glossary](../../../../glossary.md#shape); [cardinality glossary](../../../../glossary.md#term-cardinality); [target glossary](../../../../glossary.md#term-target); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names))</span>
4. `ARG_MAX` and `ARG_MIN` remain nonexecutable until the ordering task owns their INT64 cast and
   complete tie/special-value proof.<span data-glossary-navigation="true"> ([INT64 glossary](../../../../glossary.md#data-type--datatype))</span>

## Gate D: proof and regressions

1. Add capability truth-table tests proving exactly `69/46`, the exact eight newly true kinds, and
   every blocked reduction/window control above.
2. Add raw schema/native malformed tests for reduction forms, axes, correction, scatter reduction,
   tuple geometry, slice mapping, fold geometry, type mismatches, and old/stale schema/identity
   rejection. Schema remains 15 and identities remain 16.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [axes glossary](../../../../glossary.md#term-axis); [slice glossary](../../../../glossary.md#slice); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
3. Real native tests cover every admitted carrier/type, signed zeros/NaNs as inert copied payloads,
   integer extremes and modular overflow, BOOL identities and malformed bytes, uncovered fold zeros,
   negative slice steps, suffix scatter updates, duplicate/out-of-range rejection before writes,
   canaries, input preservation, repeated runs, and concurrency.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [overflow glossary](../../../../glossary.md#term-overflow); [BOOL glossary](../../../../glossary.md#data-type--datatype); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
4. Force every newly executable structural recipe through create/run/release and compare only with
   its Model result set; keep capability false for wires <span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [release glossary](../../../../glossary.md#term-release); [Model glossary](../../../../glossary.md#term-model))</span>`111..115`.
5. Add public no-skip Engine tests for each of the eight newly true kinds, lifecycle/close behavior,
   repeated sessions, concurrency, exact publication, and failure atomicity. Reuse existing transfer
   contracts; add no fallback or alternate scheduling convention.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [publication glossary](../../../../glossary.md#term-publication); [failure atomicity glossary](../../../../glossary.md#term-failure-atomicity))</span>
6. Rebuild/sign/package the native dylib and verify ABI 5, schema 15, exact thirteen exports, package
   manifest/checksums, and Gradle package consumption. Run focused Metal/Engine/CPU-contract and
   changed-module Javadocs. Per request, run no final full repository build.<span data-glossary-navigation="true"> ([dylib glossary](../../../../glossary.md#term-dylib); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [manifest glossary](../../../../glossary.md#term-manifest); [Gradle glossary](../../../../glossary.md#term-gradle); [Metal glossary](../../../../glossary.md#term-metal); [CPU glossary](../../../../glossary.md#term-cpu); [module glossary](../../../../glossary.md#term-module))</span>
7. Run independent cumulative Class C review after implementation and proof. Remediate every
   P0/P1/P2, repeat affected proof, and obtain final zero-finding approval before marking Complete.

## Completion contract

Completion requires one clean [commit](../../../../glossary.md#term-commit) chain from `f3ad5e12`, exact counts and blocker documentation,
all affected call sites/tests/docs updated, no compatibility shim or stale [alias](../../../../glossary.md#aliasing-and-arena), clean diff, and a
final independent cumulative Class C `APPROVE` with zero P0/P1/P2. Formal blockers are reported as
blockers, never hidden behind [MPSGraph](../../../../glossary.md#term-mpsgraph) availability or a narrowed test.

## Completion evidence

- The complete [Metal](../../../../glossary.md#term-metal) [module](../../../../glossary.md#term-module) test task passed against the rebuilt real [dylib](../../../../glossary.md#term-dylib), including the exact
  `69/46` [capability ledger](../../../../glossary.md#capability-ledger), `87/28` structural [registry](../../../../glossary.md#term-registry), `75/35/5` [MPSGraph](../../../../glossary.md#term-mpsgraph) catalog, and
  `46/69/0` custom catalog.
- Native proofs exercised all six replacement carrier widths and all three floating fold carriers,
  INT32 and INT64 Scatter-ND indices, signed SliceAttrs and crop placement, uncovered fold zeros,
  padding/ceiling/wide non-overlap geometry, modular INT32/INT64 PROD, canonical BOOL ALL/ANY,
  empty-axis and rank-zero identities, malformed BOOL, duplicate/out-of-range atomic rejection,
  alias rejection.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [INT32 glossary](../../../../glossary.md#data-type--datatype); [INT64 glossary](../../../../glossary.md#data-type--datatype); [Scatter-ND glossary](../../../../glossary.md#scatter-nd); [padding glossary](../../../../glossary.md#padding); [canonical glossary](../../../../glossary.md#term-canonical-form); [BOOL glossary](../../../../glossary.md#data-type--datatype); [axis glossary](../../../../glossary.md#term-axis); [rank-zero glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [alias glossary](../../../../glossary.md#aliasing-and-arena))</span>
- The public no-skip Engine proof passed under both numerical profiles across repeated independent
  sessions, same-session concurrency, duplicate/out-of-range failure recovery, and exact
  publications for every newly admitted operation kind.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [concurrency glossary](../../../../glossary.md#term-concurrency); [operation glossary](../../../../glossary.md#operation))</span>
- The dylib was rebuilt, ad-hoc signed with the fixed identifier, packaged, independently verified,
  and consumed by the public Engine proof from the verified package path. Metal Javadocs passed.
- Per request, no final full repository build was [run](../../../../glossary.md#term-run). Independent cumulative re-review of
  `55cdebc3..a4fe4754` returned `APPROVE` with zero P0/P1/P2.
