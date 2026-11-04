# Task 0010: Narrow [OpenBLAS](../../../../glossary.md#term-openblas) [BLAS](../../../../glossary.md#term-blas)-Compatible Native Route

## Status

Complete

## Goal

Add the first [CPU](../../../../glossary.md#term-cpu)-native peer route at
`io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas`. The route consumes the
existing common [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [lowering](../../../../glossary.md#lowering) and invokes the completed low-level [OpenBLAS](../../../../glossary.md#term-openblas) provider only for one
small, exactly defined dense [SGEMM](../../../../glossary.md#general-matrix-multiplication--gemm)/[DGEMM](../../../../glossary.md#general-matrix-multiplication--gemm) subset. CPU analysis retains the portable generated
candidate, filters the native candidate exactly, accounts for any one-input [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout)
[materialization](../../../../glossary.md#materialization), and selects OpenBLAS only when caller-supplied [immutable](../../../../glossary.md#term-immutable) cost facts prove the
complete native plan cheaper by both configured thresholds.

OpenBLAS remains an optional route within `BackendId("cpu")`, never another [backend](../../../../glossary.md#backend), a universal
MATMUL implementation, a preferred route, a [runtime](../../../../glossary.md#term-runtime) fallback, or a provider-owned [graph](../../../../glossary.md#graph)/lowering
pipeline. Default analysis remains portable-only.

## [Scope](../../../../glossary.md#term-scope)

- Extend CPU-private analysis inputs with one disabled-by-default typed OpenBLAS route
  configuration and immutable expected boundary-storage facts. These are availability,
  qualification, thread-configuration, alignment, and cost snapshots only; they contain no<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [immutable glossary](../../../../glossary.md#term-immutable); [qualification glossary](../../../../glossary.md#term-qualification); [alignment glossary](../../../../glossary.md#term-alignment))</span>
  [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary), segment, address, [runtime](../../../../glossary.md#term-runtime) slot, [mutable](../../../../glossary.md#term-mutable) resource, or [graph](../../../../glossary.md#graph) semantics.
- Generate the OpenBLAS candidate only after the current partition DAG, MATMUL unit, fusion,
  access, portable realization, and representation facts have been derived by common CPU
  lowering. The OpenBLAS leaf consumes those facts and performs no graph interpretation.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [fusion glossary](../../../../glossary.md#term-fusion); [lowering glossary](../../../../glossary.md#lowering); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- Admit exactly one bare, one-[node](../../../../glossary.md#node), one-unit [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) with positive static `m`, `n`, and
  `k`; same-type [`FLOAT32`](../../../../glossary.md#term-float32) or same-type [`FLOAT64`](../../../../glossary.md#term-float64) inputs/result; exact/default numerical mode;
  canonical dense row-major zero-offset output; and dimensions representable by the provider's
  32-bit <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [row-major glossary](../../../../glossary.md#term-row-major); [dimensions glossary](../../../../glossary.md#dimension))</span>`blasint` contract.
- Use one provider call with `alpha = 1` and `beta = 0`: `sgemm` for `FLOAT32`, `dgemm` for
  `FLOAT64`. The selected [CPU](../../../../glossary.md#term-cpu) route plan carries the exact type, `m`, `n`, `k`, boundary
  positions, selected thread count, representation choice, and whole-plan cost decision. It
  carries no live provider or runtime representation.
- Permit either direct native `A`, `B`, and `C`, or exactly one existing CPU affine
  materialization of <span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>`A` or `B` into a run-owned aligned native contiguous workspace. One copy may
  jointly resolve a source's non-canonical layout and non-native carrier. The output is never
  copied, two-input materialization remains rejected, and no second copy/packing framework is
  added.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [contiguous glossary](../../../../glossary.md#term-contiguous); [layout glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Extend the route-neutral [partition](../../../../glossary.md#partition) plan only enough to represent `PORTABLE` or `OPENBLAS` and
  the exact optional OpenBLAS plan. Preserve the selected portable route plan as the supported
  semantic alternative in the immutable analysis result. Finalization and Runtime never
  reselect between them.
- Add one narrow CPU-private invocation seam that borrows and strongly retains the current
  `OpenBlasLibrary` for prepared use, exposes only the exact SGEMM/DGEMM and thread-count checks
  needed by this route, and enables deterministic native-free tests. It does not load, discover,
  configure, close, restore, or otherwise own the provider handle.<span data-glossary-navigation="true"> ([SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- Extend finalization and prepared execution so a direct OpenBLAS plan produces one immutable
  native executable, while a one-copy plan uses the existing partition composite to execute the
  generated affine copy once before the native child. Generalize the composite child type only as
  far as needed to retain portable and native prepared children.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>
- Validate the route with native-free fake invocation tests and one explicit test-source native
  checkpoint using a caller-supplied absolute compatible OpenBLAS path. Ordinary tests perform no
  native lookup, skip, discovery, or environment/property-based activation.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [lookup glossary](../../../../glossary.md#term-lookup); [activation glossary](../../../../glossary.md#term-activation))</span>
- Replace the backend-conformance placeholder with one narrow ordinary conformance test that
  constructs the existing one-node MATMUL preparation projection, selects the direct OpenBLAS
  route from explicit immutable facts, finalizes it with the fake invocation seam, cold-binds
  arena-backed <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [arena glossary](../../../../glossary.md#term-arena))</span>`MemorySegmentStorage` through `CpuBorrowedBuffer` in an exact [`RunState`](../../../../glossary.md#run-state--runstate), and
  executes the prepared recipe. This test uses no Engine and performs no native-library lookup or
  call.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- Finalize the CPU backend guide, every changed Java/Javadoc contract, native route package
  documentation, glossary impact, this task, CPU master plan, and roadmap through a distinct
  clean documentation-focused context after executable Java stabilizes.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

## Exact initial eligibility

Every row is mandatory. Failure or uncertainty in any row removes only the [OpenBLAS](../../../../glossary.md#term-openblas) candidate and
leaves the already-valid portable plan selected during analysis.

| [Dimension](../../../../glossary.md#dimension) | Required OpenBLAS fact |
|---|---|
| [Partition](../../../../glossary.md#partition)/unit | Exactly one partition [node](../../../../glossary.md#node) and one execution unit; the unit is [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) only, with no [fused](../../../../glossary.md#specialization-and-fusion) [bias](../../../../glossary.md#term-bias), terminal, virtual suffix, sibling unit, or [publication](../../../../glossary.md#term-publication)-sensitive internal value. |
| [Shape](../../../../glossary.md#shape) | Resolved [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) `A[m,k]`, `B[k,n]`, and `C[m,n]`; `m`, `n`, and `k` are each in `[1, Integer.MAX_VALUE]`; all checked [element](../../../../glossary.md#term-element)/span arithmetic succeeds. |
| Type | `A`, `B`, and `C` are all [`FLOAT32`](../../../../glossary.md#term-float32), or all [`FLOAT64`](../../../../glossary.md#term-float64); no promotion or output conversion. |
| Numerical mode | Existing `EXACT_DEFAULT`; the MATMUL contract's permitted [reassociation](../../../../glossary.md#term-reassociation)/[FMA](../../../../glossary.md#term-fused-multiply-add-fma) applies, but the supplied provider [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) must have passed this task's [CPU](../../../../glossary.md#term-cpu)-native [conformance](../../../../glossary.md#term-conformance) [checkpoint](../../../../glossary.md#term-checkpoint). |
| [Operation](../../../../glossary.md#operation) [scalars](../../../../glossary.md#term-scalar) | Fixed `alpha = 1` and `beta = 0`; the route does not read an existing logical output value. |
| Output | [Canonical](../../../../glossary.md#term-canonical-form) dense [row-major](../../../../glossary.md#term-row-major), zero logical offset, [injective](../../../../glossary.md#term-injective), direct expected native segment, writable at [binding](../../../../glossary.md#term-binding), type-width aligned, and large enough for exactly `m * n` elements. |
| Direct input | Canonical dense row-major, zero logical offset, expected native segment, type-width aligned, and large enough for its complete [matrix](../../../../glossary.md#term-matrix-meanings). |
| Copied input | At most one of `A` or `B`; any current admitted non-negative affine [layout](../../../../glossary.md#layout)/[carrier](../../../../glossary.md#term-storage-carrier) that the existing affine-copy plan can read completely and copy into one canonical native workspace. |
| [Aliasing](../../../../glossary.md#term-aliasing) | `A` and `B` may overlap. Binding rejects any overlap between `C` and either effective input, or between a [materialization](../../../../glossary.md#materialization) workspace and any selected [buffer](../../../../glossary.md#term-buffer)/workspace, before a copy or [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) can write. |
| Execution | One OpenBLAS call on the invoking thread with the sole initial `SINGLE_THREAD` OpenBLAS configuration, whose provider count is exactly one. CPU portable worker ranges/species do not leak into the native plan. |
| Availability | The typed route input is enabled, marks the exact supplied provider as CPU-route-qualified, and selects `SINGLE_THREAD`. Finalization has a matching open [borrowed](../../../../glossary.md#ownership-borrowing-and-lease) invocation, and cold binding observes provider thread count one. |
| Profitability | The complete native candidate passes the deterministic whole-plan comparison below. Equality, insufficient benefit, [overflow](../../../../glossary.md#term-overflow), or missing cost facts retains portable. |

The provider still revalidates dimensions, native segments, [alignment](../../../../glossary.md#term-alignment), spans, output mutability,
and overlap. CPU binding performs its own route-specific checks first so all selected resources
fail before any materialization or output mutation.

## Whole-plan candidate and cost contract

`CpuPartitionAnalysisInputs.OpenBlasRouteConfig` is a closed, typed configuration with a
[fail-closed](../../../../glossary.md#term-fail-closed) `DISABLED` value. An enabled value contains:

- one explicit `QUALIFIED` availability value meaning that composition has associated this input
  with a compatible 32-bit-`blasint` [OpenBLAS](../../../../glossary.md#term-openblas) [binary](../../../../glossary.md#term-binary-descriptor-and-metadata) that passed the [CPU](../../../../glossary.md#term-cpu) [checkpoint](../../../../glossary.md#term-checkpoint);
- the sole initial `SINGLE_THREAD` OpenBLAS configuration, meaning provider thread count exactly
  one, already installed and externally coordinated by the caller for the complete prepared-use
  lifetime;<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime))</span>
- non-negative portable fixed, per-output, and per-multiply-accumulate cost units;
- non-negative OpenBLAS fixed, per-output, and per-multiply-accumulate cost units; and
- non-negative absolute benefit and relative basis-point thresholds, with basis points in
  `[0, 10_000]`.

The values are dimensionless [immutable](../../../../glossary.md#term-immutable) [heuristic](../../../../glossary.md#term-heuristic) inputs, not measurements, benchmark claims,
tuning-[cache](../../../../glossary.md#term-cache) entries, provider capabilities, or [Planning](../../../../glossary.md#term-planning)-[module](../../../../glossary.md#term-module) scoring. The OpenBLAS fixed term
must include the native transition/provider-call cost and the supplied thread configuration's
known fixed cost. A selected copy reuses the current `MaterializationPolicy` expected-[run](../../../../glossary.md#term-run) count,
copy fixed/per-[element](../../../../glossary.md#term-element) terms, byte ceiling, and exact `CpuMaterializationPlan`; its fixed term is
documented to include run-owned workspace allocation/[binding](../../../../glossary.md#term-binding) and one affine-copy invocation. No
copy candidate exists unless [materialization](../../../../glossary.md#materialization) policy is enabled and its exact bytes fit the
existing ceiling.

For `W = m * n * k`, `O = m * n`, expected runs `R`, and copied logical elements `E` (zero for a
direct candidate), compare checked non-negative costs:

```text
portable = R * (portableFixed + portablePerOutput * O + portablePerMac * W)

openblas = R * (openblasFixed + openblasPerOutput * O + openblasPerMac * W
                + copyFixed(if E > 0) + copyPerElement * E)
```

The candidate wins only when `openblas < portable`, `portable - openblas` meets the absolute
threshold, and its benefit meets the relative basis-point threshold against `portable`. All
arithmetic is exact and checked. [Overflow](../../../../glossary.md#term-overflow), a zero portable denominator with a nonzero relative
threshold, inconsistent expected-run facts, or any incomplete term rejects OpenBLAS rather than
[saturating](../../../../glossary.md#saturation-pole-and-extremum) or guessing.

Candidate ordering is stable: direct, copy `A`, copy `B`. Cost wins first; exact ties prefer fewer
copies, then fewer workspace bytes, then that stable order. Selection covers the entire one-unit
plan—provider call, computation, native transition, copy, workspace, and expected reuse—not an
isolated [GEMM](../../../../glossary.md#general-matrix-multiplication--gemm) timing. No preparation-time measurement, benchmark [lookup](../../../../glossary.md#term-lookup), tuning-cache access,
machine probing, or fixed “OpenBLAS first” priority is permitted.

## Provider [ownership](../../../../glossary.md#term-ownership-meanings), thread state, and failures

- Composition explicitly opens [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) by exact name or absolute path and remains its
  lifetime owner. CPU receives only an explicitly borrowed route invocation at finalization.<span data-glossary-navigation="true"> ([lifetime glossary](../../../../glossary.md#term-lifetime); [owner glossary](../../../../glossary.md#term-owner); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- [CPU](../../../../glossary.md#term-cpu) analysis selects the typed `SINGLE_THREAD` configuration from immutable inputs. The caller
  must install provider thread count one before finalization, exclude competing OpenBLAS
  setters/native users for the complete
  prepared-use lifetime, keep the handle open, and perform any desired restoration after all
  prepared work is quiescent. This task adds no discovery, global registry, hidden singleton,
  per-call setter, automatic restoration, lock, active-call counter, or claim of isolation across
  other handles/class loaders/native consumers.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [quiescent glossary](../../../../glossary.md#term-quiescent); [registry glossary](../../../../glossary.md#term-registry); [singleton glossary](../../../../glossary.md#term-singleton))</span>
- Finalization and each cold bind fail if a selected native route has no invocation, the borrowed
  provider is closed, or the observed thread count is not one. These checks
  do not make a query atomic with later native work; the external no-race contract remains
  mandatory.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [race glossary](../../../../glossary.md#term-race))</span>
- A selected native plan never falls back during finalization, binding, or execution. Closure,
  thread-state drift, inaccessible storage, wrong type/layout/provenance, misalignment, overlap,
  or provider failure propagates as a stable CPU binding/execution failure before writes where
  possible. Runtime performs no route lookup or reselection.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [layout glossary](../../../../glossary.md#layout); [provenance glossary](../../../../glossary.md#provenance); [Runtime glossary](../../../../glossary.md#term-runtime); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Prepared recipes strongly retain the invocation adapter and therefore the provider reference,
  but they do not prevent the caller from closing it incorrectly and never close it themselves.

## Numerical [conformance](../../../../glossary.md#term-conformance) boundary

The [Model](../../../../glossary.md#term-model) permits floating [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [reassociation and fused multiply-add](../../../../glossary.md#rounding-sites-fma-and-aggregate-floor), so the [OpenBLAS](../../../../glossary.md#term-openblas) route need
not be bitwise identical to the sequential portable reference. It must nevertheless preserve the
same [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) result type, full [contraction](../../../../glossary.md#contraction-dimension), [IEEE](../../../../glossary.md#term-ieee) [operation](../../../../glossary.md#operation)-class behavior, and [positive
zero](../../../../glossary.md#nan-infinity-and-signed-zero) for cases that mathematically require an unambiguous positive zero. This initial route
excludes `k == 0`, so the provider's positive-output/empty-contraction behavior is not consumed.

Native [checkpoint](../../../../glossary.md#term-checkpoint) comparisons use an independent higher-[precision](../../../../glossary.md#term-precision) sum-of-products [oracle](../../../../glossary.md#term-test-oracle). For a
[finite](../../../../glossary.md#term-finite) output, let `u` be `2^-24` for FLOAT32 or `2^-53` for FLOAT64, let
`gamma = (2 * k * u) / (1 - 2 * k * u)`, and let `sumAbs` be the higher-precision sum of the
absolute exact input products. Checkpoint cases must keep `2 * k * u < 1`. The accepted [absolute
error](../../../../glossary.md#term-absolute-error) is:

```text
max(gamma * sumAbs, 4 * ulp(target-rounded higher-precision result))
```

The four-[ULP](../../../../glossary.md#term-unit-in-the-last-place-ulp) term uses the [target](../../../../glossary.md#term-target) type's ULP, including its minimum positive value at zero. The
same formula applies to direct and one-copy results. Separate cases require [NaN](../../../../glossary.md#term-nan) when exactly one
contributing product is NaN, positive or [negative infinity](../../../../glossary.md#nan-infinity-and-signed-zero) when that is the sole infinite sign in
the sum, and positive zero for an all-positive-zero-times-positive input. They do not compare NaN
payloads or impose an [evaluation](../../../../glossary.md#term-evaluation) order forbidden by Model. The test and [CPU](../../../../glossary.md#term-cpu) guide must name this
formula and its bounded-checkpoint role and fail rather than silently disabling assertions. Do
not claim general OpenBLAS accuracy, determinism, performance, or cross-build compatibility from
the bounded checkpoint.

## Out of [scope](../../../../glossary.md#term-scope)

- Vector-vector, matrix-vector, vector-matrix, batched, right-broadcast-batched, rank greater than
  two, dynamic, unresolved, transposed, strided-batched, sparse, quantized, or complex MATMUL.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [rank glossary](../../../../glossary.md#term-rank); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- [`BFLOAT16`](../../../../glossary.md#term-bfloat16), mixed floating types, [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype), any conversion, or any operation other than
  bare MATMUL.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation))</span>
- [Bias](../../../../glossary.md#term-bias), [activation](../../../../glossary.md#term-activation), clamp, [`linear`](../../../../glossary.md#term-linear-meanings), attention, convolution lowering, higher-level BLAS pattern
  recognition, or any fused native epilogue.<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers); [lowering glossary](../../../../glossary.md#lowering); [BLAS glossary](../../../../glossary.md#term-blas); [fused glossary](../../../../glossary.md#specialization-and-fusion); [epilogue glossary](../../../../glossary.md#term-epilogue))</span>
- `k == 0`, `m == 0`, `n == 0`, output materialization, two-input copies, packing panels,
  transpose/reorder kernels, persistent packed weights, workspace pooling, or provider-side
  allocation.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [transpose glossary](../../../../glossary.md#term-transpose); [kernels glossary](../../../../glossary.md#kernel); [weights glossary](../../../../glossary.md#term-weight); [pooling glossary](../../../../glossary.md#term-pooling))</span>
- OpenBLAS discovery, platform filename selection, download, packaging, version probing, ILP64,
  additional symbols, provider API/source/test changes, or a default installed-library
  assumption.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [ILP64 glossary](../../../../glossary.md#term-ilp64); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Autotuning, tuning-cache persistence, workload cache, benchmarking, profiling, performance
  claims, relaxed numerical mode, or CPU tasks 0011 through 0017.<span data-glossary-navigation="true"> ([Autotuning glossary](../../../../glossary.md#autotuning--model-autotuning); [cache glossary](../../../../glossary.md#term-cache); [persistence glossary](../../../../glossary.md#term-persistence); [benchmarking glossary](../../../../glossary.md#term-benchmarking); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Another [`BackendId`](../../../../glossary.md#backend-identity--backendid), Planning ownership/cost changes, shared Prepare/Runtime/Engine API changes,
  runtime route choice, dynamic fallback, service loader, registry, broad native facade, or common
  vendor abstraction.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [registry glossary](../../../../glossary.md#term-registry); [facade glossary](../../../../glossary.md#term-facade))</span>
- Model, Tensor, Compiler, Training, Config public surface, capability reporting, or generated
  portable MATMUL semantics/schema/inventory changes.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Tensor glossary](../../../../glossary.md#tensor); [Compiler glossary](../../../../glossary.md#term-compiler); [schema glossary](../../../../glossary.md#term-schema))</span>
- Architecture contract, ADR, production-module dependency, shared production build, shared test
  harness, or integration-test changes. The only build/dependency change is the exact test-scoped
  backend-conformance dependency surface listed below, guarded by one focused architecture test.
  If conformance requires an Engine dependency, a shared API, or another dependency, stop and
  revise the plan instead of hiding it in this task.<span data-glossary-navigation="true"> ([Architecture contract glossary](../../../../glossary.md#architecture-contract); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module); [dependency glossary](../../../../glossary.md#term-dependency); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [architecture test glossary](../../../../glossary.md#term-architecture-test))</span>

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md), especially CPU routes, OpenBLAS provider,
  preparation/finalization ownership, resources, performance, and testing boundaries<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Runtime, Prepare, and backend boundary](../../../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend))</span>
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Performance evidence and tuning](../../../../architecture/performance-evidence-and-tuning.md)
- [Partition scoring](../../../../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../../../../glossary.md#partition))</span>
- [ADR 0002: Backend-owned lowering](../../../../design/decisions/0002-backend-owned-lowering.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Backend-owned lowering glossary](../../../../glossary.md#backend-owned-lowering))</span>
- [ADR 0008: Performance evidence and tuning boundaries](../../../../design/decisions/0008-performance-evidence-and-tuning-boundaries.md)
- [ADR 0010: Staged preparation](../../../../design/decisions/0010-staged-backend-preparation.md)
- [ADR 0011: Per-run resources](../../../../design/decisions/0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run))</span>
- [Model MATMUL semantics](../../../modules/model/tasks/0019-matmul-semantics-and-tensor-expression.md)<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- [CPU 0005A architecture reset](0005a-atomic-partition-kernel-architecture-reset.md)
- [CPU 0008F portable MATMUL](0008f-portable-matmul-execution-and-bounded-linear-epilogues.md)
- [CPU 0009F1 MATMUL route decision](0009f1-matmul-and-convolution-route-decisions.md)
- [CPU 0009G checkpoint](0009g-final-support-correctness-hygiene-and-inventory-checkpoint.md)<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- [CPU 0009G1 evidence correction](0009g1-scalar-strategy-evidence-correction.md)
- [OpenBLAS provider master plan](../../openblas-provider/master-plan.md)
- [Provider 0001 loading/binding](../../openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md)<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding))</span>
- [Provider 0002 SGEMM/DGEMM](../../openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md)<span data-glossary-navigation="true"> ([SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- [Provider 0003 thread control/checkpoint](../../openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md)

## Architecture constraints

- Planning continues to select only CPU ownership. CPU preparation owns complete route candidate
  construction, exact filtering, cost comparison, representation/materialization choice, selected
  route plan, and resource declarations.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [CPU glossary](../../../../glossary.md#term-cpu); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [materialization glossary](../../../../glossary.md#materialization))</span>
- Common CPU lowering, MATMUL geometry/access facts, fusion decisions, numerical mode, portable
  realization, and representation framework remain authoritative. The OpenBLAS leaf consumes
  them and owns only provider-specific eligibility, route configuration, invocation adaptation,
  and prepared native execution.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [fusion glossary](../../../../glossary.md#term-fusion); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- `backends/cpu -> backends/openblas-provider` is the existing allowed dependency. The provider
  remains a zero-project-dependency leaf and never imports CPU, Model, Planning, Prepare, Runtime,
  Tensor, graph, route, or configuration contracts.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [dependency leaf glossary](../../../../glossary.md#term-dependency-leaf); [Model glossary](../../../../glossary.md#term-model); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Tensor glossary](../../../../glossary.md#tensor); [graph glossary](../../../../glossary.md#graph))</span>
- `testing/backend-conformance` may depend outward on `backends/cpu` in test scope and on the
  exact shared Model, Planning, Prepare, Runtime, and Backend Contract types used to construct and
  execute the existing staged boundary. This testing edge changes no production dependency
  direction. The borrowed-invocation seam must not expose an OpenBLAS-provider type in the ABI
  consumed by this test: backend conformance supplies only a fake and adds no direct dependency
  on either the provider or Engine.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [Backend glossary](../../../../glossary.md#backend); [dependency direction glossary](../../../../glossary.md#java-platform-and-build-tools); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>
- Analysis uses only complete immutable context and performs no native query or measurement.
  Finalization verifies resources but cannot change route, thread configuration, copies, or
  declarations. Runtime invokes already-bound work and sees no route vocabulary.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- Run-owned buffers/workspaces retain current CPU ownership. The OpenBLAS handle and its mutable
  process/library thread state are externally coordinated borrowed state, not a run-owned buffer,
  workspace, provider-owned lease, or hidden global.<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run); [buffers glossary](../../../../glossary.md#term-buffer); [mutable glossary](../../../../glossary.md#term-mutable); [lease glossary](../../../../glossary.md#term-lease))</span>
- Portable generation remains the correctness baseline. Ineligibility selects portable during
  analysis; a failure after OpenBLAS has been selected is an error, not a late fallback.
- Any need to change authoritative architecture, dependency direction, shared lifecycle/API,
  Model semantics, provider surface, or production-module build structure is a stop condition.
  The exact backend-conformance test build and architecture-test changes listed in this task are
  required scope, not stop conditions.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [module glossary](../../../../glossary.md#term-module))</span>

## Package impact

Existing packages changed:

- `io.github.pho001.synaptik.backend.cpu.internal.prepare` — adds typed storage/config inputs,
  route-plan invariants, cold selection handoff, and provider-aware finalization.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.lowering` — reuses the bounded representation
  candidates and permits one native carrier/layout transition through the existing affine-copy
  plan.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout))</span>
- `io.github.pho001.synaptik.backend.cpu.internal.executable` — permits the fixed partition
  composite to retain either a portable or native prepared child.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>

Package added:

- `io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas` — the reserved
  OpenBLAS leaf for exact route filtering/configuration, immutable plan, borrowed provider
  invocation, and prepared SGEMM/DGEMM execution. It owns no graph lowering or common
  representation policy.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [leaf glossary](../../../../glossary.md#term-leaf); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [graph glossary](../../../../glossary.md#graph); [lowering glossary](../../../../glossary.md#lowering))</span>

Expected type placement:

- `CpuPartitionAnalysisInputs.OpenBlasRouteConfig` and `BoundaryStorageFact` — immutable
  route-availability/cost/thread and expected storage/alignment snapshots because they are inputs
  to CPU analysis, not provider state.<span data-glossary-navigation="true"> ([alignment glossary](../../../../glossary.md#term-alignment))</span>
- `CpuOpenBlasRouteSelector` — field-free cold candidate filtering and complete cost comparison
  over common unit/representation facts.
- `CpuOpenBlasRoutePlan` — exact [immutable](../../../../glossary.md#term-immutable) selected [SGEMM](../../../../glossary.md#general-matrix-multiplication--gemm)/[DGEMM](../../../../glossary.md#general-matrix-multiplication--gemm) and representation facts.
- `CpuOpenBlasInvocation` — minimal technically public but unsupported `.internal` cross-package
  invocation interface/test seam whose consumed ABI names no provider type. Its CPU-private
  production adapter strongly retains one borrowed <span data-glossary-navigation="true"> ([ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>[`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary), while the backend-
  conformance test can supply a deterministic fake without loading native code; no general
  native-provider abstraction.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [native code glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture))</span>
- `CpuOpenBlasPreparedExecutable` — route-specific cold [binding](../../../../glossary.md#term-binding) and one-call execution.

Do not create `nativeblas` umbrella policy types, a vendor [registry](../../../../glossary.md#term-registry)/factory, a new supported public
[CPU](../../../../glossary.md#term-cpu) type, a provider-side CPU adapter, or another native package.

## Affected files

Expected [CPU](../../../../glossary.md#term-cpu) production paths:

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionAnalysisInputs.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparationPlan.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizer.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlanner.java`
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMaterializationPlan.java`, only if the existing [identity](../../../../glossary.md#term-identity-meanings) needs one exact native-transition reason
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedPartitionExecutable.java`
- new files only under
  `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/route/nativeblas/openblas/`:
  `package-info.java`, `CpuOpenBlasRouteSelector.java`, `CpuOpenBlasRoutePlan.java`,
  `CpuOpenBlasInvocation.java`, and `CpuOpenBlasPreparedExecutable.java`

Expected CPU test/[checkpoint](../../../../glossary.md#term-checkpoint) paths:

- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProviderPublicShapeTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/CpuInternalPackageInventoryTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuRepresentationPlannerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionFinalizerTest.java`
- `backends/cpu/src/test/java/io/github/pho001/synaptik/backend/cpu/internal/executable/CpuPreparedPartitionExecutableTest.java`
- new matching route-[leaf](../../../../glossary.md#term-leaf) tests `CpuOpenBlasRouteSelectorTest.java`,
  `CpuOpenBlasPreparedExecutableTest.java`, and non-[JUnit](../../../../glossary.md#term-junit)
  `CpuOpenBlasNativeCheckpoint.java`

[Backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) and [dependency](../../../../glossary.md#term-dependency)-enforcement paths:

- `testing/backend-conformance/build.gradle.kts`
- `testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuOpenBlasRouteConformanceTest.java`
- `testing/architecture-tests/src/test/java/io/github/pho001/synaptik/testing/architecture/BackendConformanceDependencyContractTest.java`

The backend-conformance build keeps its existing
`implementation(project(":modules:backend-contract"))` edge and adds exactly these test
dependencies, in this order:

```kotlin
testImplementation(project(":modules:model"))
testImplementation(project(":modules:planning"))
testImplementation(project(":modules:runtime"))
testImplementation(project(":modules:prepare"))
testImplementation(project(":backends:cpu"))
```

Its Java [compile](../../../../glossary.md#compile) and test tasks add `jdk.incubator.vector`, matching the CPU [module](../../../../glossary.md#term-module) because the
conformance test crosses CPU's existing internal preparation types. The new [architecture test](../../../../glossary.md#term-architecture-test)
locks this exact project-dependency surface, requires the CPU edge to remain test-scoped, and
rejects any direct [OpenBLAS](../../../../glossary.md#term-openblas)-provider, [Engine](../../../../glossary.md#term-engine), or other dependency drift. Keeping provider types
out of the fake seam's consumed [ABI](../../../../glossary.md#term-application-binary-interface-abi) lets the ordinary conformance source remain provider-agnostic
and native-free; the CPU-local explicit checkpoint separately owns real-provider composition.

Documentation/[planning](../../../../glossary.md#term-planning) paths:

- `docs/backend-guide/cpu-backend.md`
- `docs/glossary.md`, only if current backend-route/OpenBLAS entries cannot express the final
  CPU-route/thread-lifecycle distinction accurately<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- this task
- `docs/planning/backends/cpu/master-plan.md`
- `docs/planning/roadmap.md`

Review-only paths include current [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) lowerer/[IR](../../../../glossary.md#term-intermediate-representation-ir)/reference/generated tests, [buffer](../../../../glossary.md#term-buffer)/workspace
representations, provider source/tests/tasks, CPU/provider [Gradle](../../../../glossary.md#term-gradle) files, [Model](../../../../glossary.md#term-model)/[Compiler](../../../../glossary.md#term-compiler)/Training/
Config/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime) APIs, architecture/ADRs and all other architecture-test sources, all other
backend-conformance sources, integration tests, and portable evidence resources. They remain
byte-for-byte unchanged unless this task is first returned to planning.

## Maximum [scope](../../../../glossary.md#term-scope)

At most 29 paths:

- 12 CPU production paths: seven existing paths (including the conditional materialization path)
  and five new OpenBLAS-leaf paths;<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [leaf glossary](../../../../glossary.md#term-leaf))</span>
- nine [CPU](../../../../glossary.md#term-cpu) test/[checkpoint](../../../../glossary.md#term-checkpoint) paths;
- three [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance)/build/[dependency](../../../../glossary.md#term-dependency)-enforcement paths;
- at most two explanatory documentation paths; and
- three [planning](../../../../glossary.md#term-planning) paths.

If `CpuMaterializationPlan.java` or `docs/glossary.md` does not need an edit, the final scope is
smaller. Compatibility constructors belong in the listed preparation inputs/plan so existing
tests do not require mechanical edits. A need for another production/test path, shared [module](../../../../glossary.md#term-module),
provider path, [Gradle](../../../../glossary.md#term-gradle) file, architecture/conformance/integration source, or more than 29 total
paths is a planning stop, not implied scope.

## Acceptance criteria

1. Default CPU analysis and every ineligible, unqualified, missing-cost, overflow, tied, or
   insufficient-benefit case select the unchanged portable plan and declare no OpenBLAS-only
   resource.<span data-glossary-navigation="true"> ([overflow glossary](../../../../glossary.md#term-overflow); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
2. Exact filtering admits only the tabled rank-two, positive, same-type FLOAT32/FLOAT64 bare
   MATMUL subset. Tests independently reject every excluded rank, batch/broadcast, type/promotion,
   empty dimension, oversized dimension, epilogue/fusion, layout, output carrier, alignment,
   copy-count, numerical-mode, and provider-qualification boundary.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [rank glossary](../../../../glossary.md#term-rank); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [dimension glossary](../../../../glossary.md#dimension); [epilogue glossary](../../../../glossary.md#term-epilogue); [fusion glossary](../../../../glossary.md#term-fusion); [layout glossary](../../../../glossary.md#layout); [carrier glossary](../../../../glossary.md#term-storage-carrier); [alignment glossary](../../../../glossary.md#term-alignment); [qualification glossary](../../../../glossary.md#term-qualification))</span>
3. Direct, copy-`A`, and copy-`B` candidates reuse common lowering and the existing typed
   materialization/resource framework. Selection is deterministic; two-input and output copies
   are absent; workspaces are exact-sized, aligned, run-owned, deduplicated, and included in the
   plan and cost.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [materialization glossary](../../../../glossary.md#materialization); [run glossary](../../../../glossary.md#term-run))</span>
4. The immutable plan retains both the common selected portable realization and exactly one
   optional OpenBLAS plan, with constructor invariants that make route/configuration/
   representation/resource disagreements unrepresentable. Finalization cannot change the
   selection.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
5. Whole-plan arithmetic follows the specified formula and thresholds with checked values.
   Table-driven tests cover direct and copied wins/losses, expected-run reuse, absolute and
   relative thresholds, ties, candidate tie-breaking, byte ceilings, and every overflow/fail-
   closed branch. No measurement or fixed vendor priority occurs in preparation.<span data-glossary-navigation="true"> ([losses glossary](../../../../glossary.md#loss))</span>
6. Finalization requires a matching open borrowed invocation and selected thread count, generates
   only selected copy artifacts, and creates the direct native executable or existing copy-first
   composite. Missing/closed/drifted provider state fails deterministically and never produces a
   portable fallback after selection.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
7. Cold binding validates exact FLOAT32/FLOAT64 native effective segments, accessibility,
   alignment, spans, output writability, workspaces, and all write overlaps before mutation. Hot
   execution performs exactly one typed provider GEMM call and no graph/IR interpretation, route
   lookup, type/carrier/layout dispatch, allocation, boxing, reflection, map/string dispatch,
   thread setter/query, synchronization, or copy.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [graph glossary](../../../../glossary.md#graph); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
8. Native-free tests prove exact `m`, `n`, `k`, `alpha`, `beta`, segment identity, call count,
   failure propagation, no call after failed bind/copy validation, repeated-run reuse, and direct
   versus one-copy semantic parity for both types.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [validation glossary](../../../../glossary.md#term-validation))</span>
9. `CpuOpenBlasRouteConformanceTest` exercises one representative direct native-storage MATMUL
   for both FLOAT32 and FLOAT64 through [`PrepareContext`](../../../../glossary.md#prepare-context-preparecontext) -> `CpuPartitionPreparer.analyze` ->
   `BackendPartitionFinalization` -> `CpuPartitionFinalizer.finalizePartition` -> cold binding ->
   bound execution. It builds the exact [`PreparedMemoryPlan`](../../../../glossary.md#prepared-memory-plan--preparedmemoryplan) and assignments from the analyzed
   requirements, then binds arena-backed <span data-glossary-navigation="true"> ([arena glossary](../../../../glossary.md#term-arena))</span>`MemorySegmentStorage` through `CpuBorrowedBuffer` in a
   [`RunState`](../../../../glossary.md#run-state--runstate). It supplies explicit qualifying storage/cost/thread facts, proves the selected route
   is `OPENBLAS`, injects a fake open single-thread invocation, checks the exact one-call
   dimensions/scalars/segment identities, and verifies the output against independently computed
   row-major MATMUL results. It uses no Engine, installed library, native lookup, native call,
   skip, environment variable, or system-property activation.<span data-glossary-navigation="true"> ([scalars glossary](../../../../glossary.md#term-scalar); [row-major glossary](../../../../glossary.md#term-row-major); [Engine glossary](../../../../glossary.md#term-engine); [activation glossary](../../../../glossary.md#term-activation))</span>
10. The backend-conformance build has only the listed project dependencies and CPU remains a
    test-scoped edge. <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>`BackendConformanceDependencyContractTest` fails closed for an Engine,
    any OpenBLAS-provider edge, a production-scoped CPU edge, missing shared test dependency,
    reordered dependency, or otherwise unlisted project edge.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
11. The explicit native checkpoint consumes exactly one caller-supplied absolute compatible
   library path, installs/restores the selected thread count in a <span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>`finally` path under the stated
   exclusive-use precondition, drives CPU analysis through prepared execution for FLOAT32 and
   FLOAT64 direct and one-copy cases, confirms OpenBLAS selection, and passes the documented
   finite/special numerical checks. It never discovers, downloads, silently skips, or converts a
   missing/incompatible binary into a passing result.<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [finite glossary](../../../../glossary.md#term-finite); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata))</span>
12. `CpuCapabilityProvider` remains the sole supported [CPU](../../../../glossary.md#term-cpu) [API](../../../../glossary.md#term-application-programming-interface-api); `BackendId("cpu")` and capability
    reporting are unchanged; the provider remains unchanged; public/internal package inventory
    records exactly the new unsupported leaf types.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf))</span>
13. A distinct clean documentation-focused context reviews the final diff, finalizes every
    affected Javadoc and package summary under the General and API/Javadoc profiles, updates the
    CPU guide under the Backend Guide profile, resolves glossary impact, and synchronizes this
    task/master/roadmap under the Planning profile. It records reasoned no-change conclusions for
    public Tensor/Compile/Training/Config APIs, Model semantics, provider API, shared Prepare/
    Runtime, architecture/ADRs other than the named dependency test, all other backend-
    conformance sources, integration, other Gradle builds, and other modules.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning); [Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module))</span>
14. CPU and backend-conformance tests, the explicit native checkpoint, the architecture suite,
    repository-wide tests, CPU Javadoc/render inspection, Markdown/scope/status checks, and
    whitespace validation pass before the task becomes <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`Complete`.

## Tests / [validation](../../../../glossary.md#term-validation)

During implementation, use the focused affected-[module](../../../../glossary.md#term-module) command as needed:

```bash
./gradlew :backends:cpu:test :testing:backend-conformance:test \
  :testing:architecture-tests:test
```

The [CPU](../../../../glossary.md#term-cpu) suite must include selector/config/plan invariants, preparation/finalization,
representation and exact resource declarations, composite sequencing, [binding](../../../../glossary.md#term-binding)/failure-before-
write, fake provider invocation, [lifecycle](../../../../glossary.md#lifecycle)/thread drift, inventory/public [shape](../../../../glossary.md#shape), and unchanged
portable fallback coverage. The [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) suite must include the two-type direct-route
preparation/finalization/binding/execution case above. The architecture suite must enforce the
exact new test-only [dependency](../../../../glossary.md#term-dependency) surface. All ordinary tests require no native access or installed
[OpenBLAS](../../../../glossary.md#term-openblas).

[Compile](../../../../glossary.md#compile) the explicit [checkpoint](../../../../glossary.md#term-checkpoint) and [run](../../../../glossary.md#term-run) it separately with the caller-supplied [binary](../../../../glossary.md#term-binary-descriptor-and-metadata):

```bash
./gradlew :backends:cpu:testClasses
java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector \
  -cp backends/cpu/build/classes/java/test:backends/cpu/build/classes/java/main:backends/openblas-provider/build/classes/java/main:modules/model/build/classes/java/main:modules/config/build/classes/java/main:modules/planning/build/classes/java/main:modules/runtime/build/classes/java/main:modules/prepare/build/classes/java/main:modules/backend-contract/build/classes/java/main:modules/trace/build/classes/java/main \
  io.github.pho001.synaptik.backend.cpu.internal.route.nativeblas.openblas.CpuOpenBlasNativeCheckpoint \
  <ABSOLUTE_COMPATIBLE_OPENBLAS_LIBRARY>
```

On Windows, use `;` as the class-path separator and pass an absolute compatible [DLL](../../../../glossary.md#term-dll) path. The
checkpoint command is mandatory evidence for completion; an absent supplied library leaves task
0010 incomplete.

Because this task adds a project dependency, changes a [Gradle](../../../../glossary.md#term-gradle) build, and spans CPU plus two
testing modules, the implementation context runs one final repository-wide suite after executable
code, build files, and tests stabilize:

```bash
./gradlew test
```

This final repository command may serve as the final CPU, backend-conformance, and architecture
evidence instead of repeating an unchanged successful focused command. Record task/test counts
for those three projects from that run. The separate explicit real-native checkpoint remains
required because it is intentionally outside [JUnit](../../../../glossary.md#term-junit).

The documentation-focused context reuses the successful repository-wide tests and native
checkpoint unless it changes executable Java, build logic, or tests, then runs:

```bash
./gradlew :backends:cpu:javadoc
git diff --check
git status --short -uall
```

It renders and inspects the changed generated [Javadoc](../../../../glossary.md#term-javadoc) pages; checks local Markdown links and
anchors, heading order/uniqueness, balanced fences, LF/final newlines, terminology, exact path
[scope](../../../../glossary.md#term-scope), provider/portable no-change, and public/internal inventory; and confirms CPU 0010 alone was
[`Ready`](../../../../glossary.md#term-ready) before implementation and CPU 0011 through 0017 remain `Draft`. It also confirms the
backend-conformance test is native-free and [Engine](../../../../glossary.md#term-engine)-free, the one architecture-test addition is
limited to the new dependency surface, and no successful Java suite is duplicated without stale
evidence. Integration validation remains deferred until Engine provides the public end-to-end
lifecycle.

No benchmark, tuning workflow, provider suite rerun, or generated portable evidence regeneration
is required. The recorded `./gradlew test` run is the task's repository-wide gate; [CI](../../../../glossary.md#term-continuous-integration-ci) remains the
final independent gate.

## Dependencies

- Complete [CPU 0005A](0005a-atomic-partition-kernel-architecture-reset.md), which reserves the
  route leaf, keeps common lowering authoritative, and establishes the portable fallback.<span data-glossary-navigation="true"> ([leaf glossary](../../../../glossary.md#term-leaf); [lowering glossary](../../../../glossary.md#lowering))</span>
- Complete [CPU 0009G1](0009g1-scalar-strategy-evidence-correction.md), after completed CPU 0009G,
  which closes the current portable checkpoint and task frontier.
- Complete [OpenBLAS provider](../../openblas-provider/master-plan.md)<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>, including completed tasks
  [0001](../../openblas-provider/tasks/0001-library-loading-and-required-symbol-binding.md),
  [0002](../../openblas-provider/tasks/0002-float32-float64-row-major-gemm-invocation.md), and
  [0003](../../openblas-provider/tasks/0003-thread-control-and-native-provider-checkpoint.md).
- Current `CpuMatmulLowering`, representation/materialization plans, run-owned native
  buffer/workspace types, staged CPU finalization, portable MATMUL reference/tests, and the
  already-declared <span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [run glossary](../../../../glossary.md#term-run); [buffer glossary](../../../../glossary.md#term-buffer); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>`backends/cpu -> backends/openblas-provider` [Gradle](../../../../glossary.md#term-gradle) [dependency](../../../../glossary.md#term-dependency).
- The existing included `testing/backend-conformance` and `testing/architecture-tests` projects.
  Backend conformance currently has only its marker type and Backend Contract dependency, so this
  task owns the first concrete CPU conformance test and its exact test-scoped dependencies.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

All prerequisites are complete and consistent. The provider [checkpoint](../../../../glossary.md#term-checkpoint) recorded a compatible
32-bit-`blasint` [SGEMM](../../../../glossary.md#general-matrix-multiplication--gemm)/[DGEMM](../../../../glossary.md#general-matrix-multiplication--gemm)/thread-control [binary](../../../../glossary.md#term-binary-descriptor-and-metadata), so the explicit task checkpoint has an
established contract even though ordinary tests remain native-free. The completed provider
master plan's statement that no [CPU](../../../../glossary.md#term-cpu) task was yet [Ready](../../../../glossary.md#term-ready) records that provider milestone's
completion-time [handoff](../../../../glossary.md#term-handoff); the CPU master plan and repository roadmap own the current CPU frontier.

## Follow-up tasks

- [CPU](../../../../glossary.md#term-cpu) 0011 remains the next `Draft` row and owns distinct oneMKL BLAS/VML peer routes only after
  task 0010 completes.<span data-glossary-navigation="true"> ([oneMKL glossary](../../../../glossary.md#term-onemkl); [BLAS glossary](../../../../glossary.md#term-blas); [VML glossary](../../../../glossary.md#term-vml))</span>
- Broader OpenBLAS shapes, batches, transposes, epilogues, two-input/output materialization,
  packing, thread candidates, or persistent tuning evidence are not implied follow-ups. Add an
  ordered bounded task only after a concrete use case and compatible evidence exist.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [shapes glossary](../../../../glossary.md#shape); [batches glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [materialization glossary](../../../../glossary.md#materialization))</span>
- CPU 0016 may later consume compatible measured selected-route evidence; this task creates no
  cache or opaque tuning handoff. CPU 0017 later owns explicit relaxed numerical candidates.<span data-glossary-navigation="true"> ([cache glossary](../../../../glossary.md#term-cache); [opaque glossary](../../../../glossary.md#term-opaque); [handoff glossary](../../../../glossary.md#term-handoff))</span>

## Architecture impact

Expected architecture impact: None. This implements the already-reserved [OpenBLAS](../../../../glossary.md#term-openblas) [CPU](../../../../glossary.md#term-cpu) route using
the existing production [dependency](../../../../glossary.md#term-dependency) and staged CPU-owned analysis/finalization boundary. The new
[backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) edges are test-only outward dependencies from a testing project, and the
focused [architecture test](../../../../glossary.md#term-architecture-test) enforces that they do not introduce [Engine](../../../../glossary.md#term-engine), production-scoped
CPU, any direct OpenBLAS-provider edge, or other dependency drift.

If implementation requires an architecture rule, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), production-[module](../../../../glossary.md#term-module) edge, dependency beyond
the exact listed backend-conformance test surface, provider contract, shared [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/
Engine surface, global thread-state coordinator, or [backend identity](../../../../glossary.md#backend-identity--backendid) change, stop and report the
exact conflict before editing outside this specification.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are the clean implementation agent for Synaptik CPU 0010. Do not use GSD, commit, or push.
Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, the CPU master plan, and
docs/planning/backends/cpu/tasks/0010-narrow-openblas-blas-compatible-native-route.md, plus every
architecture, prerequisite task, source, test, and documentation contract directly referenced by
that task. Implement the task exactly within its 29-path ceiling. Stop for any architecture,
provider, shared-lifecycle, or scope conflict instead of inventing a new boundary.

Implement the required native-free backend-conformance test and exact dependency architecture
test without adding Engine. After Java, build, conformance, repository-wide, and required native
validation stabilize, hand the exact diff and recorded evidence to a distinct clean
documentation-focused agent following documentation-rules.md. That agent must finalize affected
Javadocs, package documentation, CPU guide, glossary impact, task, master plan, roadmap, and
documentation validation without repeating successful Java tests unless it changes executable
behavior or records a concrete stale-evidence risk. Do not mark CPU 0010 Complete until the
backend-conformance test, repository-wide suite, native checkpoint, and documentation pass all
succeed.
```

## Local decisions

- Start with bare rank-two same-type FLOAT32/FLOAT64 MATMUL because it is the exact semantic
  intersection of current <span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>`CpuMatmulLowering` and the provider's row-major non-transposed
  SGEMM/DGEMM contract. Vector/rank-one/batch/epilogue support would require additional mapping or
  call structure and is deliberately excluded.<span data-glossary-navigation="true"> ([row-major glossary](../../../../glossary.md#term-row-major); [SGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [DGEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm); [Vector glossary](../../../../glossary.md#term-vector); [rank-one glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters); [epilogue glossary](../../../../glossary.md#term-epilogue))</span>
- Require positive `m`, `n`, and `k`. Portable execution remains responsible for every empty
  case, avoiding a numerical claim around the provider's `k == 0` beta behavior and pointless
  native transitions for output-empty calls.
- Reuse one existing affine-copy materialization for an input carrier/layout transition. This
  makes native transition cost and resource ownership real without adding packing, output copy,
  or a second representation system.<span data-glossary-navigation="true"> ([resource ownership glossary](../../../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [carrier glossary](../../../../glossary.md#term-storage-carrier); [layout glossary](../../../../glossary.md#layout); [ownership glossary](../../../../glossary.md#term-ownership))</span>
- Keep an explicit expected native/[alignment](../../../../glossary.md#term-alignment) fact separate from `CarrierAccess.MEMORY_SEGMENT`.
  The latter can represent a read-only heap segment whose heap base is not observable and cannot
  prove provider-native storage.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap))</span>
- Keep the live provider out of analysis. A borrowed invocation enters only finalization, just as
  the current worker group is an explicitly borrowed finalization resource.<span data-glossary-navigation="true"> ([borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Select only `SINGLE_THREAD` and verify an externally installed provider count of one without
  setting/querying it in the hot invocation. This is the smallest honest initial policy over
  process/library-global mutable state; it does not pretend CPU can coordinate unrelated native
  users or qualify arbitrary thread configurations.<span data-glossary-navigation="true"> ([mutable glossary](../../../../glossary.md#term-mutable); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Retain the selected portable route plan even when OpenBLAS wins, but permit no post-analysis
  fallback. Retention preserves a complete semantic alternative for analysis/evidence; immutable
  prepared execution preserves the Runtime boundary.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [immutable glossary](../../../../glossary.md#term-immutable); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [Runtime glossary](../../../../glossary.md#term-runtime))</span>
- Use deterministic configured cost units and strict benefit thresholds. They are safe heuristic
  selection inputs, not benchmark evidence or a performance claim.<span data-glossary-navigation="true"> ([heuristic glossary](../../../../glossary.md#term-heuristic))</span>
- Put the first concrete CPU backend-conformance case in the existing conformance package and
  exercise the staged internal boundary directly. A test-scoped CPU dependency is the smallest
  honest integration because Engine does not exist; direct provider use would bypass the route,
  and reflection or copied CPU test fixtures would not prove the real preparation/finalization
  path. Guard the exact dependency set in the architecture suite because the build edge changes.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [dependency glossary](../../../../glossary.md#term-dependency); [Engine glossary](../../../../glossary.md#term-engine); [Guard glossary](../../../../glossary.md#term-guard))</span>

## Known limitations

- Only one bare positive [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) same-type [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) can use [OpenBLAS](../../../../glossary.md#term-openblas).
- At most one input is copied. Output copy, two-input copies, transpose, packing, batching,
  broadcasting, epilogues, and empty dimensions stay portable.<span data-glossary-navigation="true"> ([transpose glossary](../../../../glossary.md#term-transpose); [broadcasting glossary](../../../../glossary.md#broadcasting); [dimensions glossary](../../../../glossary.md#dimension))</span>
- Qualification and thread-state exclusion are explicit caller/composition responsibilities.
  CPU cannot prevent another handle, class loader, or arbitrary native caller from changing the
  shared OpenBLAS state or closing an aliased owner incorrectly.<span data-glossary-navigation="true"> ([Qualification glossary](../../../../glossary.md#term-qualification); [CPU glossary](../../../../glossary.md#term-cpu); [owner glossary](../../../../glossary.md#term-owner))</span>
- The native checkpoint qualifies only the supplied binary, process, ABI, selected thread count,
  and bounded cases. It establishes no universal OpenBLAS build, platform, determinism, or
  performance guarantee.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint); [binary glossary](../../../../glossary.md#term-binary-descriptor-and-metadata); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- No public configuration or Engine integration exists yet. The route is exercised through the
  current internal staged CPU preparation/finalization contracts in both CPU-local and backend-
  conformance tests.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- The implementation [handoff](../../../../glossary.md#term-handoff) recorded
  `./gradlew :backends:cpu:test :testing:backend-conformance:test
  :testing:architecture-tests:test` passing after executable stabilization.
- The final repository gate `./gradlew test` ran exactly once and passed in 2m53s with 63
  actionable tasks. The resulting XML reports were independently inspected by documentation
  context `/root`: CPU passed 942 tests with zero failures/errors and 28 existing opt-in skips;
  backend conformance passed 3 tests with zero failures/errors/skips; architecture passed 7 tests
  with zero failures/errors/skips.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- The required explicit CPU native checkpoint first rejected the stale exact provider-version
  path rather than discovering or substituting another library. The coordinator then supplied<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
  `/opt/homebrew/Cellar/openblas/0.3.34/lib/libopenblasp-r0.3.34.dylib`; the exact documented
  checkpoint command passed and printed
  `CPU OpenBLAS native checkpoint passed; restored thread count 16`. It exercised FLOAT32 and
  FLOAT64 direct and one-copy prepared routes under the documented bounded gamma/ULP and
  exceptional-class checks.<span data-glossary-navigation="true"> ([FLOAT32 glossary](../../../../glossary.md#term-float32); [FLOAT64 glossary](../../../../glossary.md#term-float64); [ULP glossary](../../../../glossary.md#term-unit-in-the-last-place-ulp))</span>
- `javap -public` confirmed that `CpuOpenBlasInvocation` exposes only `isOpen`, `threadCount`,
  `sgemm`, and `dgemm`, using [JDK](../../../../glossary.md#term-java-development-kit-jdk) [`MemorySegment`](../../../../glossary.md#term-memorysegment) and no provider type. Production-source search
  found [`OpenBlasLibrary`](../../../../glossary.md#openblas-library-handle--openblaslibrary) only in the finalization adapter in `CpuOpenBlasRouteSelector`; the
  explicit native checkpoint is the only test-source provider reference. Backend conformance is
  provider-free, Engine-free, and native-free.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>
- Documentation context `/root` ran `./gradlew :backends:cpu:javadoc` after its initial Javadoc
  pass and reran it after the final failure-contract additions; both passed. The final run was<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [run glossary](../../../../glossary.md#term-run))</span>
  `BUILD SUCCESSFUL` in 3s with 11 actionable tasks. Its 94 total warnings comprise the expected
  Java 26 incubating-Vector notices and pre-existing missing-parameter warnings on older
  compatibility constructors in <span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector); [parameter glossary](../../../../glossary.md#term-parameter))</span>`CpuPartitionLowering` and `CpuPartitionPreparationPlan`; none
  names a new OpenBLAS declaration or this task's changed Javadocs.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- The changed generated pages for the OpenBLAS package and its four types, analysis inputs,
  preparation plan/preparer/finalizer, representation planner, and partition composite were
  rendered to plain text with Pandoc and inspected for descriptions, parameter/result/failure
  sections, ownership, route selection, provider-free ABI, copy resources, and late-fallback
  exclusions.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Documentation context `/root` ran
  `./gradlew :testing:architecture-tests:test`; it passed as up-to-date in 506ms with 3 actionable
  tasks, reusing the still-current successful architecture result because no executable test or
  build file changed during documentation review.
- The final targeted Markdown checker passed for the CPU guide, glossary, task, CPU master plan,
  and roadmap: local files/anchors, heading uniqueness, balanced fences, LF endings, final
  newlines, and trailing whitespace are valid. Final status checks show CPU 0010 `Complete`, CPU
  0011 as the next `Draft` [CPU](../../../../glossary.md#term-cpu) frontier, and CPU 0012 through 0017 still `Draft`.
- Final scope inspection found exactly 24 changed/new paths, within the 29-path ceiling and exact
  task allowlist. <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`git diff --check` passed. No ordinary Java suite, provider suite, benchmark,
  tuning workflow, generated portable evidence, or integration test was rerun by the
  documentation context.<span data-glossary-navigation="true"> ([integration test glossary](../../../../glossary.md#term-integration-test))</span>

## Implementation notes

- CPU analysis inputs now carry disabled-by-default expected native-storage facts and a typed,
  immutable qualification/thread/cost snapshot. The selector consumes only the complete common
  portable MATMUL plan and fails closed for every uncertain eligibility or checked-cost branch.<span data-glossary-navigation="true"> ([carry glossary](../../../../glossary.md#term-carry); [immutable glossary](../../../../glossary.md#term-immutable); [qualification glossary](../../../../glossary.md#term-qualification); [snapshot glossary](../../../../glossary.md#term-snapshot); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
- The selected route plan records exact type, `m`, `n`, `k`, boundary positions, single-thread
  configuration, direct/copy representation, optional existing affine materialization and exact
  workspace, and checked whole-plan cost/benefit facts. It retains no provider or run resource.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [run glossary](../../../../glossary.md#term-run))</span>
- Finalization borrows an open single-thread provider-free invocation. Direct plans create one
  native child; copied plans reuse the existing copy-first atomic composite. Cold binding checks
  provider state, carriers, type, native storage, accessibility, alignment, spans, writability,
  and all write overlaps before mutation. Hot work performs exactly one typed GEMM call and never
  falls back or reselects a route.<span data-glossary-navigation="true"> ([atomic glossary](../../../../glossary.md#resource-transaction-and-rollback); [binding glossary](../../../../glossary.md#term-binding); [alignment glossary](../../../../glossary.md#term-alignment); [GEMM glossary](../../../../glossary.md#general-matrix-multiplication--gemm))</span>
- Ordinary tests use fakes only. Backend conformance constructs the existing staged CPU boundary
  without Engine or the provider. The explicit checkpoint alone adapts the caller-supplied real
  provider and owns the bounded installed-library evidence.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Documentation context `/root` independently read the architecture, named ADRs, provider plans,
  documentation profiles, final implementation/test/build diff, and recorded reports. It
  finalized only Javadocs/package documentation in already changed production files plus the CPU
  guide, glossary, this task, CPU master plan, and roadmap; executable semantics were preserved.
- No-change review: `CpuCapabilityProvider` remains the sole supported [CPU](../../../../glossary.md#term-cpu) [API](../../../../glossary.md#term-application-programming-interface-api) and
  `BackendId("cpu")`/capability reporting are unchanged. Public Tensor, Compile, Training, and
  Config APIs; Model MATMUL semantics; provider API/source/tests/tasks; shared Prepare and Runtime;
  Engine; architecture contracts and ADRs; other backend-conformance sources; integration tests;
  other Gradle builds; other modules; and portable MATMUL/generated schema and evidence remain
  unchanged. The sole architecture addition enforces only the exact backend-conformance
  test-dependency surface.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [Compile glossary](../../../../glossary.md#compile); [Model glossary](../../../../glossary.md#term-model); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Gradle glossary](../../../../glossary.md#term-gradle); [modules glossary](../../../../glossary.md#term-module); [schema glossary](../../../../glossary.md#term-schema); [dependency glossary](../../../../glossary.md#term-dependency))</span>

## Completion summary

- Completed changes: implemented the qualified narrow OpenBLAS CPU MATMUL selection, exact direct
  or one-affine-input-copy resource path, finalization-only borrowed invocation, fail-before-write
  binding, one-call prepared execution, native-free CPU/conformance coverage, exact dependency
  enforcement, explicit native checkpoint, and synchronized Javadoc/guide/glossary/planning.<span data-glossary-navigation="true"> ([OpenBLAS glossary](../../../../glossary.md#term-openblas); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease); [binding glossary](../../../../glossary.md#term-binding); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [conformance glossary](../../../../glossary.md#term-conformance); [dependency glossary](../../../../glossary.md#term-dependency); [checkpoint glossary](../../../../glossary.md#term-checkpoint); [Javadoc glossary](../../../../glossary.md#term-javadoc); [planning glossary](../../../../glossary.md#term-planning))</span>
- Files changed or created: 11 CPU production paths, 5 CPU test/checkpoint paths, the
  backend-conformance build and conformance test, one architecture dependency test, the CPU guide,
  glossary, this task, CPU master plan, and roadmap—24 paths total. Final status/diff inspection
  recorded the exact list, all within <span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend))</span>[Affected files](#affected-files), with no provider,
  architecture-contract, ADR, integration, shared-module, or unrelated documentation path.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [module glossary](../../../../glossary.md#term-module))</span>
- Tests and validation: focused CPU/conformance/architecture tests passed; the sole final root
  test passed with the recorded 942/3/7 project counts; the caller-supplied OpenBLAS 0.3.34
  checkpoint passed and restored count 16; CPU Javadoc, generated-page inspection, ABI/source
  checks, architecture rerun, Markdown/status/scope checks, and whitespace validation passed.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [scope glossary](../../../../glossary.md#term-scope))</span>
- Documentation impact: current guide and glossary text now distinguish portable MATMUL, the
  narrow CPU-owned OpenBLAS route, and the provider-only invocation/lifecycle boundary. Every
  affected OpenBLAS/configuration/plan/finalization/execution Javadoc and the native package
  summary records ownership, constraints, resources, failures, and the no-late-fallback rule.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Unresolved issues: none within task 0010. Integration remains deliberately deferred because no
  public Engine lifecycle exists; broader native shapes, providers, and tuning remain separate
  Draft work.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [shapes glossary](../../../../glossary.md#shape))</span>
- Required follow-up: none for task 0010. [CPU](../../../../glossary.md#term-cpu) 0011 remains the next Draft CPU frontier.

Status: Complete
