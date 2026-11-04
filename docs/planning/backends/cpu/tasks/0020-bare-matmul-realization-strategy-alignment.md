# Task 0020: Bare [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) realization and strategy [alignment](../../../../glossary.md#term-alignment)

## Reader context — historical integration boundary

Vector-eligible geometry is not the same as selecting a vector realization. CPU's cold MATMUL
selector must agree with the requested compute preference, generated strategy and species width
before finalization; Runtime does not repair a mismatch. The scalar preference still computes every
matrix output and contraction, merely using scalar instructions/work units. The public INT32
regression intentionally avoids the optional FLOAT32/FLOAT64 OpenBLAS peer, so that it exercises the
portable path rather than hiding it behind native selection. Historical red/green commands below
describe the original repair, not checks rerun by this reading pass.

Current reading: [Portable MATMUL realization](../../../../backend-guide/cpu-backend.md#current-portable-matmul-family).

## Status

Complete — implemented and independently reviewed on 2026-10-10 from clean `main` at
`f4e35b45`. [CPU](../../../../glossary.md#term-cpu) 0008F, 0010M, 0018, and 0019 were Complete at launch; no conflicting CPU
preparation writer was active. CPU 0007A1D is Superseded and supplies no performance acceptance
for this correctness task.

## Change class

Class C — selected [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) loop form, generated [specialization](../../../../glossary.md#term-specialization), and prepared work domain must
describe the same hot-path algorithm. A clean implementation context and independent targeted
review/documentation context are mandatory.

## Goal

Make a bare portable [CPU](../../../../glossary.md#term-cpu) [`MATMUL`](../../../../glossary.md#matrix-multiplication--matmul) executable under the default [`SCALAR`](../../../../glossary.md#term-scalar) compute preference even
when its geometry also makes a [Vector API](../../../../glossary.md#term-vector-api) realization eligible. Keep explicit
`VECTOR_IF_ELIGIBLE` selection and existing scalar/[vector](../../../../glossary.md#term-vector) algorithms intact.

## [Scope](../../../../glossary.md#term-scope)

- Establish a failing regression before the fix for a same-type, contiguous rank-two MATMUL
  such as <span data-glossary-navigation="true"> ([regression glossary](../../../../glossary.md#term-regression); [contiguous glossary](../../../../glossary.md#term-contiguous); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>`FLOAT32 [2,63] × [63,128]`. Source audit currently shows
  `CpuMatmulCandidateSelector.select` chooses `DIRECT_N_VECTOR` from geometry without reading
  the compute preference, while `CpuPartitionPreparer` chooses vector strategy and species only
  under <span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector))</span>`VECTOR_IF_ELIGIBLE`. This is a source-level inference; record the observed failure stage
  and message rather than assuming it.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference))</span>
- Align the selected `CpuMatmulIr.Realization`, `ExecutionStrategy`, specialization strategy and
  species bits, work count, and generated route before finalization. With <span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>[`SCALAR`](../../../../glossary.md#term-scalar), choose the
  already-eligible direct or 2×2 scalar realization; with `VECTOR_IF_ELIGIBLE`, retain the
  eligible direct-N or 2×2 vector realization and its exact preferred species. Preserve the
  established geometry thresholds and candidate order unless a test proves they are themselves
  inconsistent.
- Cover the four [homogeneous](../../../../glossary.md#term-homogeneous) [Vector API](../../../../glossary.md#term-vector-api) types ([`FLOAT32`](../../../../glossary.md#term-float32), [`FLOAT64`](../../../../glossary.md#term-float64), [`INT32`](../../../../glossary.md#data-type--datatype), [`INT64`](../../../../glossary.md#data-type--datatype)), small
  scalar work, both direct and tiled eligible shapes, non-unit-stride/terminal ineligibility,
  and the portable public path. Use <span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [stride glossary](../../../../glossary.md#term-stride))</span>`INT32` for an exact public Engine regression because the
  optional OpenBLAS peer cannot mask that CPU route.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [OpenBLAS glossary](../../../../glossary.md#term-openblas); [mask glossary](../../../../glossary.md#term-boolean-mask); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- Add a cold fail-closed consistency check only if needed to prevent a future mismatch between
  the typed MATMUL realization and generated specialization. Keep it out of per-element work.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [element glossary](../../../../glossary.md#term-element))</span>

## Non-goals

No new [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) algorithm, [Vector API](../../../../glossary.md#term-vector-api) emitter rewrite, default/preference change, threshold
retuning, performance speedup claim, [OpenBLAS](../../../../glossary.md#term-openblas) selection change, mixed/low-[precision](../../../../glossary.md#term-precision) capability,
[fusion](../../../../glossary.md#term-fusion)/[epilogue](../../../../glossary.md#term-epilogue) redesign, [CPU](../../../../glossary.md#term-cpu) 0018 recognition change, or [Model](../../../../glossary.md#term-model)/[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)/[Engine](../../../../glossary.md#term-engine) production
[API](../../../../glossary.md#term-application-programming-interface-api) change. If the correction needs generated [bytecode](../../../../glossary.md#term-bytecode) or a broader [owner](../../../../glossary.md#term-owner), stop and replan.

## Contracts

- [`ARCHITECTURE.md` — Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements) — backend-owned
  route decisions, fixed prepared execution, backend conformance and end-to-end regressions.<span data-glossary-navigation="true"> ([backend glossary](../../../../glossary.md#backend); [prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- [Backend execution — Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>
  — CPU owns candidate selection, lowering, generated specialization, and route consistency.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- [Runtime/Prepare/Engine — `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine))</span>,
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  and [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — shared preparation assigns resources but does not reinterpret CPU-private realization.

If a required contract is missing or ambiguous, stop and request clarification.

## Dependencies and integration

- Depends on: [CPU](../../../../glossary.md#term-cpu) 0008F, 0010M, 0018, and 0019 (all Complete)
- Conflicts with: concurrent edits to CPU MATMUL candidate/lowering, execution-strategy or
  specialization selection, their tests, or the affected CPU guide<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [lowering glossary](../../../../glossary.md#lowering); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Parallel group: None
- Common base revision: `f4e35b45` on clean `main`; serial work
- Integration order: reproduce → align cold selection → focused tests → independent Class C
  review → integrate planning status<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning))</span>
- Integration validation: focused CPU, backend-conformance, and Engine tests below, then the
  affected CPU module suite and Javadoc once on stabilized code<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [backend glossary](../../../../glossary.md#backend); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine); [module glossary](../../../../glossary.md#term-module); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Shared-document integration owner: Main planner for CPU master plan and roadmap; implementation
  and review contexts own only this brief and affected CPU guidance<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>
- Frontier verification (2026-10-10): declared predecessors are Complete; CPU 0007A1D is
  Superseded; no other CPU implementation task is Ready or conflicting writer active; the root
  and scoped CPU/Prepare contracts are stable at <span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [Prepare glossary](../../../../glossary.md#prepare))</span>`f4e35b45`.

## Files and symbols

- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/lowering/CpuMatmulLowering.java`
  — `CpuMatmulCandidateSelector.select`, typed realization construction, and affected [Javadoc](../../../../glossary.md#term-javadoc).
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/internal/prepare/CpuPartitionPreparer.java`
  — MATMUL compute strategy, species, range geometry, specialization, and affected Javadoc only
  if consistency requires a change there.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization))</span>
- Focused `CpuMatmulCandidateSelectorTest`, `CpuMatmulLoweringTest`, and relevant finalizer or
  generated-kernel tests under <span data-glossary-navigation="true"> ([kernel glossary](../../../../glossary.md#kernel))</span>`backends/cpu/src/test/java/` — regression and scalar/vector
  controls; do not change unrelated family tests.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector))</span>
- `testing/backend-conformance/` — portable [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) strategy/realization [conformance](../../../../glossary.md#term-conformance).
- `testing/integration-tests/` — exact bare [`INT32`](../../../../glossary.md#data-type--datatype) public [Engine](../../../../glossary.md#term-engine) numerical [regression](../../../../glossary.md#term-regression).
- `docs/backend-guide/cpu-backend.md` — reconcile the existing compute-preference and MATMUL
  route explanation with the observed corrected behavior.

## Acceptance criteria

- A pre-fix focused test demonstrates the mismatch or its concrete failure for a vector-eligible
  bare MATMUL under default <span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>[`SCALAR`](../../../../glossary.md#term-scalar); retain the red evidence in the task result.
- Under `SCALAR`, typed IR realization is scalar, specialization compute is scalar with species
  bits zero, and the work domain matches the chosen direct or tiled scalar form. Preparation,
  finalization, and execution succeed with correct results.<span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [specialization glossary](../../../../glossary.md#term-specialization))</span>
- Under `VECTOR_IF_ELIGIBLE`, the same eligible geometries retain the direct-N or tiled vector
  form, matching vector strategy/preferred species and work domain. Explicit scalar, vector,
  parallel-scalar, and parallel-vector orchestration remains internally consistent where legal.
- For all four homogeneous vector-capable types, eligible direct/tiled and ineligible controls
  keep their supported behavior. The public bare <span data-glossary-navigation="true"> ([homogeneous glossary](../../../../glossary.md#term-homogeneous))</span>[`INT32`](../../../../glossary.md#data-type--datatype) graph returns exact matrix-product
  values through <span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>`Engine.standard()`; floating regressions use the existing operation/dtype
  test-only numerical oracle. OpenBLAS routes and low/mixed types remain unchanged.<span data-glossary-navigation="true"> ([operation glossary](../../../../glossary.md#operation); [dtype glossary](../../../../glossary.md#data-type--datatype); [numerical oracle glossary](../../../../glossary.md#adr-ci-and-qualification); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>
- Emission for a fixed specialization is unchanged. Selecting a different scalar realization
  naturally selects its existing distinct artifact identity and bytes, without changing the
  generator schema. If the emitter itself must change, stop and replan with generated-versus-
  optimal-direct Java shape and performance evidence rather than silently broadening this task.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact); [identity glossary](../../../../glossary.md#term-identity-meanings); [schema glossary](../../../../glossary.md#term-schema); [shape glossary](../../../../glossary.md#shape))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker validation after the [regression](../../../../glossary.md#term-regression) is red and after the correction stabilizes:

```bash
./gradlew :backends:cpu:test --tests '*CpuMatmulCandidateSelectorTest' --tests '*CpuMatmulLoweringTest' --tests '*CpuMatmulGeneratedKernelTest'
./gradlew :testing:backend-conformance:test --tests '*CpuMatmulStrategyConformanceTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuMatmulStrategyIntegrationTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
git diff --check
```

The integration [owner](../../../../glossary.md#term-owner) reuses those results on the exact integrated diff, checks links, glossary
impact, and status. No repository-wide suite is required unless implementation changes a shared
contract, [dependency](../../../../glossary.md#term-dependency), build configuration, or another production [module](../../../../glossary.md#term-module).

## Documentation and review impact

Review the [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) guide and Javadocs for compute-preference, selected realization, and
[specialization](../../../../glossary.md#term-specialization) semantics. Search the glossary for [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), route, and [Vector API](../../../../glossary.md#term-vector-api) terminology;
add no new term merely for this fix. Independent Class C review must inspect the final source,
test, and documentation diff and reuse successful Java evidence unless executable code changes.

## Result

Implementation pass: cold [CPU](../../../../glossary.md#term-cpu) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) selection now consumes the typed compute preference.
Default [`SCALAR`](../../../../glossary.md#term-scalar) chooses the direct or threshold-eligible 2×2 scalar realization; explicit
`VECTOR_IF_ELIGIBLE` preserves eligible direct-N and 2×2 [vector](../../../../glossary.md#term-vector) forms. Preparation checks that
typed realization, generated compute strategy, and preferred species bits agree before
finalization. No emitter, [schema](../../../../glossary.md#term-schema), shared-[module](../../../../glossary.md#term-module), or public [API](../../../../glossary.md#term-application-programming-interface-api) change was needed.

Pre-fix red evidence on `f4e35b45` production source, after adding only the focused test:
`./gradlew :backends:cpu:test --tests '*CpuMatmulLoweringTest.defaultScalarPreferenceKeepsEligibleBareMatmulScalar'`
failed (1 test, 1 failure) at preparation-plan assertion in
`CpuMatmulLoweringTest.java:113`: `expected: <DIRECT_SCALAR> but was: <DIRECT_N_VECTOR>`.
That is the observed failure stage; public [INT32](../../../../glossary.md#data-type--datatype) execution was not measured before the fix, so
this evidence does not claim a pre-fix [Engine](../../../../glossary.md#term-engine) exception or wrong numerical result.

Focused CPU, [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance), and public exact-INT32 Engine tests exercise four [homogeneous](../../../../glossary.md#term-homogeneous)
[Vector API](../../../../glossary.md#term-vector-api) types, direct/tiled choices, scalar and parallel orchestration, small and ineligible
controls. The INT32 public path cannot select the [FLOAT32](../../../../glossary.md#term-float32)/[FLOAT64](../../../../glossary.md#term-float64)-only [OpenBLAS](../../../../glossary.md#term-openblas) peer, so the
Engine [regression](../../../../glossary.md#term-regression) covers portable CPU finalization and execution.

[Validation](../../../../glossary.md#term-validation): the focused CPU command passed 22 tests (0 skipped); the focused conformance command
passed 3 (0 skipped); and the Engine command passed 1 (0 skipped). The final
`./gradlew :backends:cpu:test :backends:cpu:javadoc` passed: CPU test XML records 1,049 tests in
201 suites, 28 skipped, 0 failures, 0 errors. [Javadoc](../../../../glossary.md#term-javadoc) succeeded with 94 warnings; it was
regenerated after a Javadoc-only selector tag correction with the same warning count.
`git diff --check` passed. The glossary's existing `MATMUL`, backend-route, and CPU-portable-route
entries were reviewed; no term changed. Independent Class C review found one Javadoc omission,
which was corrected; final review returned `APPROVE` with zero P0/P1/P2 findings. Post-correction
`./gradlew :backends:cpu:javadoc` passed with the same 94 warnings and `git diff --check` passed.
No executable Java changed after the successful test [run](../../../../glossary.md#term-run). No unresolved issue or required follow-up.

Status: Complete
