# Task 0009E1B1: Log-Sum-Exp Route Decision

## Status

Ready

## Goal

Decide whether the current CPU `LOG_SUM_EXP` floating reduction should retain its specialized generated route or be replaced by finite direct typed Java. Select direct Java only when it lowers the total implementation and verification cost; this task has no benchmark or performance gate.

## Scope

Inspect exactly one static, one-input/one-output advanced-reduction meaning: `AggregateReductionKind.LOG_SUM_EXP` with `MultiAxisReductionAttrs`, identical `FLOAT64`, `FLOAT32`, or `BFLOAT16` input/output type, normalized ordered selected axes, and either retained or removed selected dimensions. The decision covers the current resolved non-negative dense-linear and general-odometer layouts, heap, `MemorySegment`, and mixed typed carriers, injective output, and scalar or disjoint complete-output-cell ranges.

Preserve the current shared `CpuAdvancedReductionIr` and `CpuAdvancedReductionLowering.Geometry` boundary: two buffers, cold packed bases/geometry, complete output-cell ownership, static selected-domain count, no materialization, no partial or combine state, and zero log-sum-exp scratch. Preserve the existing two-pass maximum-shift and compensated exponential-sum algorithm, singleton represented-value path, empty-domain negative infinity, canonical NaN, positive infinity, signed-zero, narrowing, and cold validation before mutation or worker submission.

Compare `CpuLogSumExpEmitter` together with its shared carrier/address/classification helpers, generated artifact identity and invocation, finalization, executable binding, reference oracle, and focused inventory/evidence owners against a complete finite direct replacement and retirement proof. A selected direct route must cold-select finite typed array/segment, dense/general entry forms and must not introduce generic per-element carrier, kind, layout, or reflection dispatch. A retained route changes no executable production/test path and records why replacement costs more overall.

The task must not decide variance, standard deviation, L1 norm, or L2 norm. Those owners share geometry but have materially separate corrected/exact-state or norm numerical algorithms and remain ordered E1B2/E1B3 summaries.

## Out of scope

`VARIANCE`, `STANDARD_DEVIATION`, `L1_NORM`, `L2_NORM`, masked and ordinary reductions, softmax/log-softmax, normalizations, losses, vector/native routes, Model/Compiler semantics or gradients, shared Prepare/Runtime contracts, architecture, build files, benchmarks, and new performance claims. Do not create detailed E1B2, E1B3, E1C, E2, E3, F, or G specifications. Legacy is read-only behavioral evidence; do not copy its source.

## Architecture references

Read [the architecture contract](../../../../../ARCHITECTURE.md), [the current architecture plan](../../../../architecture/current-architecture-plan.md), [the Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md), the CPU master plan, parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [CPU 0009E1](0009e1-partial-integral-reduction-direct-java-migration.md), completed [CPU 0009E1A](0009e1a-masked-reduction-route-decision.md), and completed [CPU 0007D](0007d-portable-logarithmic-statistical-and-norm-reduction-coverage.md).

## Architecture constraints

`ARCHITECTURE.md` is authoritative. CPU Prepare owns lowering, route selection, exact-resource declaration, and finalization after shared slot assignment; Runtime invokes immutable prepared work only. Preserve borrowed carriers, run-owned resources, one `RunState` per active run, cold validation before writes, and prepared-region ownership. A retained generated loop remains subject to the optimal-clean-Java oracle and hot-loop hygiene: no avoidable allocation, boxing, reflection, string/map dispatch, semantic dispatch, or per-element virtual indirection. Do not add a public/shared dependency, resource kind, generic executor, architecture rule, or build change; stop and report any need for one.

## Package impact

Existing packages used:

- `internal.ir` — shared advanced-reduction structural identity.
- `internal.lowering` — shared static output-cell geometry.
- `internal.codegen.emit` — generated log-sum-exp body and its current private helper seams.
- `internal.prepare`, `internal.executable`, and `internal.cache` — cold route realization, artifact finalization, and invocation.
- `internal.reference` — independent mathematical oracle.

Packages added or changed:

- None for a retained route. A selected direct route may add one narrowly named CPU-private direct log-sum-exp owner in an existing `internal` package only after its finite entry/type map is documented in this task and the CPU master plan.

Type placement:

- No new type is authorized for a retained route.
- A selected direct owner, if needed, belongs beside the existing CPU-private executable route; it must not be public or shared and must not absorb statistical or norm algorithms.

## Affected files

Expected decision/planning paths:

- this task, `docs/planning/backends/cpu/master-plan.md`, parent CPU 0009, and `docs/planning/roadmap.md`.

Review-only implementation seams:

- `CpuAdvancedReductionIr`, `CpuAdvancedReductionLowering`, `CpuLogSumExpEmitter`, its shared `CpuNormEmitter` helper seam, `CpuGeneratedKernel`, generated artifact/cache identity, partition preparation/finalization, `CpuPreparedExecutable`, `CpuAdvancedReductionReferenceKernel`, and their focused capability/IR/lowering/generated/reference/prepare/executable/inventory/evidence tests.

## Maximum scope

The implementation, tests, documentation, and planning may modify or create at most 28 repository paths: at most 15 production/Javadoc paths, eight test paths, and the four named planning paths. At most one new CPU-private direct log-sum-exp production type is permitted.

A 29th path, a second new production type, any change to shared advanced geometry that changes statistics or norms, a new workspace/resource kind, materialization, partial/combine state, a shared/public/build/architecture/conformance/integration path, or a direct route with generic element dispatch is a stop-and-replan condition.

## Acceptance criteria

- The final decision covers the complete current `LOG_SUM_EXP` matrix and states whether the generated route is retained or a finite direct typed route replaces it; it gives a concrete implementation-plus-verification cost comparison and makes no performance claim.
- `LOG_SUM_EXP` preserves the current attribute/type/layout/carrier/range admission and exact rejection boundary; statistics and norms remain untouched and independently ordered.
- The selected implementation preserves the exact two-pass max-shift, compensated-sum, singleton, empty-domain, special-value, represented narrowing, one-final-store, complete-cell range, and cold-failure-before-write behavior documented by the current source and tests.
- If direct is selected, finite cold route selection and typed entry descriptors cover the current carrier/layout forms without generic per-element dispatch; every selected generated artifact, cache route, invocation path, inventory row, and test/evidence reference is retired or updated.
- If generation is retained, production Java and executable tests remain unchanged and the task records the retained emitter/helper, typed carrier/layout, artifact, binding, oracle, and retirement costs that make direct replacement larger.
- Focused semantic, special-value, layout/carrier, range/canary, prepare/finalization, executable, reference, inventory, and route-retirement tests pass for the selected route. Generated code, when retained, receives proportionate class-file/member/hot-loop inspection against a matching optimal clean-Java oracle; direct code receives equivalent source/call-site hygiene review.
- A separate documentation-focused clean context finalizes any affected Javadocs, explanatory documentation, glossary impact, planning status, and documentation validation in the same overall change. If a retained decision changes only planning, it records the explicit no-change conclusion for Javadocs, glossary, and executable documentation.

## Tests / validation

The implementation context runs focused tests selected from the existing owners after executable code stabilizes, including the advanced IR/lowering/generated/reference suites and the affected capability, prepare/finalizer, executable, inventory, and evidence suites. It must record exact commands and results in this task; do not run a repository-wide suite or benchmarks.

The documentation-focused context reuses stable Java evidence unless it changes executable Java. It validates Markdown links, anchors, fences, terminology, whitespace, exact changed paths, status/dependency/frontier consistency, and then runs:

```bash
git diff --check
git status --short -uall
```

Repository-wide, architecture, backend-conformance, integration, and performance validation remain deferred to CPU 0009's final checkpoint or CI because this task changes no shared boundary.

## Dependencies

- CPU 0009E1A is Complete and retains the masked generated route.
- Complete CPU 0007D supplies the existing advanced IR/lowering, five-kind emitter split, reference oracle, static floating matrix, and historical generated evidence.
- Existing Prepare/Runtime contracts already carry the two-buffer, zero-workspace log-sum-exp route without modification.

## Follow-up tasks

- CPU 0009E1B2 remains a Draft summary for variance and standard deviation; it depends on this decision because its corrected two-pass exact-mean workspace route must not be folded into it.
- CPU 0009E1B3 remains a Draft summary for L1/L2 norms; it follows E1B2 and keeps its separate exact-absolute-sum/scaled-squares numerical ownership.
- CPU 0009E1C, E2, E3, F, and G remain ordered summaries only.

## Architecture impact

Expected impact: None.

If the decision requires a public/shared contract, a dependency, a resource kind, a module change, or an architecture rule, stop and report the conflict rather than changing that boundary.

## Implementation prompt

Use this prompt in a separate clean-context implementation task/thread:

```text
You are the clean implementation agent for Synaptik CPU task 0009E1B1. Work on the existing worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, the current architecture plan, planning guide, CPU master plan, parent CPU 0009, completed E1/E1A/0007D, this task, current advanced-reduction source and focused tests, documentation rules, and the General and Planning profiles.

Implement exactly this route decision. Choose finite direct typed Java only if it lowers total implementation and verification complexity; otherwise retain generation with concrete evidence. Preserve the complete current LOG_SUM_EXP boundary and do not decide statistics or norms. Stop on an architecture or scope conflict. Hand any stabilized executable diff and test evidence to a separate documentation-focused clean context before marking the task Complete. Do not run benchmarks or repository-wide tests. Do not commit, push, or stage.
```

## Local decisions

- Split E1B by numerical ownership, not by shared geometry. `LOG_SUM_EXP` owns a two-pass max-shift/compensated exponential sum with zero workspace; statistics own corrected exact-mean state and a distinct second pass; norms own exact absolute summation or scaled squares. A single migration decision would exceed the useful review and verification boundary.
- The first detailed child is log-sum-exp because it is the only zero-workspace advanced member and therefore exposes the direct-versus-generated route question without statistical scratch or norm state decisions.

## Known limitations

- This is a route decision, not a new numerical algorithm, public API, or performance result.
- Only the already supported static resolved-layout scalar/parallel-scalar CPU route is in scope; dynamic Shapes, vectorization, fusion, materialization, partial/combine execution, and native routes remain outside it.

## Validation evidence

Planning-only creation: source and focused-test seams were inspected before setting this task Ready. No Java command is run because this change creates no executable behavior. The implementing and documentation contexts must replace this section with exact selected-route evidence before marking the task Complete.

## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
