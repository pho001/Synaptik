# Task 0009E1B2: Statistical Reduction Route Decision

## Status

Ready

## Goal

Decide whether the current CPU `VARIANCE` and `STANDARD_DEVIATION` reductions retain their specialized generated route or are replaced by finite direct typed Java. Select direct Java only when its total implementation and verification cost is lower. This task has no benchmark, performance gate, or performance claim.

## Scope

Inspect exactly the two current one-input/one-output statistical meanings: `AggregateReductionKind.VARIANCE` and `AggregateReductionKind.STANDARD_DEVIATION`, each with `StatisticalReductionAttrs`. Admission is identical `FLOAT64`, `FLOAT32`, or `BFLOAT16` input and output; non-null, normalized, ordered, distinct non-negative selected axes (including an empty list); either retained or removed selected dimensions; and non-negative `correction`. The reduced output Shape must be exact, and checked selected-domain count `N` must exceed `correction`.

The decision covers the current static resolved non-negative layouts: dense-linear or general-odometer input, and dense-linear or general-odometer injective output; heap, `MemorySegment`, and ordered mixed typed input/output carriers; two borrowed boundary carriers; and scalar or disjoint parallel-scalar ranges that each own complete output cells. It preserves the shared `CpuAdvancedReductionIr` / `CpuAdvancedReductionLowering.Geometry` boundary: cold packed input/output bases and static geometry, two passes, no materialized input or partial/combine state, exactly one final typed store for every owned cell, and exact scratch state for every concurrently selected range when output cells exist.

Preserve the current corrected two-pass algorithm exactly: first pass classifies values and uses the per-range `CpuExactSumEmitter` state to form the exact represented mean rounded at the current represented seam; second pass uses compensated deviation and square sums, subtracts `deviations² / N`, clamps a negative numerator to positive zero, divides by `N - correction`, and applies `Math.sqrt` only for standard deviation. Empty selected domains and `N <= correction` are rejected cold for statistics; an empty axes list is a point domain and produces zero only when its `N = 1` satisfies `N > correction` (therefore, under the current rule, only with correction zero). Canonical NaN input or any infinity produces canonical NaN; negative-zero-only variance is positive zero; overflowing finite squares produce positive infinity unless a special-value rule overrides it. Preserve FLOAT32/BFLOAT16 narrowing, including canonical NaN narrowing.

Compare `CpuStatisticalReductionEmitter`, `CpuExactSumEmitter`, shared typed carrier/address and selected-domain helpers, generator/artifact identity, Prepare declaration/finalization, immutable prepared binding/invocation, independent reference oracle, and current semantic/inventory/evidence owners with a complete finite direct replacement and retirement proof. A direct replacement must cold-select finite typed array/segment and dense/general forms and bind an immutable range-owning entry; it must not perform generic carrier, type, layout, kind, reflection, map/string, or virtual semantic dispatch per element. If generation remains, retain a matching optimal clean-Java oracle: the same two passes, exact-mean state and compensation order, special handling, narrowing, and one store, with no avoidable allocation, boxing, reflection, helper semantic dispatch, or per-element virtual indirection in the generated hot loop.

## Out of scope

`LOG_SUM_EXP` (Complete E1B1), `L1_NORM`/`L2_NORM` (Draft E1B3), softmax/log-softmax, normalization, loss, masked or ordinary reductions, public Model/Compiler semantics or gradients, architecture/build/shared-boundary changes, new resource kinds, vector/native routes, benchmarks, and performance assertions. Do not create detailed E1B3 or later specifications. Legacy `pre-rewrite` is read-only behavioral evidence; do not copy source, packages, or dependencies.

## Architecture references and constraints

Read [the architecture contract](../../../../../ARCHITECTURE.md), [current architecture plan](../../../../architecture/current-architecture-plan.md), [Planning Guide](../../../planning-guide.md), [documentation rules](../../../../developer-guide/documentation-rules.md), CPU master plan, parent [CPU 0009](0009-portable-generated-coverage-closure-checkpoint.md), completed [E1](0009e1-partial-integral-reduction-direct-java-migration.md), [E1A](0009e1a-masked-reduction-route-decision.md), [E1B1](0009e1b1-log-sum-exp-route-decision.md), and [0007D](0007d-portable-logarithmic-statistical-and-norm-reduction-coverage.md).

`ARCHITECTURE.md` is authoritative. CPU Prepare alone owns lowering, finite route selection, resource declaration, and finalization after shared slot assignment; Runtime only invokes immutable prepared work. Preserve borrowed caller carriers, run-owned scratch, one `RunState` for an active run, cold validation before mutation or worker submission, and prepared-region ownership. Do not add a public/shared dependency, generic executor, workspace/resource kind, architecture rule, or build change; stop and report any need for one.

## Package impact

Existing packages only:

- `internal.ir` and `internal.lowering` retain structural identity and static geometry.
- `internal.codegen.emit` retains the generated statistical body and exact-state/helper seams if generation is selected.
- `internal.prepare`, `internal.executable`, and `internal.cache` retain cold selection, resource finalization, artifact identity, and immutable invocation.
- `internal.reference` retains the independent oracle.

A retained route adds no type. A selected direct route may add at most one narrowly named CPU-private statistical executable owner in an existing package; it must not absorb norm or log-sum-exp algorithms.

## Affected files and maximum scope

Expected planning paths are this task, CPU master plan, parent CPU 0009, and roadmap. Review-only seams are `CpuCapabilityProvider`, `CpuAdvancedReductionIr`, `CpuAdvancedReductionLowering`, `CpuStatisticalReductionEmitter`, `CpuExactSumEmitter`, `CpuNormEmitter`, `CpuClassFileKernelGenerator`, preparation/finalization and `CpuPreparedExecutable`, `CpuAdvancedReductionReferenceKernel`, and focused capability/IR/lowering/generated/reference/prepare/executable/inventory/evidence tests.

Implementation may modify or create at most 30 repository paths: at most 16 production/Javadoc, ten test, and these four planning paths. At most one new CPU-private direct production type is permitted. A 31st path, a second new production type, changed shared advanced geometry/resource contract, materialization, partial/combine state, shared/public/build/architecture/conformance/integration path, or generic element dispatch is a stop-and-replan condition.

## Acceptance criteria

- The final decision covers the complete current two-kind matrix and says whether generation is retained or a finite direct typed route replaces it, with concrete implementation-plus-verification cost evidence and no performance claim.
- The chosen route preserves `StatisticalReductionAttrs` axes, keep-dimensions, correction, `N > correction` admission, Shape/layout/carrier/range boundary, complete-cell ownership, run-owned aligned exact-state slice isolation, and cold failure before writes/submission.
- It preserves corrected exact-mean two-pass semantics, empty/point-domain distinction, singleton, invalid denominator, canonical NaN/infinity/signed-zero/overflow behavior, represented narrowing, and one final output store exactly as current source and tests require.
- A direct route has finite cold typed carrier/layout selection and immutable binding with no generic per-element dispatch; generation is retired only after all selected artifact, cache, invocation, inventory, evidence, and no-selected-reference proof is updated.
- A retained route changes no executable path and records the typed carrier/layout, scratch, emitter/helper, artifact/binding, oracle, and retirement costs that make replacement larger.
- Focused semantic, special-value, correction/rejection, type/narrowing, layout/carrier, range/canary, scratch isolation, preparation/finalization, invocation, reference, inventory, and route-retirement tests pass. Retained generation receives proportionate class-file/member and hot-loop inspection against the clean-Java oracle; direct code receives equivalent source and call-site hygiene review.
- Before `Complete`, a separate documentation-focused clean context finalizes affected Javadocs, explanatory documentation, glossary impact, planning status, and documentation validation.

## Tests / validation

After executable stabilization, the implementation context runs focused CPU tests selected from the existing statistical owners, including capability, advanced IR/lowering/generated/reference, specialized matrix/direct-evidence/coverage inventory, and affected prepare/finalizer/executable tests. Record exact commands and results here. Do not run benchmarks or repository-wide tests.

The documentation context reuses stable Java evidence unless it changes executable Java; it checks Markdown links, anchors, fences, terminology/glossary impact, changed-path scope, statuses, dependencies, and frontier consistency, then runs:

```bash
git diff --check
git status --short -uall
```

Repository-wide, architecture, backend-conformance, integration, and performance validation are deferred to CPU 0009 final checkpoint or CI because this task must not change a shared boundary.

## Dependencies and follow-up tasks

- E1, E1A, and E1B1 are Complete; E1B1 retains the zero-workspace log-sum-exp route.
- Complete 0007D supplies the existing advanced IR/lowering, statistical generated emitter, exact-state workspace, reference oracle, and focused evidence.
- E1B3 remains Draft and depends on this decision; E1C, E2, E3, F, and G remain ordered Draft summaries only.

## Architecture impact

Expected impact: None. Stop and report any public, dependency, resource, module, or architecture conflict rather than changing that boundary.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU task 0009E1B2. Work on the existing worktree without committing, pushing, staging, resetting, reverting, deleting, or modifying unrelated work. Do not use a GSD skill or workflow.

Read AGENTS.md, ARCHITECTURE.md, current architecture plan, Planning Guide, CPU master plan, parent CPU 0009, completed E1/E1A/E1B1/0007D, this task, current statistical advanced-reduction source and focused tests, documentation rules, and General/Planning profiles. Implement exactly this route decision. Choose finite direct typed Java only if total implementation and verification cost is lower; otherwise retain generation with concrete evidence. Preserve the complete current VARIANCE/STANDARD_DEVIATION boundary. Stop on an architecture or scope conflict. Hand stabilized executable diff and exact test evidence to a separate documentation-focused clean context before marking Complete. Do not run benchmarks or repository-wide tests. Do not commit, push, or stage.
```

## Local decisions

E1B2 owns both statistics because they share the corrected exact-mean state, two-pass traversal, denominator correction, and only differ in the final square root. It follows E1B1 because the logarithmic route has zero workspace; it precedes E1B3 because norms have different numerical state and finishing rules despite shared geometry.

## Known limitations

This is a route decision, not a numerical redesign, public API change, or performance result. Only current static scalar/parallel-scalar resolved-layout forms are in scope; dynamic Shapes, vectorization, fusion, materialization, partial/combine execution, and native routes remain out.

## Validation evidence

Empty until implemented.
## Implementation notes

Empty until implemented.

## Completion summary

Empty until implemented.
