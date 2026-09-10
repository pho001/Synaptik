# Task 0009F: Hybrid Route Decisions for Specialized Compute Families

## Status

Complete

## Goal

Close the next CPU 0009 route-decision frontier through ordered, independent decisions for the
existing MATMUL/convolution, pooling, attention, and batch-normalization families. Each decision
chooses the least costly maintainable complete route: retain bounded generated compute when a
replacement costs more across implementation, verification, and retirement; use finite direct
typed Java only for demonstrably cheaper static orchestration or window-loop work. Equality or
uncertainty retains the existing generated route.

## Scope

This parent governs decisions only; its ordered children each inspect their complete current
family boundary before making a route decision.

| Child | Status | Family boundary and decision responsibility |
| --- | --- | --- |
| 0009F1 | Complete | [MATMUL and convolution route decisions](0009f1-matmul-and-convolution-route-decisions.md). Separately retained MATMUL, Conv1d visible composition, and direct Conv2d/Conv3d generated bodies after source/test-backed complete-route decisions; no performance or JIT claim. |
| 0009F2 | Complete | [Pooling route decisions](0009f2-pooling-route-decisions.md). Independently retained exact Pool1d composition and the separate direct generated Pool2d/Pool3d routes; no performance or JIT claim. |
| 0009F3 | Complete | [Attention and BatchNorm route decisions](0009f3-attention-and-batch-normalization-route-decisions.md). Independently retains generated attention (schema 57), inference BatchNorm (schema 49), and training/statistic-transition BatchNorm (schema 50) on complete non-performance cost grounds. |

The current source-backed boundaries are deliberately narrow:

- MATMUL is static vector/matrix/batched/right-broadcast execution with bounded scalar/vector
  full-K realizations and exact recognized linear epilogues. Conv1d remains the visible virtual
  singleton Conv2d composition; Conv2d is grouped NCHW and Conv3d is grouped NCDHW direct
  generated execution. No child may merge those rank-specific contracts or infer a new operation.
- Pool1d is only the exact `EXPAND_DIMS(axis 2) -> POOL2D -> SQUEEZE(axis 2)` composition.
  Pool2d and Pool3d are first-class floating max/fixed-count-average output-cell window routes;
  the latter are NCHW and NCDHW respectively.
- Attention is the current static one- or two-output scaled-dot-product subset with optional
  right-broadcast BOOL mask, top-left causal eligibility, frozen stable normalization, and
  per-range score/weight scratch. It remains distinct from general reductions or softmax.
- BatchNorm inference is the five-input/one-output arbitrary-channel-axis route with selected
  channel/non-channel range ownership and no workspace. BatchNorm training is the five-input/
  five-output complete-channel, three-pass statistic-transition route with exact-state scratch.
  Neither is Layer/RMS normalization or a generic reduction migration.

For every retained generated hot path, preserve the `AGENTS.md` requirement for a well-written,
optimal clean-Java semantic and hot-loop/dataflow oracle. Generated code must retain the oracle's
semantic algorithm, loop/dataflow shape, and avoidable-overhead profile; Class-File/decompilation
inspection remains proportionate evidence, not a universal verifier. For every migration, replace
the complete selected generated subfamily, including lowering/selection, cold binding, invocation,
artifact/cache/inventory and generated-only evidence as applicable. No hidden fallback or
dual-maintained route may remain.

## Out of scope

- A universal structural verifier, common route mandate, big-bang rewrite, or new generic
  execution framework.
- A performance gate, benchmark campaign, automatic speed claim, or a conclusion about JIT or
  whole-backend performance.
- New CPU capability, operation semantics, native route, module/public API, package boundary,
  shared Prepare/Runtime contract, or dynamic-Shape support.
- CPU 0009G's repository-wide support, correctness, hygiene, inventory, documentation,
  conformance, and integration checkpoint.

## Architecture references and constraints

`ARCHITECTURE.md` is authoritative. CPU backend analysis/prepare owns family lowering, route
selection, specialization, and exact resource declarations; finalization constructs the prepared
executable after shared slot assignment; Runtime invokes prepared work and performs neither graph
interpretation nor route selection. Keep selection and typed carrier/layout/range binding cold.
Element loops must not introduce allocation, boxing, reflection, string/map dispatch,
synchronization, or avoidable virtual dispatch; parallelism remains outside those loops.

The CPU backend may use only its allowed module dependencies. This planning task changes no
architecture or dependency rule. Stop and report rather than inventing an architecture change if a
family cannot be completely replaced or retained within these constraints.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.lowering` owns family admission and cold
  geometry/lowering.
- `...internal.ir`, `...internal.codegen.emit`, `...internal.reference`, `...internal.prepare`,
  `...internal.executable`, and `...internal.cache` own, respectively, identity, generated body,
  clean-Java oracle, prepared selection/finalization, typed invocation, and generated artifacts.
- Mirrored CPU test packages own focused semantic, generated-kernel, evidence, lowering,
  preparation, finalization, executable, cache, and capability coverage.

Packages added or changed:

- None planned. A child must stop and update this parent/master-plan package map before adding a
  package or moving a published/completed type.

Type placement:

- No types are planned by this parent. Any direct-Java replacement remains in the existing
  family-owning internal package; it must not create a shared catch-all route abstraction.

## Affected files

Expected, per child after source inspection:

- The applicable family-owned CPU lowerer, IR, emitter or direct replacement, reference kernel,
  preparation/finalization/executable binding, specialization/artifact-cache seams, and their
  focused tests and evidence owners.
- This parent, its detailed active child, CPU master plan, and roadmap status records.
- Only affected Javadocs, explanatory documentation, and glossary entries when a completed route
  changes behavior, terminology, or a public/implementation contract.

## Maximum scope

Each child may modify at most 18 production, test, documentation, and planning paths. This parent
is planning coordination only. If a complete family replacement exceeds that bound, retain the
generated route when its existing evidence is sound, or stop and propose a separately bounded
follow-up; do not split a selected subfamily into a permanent dual route.

## Acceptance criteria

- 0009F1, then 0009F2, then 0009F3 complete in order; no later child is detailed or executed
  early without a recorded Planning Guide exception.
- Each child records a separate, source- and test-backed cost decision for every family in its
  boundary. A family is retained when generated replacement plus verification and retirement is
  more costly, equal, or uncertain; direct finite Java is selected only when its whole route is
  demonstrably strictly cheaper.
- F1 accounts for complete MATMUL and Conv1d/Conv2d/Conv3d boundaries; F2 accounts for
  Pool1d/Pool2d/Pool3d; F3 accounts separately for attention, inference BatchNorm, and training
  BatchNorm/statistic transition.
- A retained generated hot path has current clean-Java semantic/hot-loop/dataflow equivalence
  evidence and proportionate generated Class-File/decompilation checks. A migration fully retires
  its selected generated subfamily with no selected fallback, dual maintenance, stale artifact
  schema/inventory/evidence, or unverified call path.
- Each child runs focused CPU validation once after executable changes stabilize and records exact
  results. A separate clean documentation-focused context finalizes the affected Javadocs,
  explanatory documentation, glossary impact, and documentation validation in the same change.
- No child introduces a performance gate, benchmark campaign, or automatic performance claim.

## Tests / validation

Each detailed child selects its focused CPU command from the current owner tests and runs it once
after executable work stabilizes. It must cover the changed route's lowering/admission, semantic
oracle, generated or direct invocation, aliases/canaries and failure-before-write behavior where
applicable, preparation/finalization, selected-route/retirement evidence, and capability/inventory
truthfulness. Retained-only decisions reuse existing focused evidence only after independently
inspecting the source, tests, and evidence owner; they do not rerun Java merely to duplicate it.

The documentation pass checks local links, headings/anchors, fences, terminology and glossary
impact, task/master/roadmap status and frontier consistency, package placement, final newlines,
and exact changed-path scope. It runs:

```bash
git diff --check
git status --short -uall
```

Repository-wide validation, architecture tests, backend conformance, and integration tests defer
to CPU 0009G/CI unless a child changes a shared dependency, architecture boundary, backend
conformance contract, or end-to-end behavior; that exception requires the applicable validation
in the same child.

## Dependencies and follow-up tasks

- Depends on Complete CPU 0009E3 and the established family evidence: CPU 0008F (MATMUL),
  0008/0008A (Conv2d/Conv1d/Conv3d), 0008G/0008G1 (pooling), 0008H (attention), and
  0007F1/0007F2 (BatchNorm inference/training).
- 0009F1, F2, and F3 are Complete in order. CPU 0009G is the sole next Draft frontier.

## Architecture impact

Expected impact: None. If a child requires an architecture, public API, module dependency, shared
Prepare/Runtime, or resource-lifecycle change, stop and report the conflict.

## Implementation prompt

```text
You are the clean implementation agent for Synaptik CPU 0009F. Read AGENTS.md, ARCHITECTURE.md,
docs/architecture/current-architecture-plan.md, docs/planning/planning-guide.md, the CPU master
plan, CPU 0009, this task, and the one current detailed child. Implement only that child exactly.
Inspect the complete family source, tests, Javadocs, and existing evidence before deciding. Stop on
an architecture or scope conflict. Do not commit, push, stage, use GSD, add a benchmark gate, or
run repository-wide validation unless the child changes a shared boundary. Hand the stabilized
diff and exact focused-test evidence to a separate clean documentation-focused context, which
follows docs/developer-guide/documentation-rules.md and finalizes documentation/Javadoc/glossary
impact and documentation validation without duplicating stable Java tests. Do not mark the child
Complete until that pass is recorded.
```

## Local decisions

F1 independently retained generated MATMUL, visible virtual Conv1d-through-Conv2d composition,
and direct generated Conv2d and Conv3d. The decisions are separate; Conv1d is not a generated or
direct kernel family, and no performance/JIT conclusion is made.

## Known limitations

F1 changes no executable source, Javadoc, architecture, glossary, build, conformance, or
integration contract. Its clean documentation-focused review reused stable focused Java evidence
after independent source/test inspection, as permitted when no executable Java changes.

## Validation evidence

F1's clean documentation-focused context finalized its decision record, removed dependence on a
temporary evidence path, checked the planning links/status/frontier/scope, and ran final diff and
status checks. No focused Java suite was rerun because no executable Java changed and no concrete
discrepancy was found.

## Implementation notes

F1, F2, and F3 are Complete. This parent is complete; CPU 0009G is the sole next Draft frontier.

## Completion summary

F1 retained its current routes and F2 retained exact Pool1d composition plus separate Pool2d/Pool3d
generated routes. F3 independently retained schema-57 attention, schema-49 inference BatchNorm,
and schema-50 training BatchNorm. No route migrated or retired, and no performance/JIT claim is
made. CPU 0009G is the sole next Draft frontier.

Status: Complete
