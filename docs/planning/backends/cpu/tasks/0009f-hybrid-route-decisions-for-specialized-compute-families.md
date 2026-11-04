# Task 0009F: Hybrid Route Decisions for Specialized Compute Families

## Reader context

This task records coordination of separate specialized-family decisions, not one generic migration.
F1 covers MATMUL and rank-specific convolution; F2 distinguishes exact Pool1d composition from
generated Pool2d/Pool3d; F3 separates attention, inference BatchNorm, and
training/statistic-transition BatchNorm. Retained clean-Java semantic oracles describe each family's
algorithm and loop dataflow; proportionate Class-File inspection supplies structural evidence rather
than universal verification.

Here, CPU analysis/Prepare owns admission, selection, specialization, and exact resources.
Finalization follows shared slot assignment; Runtime only invokes prepared work. Attention
score/weight scratch and training BatchNorm exact-state scratch are run resources, not persistent
model state. Inference BatchNorm declares no workspace. Complete replacement would include binding,
invocation, artifacts, cache, inventory, and evidence; the parent records that no route migrated or
retired.

Current-topic navigation (not a replacement for the historical record):

- [current CPU preparation and invocation handoffs](../../../../explanation/packages/cpu.md#read-the-pipeline-before-the-packages) — Names the current analysis, shared assignment, finalization, cold binding, and Runtime participants.
- [backend ownership versus implementation route](../../../../design/notes/cpu-kernel-strategy.md#strategy) — Separates Planning ownership, CPU route choice, finalization, and Runtime invocation.

## Status

Complete

## Goal

Close the next [CPU](../../../../glossary.md#term-cpu) 0009 route-decision frontier through ordered, independent decisions for the
existing [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[convolution](../../../../glossary.md#term-channels-first-convolution-layers), [pooling](../../../../glossary.md#term-pooling), attention, and [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)-[normalization](../../../../glossary.md#term-normalization) families. Each decision
chooses the least costly maintainable complete route: retain bounded generated compute when a
replacement costs more across implementation, verification, and retirement; use [finite](../../../../glossary.md#term-finite) direct
typed Java only for demonstrably cheaper static orchestration or window-loop work. Equality or
uncertainty retains the existing generated route.

## [Scope](../../../../glossary.md#term-scope)

This parent governs decisions only; its ordered children each inspect their complete current
family boundary before making a route decision.

| Child | Status | Family boundary and decision responsibility |
| --- | --- | --- |
| 0009F1 | Complete | [MATMUL and convolution route decisions](0009f1-matmul-and-convolution-route-decisions.md)<span data-glossary-navigation="true"> ([convolution glossary](../../../../glossary.md#term-channels-first-convolution-layers))</span>. Separately retained [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), Conv1d visible composition, and direct [Conv2d](../../../../glossary.md#convolution--conv2d)/[Conv3d](../../../../glossary.md#conv3d--ncdhw) generated bodies after source/test-backed complete-route decisions; no performance or [JIT](../../../../glossary.md#term-just-in-time-compilation-jit) claim. |
| 0009F2 | Complete | [Pooling route decisions](0009f2-pooling-route-decisions.md)<span data-glossary-navigation="true"> ([Pooling glossary](../../../../glossary.md#term-pooling))</span>. Independently retained exact [Pool1d composition](../../../../glossary.md#pool1d-composition) and the separate direct generated Pool2d/[Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling) routes; no performance or JIT claim. |
| 0009F3 | Complete | [Attention and BatchNorm route decisions](0009f3-attention-and-batch-normalization-route-decisions.md). Independently retains generated attention ([schema](../../../../glossary.md#term-schema) 57), [inference](../../../../glossary.md#term-inference) BatchNorm (schema 49), and training/[statistic](../../../../glossary.md#term-statistic)-transition BatchNorm (schema 50) on complete non-performance cost grounds. |

The current source-backed boundaries are deliberately narrow:

- MATMUL is static vector/matrix/batched/right-broadcast execution with bounded scalar/vector
  full-K realizations and exact recognized linear epilogues. Conv1d remains the visible virtual
  singleton Conv2d composition; Conv2d is grouped NCHW and Conv3d is grouped NCDHW direct
  generated execution. No child may merge those rank-specific contracts or infer a new operation.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [scalar glossary](../../../../glossary.md#term-scalar); [linear glossary](../../../../glossary.md#term-linear-meanings); [singleton glossary](../../../../glossary.md#term-singleton); [NCHW glossary](../../../../glossary.md#term-nchw); [NCDHW glossary](../../../../glossary.md#conv3d--ncdhw); [rank glossary](../../../../glossary.md#term-rank); [operation glossary](../../../../glossary.md#operation))</span>
- Pool1d is only the exact `EXPAND_DIMS(axis 2) -> POOL2D -> SQUEEZE(axis 2)` composition.
  Pool2d and Pool3d are first-class floating max/fixed-count-average output-cell window routes;
  the latter are NCHW and NCDHW respectively.
- Attention is the current static one- or two-output scaled-dot-product subset with optional
  right-broadcast BOOL mask, top-left causal eligibility, frozen stable normalization, and
  per-range score/weight scratch. It remains distinct from general reductions or softmax.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [mask glossary](../../../../glossary.md#term-boolean-mask); [normalization glossary](../../../../glossary.md#term-normalization); [weight glossary](../../../../glossary.md#term-weight); [softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>
- BatchNorm inference is the five-input/one-output arbitrary-channel-axis route with selected
  channel/non-channel range ownership and no workspace. BatchNorm training is the five-input/
  five-output complete-channel, three-pass statistic-transition route with exact-state scratch.
  Neither is Layer/RMS normalization or a generic reduction migration.<span data-glossary-navigation="true"> ([channel glossary](../../../../glossary.md#term-channel); [axis glossary](../../../../glossary.md#term-axis); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [RMS normalization glossary](../../../../glossary.md#rms-normalization))</span>

For every retained generated [hot path](../../../../glossary.md#term-hot-path), preserve the `AGENTS.md` requirement for a well-written,
optimal clean-Java semantic and hot-loop/[dataflow](../../../../glossary.md#term-dataflow) [oracle](../../../../glossary.md#term-test-oracle). Generated code must retain the oracle's
semantic algorithm, loop/dataflow [shape](../../../../glossary.md#shape), and avoidable-overhead profile; Class-File/[decompilation](../../../../glossary.md#term-decompilation)
inspection remains proportionate evidence, not a universal verifier. For every migration, replace
the complete selected generated subfamily, including [lowering](../../../../glossary.md#lowering)/selection, cold [binding](../../../../glossary.md#term-binding), invocation,
[artifact](../../../../glossary.md#term-artifact)/[cache](../../../../glossary.md#term-cache)/inventory and generated-only evidence as applicable. No hidden fallback or
dual-maintained route may remain.

## Out of [scope](../../../../glossary.md#term-scope)

- A universal structural verifier, common route mandate, big-bang rewrite, or new generic
  execution framework.
- A performance gate, benchmark campaign, automatic speed claim, or a conclusion about JIT or
  whole-backend performance.<span data-glossary-navigation="true"> ([JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit); [backend glossary](../../../../glossary.md#backend))</span>
- New CPU capability, operation semantics, native route, module/public API, package boundary,
  shared Prepare/Runtime contract, or dynamic-Shape support.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [operation glossary](../../../../glossary.md#operation); [module glossary](../../../../glossary.md#term-module); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Shape glossary](../../../../glossary.md#shape))</span>
- CPU 0009G's repository-wide support, correctness, hygiene, inventory, documentation,
  conformance, and integration checkpoint.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Architecture references and constraints

`ARCHITECTURE.md` is authoritative. [CPU](../../../../glossary.md#term-cpu) [backend](../../../../glossary.md#backend) analysis/[prepare](../../../../glossary.md#prepare) owns family [lowering](../../../../glossary.md#lowering), route
selection, [specialization](../../../../glossary.md#term-specialization), and exact resource declarations; finalization constructs the [prepared
executable](../../../../glossary.md#prepared-executable--preparedexecutable) after shared slot assignment; [Runtime](../../../../glossary.md#term-runtime) invokes prepared work and performs neither [graph](../../../../glossary.md#graph)
interpretation nor route selection. Keep selection and typed [carrier](../../../../glossary.md#term-storage-carrier)/[layout](../../../../glossary.md#layout)/range [binding](../../../../glossary.md#term-binding) cold.
[Element](../../../../glossary.md#term-element) loops must not introduce allocation, boxing, reflection, string/map [dispatch](../../../../glossary.md#term-dispatch),
[synchronization](../../../../glossary.md#term-synchronization), or avoidable virtual dispatch; parallelism remains outside those loops.

The CPU backend may use only its allowed [module](../../../../glossary.md#term-module) dependencies. This [planning](../../../../glossary.md#term-planning) task changes no
architecture or [dependency](../../../../glossary.md#term-dependency) rule. Stop and report rather than inventing an architecture change if a
family cannot be completely replaced or retained within these constraints.

## Package impact

Existing packages used:

- `io.github.pho001.synaptik.backend.cpu.internal.lowering` owns family admission and cold
  geometry/lowering.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [lowering glossary](../../../../glossary.md#lowering))</span>
- `...internal.ir`, `...internal.codegen.emit`, `...internal.reference`, `...internal.prepare`,
  `...internal.executable`, and `...internal.cache` own, respectively, identity, generated body,
  clean-Java oracle, prepared selection/finalization, typed invocation, and generated artifacts.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [oracle glossary](../../../../glossary.md#term-test-oracle); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Mirrored CPU test packages own focused semantic, generated-kernel, evidence, lowering,
  preparation, finalization, executable, cache, and capability coverage.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [kernel glossary](../../../../glossary.md#kernel); [cache glossary](../../../../glossary.md#term-cache))</span>

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
  focused tests and evidence owners.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [kernel glossary](../../../../glossary.md#kernel); [binding glossary](../../../../glossary.md#term-binding); [specialization glossary](../../../../glossary.md#term-specialization); [artifact glossary](../../../../glossary.md#term-artifact); [cache glossary](../../../../glossary.md#term-cache))</span>
- This parent, its detailed active child, [CPU](../../../../glossary.md#term-cpu) master plan, and roadmap status records.
- Only affected Javadocs, explanatory documentation, and glossary entries when a completed route
  changes behavior, terminology, or a public/implementation contract.

## Maximum [scope](../../../../glossary.md#term-scope)

Each child may modify at most 18 production, test, documentation, and [planning](../../../../glossary.md#term-planning) paths. This parent
is planning coordination only. If a complete family replacement exceeds that bound, retain the
generated route when its existing evidence is sound, or stop and propose a separately bounded
follow-up; do not split a selected subfamily into a permanent dual route.

## Acceptance criteria

- 0009F1, then 0009F2, then 0009F3 complete in order; no later child is detailed or executed
  early without a recorded Planning Guide exception.<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- Each child records a separate, source- and test-backed cost decision for every family in its
  boundary. A family is retained when generated replacement plus verification and retirement is
  more costly, equal, or uncertain; direct finite Java is selected only when its whole route is
  demonstrably strictly cheaper.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite))</span>
- F1 accounts for complete MATMUL and Conv1d/Conv2d/Conv3d boundaries; F2 accounts for
  Pool1d/Pool2d/Pool3d; F3 accounts separately for attention, inference BatchNorm, and training
  BatchNorm/statistic transition.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [inference glossary](../../../../glossary.md#term-inference); [statistic glossary](../../../../glossary.md#term-statistic))</span>
- A retained generated hot path has current clean-Java semantic/hot-loop/dataflow equivalence
  evidence and proportionate generated Class-File/decompilation checks. A migration fully retires
  its selected generated subfamily with no selected fallback, dual maintenance, stale artifact
  schema/inventory/evidence, or unverified call path.<span data-glossary-navigation="true"> ([hot path glossary](../../../../glossary.md#term-hot-path); [dataflow glossary](../../../../glossary.md#term-dataflow); [decompilation glossary](../../../../glossary.md#term-decompilation); [artifact glossary](../../../../glossary.md#term-artifact); [schema glossary](../../../../glossary.md#term-schema))</span>
- Each child runs focused CPU validation once after executable changes stabilize and records exact
  results. A separate clean documentation-focused context finalizes the affected Javadocs,
  explanatory documentation, glossary impact, and documentation validation in the same change.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [validation glossary](../../../../glossary.md#term-validation))</span>
- No child introduces a performance gate, benchmark campaign, or automatic performance claim.

## Tests / [validation](../../../../glossary.md#term-validation)

Each detailed child selects its focused [CPU](../../../../glossary.md#term-cpu) command from the current [owner](../../../../glossary.md#term-owner) tests and runs it once
after executable work stabilizes. It must cover the changed route's [lowering](../../../../glossary.md#lowering)/[admission](../../../../glossary.md#term-admission), semantic
[oracle](../../../../glossary.md#term-test-oracle), generated or direct invocation, [aliases](../../../../glossary.md#aliasing-and-arena)/canaries and failure-before-write behavior where
applicable, preparation/finalization, selected-route/retirement evidence, and capability/inventory
truthfulness. Retained-only decisions reuse existing focused evidence only after independently
inspecting the source, tests, and evidence owner; they do not rerun Java merely to duplicate it.

The documentation pass checks local links, headings/anchors, fences, terminology and glossary
impact, task/master/roadmap status and frontier consistency, package placement, final newlines,
and exact changed-path [scope](../../../../glossary.md#term-scope). It runs:

```bash
git diff --check
git status --short -uall
```

Repository-wide validation, architecture tests, [backend](../../../../glossary.md#backend) [conformance](../../../../glossary.md#term-conformance), and integration tests defer
to CPU 0009G/[CI](../../../../glossary.md#term-continuous-integration-ci) unless a child changes a shared [dependency](../../../../glossary.md#term-dependency), architecture boundary, backend
conformance contract, or end-to-end behavior; that exception requires the applicable validation
in the same child.

## Dependencies and follow-up tasks

- Depends on Complete CPU 0009E3 and the established family evidence: CPU 0008F (MATMUL),
  0008/0008A (Conv2d/Conv1d/Conv3d), 0008G/0008G1 (pooling), 0008H (attention), and
  0007F1/0007F2 (BatchNorm inference/training).<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [pooling glossary](../../../../glossary.md#term-pooling); [inference glossary](../../../../glossary.md#term-inference))</span>
- 0009F1, F2, and F3 are Complete in order. [CPU](../../../../glossary.md#term-cpu) 0009G is the sole next Draft frontier.

## Architecture impact

Expected impact: None. If a child requires an architecture, public [API](../../../../glossary.md#term-application-programming-interface-api), [module](../../../../glossary.md#term-module) [dependency](../../../../glossary.md#term-dependency), shared
[Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime), or resource-[lifecycle](../../../../glossary.md#lifecycle) change, stop and report the conflict.

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

F1 independently retained generated [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), visible virtual Conv1d-through-[Conv2d](../../../../glossary.md#convolution--conv2d) composition,
and direct generated Conv2d and [Conv3d](../../../../glossary.md#conv3d--ncdhw). The decisions are separate; Conv1d is not a generated or
direct [kernel](../../../../glossary.md#kernel) family, and no performance/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit) conclusion is made.

## Known limitations

F1 changes no executable source, [Javadoc](../../../../glossary.md#term-javadoc), architecture, glossary, build, [conformance](../../../../glossary.md#term-conformance), or
integration contract. Its clean documentation-focused review reused stable focused Java evidence
after independent source/test inspection, as permitted when no executable Java changes.

## [Validation](../../../../glossary.md#term-validation) evidence

F1's clean documentation-focused context finalized its decision record, removed dependence on a
temporary evidence path, checked the [planning](../../../../glossary.md#term-planning) links/status/frontier/[scope](../../../../glossary.md#term-scope), and ran final diff and
status checks. No focused Java suite was rerun because no executable Java changed and no concrete
discrepancy was found.

## Implementation notes

F1, F2, and F3 are Complete. This parent is complete; [CPU](../../../../glossary.md#term-cpu) 0009G is the sole next Draft frontier.

## Completion summary

F1 retained its current routes and F2 retained exact [Pool1d composition](../../../../glossary.md#pool1d-composition) plus separate Pool2d/[Pool3d](../../../../glossary.md#pool3d--ncdhw-pooling)
generated routes. F3 independently retained [schema](../../../../glossary.md#term-schema)-57 attention, schema-49 [inference](../../../../glossary.md#term-inference) BatchNorm,
and schema-50 training BatchNorm. No route migrated or retired, and no performance/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit) claim is
made. [CPU](../../../../glossary.md#term-cpu) 0009G is the sole next Draft frontier.

Status: Complete
