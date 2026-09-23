# Task 0010L: CPU execution and public lifecycle status reconciliation

## Status

Complete

Readiness verification: clean `main` at `b188dd6`; CPU 0010K, Compiler 0006B10, Model 0025M,
and Engine 0013 are Complete. The user authorized this sole serial CPU frontier; no CPU task was
`Ready` or `In progress`. Review-needed, blocked, and Draft CPU rows remain unchanged.

## Change class

Class A — explanatory status and one public type's Javadoc are reconciled to current source with
no change to semantics, numerics, API, lifecycle, ownership, concurrency, persistence, ABI,
capability, dependencies, workflow, hot paths, or architecture authority. One context suffices.

## Goal

Reconcile the finite confirmed current-status drift without broadening capability: bounded CPU
execution paragraphs for MATMUL, attention, Conv2d/Conv3d forward, reductions, scans, ordering,
floating classifications, Softmax/LogSoftmax, normalizations, losses, movement/layout,
indexing/scatter, composition, and supported unfold/fold; the public Engine compile and training
status; and the CPU complete-partition Javadoc inventory.

## Scope

- Correct the confirmed `docs/api/compile-api.md` current-status paragraphs for bounded CPU
  MATMUL; scaled-dot-product attention; Conv2d and Conv3d forward; ordinary, masked, arg-extrema,
  advanced, and sum-to-shape reductions; cumulative scans; stable SORT/ARGSORT/TOP_K; floating
  classifications; Softmax/LogSoftmax; normalizations; losses; supported movement/layout;
  gather/indexing; scatter; concat/stack; and supported unfold/fold subsets.
- Correct the false public Engine compile-status sentence in `docs/api/compile-api.md`.
- Give every CPU correction exact bounded static/resolved-layout/fail-closed wording while
  retaining each paragraph's truthful Model, Compiler, dynamic, gradient, and other-backend limits.
- Correct only the false Compiler-autograd/executable-runtime status in
  `docs/user-guide/training.md`; retain its conceptual planned training workflow.
- Complete the closed complete-partition inventory in the class Javadoc of
  `CpuCapabilityProvider`; preserve the truthful one-through-sixteen semantic input-occurrence
  limit for supported composition operations.
- Review the execution-status wording in `docs/api/tensor-api.md` and record a no-change conclusion:
  its relevant statements are scoped to Model semantics rather than unqualified repository status.
- Synchronize this brief, the CPU master plan, and the roadmap after execution.

## Non-goals

- No executable Java, signature, test, Gradle, dependency, behavior, numerics, capability, route,
  lifecycle, ownership, architecture-authority, ADR, glossary, or unrelated documentation change.
- No generic, dynamic-Shape, mixed/other-backend, gradient, fusion, vector, native, performance, or
  universal-execution claim.
- Preserve planned recurrent execution, Conv3d backward/adjoints, training sessions, optimizers,
  and the Training extension.
- Do not edit `docs/api/tensor-api.md`, sweep beyond the confirmed Compile API paragraphs, add a
  historical chronicle, or create parallel/follow-up tasks.

## Contracts

- [`ARCHITECTURE.md` headings `Core lifecycle`](../../../../../ARCHITECTURE.md#core-lifecycle) and
  [`Core invariants`](../../../../../ARCHITECTURE.md#core-invariants) — preserve the current public
  Engine compile/prepare/run lifecycle and Compiler/backend separation.
- [`compiler-autograd.md` headings `modules/compiler`](../../../../architecture/contracts/compiler-autograd.md#modulescompiler)
  and [`Compiler-owned automatic differentiation`](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  — describe current bounded Compiler compilation and autograd without moving execution into it.
- [`runtime-prepare-engine.md` headings `Run lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)
  and [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — describe the current public Engine lifecycle without implying generic backend composition.
- [`backend-execution.md` heading `CPU backend routes`](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  — keep CPU support exact, bounded, route-private, and fail-closed.
- [`extensions-training.md` headings `extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining)
  and [`Optimizer/training lifecycle`](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
  — preserve the planned optimizer/session boundary while acknowledging current prerequisites.
- [`recurrent-scan.md` heading `Fixed recurrent scan without graph regions`](../../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)
  — retain the truthful planned concrete-backend recurrent execution status.

If an applicable contract is missing or ambiguous, stop and report it.

## Dependencies and integration

- Depends on: CPU 0010K; Compiler 0006B10; Model 0025M; Engine 0013
- Conflicts with: concurrent edits to the six allowed changed paths or their status claims
- Parallel group: None
- Common base revision: N/A — serial task; readiness verified on clean `main` at `b188dd6`
- Integration order: Any
- Integration validation: CPU Javadoc generation, scoped Markdown validation, exact-path audit, and
  `git diff --check` recorded below
- Shared-document integration owner: N/A — this single task exclusively owns all scoped edits

## Files and symbols

Exact changed-path allowlist:

- `docs/api/compile-api.md` — the false public Engine compile-status sentence and finite stale CPU
  current-status paragraphs for MATMUL, attention, Conv2d/Conv3d forward, ordinary/masked/
  arg-extrema/advanced/sum-to-shape reductions, scans, stable ordering/TOP_K, floating
  classifications, Softmax/LogSoftmax, normalizations, losses, supported movement/layout,
  gather/indexing, scatter, concat/stack, and supported unfold/fold subsets.
- `docs/user-guide/training.md` — the opening current-versus-planned status statement only.
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java` —
  class Javadoc complete-partition inventory only.
- this task brief — final status and compact result evidence.
- `docs/planning/backends/cpu/master-plan.md` — 0010L row and frontier summary only.
- `docs/planning/roadmap.md` — CPU row and authorized-frontier summary only.

Read-only evidence includes `docs/api/tensor-api.md`, `CpuPartitionLowering`,
`CpuPartitionDagDecomposer`, current Engine compile/prepare/run and backward surfaces, focused CPU
family tests, and targeted glossary entries. No path outside the six-file allowlist may change.

## Acceptance criteria

- The Compile API no longer labels current CPU provider/lowering/preparation/execution as planned
  for supported MATMUL, attention, Conv2d/Conv3d forward, ordinary/masked/arg-extrema/advanced/
  sum-to-shape reductions, cumulative scans, SORT/ARGSORT/TOP_K, floating classifications,
  SOFTMAX/LOG_SOFTMAX, normalizations, BatchNorm inference/training, losses, movement/layout,
  gather/indexing, scatter, concat/stack, and unfold/fold subsets.
- Every corrected family paragraph describes only its exact bounded fully static,
  resolved-layout, fail-closed CPU subset and retains its family-specific Shape, data-type,
  layout, role, value, topology, and composition limits. It does not imply generic, dynamic,
  recurrent, mixed-backend, Metal/CUDA, other-backend, gradient, fusion, native, or performance
  support.
- Truthful planned statements remain planned, including dynamic binding, unsupported gradients
  and family-specific adjoints, Conv3d backward pending Compiler 0006C, other-backend execution,
  and recurrent execution. Model-semantics-scoped statements are not rewritten as backend claims.
- The Compile API no longer says `GraphCompilationPort` is the only public compile call or that no
  Engine `CompiledGraph` exists. It distinguishes ordinary and advanced public Engine compile
  surfaces from the narrow integration SPI and package-private Compiler implementation.
- The training guide distinguishes current bounded Compiler autograd and current public CPU-only
  Engine compile/prepare/run/backward execution from the still-planned Training extension,
  training session, optimizer API, optimizer update, and complete training workflow. Its example
  remains explicitly conceptual.
- `CpuCapabilityProvider` class Javadoc gives a source-backed closed complete-partition inventory,
  including movement, indexing, scatter, fold, ordering, explicit-state random, cumulative scan,
  ordinary/arg-extrema/masked/advanced reductions, softmax, Layer/RMS normalization, BatchNorm
  inference/training, Conv2d/Conv3d, MATMUL, Pool2d/Pool3d, scaled-dot-product attention, loss,
  recognized Conv1d/Pool1d compositions, affine units, and pointwise units as admitted by current
  lowering/decomposition. It does not turn occurrence-local support into arbitrary partition
  support.
- The existing truthful one-through-sixteen semantic input-occurrence bound for supported
  `CONCAT`/`STACK` composition remains unchanged.
- Targeted review records that `docs/api/tensor-api.md` needs no edit because its relevant
  non-execution statements are Model-contract scoped. Truthful planned recurrent execution and
  Conv3d backward wording remains present.
- No authority or executable behavior changes, no new reusable term requires a glossary edit, and
  no unrelated claim is rewritten.
- Only the six allowed paths change; task/master/roadmap status remains synchronized; this brief
  stays at most 200 lines and 15 KB; links, anchors, unique headings, fences, final newlines,
  whitespace, and `git diff --check` pass.

## Validation

Worker validation after final documentation edits:

```bash
./gradlew :backends:cpu:javadoc
python3 /tmp/validate_synaptik_markdown.py \
  docs/api/compile-api.md \
  docs/user-guide/training.md \
  docs/planning/backends/cpu/tasks/0010l-cpu-execution-and-public-lifecycle-status-reconciliation.md \
  docs/planning/backends/cpu/master-plan.md \
  docs/planning/roadmap.md
wc -l -c docs/planning/backends/cpu/tasks/0010l-cpu-execution-and-public-lifecycle-status-reconciliation.md
git diff --name-only
git diff --check
```

If the temporary Markdown validator is absent, create an equivalent validator outside the
repository. It must check local Markdown targets and generated heading anchors, unique headings,
balanced fences, trailing whitespace, and final newlines. The exact-path audit must equal the
six-file allowlist, and final diff review must confirm that the only Java hunk is class Javadoc and
that `docs/api/tensor-api.md` is unchanged.

Integration/repository validation: the recorded worker checkpoint is sufficient because this
serial Class A task changes explanatory documentation and Javadoc only; repository-wide tests are
deferred to CI.

## Documentation and review impact

- Apply General + API and Javadoc to `compile-api.md` and `CpuCapabilityProvider`; General + User
  guide to `training.md`; and Planning style to planning files.
- Perform targeted glossary review for Compiler autograd, Engine lifecycle, CPU route, and training
  terms. No glossary change is expected because meanings and boundaries do not change.
- The implementer performs the final targeted documentation/Javadoc review. A separate clean
  review is not required for this Class A status-only correction.

## Result

- Changed `compile-api.md` (including review follow-up for sum-to-shape, ordering/TOP_K, floating
  classifications, and Softmax/LogSoftmax), `training.md`, and provider Javadoc; synchronized plans.
- No change: `docs/api/tensor-api.md` remains correctly Model-contract scoped; targeted glossary
  review found no new or changed Compiler, autograd, Engine, CPU-route, or training term.
- `./gradlew :backends:cpu:javadoc`: `BUILD SUCCESSFUL in 4s`; 11 tasks, 2 executed and 9
  up-to-date; 94 warnings were confined to unchanged lowering/preparation constructor Javadocs.
- `python3 /tmp/validate_synaptik_markdown.py ...`: `validated 5 Markdown files`; local targets,
  generated anchors, unique headings, fences, trailing whitespace, and final newlines passed.
- `wc -l -c .../0010l-...md`: `199 12805`, within 200 lines and 15 KB.
- Exact-path audit matched the six-path allowlist; targeted status-band stale search found only
  scoped/planned limits; `git diff --check` produced no output; the sole Java hunk remains Javadoc.
- No signature, executable token, test, Gradle, architecture, glossary, or behavior changed.
- Limitations or unresolved issues: none.

Status: Complete
