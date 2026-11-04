# Task 0010L: [CPU](../../../../glossary.md#term-cpu) execution and public [lifecycle](../../../../glossary.md#lifecycle) status reconciliation

## Status

Complete

Readiness verification: clean `main` at `b188dd6`; [CPU](../../../../glossary.md#term-cpu) 0010K, [Compiler](../../../../glossary.md#term-compiler) 0006B10, [Model](../../../../glossary.md#term-model) 0025M,
and [Engine](../../../../glossary.md#term-engine) 0013 are Complete. The user authorized this sole serial CPU frontier; no CPU task was
[`Ready`](../../../../glossary.md#term-ready) or `In progress`. Review-needed, blocked, and Draft CPU rows remain unchanged.

## Change class

Class A — explanatory status and one public type's [Javadoc](../../../../glossary.md#term-javadoc) are reconciled to current source with
no change to semantics, numerics, [API](../../../../glossary.md#term-application-programming-interface-api), [lifecycle](../../../../glossary.md#lifecycle), [ownership](../../../../glossary.md#term-ownership-meanings), [concurrency](../../../../glossary.md#term-concurrency), [persistence](../../../../glossary.md#term-persistence), [ABI](../../../../glossary.md#term-application-binary-interface-abi),
capability, dependencies, workflow, hot paths, or architecture authority. One context suffices.

## Goal

Reconcile the [finite](../../../../glossary.md#term-finite) confirmed current-status drift without broadening capability: bounded [CPU](../../../../glossary.md#term-cpu)
execution paragraphs for [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), attention, [Conv2d](../../../../glossary.md#convolution--conv2d)/[Conv3d](../../../../glossary.md#conv3d--ncdhw) forward, reductions, scans, ordering,
floating classifications, [Softmax](../../../../glossary.md#softmax--log-softmax)/LogSoftmax, normalizations, [losses](../../../../glossary.md#loss), movement/[layout](../../../../glossary.md#layout),
indexing/scatter, composition, and supported unfold/fold; the public [Engine](../../../../glossary.md#term-engine) [compile](../../../../glossary.md#compile) and training
status; and the CPU complete-[partition](../../../../glossary.md#partition) [Javadoc](../../../../glossary.md#term-javadoc) inventory.

## [Scope](../../../../glossary.md#term-scope)

- Correct the confirmed `docs/api/compile-api.md` current-status paragraphs for bounded CPU
  MATMUL; scaled-dot-product attention; Conv2d and Conv3d forward; ordinary, masked, arg-extrema,
  advanced, and sum-to-shape reductions; cumulative scans; stable SORT/ARGSORT/TOP_K; floating
  classifications; Softmax/LogSoftmax; normalizations; losses; supported movement/layout;
  gather/indexing; scatter; concat/stack; and supported unfold/fold subsets.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [sum-to-shape glossary](../../../../glossary.md#sum-to-shape); [Softmax glossary](../../../../glossary.md#softmax--log-softmax); [losses glossary](../../../../glossary.md#loss); [layout glossary](../../../../glossary.md#layout); [gather glossary](../../../../glossary.md#gather))</span>
- Correct the false public [Engine](../../../../glossary.md#term-engine) [compile](../../../../glossary.md#compile)-status sentence in `docs/api/compile-api.md`.
- Give every CPU correction exact bounded static/resolved-layout/fail-closed wording while
  retaining each paragraph's truthful Model, Compiler, dynamic, gradient, and other-backend limits.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [Model glossary](../../../../glossary.md#term-model); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend))</span>
- Correct only the false [Compiler](../../../../glossary.md#term-compiler)-[autograd](../../../../glossary.md#autograd)/executable-[runtime](../../../../glossary.md#term-runtime) status in
  `docs/user-guide/training.md`; retain its conceptual planned training workflow.
- Complete the closed complete-[partition](../../../../glossary.md#partition) inventory in the class [Javadoc](../../../../glossary.md#term-javadoc) of
  `CpuCapabilityProvider`; preserve the truthful one-through-sixteen semantic input-occurrence
  limit for supported composition operations.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Review the execution-status wording in `docs/api/tensor-api.md` and record a no-change conclusion:
  its relevant statements are scoped to Model semantics rather than unqualified repository status.
- Synchronize this brief, the [CPU](../../../../glossary.md#term-cpu) master plan, and the roadmap after execution.

## Non-goals

- No executable Java, signature, test, Gradle, dependency, behavior, numerics, capability, route,
  lifecycle, ownership, architecture-authority, ADR, glossary, or unrelated documentation change.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency); [lifecycle glossary](../../../../glossary.md#lifecycle); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- No generic, dynamic-Shape, mixed/other-backend, gradient, fusion, vector, native, performance, or
  universal-execution claim.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape); [backend glossary](../../../../glossary.md#backend); [gradient glossary](../../../../glossary.md#term-gradient); [fusion glossary](../../../../glossary.md#term-fusion); [vector glossary](../../../../glossary.md#term-vector))</span>
- Preserve planned recurrent execution, Conv3d backward/adjoints, training sessions, optimizers,
  and the Training extension.<span data-glossary-navigation="true"> ([Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [optimizers glossary](../../../../glossary.md#term-optimizer))</span>
- Do not edit `docs/api/tensor-api.md`, sweep beyond the confirmed Compile API paragraphs, add a
  historical chronicle, or create parallel/follow-up tasks.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Contracts

- [`ARCHITECTURE.md` headings `Core lifecycle`](../../../../../ARCHITECTURE.md#core-lifecycle) and
  [`Core invariants`](../../../../../ARCHITECTURE.md#core-invariants) — preserve the current public
  Engine compile/prepare/run lifecycle and Compiler/backend separation.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [compile glossary](../../../../glossary.md#compile); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [lifecycle glossary](../../../../glossary.md#lifecycle); [Compiler glossary](../../../../glossary.md#term-compiler); [backend glossary](../../../../glossary.md#backend))</span>
- [`compiler-autograd.md` headings `modules/compiler`](../../../../architecture/contracts/compiler-autograd.md#modulescompiler)
  and [`Compiler-owned automatic differentiation`](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
  — describe current bounded Compiler compilation and autograd without moving execution into it.<span data-glossary-navigation="true"> ([compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [autograd glossary](../../../../glossary.md#autograd))</span>
- [`runtime-prepare-engine.md` headings `Run lifecycle`](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)
  and [`modules/engine`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)
  — describe the current public Engine lifecycle without implying generic backend composition.<span data-glossary-navigation="true"> ([backend composition glossary](../../../../glossary.md#backend-composition))</span>
- [`backend-execution.md` heading `CPU backend routes`](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)
  — keep CPU support exact, bounded, route-private, and fail-closed.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [fail-closed glossary](../../../../glossary.md#term-fail-closed))</span>
- [`extensions-training.md` headings `extensions/training`](../../../../architecture/contracts/extensions-training.md#extensionstraining)
  and [`Optimizer/training lifecycle`](../../../../architecture/contracts/extensions-training.md#optimizertraining-lifecycle)
  — preserve the planned optimizer/session boundary while acknowledging current prerequisites.<span data-glossary-navigation="true"> ([optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- [`recurrent-scan.md` heading `Fixed recurrent scan without graph regions`](../../../../architecture/contracts/recurrent-scan.md#fixed-recurrent-scan-without-graph-regions)
  — retain the truthful planned concrete-backend recurrent execution status.

If an applicable contract is missing or ambiguous, stop and report it.

## Dependencies and integration

- Depends on: [CPU](../../../../glossary.md#term-cpu) 0010K; [Compiler](../../../../glossary.md#term-compiler) 0006B10; [Model](../../../../glossary.md#term-model) 0025M; [Engine](../../../../glossary.md#term-engine) 0013
- Conflicts with: concurrent edits to the six allowed changed paths or their status claims
- Parallel group: None
- Common base revision: N/A — serial task; readiness verified on clean `main` at `b188dd6`
- Integration order: Any
- Integration [validation](../../../../glossary.md#term-validation): CPU [Javadoc](../../../../glossary.md#term-javadoc) generation, scoped Markdown validation, exact-path audit, and
  `git diff --check` recorded below
- Shared-document integration [owner](../../../../glossary.md#term-owner): N/A — this single task exclusively owns all scoped edits

## Files and symbols

Exact changed-path allowlist:

- `docs/api/compile-api.md` — the false public Engine compile-status sentence and finite stale CPU
  current-status paragraphs for MATMUL, attention, Conv2d/Conv3d forward, ordinary/masked/
  arg-extrema/advanced/sum-to-shape reductions, scans, stable ordering/TOP_K, floating
  classifications, Softmax/LogSoftmax, normalizations, losses, supported movement/layout,
  gather/indexing, scatter, concat/stack, and supported unfold/fold subsets.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [sum-to-shape glossary](../../../../glossary.md#sum-to-shape); [Softmax glossary](../../../../glossary.md#softmax--log-softmax); [losses glossary](../../../../glossary.md#loss); [layout glossary](../../../../glossary.md#layout); [gather glossary](../../../../glossary.md#gather))</span>
- `docs/user-guide/training.md` — the opening current-versus-planned status statement only.
- `backends/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/CpuCapabilityProvider.java` —
  class Javadoc complete-partition inventory only.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [partition glossary](../../../../glossary.md#partition))</span>
- this task brief — final status and compact result evidence.
- `docs/planning/backends/cpu/master-plan.md` — 0010L row and frontier summary only.
- `docs/planning/roadmap.md` — [CPU](../../../../glossary.md#term-cpu) row and authorized-frontier summary only.

Read-only evidence includes `docs/api/tensor-api.md`, `CpuPartitionLowering`,
`CpuPartitionDagDecomposer`, current [Engine](../../../../glossary.md#term-engine) [compile](../../../../glossary.md#compile)/[prepare](../../../../glossary.md#prepare)/[run](../../../../glossary.md#term-run) and backward surfaces, focused CPU
family tests, and targeted glossary entries. No path outside the six-file allowlist may change.

## Acceptance criteria

- The Compile API no longer labels current CPU provider/lowering/preparation/execution as planned
  for supported MATMUL, attention, Conv2d/Conv3d forward, ordinary/masked/arg-extrema/advanced/
  sum-to-shape reductions, cumulative scans, SORT/ARGSORT/TOP_K, floating classifications,
  SOFTMAX/LOG_SOFTMAX, normalizations, BatchNorm inference/training, losses, movement/layout,
  gather/indexing, scatter, concat/stack, and unfold/fold subsets.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [lowering glossary](../../../../glossary.md#lowering); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [Conv3d glossary](../../../../glossary.md#conv3d--ncdhw); [extrema glossary](../../../../glossary.md#saturation-pole-and-extremum); [sum-to-shape glossary](../../../../glossary.md#sum-to-shape); [SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [inference glossary](../../../../glossary.md#term-inference); [losses glossary](../../../../glossary.md#loss); [layout glossary](../../../../glossary.md#layout); [gather glossary](../../../../glossary.md#gather))</span>
- Every corrected family paragraph describes only its exact bounded fully static,
  resolved-layout, fail-closed CPU subset and retains its family-specific Shape, data-type,
  layout, role, value, topology, and composition limits. It does not imply generic, dynamic,
  recurrent, mixed-backend, Metal/CUDA, other-backend, gradient, fusion, native, or performance
  support.<span data-glossary-navigation="true"> ([fail-closed glossary](../../../../glossary.md#term-fail-closed); [Shape glossary](../../../../glossary.md#shape); [topology glossary](../../../../glossary.md#term-topology); [backend glossary](../../../../glossary.md#backend); [Metal glossary](../../../../glossary.md#term-metal); [CUDA glossary](../../../../glossary.md#term-cuda); [gradient glossary](../../../../glossary.md#term-gradient); [fusion glossary](../../../../glossary.md#term-fusion))</span>
- Truthful planned statements remain planned, including dynamic binding, unsupported gradients
  and family-specific adjoints, Conv3d backward pending Compiler 0006C, other-backend execution,
  and recurrent execution. Model-semantics-scoped statements are not rewritten as backend claims.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Compiler glossary](../../../../glossary.md#term-compiler); [Model glossary](../../../../glossary.md#term-model))</span>
- The [Compile](../../../../glossary.md#compile) [API](../../../../glossary.md#term-application-programming-interface-api) no longer says `GraphCompilationPort` is the only public compile call or that no
  Engine <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine))</span>`CompiledGraph` exists. It distinguishes ordinary and advanced public Engine compile
  surfaces from the narrow integration SPI and package-private Compiler implementation.<span data-glossary-navigation="true"> ([SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>
- The training guide distinguishes current bounded Compiler autograd and current public CPU-only
  Engine compile/prepare/run/backward execution from the still-planned Training extension,
  training session, optimizer API, optimizer update, and complete training workflow. Its example
  remains explicitly conceptual.<span data-glossary-navigation="true"> ([autograd glossary](../../../../glossary.md#autograd); [prepare glossary](../../../../glossary.md#prepare); [run glossary](../../../../glossary.md#term-run); [training session glossary](../../../../glossary.md#training-session--trainingsession); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>
- `CpuCapabilityProvider` class Javadoc gives a source-backed closed complete-partition inventory,
  including movement, indexing, scatter, fold, ordering, explicit-state random, cumulative scan,
  ordinary/arg-extrema/masked/advanced reductions, softmax, Layer/RMS normalization, BatchNorm
  inference/training, Conv2d/Conv3d, MATMUL, Pool2d/Pool3d, scaled-dot-product attention, loss,
  recognized Conv1d/Pool1d compositions, affine units, and pointwise units as admitted by current
  lowering/decomposition. It does not turn occurrence-local support into arbitrary partition
  support.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [partition glossary](../../../../glossary.md#partition); [cumulative scan glossary](../../../../glossary.md#cumulative-scan); [RMS normalization glossary](../../../../glossary.md#rms-normalization); [Pool3d glossary](../../../../glossary.md#pool3d--ncdhw-pooling); [pointwise glossary](../../../../glossary.md#term-pointwise); [decomposition glossary](../../../../glossary.md#term-decomposition))</span>
- The existing truthful one-through-sixteen semantic input-[occurrence](../../../../glossary.md#term-occurrence) bound for supported
  `CONCAT`/`STACK` composition remains unchanged.
- Targeted review records that `docs/api/tensor-api.md` needs no edit because its relevant
  non-execution statements are Model-contract scoped. Truthful planned recurrent execution and
  Conv3d backward wording remains present.
- No authority or executable behavior changes, no new reusable term requires a glossary edit, and
  no unrelated claim is rewritten.
- Only the six allowed paths change; task/master/roadmap status remains synchronized; this brief
  stays at most 200 lines and 15 KB; links, anchors, unique headings, fences, final newlines,
  whitespace, and `git diff --check` pass.

## [Validation](../../../../glossary.md#term-validation)

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
repository. It must check local Markdown [targets](../../../../glossary.md#term-target) and generated heading anchors, unique headings,
balanced fences, trailing whitespace, and final newlines. The exact-path audit must equal the
six-file allowlist, and final diff review must confirm that the only Java hunk is class [Javadoc](../../../../glossary.md#term-javadoc) and
that `docs/api/tensor-api.md` is unchanged.

Integration/repository validation: the recorded worker [checkpoint](../../../../glossary.md#term-checkpoint) is sufficient because this
serial Class A task changes explanatory documentation and Javadoc only; repository-wide tests are
deferred to [CI](../../../../glossary.md#term-continuous-integration-ci).

## Documentation and review impact

- Apply General + [API](../../../../glossary.md#term-application-programming-interface-api) and [Javadoc](../../../../glossary.md#term-javadoc) to `compile-api.md` and `CpuCapabilityProvider`; General + User
  guide to `training.md`; and [Planning](../../../../glossary.md#term-planning) style to planning files.
- Perform targeted glossary review for Compiler autograd, Engine lifecycle, CPU route, and training
  terms. No glossary change is expected because meanings and boundaries do not change.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- The implementer performs the final targeted documentation/Javadoc review. A separate clean
  review is not required for this Class A status-only correction.

## Result

- Changed `compile-api.md` (including review follow-up for sum-to-shape, ordering/TOP_K, floating
  classifications, and Softmax/LogSoftmax), <span data-glossary-navigation="true"> ([sum-to-shape glossary](../../../../glossary.md#sum-to-shape); [Softmax glossary](../../../../glossary.md#softmax--log-softmax))</span>`training.md`, and provider [Javadoc](../../../../glossary.md#term-javadoc); synchronized plans.
- No change: `docs/api/tensor-api.md` remains correctly Model-contract scoped; targeted glossary
  review found no new or changed Compiler, autograd, Engine, CPU-route, or training term.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler); [autograd glossary](../../../../glossary.md#autograd); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu))</span>
- `./gradlew :backends:cpu:javadoc`: `BUILD SUCCESSFUL in 4s`; 11 tasks, 2 executed and 9
  up-to-date; 94 warnings were confined to unchanged lowering/preparation constructor Javadocs.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering))</span>
- `python3 /tmp/validate_synaptik_markdown.py ...`: `validated 5 Markdown files`; local targets,
  generated anchors, unique headings, fences, trailing whitespace, and final newlines passed.<span data-glossary-navigation="true"> ([targets glossary](../../../../glossary.md#term-target))</span>
- `wc -l -c .../0010l-...md`: `199 12805`, within 200 lines and 15 KB.
- Exact-path audit matched the six-path allowlist; targeted status-band stale search found only
  scoped/planned limits; `git diff --check` produced no output; the sole Java hunk remains Javadoc.
- No [signature](../../../../glossary.md#term-signature), executable [token](../../../../glossary.md#term-token), test, [Gradle](../../../../glossary.md#term-gradle), architecture, glossary, or behavior changed.
- Limitations or unresolved issues: none.

Status: Complete
