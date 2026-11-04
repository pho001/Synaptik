# Task 0016: [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) Mixed-[Owner](../../../../glossary.md#term-owner) Schedule and Transfer

## Status

Complete

Frontier verification: the user explicitly authorized this serial Class C frontier in one isolated
worktree from exact clean revision `0fe35a845a18fb9c70cdf87081b6a6c688fece47`. [Engine](../../../../glossary.md#term-engine) 0015,
[Prepare](../../../../glossary.md#prepare) 0006, [Runtime](../../../../glossary.md#term-runtime) 0016, and [Metal](../../../../glossary.md#term-metal) 0004 are Complete. [ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0016 and the authoritative mixed-[owner](../../../../glossary.md#term-owner)
contract were accepted before production edits. This task exclusively owns the cross-owner
Prepare/Runtime/Engine, [CPU](../../../../glossary.md#term-cpu)/Metal contribution, transfer, and shared documentation [scope](../../../../glossary.md#term-scope) through
its mandatory independent review.

## Change class

Class C — this task changes mixed-[backend composition](../../../../glossary.md#backend-composition), shared [Prepare](../../../../glossary.md#prepare) assignments and [transaction](../../../../glossary.md#term-transaction)
boundaries, [Runtime](../../../../glossary.md#term-runtime) transfer [binding](../../../../glossary.md#term-binding), concrete native transfer [lifetime](../../../../glossary.md#term-lifetime), [Engine](../../../../glossary.md#term-engine) handle routing, and
the no-[lookup](../../../../glossary.md#term-lookup) hot-path [invariant](../../../../glossary.md#term-invariant).

## Goal

[Prepare](../../../../glossary.md#prepare) and repeatedly [run](../../../../glossary.md#term-run) a [compiled graph](../../../../glossary.md#compiled-graph--compiledgraphmodel) containing [CPU](../../../../glossary.md#term-cpu)- and [Metal](../../../../glossary.md#term-metal)-owned partitions through
public `Engine.builder()` composition, with explicit bidirectional intermediate transfer, exact
ordering, deterministic [ownership](../../../../glossary.md#term-ownership-meanings), transactional cleanup, and no post-prepare [registry](../../../../glossary.md#term-registry) [lookup](../../../../glossary.md#term-lookup).

## [Scope](../../../../glossary.md#term-scope)

- Route every planned [partition](../../../../glossary.md#partition) by exact [`BackendId`](../../../../glossary.md#backend-identity--backendid) and prepare the ordered complete set through
  one shared <span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>[`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation) [transaction](../../../../glossary.md#term-transaction).
- Assign one representation position per distinct participating owner on each shared logical buffer
  and pass it through backend finalization.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [buffer glossary](../../../../glossary.md#term-buffer); [backend glossary](../../../../glossary.md#backend))</span>
- Adapt CPU and Metal schedule assemblers to contribute externally assigned buffer/workspace
  representation creators; assemble one creation prefix, ordered executions/transfers, and one
  publication suffix.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal))</span>
- Add direct prepared CPU-to-Metal upload and Metal-to-CPU download for fully static canonical
  contiguous <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>[`FLOAT32`](../../../../glossary.md#term-float32) intermediate values, using run-owned CPU native storage as reusable host
  staging.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [host glossary](../../../../glossary.md#term-host))</span>
- [Capture](../../../../glossary.md#term-capture-meanings) direct input and [publication](../../../../glossary.md#term-publication) adapters per [occurrence](../../../../glossary.md#term-occurrence) in outward prepared/result handles.
- Preserve CPU-only, single-owner CPU, single-owner Metal, repeated prepared runs, public
  materialization, and CPU-only tuning behavior.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization))</span>
- Add focused shared-contract, lifecycle, integration, architecture, documentation, and real Metal
  evidence.<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Non-goals

- No conversion, non-[contiguous](../../../../glossary.md#term-contiguous) transfer, non-[`FLOAT32`](../../../../glossary.md#term-float32) transfer, heap staging, CUDA transfer,
  generic plugin transfer service-provider interface, discovery, fallback, retry, owner
  substitution, asynchronous execution, pooling, aliasing, or native ABI expansion.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [CUDA glossary](../../../../glossary.md#term-cuda); [plugin glossary](../../../../glossary.md#term-plugin); [owner glossary](../../../../glossary.md#term-owner); [asynchronous glossary](../../../../glossary.md#term-asynchronous); [pooling glossary](../../../../glossary.md#term-pooling); [aliasing glossary](../../../../glossary.md#term-aliasing); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- No registry/provider/backend-ID query after successful preparation and no second physical-memory
  or schedule convention.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [backend glossary](../../../../glossary.md#backend))</span>
- No change to Planning ownership scoring, Metal operation capability beyond current NEG, or the
  CPU-only <span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [Metal glossary](../../../../glossary.md#term-metal); [operation glossary](../../../../glossary.md#operation); [CPU glossary](../../../../glossary.md#term-cpu))</span>`AdvancedEngine` and [autotuning](../../../../glossary.md#autotuning--model-autotuning) surfaces.

## Contracts

- [`ARCHITECTURE.md` headings `Core invariants` and `Module-ownership routing`](../../../../../ARCHITECTURE.md#core-invariants)
  require shared prepared representation positions, explicit transfer steps, Runtime-only schedule
  execution, and Engine composition.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine))</span>
- [`runtime-prepare-engine.md` headings `modules/runtime`, `modules/prepare`, `Prepare lifecycle`,
  `Run lifecycle`, and `Compile ownership and mixed-owner Prepare routing`](../../../../architecture/contracts/runtime-prepare-engine.md#compile-ownership-and-mixed-owner-prepare-routing)
  own assignment, contribution, schedule order, direct binding, transaction, handle capture, and
  no-lookup behavior.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [transaction glossary](../../../../glossary.md#term-transaction); [capture glossary](../../../../glossary.md#term-capture-meanings); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [`backend-execution.md` heading `Metal backend`](../../../../architecture/contracts/backend-execution.md#metal-backend)
  owns exact CPU/Metal host-staged physical transfer and native validation.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [host glossary](../../../../glossary.md#term-host); [validation glossary](../../../../glossary.md#term-validation))</span>
- [ADR 0016](../../../../design/decisions/0016-cpu-metal-mixed-owner-schedule.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span> records the selected
  owner-indexed representation and shared-schedule design.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner))</span>

If an applicable contract is missing or ambiguous, stop and report it.

## Dependencies and integration

- Depends on: [Engine](../../../../glossary.md#term-engine) 0015; [Prepare](../../../../glossary.md#prepare) 0006; [Runtime](../../../../glossary.md#term-runtime) 0016; [Metal](../../../../glossary.md#term-metal) 0004
- Conflicts with: concurrent changes to Engine registry/adapters/handles, Prepare assignment or
  schedule context, Runtime transfer contracts, CPU/Metal representation creation or native copy,
  and affected shared architecture/planning/explanatory documents<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [CPU glossary](../../../../glossary.md#term-cpu); [planning glossary](../../../../glossary.md#term-planning))</span>
- Parallel group: None
- Common base revision: `0fe35a845a18fb9c70cdf87081b6a6c688fece47`
- Integration order: [atomic](../../../../glossary.md#resource-transaction-and-rollback) serial contract-first vertical [slice](../../../../glossary.md#slice)
- Integration validation: native Metal build; affected Engine, Prepare, Runtime, CPU, Metal,
  integration, conformance, and architecture suites; real public mixed lifecycle; Javadocs;
  documentation validation; full repository test gate<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): this isolated task worktree

## Files and symbols

- `modules/prepare` — representation-owner and workspace assignment metadata, backend physical
  contribution contract, <span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [metadata glossary](../../../../glossary.md#term-metadata); [backend glossary](../../../../glossary.md#backend))</span>[`PreparedScheduleContext`](../../../../glossary.md#prepared-schedule-context--preparedschedulecontext), [`GraphPreparation`](../../../../glossary.md#graph-preparation--graphpreparation), and [transaction](../../../../glossary.md#term-transaction) [validation](../../../../glossary.md#term-validation).
- `backends/{cpu,metal}` — finalizer representation selections, external schedule contributions,
  direct transfer binders, and exact F32 dense physical copy validation.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- `modules/engine` — registry complete-set routing/preflight, shared mixed schedule assembler,
  prepared transfer recipe, and per-input/per-publication direct adapter capture.<span data-glossary-navigation="true"> ([registry glossary](../../../../glossary.md#term-registry); [preflight glossary](../../../../glossary.md#term-preflight); [publication glossary](../../../../glossary.md#term-publication); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- `modules/runtime` — only Javadoc or focused validation if the existing direct cold-binding and
  validity contracts remain sufficient.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [binding glossary](../../../../glossary.md#term-binding))</span>
- `testing/{integration-tests,architecture-tests,backend-conformance}` and focused [module](../../../../glossary.md#term-module) tests.
- Authoritative contracts, ADR 0016, Engine master plan/task, lifecycle/boundary/API/backend/user
  explanations, package Javadocs, and targeted glossary terms.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Engine glossary](../../../../glossary.md#term-engine); [lifecycle glossary](../../../../glossary.md#lifecycle); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>

## Acceptance criteria

- A non-empty plan resolves every [partition](../../../../glossary.md#partition) by equal [`BackendId`](../../../../glossary.md#backend-identity--backendid); missing owners and unsupported
  cross-owner direction, positive rank-1..16 extent, layout, type, or checked element/byte
  geometry fail before backend analysis. Registration order stays the deterministic compile
  inventory and never becomes execution fallback.<span data-glossary-navigation="true"> ([rank glossary](../../../../glossary.md#term-rank); [extent glossary](../../../../glossary.md#term-extent); [layout glossary](../../../../glossary.md#layout); [element glossary](../../../../glossary.md#term-element); [backend glossary](../../../../glossary.md#backend); [compile glossary](../../../../glossary.md#compile))</span>
- One logical buffer slot has owner positions ordered producer first (or first consumer for an
  input), then distinct consumers. Shared buffer assignment begins with bindable input IDs in exact
  Compiler occurrence order, independent of partition/declaration order. Each finalizer selects its
  assigned position. Backend contributions cover each assigned buffer/workspace coordinate exactly
  once.<span data-glossary-navigation="true"> ([buffer slot glossary](../../../../glossary.md#buffer-slot--bufferslot); [producer glossary](../../../../glossary.md#term-producer); [buffer glossary](../../../../glossary.md#term-buffer); [bindable input glossary](../../../../glossary.md#bindable-input); [Compiler glossary](../../../../glossary.md#term-compiler); [occurrence glossary](../../../../glossary.md#term-occurrence); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- The immutable schedule creates representations first, executes partitions in compile order,
  places one producer-to-destination transfer before each distinct destination owner's first
  consumer, handles fan-out/interleaving, and publishes the exact final suffix.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>
- CPU-to-Metal and Metal-to-CPU intermediate transfers execute for current real Metal NEG on fully
  static canonical contiguous <span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [contiguous glossary](../../../../glossary.md#term-contiguous))</span>[`FLOAT32`](../../../../glossary.md#term-float32) values. A transfer failure keeps source valid and
  destination invalid; run cleanup is reverse, attempt-all, primary-preserving, and closes no
  borrowed storage.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [borrowed glossary](../../../../glossary.md#ownership-borrowing-and-lease))</span>
- Preparation finalizer/contribution/transfer/schedule/outer-publication failures roll back unique
  persistent resources once in reverse partition/acquisition order, with distinct suppression and
  no self-suppression. Representation-creation and close failures retain existing run transaction
  semantics.<span data-glossary-navigation="true"> ([publication glossary](../../../../glossary.md#term-publication); [transaction glossary](../../../../glossary.md#term-transaction))</span>
- Prepared executions are immutable and reusable. Repeated and concurrent runs own distinct
  borrowed wrappers, created representations, and staging. Result materialization uses the direct
  captured publication adapter and remains valid only for the open result lifetime; detached host
  values outlive it.<span data-glossary-navigation="true"> ([materialization glossary](../../../../glossary.md#materialization); [lifetime glossary](../../../../glossary.md#term-lifetime); [detached glossary](../../../../glossary.md#detached-model-results); [host glossary](../../../../glossary.md#term-host))</span>
- Runtime's bound action traversal contains direct executable, transfer, and publication references
  and performs no Engine, registry, backend-ID, provider, discovery, reflection, or representation
  map lookup. A poisoned registry after preparation cannot affect run or materialization.<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [registry glossary](../../../../glossary.md#term-registry); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- [CPU](../../../../glossary.md#term-cpu)-only and single-[owner](../../../../glossary.md#term-owner) CPU/[Metal](../../../../glossary.md#term-metal) behavior remains current. `prepareTuned` accepts only a
  complete CPU-owned plan and rejects mixed ownership before representative borrowing or trials.<span data-glossary-navigation="true"> ([ownership glossary](../../../../glossary.md#term-ownership-meanings); [borrowing glossary](../../../../glossary.md#term-borrowing))</span>
- Durable tests cover both directions, multiple/interleaved partitions, executed fan-out and exact
  ordering, reversed partition-versus-caller input order, two representations with one logical
  input occurrence, repeated runs, oversized and out-of-domain preflight with both backend analysis
  counters zero, rollback/close failure, exact per-input/per-publication routing, mixed publications
  after registry poison, and no post-prepare lookup.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight); [rollback glossary](../../../../glossary.md#term-rollback); [prepare glossary](../../../../glossary.md#prepare))</span>

## [Validation](../../../../glossary.md#term-validation)

```bash
./native/metal-macos-arm64/build.sh
./gradlew :modules:prepare:test :modules:runtime:test :modules:engine:test \
  :backends:cpu:test :backends:metal:test :testing:backend-conformance:test \
  :testing:architecture-tests:test
./gradlew check
./gradlew :modules:prepare:javadoc :modules:runtime:javadoc :modules:engine:javadoc \
  :backends:cpu:javadoc :backends:metal:javadoc
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/native/metal-macos-arm64/build/libsynaptik_metal_foundation.dylib" \
  ./gradlew :testing:integration-tests:test \
  --tests '*EngineExplicitCompositionMetalIntegrationTest'
git diff --check
```

The repository defines no formatter task or standalone documentation-validator script.
`./gradlew tasks --all` confirmed that `check` is the configured repository validation task.
A targeted Python 3 Markdown validator checked all 26 changed Markdown files, 646 local
links/anchors, and reported zero failures. A separate whitespace [scan](../../../../glossary.md#term-cumulative-scan) checked all eight untracked
new files and reported zero trailing-whitespace or final-newline failures.

The implementation context records exact outcomes. The independent Class C review reuses
successful executable evidence unless it changes production behavior.

## Documentation and review impact

Architecture, decision-record, [planning](../../../../glossary.md#term-planning), [API](../../../../glossary.md#term-application-programming-interface-api)/[Javadoc](../../../../glossary.md#term-javadoc), [backend](../../../../glossary.md#backend)-guide, and user-guide profiles apply.
The independent review must inspect contract-to-code [alignment](../../../../glossary.md#term-alignment), [owner](../../../../glossary.md#term-owner)/index mapping, transfer
capability [preflight](../../../../glossary.md#term-preflight), schedule order, direct-reference [hot path](../../../../glossary.md#term-hot-path), persistent/[run](../../../../glossary.md#term-run) [ownership](../../../../glossary.md#term-ownership-meanings),
[rollback](../../../../glossary.md#term-rollback)/suppression, real native evidence, stale single-owner wording, and current links/anchors.
Independent Class C review is complete; the approved branch is [ready](../../../../glossary.md#term-ready) for its authorized [commit](../../../../glossary.md#term-commit).

## Result

Implementation, [validation](../../../../glossary.md#term-validation), and independent Class C review are complete. The review trail was:
initial BLOCK with three P1 findings; re-review BLOCK with two P1 findings; final executable review
APPROVE with one remaining documentation/[Javadoc](../../../../glossary.md#term-javadoc) P1; narrow documentation review BLOCK with two
[residual](../../../../glossary.md#scan-residual-port-and-history-demand) current-surface claims; and final narrow documentation re-review APPROVE after both
residuals were reconciled. No executable finding remains open.

- Transfer [preflight](../../../../glossary.md#term-preflight) enforces positive [rank](../../../../glossary.md#term-rank)-1..16 fully static [canonical](../../../../glossary.md#term-canonical-form) [contiguous](../../../../glossary.md#term-contiguous) [`FLOAT32`](../../../../glossary.md#term-float32) with
  checked element and byte geometry before either backend analysis. Focused oversized, zero-extent,
  over-rank, executed repeated/interleaved schedule, and registry-poison regressions pass.<span data-glossary-navigation="true"> ([element glossary](../../../../glossary.md#term-element); [backend glossary](../../../../glossary.md#backend); [extent glossary](../../../../glossary.md#term-extent); [registry glossary](../../../../glossary.md#term-registry))</span>
- Shared preparation now places declared bindable buffers first in exact Compiler caller-input
  occurrence order, independently of partition analysis/declaration order, while preserving the
  existing missing-assignment validation. Focused Prepare ordering and validation tests completed<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [Compiler glossary](../../../../glossary.md#term-compiler); [occurrence glossary](../../../../glossary.md#term-occurrence); [partition glossary](../../../../glossary.md#partition); [Prepare glossary](../../../../glossary.md#prepare))</span>
  `BUILD SUCCESSFUL` with all 17 actionable tasks executed.
- The affected Prepare, Runtime, Engine, CPU, Metal, backend-conformance, and architecture suite
  completed <span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine); [CPU glossary](../../../../glossary.md#term-cpu); [conformance glossary](../../../../glossary.md#term-conformance))</span>`BUILD SUCCESSFUL` with all 44 actionable tasks executed.
- `./gradlew check` completed `BUILD SUCCESSFUL` with all 72 actionable tasks executed.
- Affected Javadocs completed `BUILD SUCCESSFUL` with 19 actionable tasks; the final targeted Metal
  Javadoc rerun completed `BUILD SUCCESSFUL` with 10 actionable tasks. Existing CPU Javadoc
  warnings remain non-fatal.
- The final real native [Metal](../../../../glossary.md#term-metal) integration completed `BUILD SUCCESSFUL` with all 33 actionable
  tasks executed. In addition to both transfer directions, repeated execution, fan-out, mixed
  publications, and registry poison, it executes <span data-glossary-navigation="true"> ([fan-out glossary](../../../../glossary.md#term-fan-out))</span>`left.add(right.neg())` under Metal-first
  composition and proves repeated run/materialization with caller order opposite partition order.<span data-glossary-navigation="true"> ([run glossary](../../../../glossary.md#term-run); [materialization glossary](../../../../glossary.md#materialization))</span>
- A targeted validator checked all 28 changed Markdown files and 673 local links/anchors with zero
  failures. Root status, the design-record index, Engine package/API, CPU/Metal integration
  Javadocs, and backend guides now match ADR 0016; ADR 0015 remains historical. <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>`git diff --check`
  and all nine untracked files' trailing-whitespace/final-newline checks also passed.
- The final narrow independent documentation re-review returned `APPROVE`; the task is Complete.
