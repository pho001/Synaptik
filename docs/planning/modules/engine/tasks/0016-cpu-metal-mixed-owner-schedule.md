# Task 0016: CPU/Metal Mixed-Owner Schedule and Transfer

## Status

Complete

Frontier verification: the user explicitly authorized this serial Class C frontier in one isolated
worktree from exact clean revision `0fe35a845a18fb9c70cdf87081b6a6c688fece47`. Engine 0015,
Prepare 0006, Runtime 0016, and Metal 0004 are Complete. ADR 0016 and the authoritative mixed-owner
contract were accepted before production edits. This task exclusively owns the cross-owner
Prepare/Runtime/Engine, CPU/Metal contribution, transfer, and shared documentation scope through
its mandatory independent review.

## Change class

Class C — this task changes mixed-backend composition, shared Prepare assignments and transaction
boundaries, Runtime transfer binding, concrete native transfer lifetime, Engine handle routing, and
the no-lookup hot-path invariant.

## Goal

Prepare and repeatedly run a compiled graph containing CPU- and Metal-owned partitions through
public `Engine.builder()` composition, with explicit bidirectional intermediate transfer, exact
ordering, deterministic ownership, transactional cleanup, and no post-prepare registry lookup.

## Scope

- Route every planned partition by exact `BackendId` and prepare the ordered complete set through
  one shared `GraphPreparation` transaction.
- Assign one representation position per distinct participating owner on each shared logical buffer
  and pass it through backend finalization.
- Adapt CPU and Metal schedule assemblers to contribute externally assigned buffer/workspace
  representation creators; assemble one creation prefix, ordered executions/transfers, and one
  publication suffix.
- Add direct prepared CPU-to-Metal upload and Metal-to-CPU download for fully static canonical
  contiguous `FLOAT32` intermediate values, using run-owned CPU native storage as reusable host
  staging.
- Capture direct input and publication adapters per occurrence in outward prepared/result handles.
- Preserve CPU-only, single-owner CPU, single-owner Metal, repeated prepared runs, public
  materialization, and CPU-only tuning behavior.
- Add focused shared-contract, lifecycle, integration, architecture, documentation, and real Metal
  evidence.

## Non-goals

- No conversion, non-contiguous transfer, non-`FLOAT32` transfer, heap staging, CUDA transfer,
  generic plugin transfer service-provider interface, discovery, fallback, retry, owner
  substitution, asynchronous execution, pooling, aliasing, or native ABI expansion.
- No registry/provider/backend-ID query after successful preparation and no second physical-memory
  or schedule convention.
- No change to Planning ownership scoring, Metal operation capability beyond current NEG, or the
  CPU-only `AdvancedEngine` and autotuning surfaces.

## Contracts

- [`ARCHITECTURE.md` headings `Core invariants` and `Module-ownership routing`](../../../../../ARCHITECTURE.md#core-invariants)
  require shared prepared representation positions, explicit transfer steps, Runtime-only schedule
  execution, and Engine composition.
- [`runtime-prepare-engine.md` headings `modules/runtime`, `modules/prepare`, `Prepare lifecycle`,
  `Run lifecycle`, and `Compile ownership and mixed-owner Prepare routing`](../../../../architecture/contracts/runtime-prepare-engine.md#compile-ownership-and-mixed-owner-prepare-routing)
  own assignment, contribution, schedule order, direct binding, transaction, handle capture, and
  no-lookup behavior.
- [`backend-execution.md` heading `Metal backend`](../../../../architecture/contracts/backend-execution.md#metal-backend)
  owns exact CPU/Metal host-staged physical transfer and native validation.
- [ADR 0016](../../../../design/decisions/0016-cpu-metal-mixed-owner-schedule.md) records the selected
  owner-indexed representation and shared-schedule design.

If an applicable contract is missing or ambiguous, stop and report it.

## Dependencies and integration

- Depends on: Engine 0015; Prepare 0006; Runtime 0016; Metal 0004
- Conflicts with: concurrent changes to Engine registry/adapters/handles, Prepare assignment or
  schedule context, Runtime transfer contracts, CPU/Metal representation creation or native copy,
  and affected shared architecture/planning/explanatory documents
- Parallel group: None
- Common base revision: `0fe35a845a18fb9c70cdf87081b6a6c688fece47`
- Integration order: atomic serial contract-first vertical slice
- Integration validation: native Metal build; affected Engine, Prepare, Runtime, CPU, Metal,
  integration, conformance, and architecture suites; real public mixed lifecycle; Javadocs;
  documentation validation; full repository test gate
- Shared-document integration owner: this isolated task worktree

## Files and symbols

- `modules/prepare` — representation-owner and workspace assignment metadata, backend physical
  contribution contract, `PreparedScheduleContext`, `GraphPreparation`, and transaction validation.
- `backends/{cpu,metal}` — finalizer representation selections, external schedule contributions,
  direct transfer binders, and exact F32 dense physical copy validation.
- `modules/engine` — registry complete-set routing/preflight, shared mixed schedule assembler,
  prepared transfer recipe, and per-input/per-publication direct adapter capture.
- `modules/runtime` — only Javadoc or focused validation if the existing direct cold-binding and
  validity contracts remain sufficient.
- `testing/{integration-tests,architecture-tests,backend-conformance}` and focused module tests.
- Authoritative contracts, ADR 0016, Engine master plan/task, lifecycle/boundary/API/backend/user
  explanations, package Javadocs, and targeted glossary terms.

## Acceptance criteria

- A non-empty plan resolves every partition by equal `BackendId`; missing owners and unsupported
  cross-owner direction, positive rank-1..16 extent, layout, type, or checked element/byte
  geometry fail before backend analysis. Registration order stays the deterministic compile
  inventory and never becomes execution fallback.
- One logical buffer slot has owner positions ordered producer first (or first consumer for an
  input), then distinct consumers. Shared buffer assignment begins with bindable input IDs in exact
  Compiler occurrence order, independent of partition/declaration order. Each finalizer selects its
  assigned position. Backend contributions cover each assigned buffer/workspace coordinate exactly
  once.
- The immutable schedule creates representations first, executes partitions in compile order,
  places one producer-to-destination transfer before each distinct destination owner's first
  consumer, handles fan-out/interleaving, and publishes the exact final suffix.
- CPU-to-Metal and Metal-to-CPU intermediate transfers execute for current real Metal NEG on fully
  static canonical contiguous `FLOAT32` values. A transfer failure keeps source valid and
  destination invalid; run cleanup is reverse, attempt-all, primary-preserving, and closes no
  borrowed storage.
- Preparation finalizer/contribution/transfer/schedule/outer-publication failures roll back unique
  persistent resources once in reverse partition/acquisition order, with distinct suppression and
  no self-suppression. Representation-creation and close failures retain existing run transaction
  semantics.
- Prepared executions are immutable and reusable. Repeated and concurrent runs own distinct
  borrowed wrappers, created representations, and staging. Result materialization uses the direct
  captured publication adapter and remains valid only for the open result lifetime; detached host
  values outlive it.
- Runtime's bound action traversal contains direct executable, transfer, and publication references
  and performs no Engine, registry, backend-ID, provider, discovery, reflection, or representation
  map lookup. A poisoned registry after preparation cannot affect run or materialization.
- CPU-only and single-owner CPU/Metal behavior remains current. `prepareTuned` accepts only a
  complete CPU-owned plan and rejects mixed ownership before representative borrowing or trials.
- Durable tests cover both directions, multiple/interleaved partitions, executed fan-out and exact
  ordering, reversed partition-versus-caller input order, two representations with one logical
  input occurrence, repeated runs, oversized and out-of-domain preflight with both backend analysis
  counters zero, rollback/close failure, exact per-input/per-publication routing, mixed publications
  after registry poison, and no post-prepare lookup.

## Validation

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
links/anchors, and reported zero failures. A separate whitespace scan checked all eight untracked
new files and reported zero trailing-whitespace or final-newline failures.

The implementation context records exact outcomes. The independent Class C review reuses
successful executable evidence unless it changes production behavior.

## Documentation and review impact

Architecture, decision-record, planning, API/Javadoc, backend-guide, and user-guide profiles apply.
The independent review must inspect contract-to-code alignment, owner/index mapping, transfer
capability preflight, schedule order, direct-reference hot path, persistent/run ownership,
rollback/suppression, real native evidence, stale single-owner wording, and current links/anchors.
Independent Class C review is complete; the approved branch is ready for its authorized commit.

## Result

Implementation, validation, and independent Class C review are complete. The review trail was:
initial BLOCK with three P1 findings; re-review BLOCK with two P1 findings; final executable review
APPROVE with one remaining documentation/Javadoc P1; narrow documentation review BLOCK with two
residual current-surface claims; and final narrow documentation re-review APPROVE after both
residuals were reconciled. No executable finding remains open.

- Transfer preflight enforces positive rank-1..16 fully static canonical contiguous `FLOAT32` with
  checked element and byte geometry before either backend analysis. Focused oversized, zero-extent,
  over-rank, executed repeated/interleaved schedule, and registry-poison regressions pass.
- Shared preparation now places declared bindable buffers first in exact Compiler caller-input
  occurrence order, independently of partition analysis/declaration order, while preserving the
  existing missing-assignment validation. Focused Prepare ordering and validation tests completed
  `BUILD SUCCESSFUL` with all 17 actionable tasks executed.
- The affected Prepare, Runtime, Engine, CPU, Metal, backend-conformance, and architecture suite
  completed `BUILD SUCCESSFUL` with all 44 actionable tasks executed.
- `./gradlew check` completed `BUILD SUCCESSFUL` with all 72 actionable tasks executed.
- Affected Javadocs completed `BUILD SUCCESSFUL` with 19 actionable tasks; the final targeted Metal
  Javadoc rerun completed `BUILD SUCCESSFUL` with 10 actionable tasks. Existing CPU Javadoc
  warnings remain non-fatal.
- The final real native Metal integration completed `BUILD SUCCESSFUL` with all 33 actionable
  tasks executed. In addition to both transfer directions, repeated execution, fan-out, mixed
  publications, and registry poison, it executes `left.add(right.neg())` under Metal-first
  composition and proves repeated run/materialization with caller order opposite partition order.
- A targeted validator checked all 28 changed Markdown files and 673 local links/anchors with zero
  failures. Root status, the design-record index, Engine package/API, CPU/Metal integration
  Javadocs, and backend guides now match ADR 0016; ADR 0015 remains historical. `git diff --check`
  and all nine untracked files' trailing-whitespace/final-newline checks also passed.
- The final narrow independent documentation re-review returned `APPROVE`; the task is Complete.
