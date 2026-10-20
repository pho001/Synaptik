# Task NN0021B Fixed recurrent scan forward execution program

## Status

Complete — main accepted CPU0028 and this bounded program after independent approval and the
successful final root gate on 2026-10-20. Authorized CPU/NN frontiers: None.

## Change class

Class C program — concrete backend capability, hot-loop invariants, reusable prepared execution,
runtime value validation and multi-output publication. This document changes no authority.

## Goal

Make the existing flat Model scan executable through ordinary CPU compile → prepare → run.
One scan occurrence prepares one reusable kernel; explicit input states and fresh valid lengths
produce dense outputs and final states without static unrolling or hidden carried state.

## Scope and ownership

- Model0025E/0025F and Compiler0006A already supply the family, canonical output provenance,
  static inference, materialized layout closure and fail-closed backpropagation through time
  (BPTT). CP0010/CP0011 already supply cold persistence. Do not reimplement those capabilities.
- [CPU0028](../../../backends/cpu/tasks/0028-fixed-recurrent-scan-forward-execution.md) owns the
  complete CPU admission, private lowering/Java kernel, resources, binding, tests and documentation.
- NN0021B owns coordination and final acceptance, not NN or CPU implementation. There is no
  source-backed prerequisite in Planning, Prepare, Runtime or Engine.

## Non-goals

No NN builder/sequence migration, Data module/API, runtime-length NN convenience, training/BPTT,
checkpoint wire change, public wrapper/session, execution hierarchy, registry/service, graph region,
loop-body API or second pipeline. No native route, tuning search, performance campaign or five-fork
matrix. Current static Java `long[]` sequences and constituent-operation training stay unchanged.

## Contracts

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules),
  [Explicit non-goals](../../../../../ARCHITECTURE.md#explicit-non-goals) and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Read the complete [recurrent-scan contract](../../../../architecture/contracts/recurrent-scan.md):
  `Fixed recurrent scan without graph regions`, `Fixed family and current Model surface`,
  `Static Shape and runtime-value boundary`, `Valid lengths, traversal, and outputs`,
  `Purity and lifecycle ownership`, `Performance and migration invariants`.
- Backend [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).
- Lifecycle [modules/runtime](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
  [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle),
  [modules/engine](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine) and
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution).
- Compiler [modules/compiler](../../../../architecture/contracts/compiler-autograd.md#modulescompiler),
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle),
  plus the recurrent contract's BPTT boundary. Model ownership and formulas remain in
  [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel) and
  [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics).

Stop and report an actual source-backed contract gap; planning cannot enact authority changes.

## Dependencies and integration

- Depends on: NN0021A; Model0025E/0025F; Compiler0006A; CPU0027; Engine0021; CP0010/CP0011 — Complete.
- Conflicts with: CPU admission/unit-plan/finalization, shared scan/lifecycle contracts and shared
  documentation; no active conflicting writer reported by main.
- Parallel group: None; independent read-only planning/review may overlap.
- Common base revision: `e9b00bafddb36c28b12a9bb9fa5add3f19978547` (`main == origin/main`).
- Integration order: main inspects verified CPU0028 Ready/frontier → clean
  CPU implementation → independent targeted code/documentation review → main's integration gate
  → NN0021B acceptance. CPU0028 does not depend on NN0021B being Complete.
- Integration validation: CPU0028 worker evidence, conformance/public Engine tests, then main's
  single source-matched repository `test javadoc` checkpoint as recorded in CPU0028.
- Shared-document integration owner: main; CPU worker may draft only its assigned explanations.

Launch verified 2026-10-20: user authorization, Ready/frontier, Complete predecessors/reviews,
recorded base, unchanged authority, inactive conflicts and all seven integration fields.
No commit/push; the two unrelated dirty files identified in CPU0028 remain protected.

## Source facts and selected first capability

`GraphCompiler` already queries all ordered ports; `OperationCapabilityQuery` is family-neutral.
`StaticResultLogicalLayoutClosure` explicitly includes `RecurrentScanKind` and closes every output.
`GraphPreparation.Projection.context` projects all inputs and siblings; publication validation
preserves requested occurrences. `PreparedExecutionRunner` binds first and executes later, in order.
`AdvancedEngine.runOrdinaryOpen` uses typed input/publication lists without a common-dtype or
single-output assumption; `runValues` accepts fresh exact-descriptor values. Shared executable
code is unchanged. Produced lengths are checked during execution, not global cold binding.

The selected CPU baseline is homogeneous FLOAT32 floating roles with arbitrary valid gradient
flags and exact output OR, non-gradient INT64 lengths, canonical layouts and checked geometry.
It includes RNN_TANH, GRU_RESET_AFTER and LSTM, both directions, bias/no bias, all canonical outputs
and zero time/batch. Scalar sequential Java kernels use exact segment access for heap, native and
mixed canonical storage. CPU0028 specifies exclusions and validation. FLOAT64/low types, arbitrary
layouts, vector/parallel scan strategies and tuning are not implied by FLOAT32 support.
Gradient eligibility is not a backward request: true PARAMETER weights, including restored scans,
must execute FORWARD_ONLY while all backward-capable scan requests remain Compiler-rejected.

CPU0028 supplies exact recurrent admission, mutually exclusive direct recurrent facts in existing
private records, and ordinary Java finalization through the existing Runtime base/composite.
It adds no generic execution hierarchy or generated scan artifact.

## Acceptance criteria

1. CPU0028 passes exact provider, independent-equation, failure-before-write, produced-length,
   repeated/fresh-input, carrier, isolation and public multi-output publication acceptance.
2. The entire length vector and access preconditions are validated before any scan result write.
   No padded recurrent arithmetic occurs; padding is positive zero; zero-length final states are
   raw initial values; reverse traversal uses only the valid prefix at original time positions.
3. Static descriptors, resources and kernel count are independent of length values. Kernel/node
   count is constant in time; dense output memory is still proportional to time × batch × hidden.
4. Current NN/static-unrolled training and cold wire formats remain unchanged. BPTT remains
   Compiler-rejected. No Data/BPTT/NN migration successor becomes a dependency of this program.
5. Main records code/docs review and integration evidence before marking the program Complete.

## Validation and review impact

CPU0028 owns exact worker/integration/root commands, outcomes and execution-status documentation.
Independent code/docs review reused successful executable evidence; main ran the final root gate.
Closure checks local links/anchors, sizes, metadata/status/DAG consistency, whitespace and protected
hashes only, without replaying tests/builds. The general and planning documentation profiles apply.
Affected Javadocs and explanations are finalized in CPU0028; glossary execution status is current,
with no new term or definition. No architecture, ADR, public API, schema or ABI change.

## Result

Main accepted all five program criteria on 2026-10-20. Completed
[CPU0028 Result](../../../backends/cpu/tasks/0028-fixed-recurrent-scan-forward-execution.md#result)
owns exact implementation, changed areas, commands, review and validation evidence.
Delivered canonical FLOAT32 scalar direct Java RNN/GRU/LSTM, all six signatures/directions,
heap/native/mixed storage, all outputs, exact output-flag OR and fresh runtime INT64 lengths.
One immutable prepared kernel per occurrence uses isolated run scratch; full length/access
preflight follows producers and precedes every scan write. Dense output remains O(T*B*H).
Worker CPU: 1,101 total/1,071 passed/30 existing opt-in skips; focused CPU 55, conformance/ledger 7,
Engine 5 and Fluent 5 passed; corrected cell-oracle tests 7 CPU + 1 conformance passed.
Dewey final independent code/docs APPROVE, zero findings, on unchanged 32-file manifest
`f8b00fbb3f32cd3d0be3e0fa78eded76d1cc40ddc74925304ccc62ec63ba33e1`.
Main's configured JDK 26.0.1/Metal `./gradlew test javadoc`: exit 0 in 5m44s; 107 tasks,
35 executed/72 UP-TO-DATE. Current mapped-project XML: 4,328 tests/647 suites, 41 skips,
zero failures/errors, including reused UP-TO-DATE evidence; stale old project paths excluded.
CPU0028's affected documentation/Javadocs and final root generation passed; closure updates only
the two briefs, CPU/NN master plans and roadmap. No NN source, public API or architecture edit.
Static NN sequences/training, cold wire formats and Compiler BPTT rejection remain unchanged.
No Data/NN migration, FLOAT64/low types, noncontiguous layouts, vector/parallel/native scan route,
tuning or performance claim is added. No unresolved issues or required follow-up.
Authorized CPU/NN frontiers: None; no successor authorized; protected files unchanged; no commit/push.
Status: Complete
