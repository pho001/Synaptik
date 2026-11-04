# Task NN0021B Fixed recurrent [scan](../../../../glossary.md#term-cumulative-scan) forward execution program

> **Editorial reading context — not a status update.** This packet coordinates acceptance;
> CPU owns the implementation of the prepared recurrent loop. One Model occurrence is not
> one graph node per time step, even though its backend implementation visits valid steps.
> Input states and returned final states remain explicit, so repeated runs do not silently
> carry state between batches. Existing cold persistence is a prerequisite, not proof the
> saved source executes. Today's [runtime-length construction/execution distinction](../../../../reference/nn/recurrent.md#nn-runtime-length-recurrent-composition)
> and [owned execution contract](../../../../reference/nn/owned-model.md#owned-tensor-model-capability)
> provide current context without extending the original CPU/type/gradient qualification.

## Status

Complete — main accepted CPU0028 and this bounded program after independent approval and the
successful final root gate on 2026-10-20. Authorized [CPU](../../../../glossary.md#term-cpu)/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) frontiers: None.

## Change class

Class C program — [concrete backend](../../../../glossary.md#concrete-backend) capability, hot-loop invariants, reusable [prepared execution](../../../../glossary.md#prepared-execution--preparedexecution),
[runtime](../../../../glossary.md#term-runtime) value [validation](../../../../glossary.md#term-validation) and multi-output [publication](../../../../glossary.md#term-publication). This document changes no authority.

## Goal

Make the existing flat [Model](../../../../glossary.md#term-model) [scan](../../../../glossary.md#term-cumulative-scan) executable through ordinary [CPU](../../../../glossary.md#term-cpu) [compile](../../../../glossary.md#compile) → [prepare](../../../../glossary.md#prepare) → [run](../../../../glossary.md#term-run).
One scan [occurrence](../../../../glossary.md#term-occurrence) prepares one reusable [kernel](../../../../glossary.md#kernel); explicit input states and fresh valid lengths
produce dense outputs and final states without static [unrolling](../../../../glossary.md#term-unrolling) or hidden carried state.

## [Scope](../../../../glossary.md#term-scope) and [ownership](../../../../glossary.md#term-ownership-meanings)

- Model0025E/0025F and Compiler0006A already supply the family, canonical output provenance,
  static inference, materialized layout closure and fail-closed backpropagation through time
  (BPTT). CP0010/CP0011 already supply cold persistence. Do not reimplement those capabilities.<span data-glossary-navigation="true"> ([BPTT glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state))</span><span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [provenance glossary](../../../../glossary.md#provenance); [inference glossary](../../../../glossary.md#term-inference); [layout glossary](../../../../glossary.md#layout); [fail-closed glossary](../../../../glossary.md#term-fail-closed); [backpropagation through time glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [persistence glossary](../../../../glossary.md#term-persistence))</span>
- [CPU0028](../../../backends/cpu/tasks/0028-fixed-recurrent-scan-forward-execution.md) owns the
  complete CPU admission, private lowering/Java kernel, resources, binding, tests and documentation.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [admission glossary](../../../../glossary.md#term-admission); [lowering glossary](../../../../glossary.md#lowering); [kernel glossary](../../../../glossary.md#kernel); [binding glossary](../../../../glossary.md#term-binding))</span>
- NN0021B owns coordination and final acceptance, not NN or CPU implementation. There is no
  source-backed prerequisite in Planning, Prepare, Runtime or Engine.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Planning glossary](../../../../glossary.md#term-planning); [Prepare glossary](../../../../glossary.md#prepare); [Runtime glossary](../../../../glossary.md#term-runtime); [Engine glossary](../../../../glossary.md#term-engine))</span>

## Non-goals

No [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) builder/sequence migration, Data [module](../../../../glossary.md#term-module)/[API](../../../../glossary.md#term-application-programming-interface-api), [runtime](../../../../glossary.md#term-runtime)-length NN convenience, training/[BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt),
[checkpoint](../../../../glossary.md#term-checkpoint) wire change, public wrapper/session, execution hierarchy, [registry](../../../../glossary.md#term-registry)/service, [graph](../../../../glossary.md#graph) region,
loop-body API or second pipeline. No native route, tuning search, performance campaign or five-[fork](../../../../glossary.md#term-fork-meanings)
[matrix](../../../../glossary.md#term-matrix-meanings). Current static Java `long[]` sequences and constituent-[operation](../../../../glossary.md#operation) training stay unchanged.

## Contracts

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>,
  [Explicit non-goals](../../../../../ARCHITECTURE.md#explicit-non-goals) and
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Read the complete [recurrent-scan contract](../../../../architecture/contracts/recurrent-scan.md)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>:
  `Fixed recurrent scan without graph regions`, `Fixed family and current Model surface`,
  `Static Shape and runtime-value boundary`, `Valid lengths, traversal, and outputs`,
  `Purity and lifecycle ownership`, `Performance and migration invariants`.
- [Backend](../../../../glossary.md#backend) [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>.
- [Lifecycle](../../../../glossary.md#lifecycle) [modules/runtime](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime)<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>,
  [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>,
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>,
  [modules/engine](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)<span data-glossary-navigation="true"> ([engine glossary](../../../../glossary.md#term-engine))</span> and
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>.
- [Compiler](../../../../glossary.md#term-compiler) [modules/compiler](../../../../architecture/contracts/compiler-autograd.md#modulescompiler),
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>,
  plus the recurrent contract's BPTT boundary. Model ownership and formulas remain in<span data-glossary-navigation="true"> ([BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [Model glossary](../../../../glossary.md#term-model); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
  [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel) and
  [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../../../../glossary.md#profile-free-numerical-semantics))</span>.

Stop and report an actual source-backed contract gap; [planning](../../../../glossary.md#term-planning) cannot enact authority changes.

## Dependencies and integration

- Depends on: NN0021A; Model0025E/0025F; Compiler0006A; CPU0027; Engine0021; CP0010/CP0011 — Complete.
- Conflicts with: CPU admission/unit-plan/finalization, shared scan/lifecycle contracts and shared
  documentation; no active conflicting writer reported by main.<span data-glossary-navigation="true"> ([admission glossary](../../../../glossary.md#term-admission); [scan glossary](../../../../glossary.md#term-cumulative-scan); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Parallel group: None; independent read-only [planning](../../../../glossary.md#term-planning)/review may overlap.
- Common base revision: `e9b00bafddb36c28b12a9bb9fa5add3f19978547` (`main == origin/main`).
- Integration order: main inspects verified CPU0028 Ready/frontier → clean
  CPU implementation → independent targeted code/documentation review → main's integration gate
  → NN0021B acceptance. CPU0028 does not depend on NN0021B being Complete.
- Integration validation: CPU0028 worker evidence, conformance/public Engine tests, then main's
  single source-matched repository <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [conformance glossary](../../../../glossary.md#term-conformance); [Engine glossary](../../../../glossary.md#term-engine))</span>`test javadoc` [checkpoint](../../../../glossary.md#term-checkpoint) as recorded in CPU0028.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main; [CPU](../../../../glossary.md#term-cpu) worker may draft only its assigned explanations.

Launch verified 2026-10-20: user authorization, [Ready](../../../../glossary.md#term-ready)/frontier, Complete predecessors/reviews,
recorded base, unchanged authority, inactive conflicts and all seven integration fields.
No [commit](../../../../glossary.md#term-commit)/push; the two unrelated dirty files identified in CPU0028 remain protected.

## Source facts and selected first capability

`GraphCompiler` already queries all ordered ports; [`OperationCapabilityQuery`](../../../../glossary.md#operation-capability-query--operationcapabilityquery) is family-neutral.
`StaticResultLogicalLayoutClosure` explicitly includes `RecurrentScanKind` and closes every output.
`GraphPreparation.Projection.context` projects all inputs and siblings; [publication](../../../../glossary.md#term-publication) [validation](../../../../glossary.md#term-validation)
preserves requested occurrences. [`PreparedExecutionRunner`](../../../../glossary.md#prepared-execution-runner--preparedexecutionrunner) binds first and executes later, in order.
`AdvancedEngine.runOrdinaryOpen` uses typed input/publication lists without a common-[dtype](../../../../glossary.md#data-type--datatype) or
single-output assumption; `runValues` accepts fresh exact-[descriptor](../../../../glossary.md#term-descriptor) values. Shared executable
code is unchanged. Produced lengths are checked during execution, not global cold [binding](../../../../glossary.md#term-binding).

The selected [CPU](../../../../glossary.md#term-cpu) baseline is [homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32) floating roles with arbitrary valid [gradient](../../../../glossary.md#term-gradient)
flags and exact output OR, non-gradient [INT64](../../../../glossary.md#data-type--datatype) lengths, [canonical](../../../../glossary.md#term-canonical-form) [layouts](../../../../glossary.md#layout) and checked geometry.
It includes RNN_TANH, GRU_RESET_AFTER and [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm), both directions, [bias](../../../../glossary.md#term-bias)/no bias, all canonical outputs
and zero time/[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters). [Scalar](../../../../glossary.md#term-scalar) sequential Java [kernels](../../../../glossary.md#kernel) use exact segment access for [heap](../../../../glossary.md#term-heap), native and
mixed canonical storage. CPU0028 specifies exclusions and validation. [FLOAT64](../../../../glossary.md#term-float64)/low types, arbitrary
layouts, [vector](../../../../glossary.md#term-vector)/parallel [scan](../../../../glossary.md#term-cumulative-scan) strategies and tuning are not implied by FLOAT32 support.
Gradient eligibility is not a backward request: true [PARAMETER](../../../../glossary.md#term-parameter) [weights](../../../../glossary.md#term-weight), including restored scans,
must execute FORWARD_ONLY while all backward-capable scan requests remain [Compiler](../../../../glossary.md#term-compiler)-rejected.

CPU0028 supplies exact recurrent [admission](../../../../glossary.md#term-admission), mutually exclusive direct recurrent facts in existing
private records, and ordinary Java finalization through the existing [Runtime](../../../../glossary.md#term-runtime) base/composite.
It adds no generic execution hierarchy or generated scan [artifact](../../../../glossary.md#term-artifact).

## Acceptance criteria

1. CPU0028 passes exact provider, independent-equation, failure-before-write, produced-length,
   repeated/fresh-input, carrier, isolation and public multi-output publication acceptance.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [publication glossary](../../../../glossary.md#term-publication))</span>
2. The entire length vector and access preconditions are validated before any scan result write.
   No padded recurrent arithmetic occurs; padding is positive zero; zero-length final states are
   raw initial values; reverse traversal uses only the valid prefix at original time positions.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [scan glossary](../../../../glossary.md#term-cumulative-scan); [padding glossary](../../../../glossary.md#padding); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
3. Static descriptors, resources and kernel count are independent of length values. Kernel/node
   count is constant in time; dense output memory is still proportional to time × batch × hidden.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [kernel glossary](../../../../glossary.md#kernel); [node glossary](../../../../glossary.md#node); [constant glossary](../../../../glossary.md#term-constant); [batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>
4. Current NN/static-unrolled training and cold wire formats remain unchanged. BPTT remains
   Compiler-rejected. No Data/BPTT/NN migration successor becomes a dependency of this program.<span data-glossary-navigation="true"> ([NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt); [Compiler glossary](../../../../glossary.md#term-compiler); [dependency glossary](../../../../glossary.md#term-dependency))</span>
5. Main records code/docs review and integration evidence before marking the program Complete.

## [Validation](../../../../glossary.md#term-validation) and review impact

CPU0028 owns exact worker/integration/root commands, outcomes and execution-status documentation.
Independent code/docs review reused successful executable evidence; main ran the final root gate.
Closure checks local links/anchors, sizes, [metadata](../../../../glossary.md#term-metadata)/status/[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) consistency, whitespace and protected
[hashes](../../../../glossary.md#term-hash) only, without replaying tests/builds. The general and [planning](../../../../glossary.md#term-planning) documentation profiles apply.
Affected Javadocs and explanations are finalized in CPU0028; glossary execution status is current,
with no new term or definition. No architecture, [ADR](../../../../glossary.md#term-architecture-decision-record-adr), public [API](../../../../glossary.md#term-application-programming-interface-api), [schema](../../../../glossary.md#term-schema) or [ABI](../../../../glossary.md#term-application-binary-interface-abi) change.

## Result

Main accepted all five program criteria on 2026-10-20. Completed
[CPU0028 Result](../../../backends/cpu/tasks/0028-fixed-recurrent-scan-forward-execution.md#result)
owns exact implementation, changed areas, commands, review and [validation](../../../../glossary.md#term-validation) evidence.
Delivered [canonical](../../../../glossary.md#term-canonical-form) [FLOAT32](../../../../glossary.md#term-float32) [scalar](../../../../glossary.md#term-scalar) direct Java [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm), all six [signatures](../../../../glossary.md#term-signature)/directions,
[heap](../../../../glossary.md#term-heap)/native/mixed storage, all outputs, exact output-flag OR and fresh [runtime](../../../../glossary.md#term-runtime) [INT64](../../../../glossary.md#data-type--datatype) lengths.
One [immutable](../../../../glossary.md#term-immutable) prepared [kernel](../../../../glossary.md#kernel) per [occurrence](../../../../glossary.md#term-occurrence) uses isolated [run](../../../../glossary.md#term-run) scratch; full length/access
[preflight](../../../../glossary.md#term-preflight) follows [producers](../../../../glossary.md#term-producer) and precedes every [scan](../../../../glossary.md#term-cumulative-scan) write. Dense output remains O(T*B*H).
Worker [CPU](../../../../glossary.md#term-cpu): 1,101 total/1,071 passed/30 existing opt-in skips; focused CPU 55, [conformance](../../../../glossary.md#term-conformance)/ledger 7,
[Engine](../../../../glossary.md#term-engine) 5 and Fluent 5 passed; corrected cell-[oracle](../../../../glossary.md#term-test-oracle) tests 7 CPU + 1 conformance passed.
Dewey final independent code/docs APPROVE, zero findings, on unchanged 32-file [manifest](../../../../glossary.md#term-manifest)
`f8b00fbb3f32cd3d0be3e0fa78eded76d1cc40ddc74925304ccc62ec63ba33e1`.
Main's configured [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26.0.1/[Metal](../../../../glossary.md#term-metal) `./gradlew test javadoc`: exit 0 in 5m44s; 107 tasks,
35 executed/72 UP-TO-DATE. Current mapped-project XML: 4,328 tests/647 suites, 41 skips,
zero failures/errors, including reused UP-TO-DATE evidence; stale old project paths excluded.
CPU0028's affected documentation/Javadocs and final root generation passed; closure updates only
the two briefs, CPU/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) master plans and roadmap. No NN source, public [API](../../../../glossary.md#term-application-programming-interface-api) or architecture edit.
Static NN sequences/training, cold wire formats and [Compiler](../../../../glossary.md#term-compiler) [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) rejection remain unchanged.
No Data/NN migration, [FLOAT64](../../../../glossary.md#term-float64)/low types, noncontiguous [layouts](../../../../glossary.md#layout), [vector](../../../../glossary.md#term-vector)/parallel/native scan route,
tuning or performance claim is added. No unresolved issues or required follow-up.
Authorized CPU/NN frontiers: None; no successor authorized; protected files unchanged; no [commit](../../../../glossary.md#term-commit)/push.
Status: Complete
