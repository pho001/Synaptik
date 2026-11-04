# Task CPU0028 Fixed recurrent [scan](../../../../glossary.md#term-cumulative-scan) forward execution

## Status

Complete — main accepted the independently approved implementation and root gate on 2026-10-20.

## Change class

Class C — new [backend](../../../../glossary.md#backend) execution, pre-write value [validation](../../../../glossary.md#term-validation), per-[run](../../../../glossary.md#term-run) scratch and hot-loop invariants.
Clean implementation and independent code/docs review required.

## Goal

Execute each existing flat recurrent [occurrence](../../../../glossary.md#term-occurrence) once per [run](../../../../glossary.md#term-run) through the ordinary [CPU](../../../../glossary.md#term-cpu) [partition](../../../../glossary.md#partition)
[lifecycle](../../../../glossary.md#lifecycle), using a reusable [backend](../../../../glossary.md#backend)-private Java [kernel](../../../../glossary.md#kernel) and explicit run-owned state and outputs.

## [Scope](../../../../glossary.md#term-scope) and first support set

Every admitted [occurrence](../../../../glossary.md#term-occurrence) must satisfy all rows below; support is not inferred across types/routes.

| [Dimension](../../../../glossary.md#dimension) | Admitted set |
|---|---|
| Meaning | RNN_TANH, GRU_RESET_AFTER, [LSTM](../../../../glossary.md#term-long-short-term-memory-lstm); [bias](../../../../glossary.md#term-bias)/no bias; FORWARD/REVERSE; exact [Model](../../../../glossary.md#term-model) port order. |
| Types and flags | [FLOAT32](../../../../glossary.md#term-float32) floating roles with arbitrary valid [gradient](../../../../glossary.md#term-gradient) flags; every output flag is their exact OR. Lengths [INT64](../../../../glossary.md#data-type--datatype)[B], `requiresGrad == false`. Flags do not request backward execution. |
| Geometry | Fully static `[T,B,I]`, states `[B,H]`, exact packed [weights](../../../../glossary.md#term-weight)/bias and outputs; I/H positive, T/B may be zero. Each dimension, packed gate [extent](../../../../glossary.md#term-extent) and each boundary [element](../../../../glossary.md#term-element) count fits `int`; all products/byte spans/workspace sizes checked. |
| [Layout](../../../../glossary.md#layout) | Every [descriptor](../../../../glossary.md#term-descriptor) equals `LayoutDescriptor.contiguous(shape)`, including zero offset; no unresolved, strided, broadcast or [view](../../../../glossary.md#term-view-meanings) layout. Segment slices may supply a different physical base with this same descriptor. |
| Storage | Exact accessible aligned segment access for [canonical](../../../../glossary.md#term-canonical-form) [heap](../../../../glossary.md#term-heap) float[]/long[], native segments and mixtures, including read-only inputs; [run](../../../../glossary.md#term-run)-owned writable outputs/scratch. No [carrier](../../../../glossary.md#term-storage-carrier) copy. |
| Strategy | One sequential [SCALAR](../../../../glossary.md#term-scalar) [scan](../../../../glossary.md#term-cumulative-scan) range, species bits zero, on the invoking thread; [vector](../../../../glossary.md#term-vector) preference/parallel budgets do not promote the scan. Other units retain their existing strategies. |

FLOAT32 avoids new low [rounding](../../../../glossary.md#term-rounding)/[FLOAT64](../../../../glossary.md#term-float64) [qualification](../../../../glossary.md#term-qualification). Other types/layouts, backward execution,
scan [SIMD](../../../../glossary.md#simd)/parallel/native, [fusion](../../../../glossary.md#term-fusion)/compaction, tuning and speed claims are excluded. Scan-containing
partitions use existing direct [decomposition](../../../../glossary.md#term-decomposition), without complete-plan/copy enumeration. Discovery
reports incomplete/unsupported as for direct long plans; ordinary untuned preparation works.

## Non-goals

No shared executable edits, public wrapper/session, new hierarchy, Data/[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) migration, training/[BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt),
[wire format](../../../../glossary.md#term-wire-format) or [dependency](../../../../glossary.md#term-dependency) change.
No static per-step [graph](../../../../glossary.md#graph), [callback](../../../../glossary.md#term-callback)/body/region, hidden state, workspace pool, generated [scan](../../../../glossary.md#term-cumulative-scan) code,
benchmark campaign or cleanup of unrelated [CPU](../../../../glossary.md#term-cpu) code.

## Contracts

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>,
  [Explicit non-goals](../../../../../ARCHITECTURE.md#explicit-non-goals),
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Read the complete [recurrent-scan contract](../../../../architecture/contracts/recurrent-scan.md)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan))</span>,
  including `Fixed recurrent scan without graph regions`, `Fixed family and current Model surface`,
  `Static Shape and runtime-value boundary`, `Valid lengths, traversal, and outputs`,
  `Purity and lifecycle ownership`, `Performance and migration invariants`.
- [Backend](../../../../glossary.md#backend) [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)<span data-glossary-navigation="true"> ([Concrete backend glossary](../../../../glossary.md#concrete-backend); [modules glossary](../../../../glossary.md#term-module))</span>
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span>.
- [Lifecycle](../../../../glossary.md#lifecycle) [modules/runtime](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime)<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime))</span>,
  [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([prepare glossary](../../../../glossary.md#prepare))</span>,
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>,
  [modules/engine](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine)<span data-glossary-navigation="true"> ([engine glossary](../../../../glossary.md#term-engine))</span>,
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)<span data-glossary-navigation="true"> ([prepared execution glossary](../../../../glossary.md#prepared-execution--preparedexecution))</span>.
- [Model](../../../../glossary.md#term-model) [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel),
  [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../../../../glossary.md#profile-free-numerical-semantics))</span>
  (family formulas, primitive/aggregate/composite sites); Compiler<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span> and
  recurrent contract's explicit BPTT rejection. Stop on an actual contract gap, not a hypothetical one.<span data-glossary-navigation="true"> ([BPTT glossary](../../../../glossary.md#term-backpropagation-through-time-bptt))</span>

## Dependencies and integration

- Depends on: NN0021A; Model0025E/0025F; Compiler0006A; CPU0027; Engine0021; CP0010/CP0011 — Complete.
- Conflicts with: CPU capability, unit facts/decomposition/finalization and shared scan/lifecycle
  documentation; CPU0023/other route work must not overlap these scopes.<span data-glossary-navigation="true"> ([decomposition glossary](../../../../glossary.md#term-decomposition); [scan glossary](../../../../glossary.md#term-cumulative-scan); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Parallel group: None.
- Common base revision: `e9b00bafddb36c28b12a9bb9fa5add3f19978547`, `main == origin/main`.
- Integration order: main Ready verification → clean CPU worker → independent review → main gate
  → NN0021B acceptance; no dependency on NN0021B completion.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency))</span>
- Integration [validation](../../../../glossary.md#term-validation): focused [conformance](../../../../glossary.md#term-conformance)/[Engine](../../../../glossary.md#term-engine) gate and one main-owned root `test javadoc`.
- Shared-document integration [owner](../../../../glossary.md#term-owner): main; assign explanatory docs to one writer during review.

Launch verified 2026-10-20: user authorization, [Ready](../../../../glossary.md#term-ready)/frontier, Complete predecessors/reviews,
recorded base, unchanged authority, inactive conflicts and all seven integration fields.
Closure preserves unrelated THIRD_PARTY_NOTICES.md and CpuOrdinaryPointwiseCastCleanJavaOracle.java;
no [commit](../../../../glossary.md#term-commit)/push. Authorized [CPU](../../../../glossary.md#term-cpu) frontier after acceptance: None.

## Files and symbols

[CPU](../../../../glossary.md#term-cpu) paths are relative to `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.

- `CpuCapabilityProvider.supports`: exact recurrent [admission](../../../../glossary.md#term-admission) before one-output defaults.
- `internal/lowering/CpuPartitionLowering.LoweredPartition`, `CpuPartitionDagDecomposer.Unit`,
  `internal/prepare/CpuPartitionPreparer.analyze/analyzeUnit/analyzeDag` and
  `CpuPartitionPreparationPlan.ExecutionUnitPlan/UnitRuntimeFacts`: mutually exclusive direct
  recurrent versus generated portable facts; existing generated consumers remain guarded.
- `internal/prepare/CpuPartitionFinalizer` and `internal/executable/CpuPreparedPartitionExecutable`:
  direct scan children after slot assignment, without generated artifacts or weaker access checks.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
- Added `internal/lowering/CpuRecurrentScanLowering` ([immutable](../../../../glossary.md#term-immutable) `Facts`/`Geometry`),
  `internal/executable/CpuRecurrentScanPreparedExecutable` (private `Invocation`) and
  `internal/route/portable/CpuRecurrentScanKernel`: checked geometry/roles, binding/preflight and
  direct FLOAT32 family loops using existing Runtime, segment and workspace ownership.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [preflight glossary](../../../../glossary.md#term-preflight); [FLOAT32 glossary](../../../../glossary.md#term-float32); [Runtime glossary](../../../../glossary.md#term-runtime); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- CPU tests: `CpuRecurrentScanLoweringTest`, `CpuRecurrentScanExecutionTest`, capability,
  composed validation, partition DAG, preparation/finalization and internal package inventory.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [partition glossary](../../../../glossary.md#partition); [DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag))</span>
- Testing projects: `CpuRecurrentScanConformanceTest`, `EngineCpuRecurrentScanIntegrationTest`,
  v2 low-precision capability ledger/test; v1 unchanged. Shared executable seams remain read-only.<span data-glossary-navigation="true"> ([precision glossary](../../../../glossary.md#term-precision); [capability ledger glossary](../../../../glossary.md#capability-ledger))</span>

## Acceptance criteria

1. Provider and lowerer agree on the table, all six signatures and two directions, exact shapes,
   flags/layouts, checked overflow and malformed negatives. Prepare reads no length values and
   retains no prototype storage/length specialization; repeated lengths use the same preparation.<span data-glossary-navigation="true"> ([signatures glossary](../../../../glossary.md#term-signature); [shapes glossary](../../../../glossary.md#shape); [layouts glossary](../../../../glossary.md#layout); [overflow glossary](../../../../glossary.md#term-overflow); [Prepare glossary](../../../../glossary.md#prepare); [prototype glossary](../../../../glossary.md#term-prototype); [specialization glossary](../../../../glossary.md#term-specialization))</span>
2. One immutable prepared scan kernel per occurrence, independent of T. Deduplicate repeated input
   ValueIds but retain ordered roles; preserve every sibling output, subset/reordered/aliased roots,
   fan-out and composition. Dense output allocation remains O(T<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [scan glossary](../../../../glossary.md#term-cumulative-scan); [kernel glossary](../../../../glossary.md#kernel); [occurrence glossary](../../../../glossary.md#term-occurrence); [fan-out glossary](../../../../glossary.md#term-fan-out))</span>*B*H), not [constant](../../../../glossary.md#term-constant) memory.
3. Declare buffers and per-run gate/state scratch before slots. Derive checked exact scratch for
   the chosen row-wise loop (O(gates*H), independent of T and lengths); omit unused zero-domain
   scratch. No hidden prepared mutable state, per-step allocations, graph growth or hot maps,
   boxing, reflection, operation/backend lookup. Finish all recurrent projections from prior h
   before overwriting h; explicitly carry LSTM c. No output/scratch overlap with inputs or each other.<span data-glossary-navigation="true"> ([buffers glossary](../../../../glossary.md#term-buffer); [run glossary](../../../../glossary.md#term-run); [mutable glossary](../../../../glossary.md#term-mutable); [graph glossary](../../../../glossary.md#graph); [operation glossary](../../../../glossary.md#operation); [backend glossary](../../../../glossary.md#backend); [lookup glossary](../../../../glossary.md#term-lookup); [carry glossary](../../../../glossary.md#term-carry); [LSTM glossary](../../../../glossary.md#term-long-short-term-memory-lstm))</span>
4. Cold bind validates complete dtype/span/capacity/alignment, liveness/current-thread access,
   writability and overlap, without reading values. Bound execution, after any length producer,
   validates the entire INT64 vector in [0,T] and all remaining access preconditions before ANY
   result representation write, even padding/zero-domain initialization. Invalid last row leaves
   sentinel bytes of every scan output unchanged and returns no public result; earlier producers
   need not roll back. Repairing lengths permits a fresh run of the same preparation.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [alignment glossary](../../../../glossary.md#term-alignment); [liveness glossary](../../../../glossary.md#term-liveness); [producer glossary](../../../../glossary.md#term-producer); [INT64 glossary](../../../../glossary.md#data-type--datatype); [vector glossary](../../../../glossary.md#term-vector); [padding glossary](../../../../glossary.md#padding))</span>
5. Use existing unchecked failures: invalid bounds throw `IllegalArgumentException` with
   row/value/T context; access errors follow existing CPU checks. Engine propagates through its
   existing cleanup convention; no new public recurrent exception/translation guarantee or type.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [Engine glossary](../../../../glossary.md#term-engine))</span>
6. Independent test-only equations (not production kernels) cover H>1 and non-diagonal hidden
   weights, every contributor once, correct packing, exactly one optional input-side bias,
   GRU reset AFTER candidate recurrent projection and <span data-glossary-navigation="true"> ([kernels glossary](../../../../glossary.md#kernel); [weights glossary](../../../../glossary.md#term-weight); [bias glossary](../../../../glossary.md#term-bias); [GRU glossary](../../../../glossary.md#term-gated-recurrent-unit-gru))</span>`n + z*(h-n)`, and LSTM explicit c/h updates.
   Qualify finite FLOAT32 arithmetic with a named test oracle/metric/threshold; test special classes,
   guards, signs and raw movement separately, without blanket final-state tolerance.<span data-glossary-navigation="true"> ([finite glossary](../../../../glossary.md#term-finite); [FLOAT32 glossary](../../../../glossary.md#term-float32); [test oracle glossary](../../../../glossary.md#term-test-oracle); [metric glossary](../../../../glossary.md#term-metric); [tolerance glossary](../../../../glossary.md#term-tolerance))</span>
7. Test lengths 0,1,T and mixed rows; T=0/B=0; FORWARD/REVERSE valid-prefix traversal at original
   time positions. Padded NaN/Inf inputs must not enter dot products/gates. Store exact +0 padding;
   copy raw initial h/c for L=0, including negative zero/NaN bits. Include active NaN/Inf/extreme
   activation cases and stable sign-guarded sigmoid; do not add a finite-input restriction.<span data-glossary-navigation="true"> ([NaN glossary](../../../../glossary.md#term-nan); [dot products glossary](../../../../glossary.md#linear-algebra-building-blocks); [negative zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [activation glossary](../../../../glossary.md#term-activation); [sigmoid glossary](../../../../glossary.md#term-sigmoid))</span>
8. Exercise heap/native/mixed canonical segments, sliced physical bases, read-only inputs and
   rejected dead/undersized/misaligned/overlapping representations. Test produced INT64 lengths
   (ordinary FLOAT32→INT64 cast), poisoned pre-producer storage, repeated/fresh lengths, two
   scan occurrences and a >8-node composed partition. Shared validation stays recurrence-blind.<span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [canonical glossary](../../../../glossary.md#term-canonical-form); [node glossary](../../../../glossary.md#node); [partition glossary](../../../../glossary.md#partition); [validation glossary](../../../../glossary.md#term-validation))</span>
9. Public compile/prepare/run/runValues cover mixed dtypes, port orders, finalCell-only roots and
   legal distinct-root aliased publications (or an explicitly inward duplicate-occurrence fixture).
   Preserve public rejection of identical Tensor roots/duplicate logical ports. Cover downstream
   consumers, isolated concurrent scratch/results, repeated runs with earlier results open, true
   PARAMETER weights (CP0011 save/load), floating flags/output OR and FORWARD_ONLY success. Keep
   static NN compute/train and Compiler rejection of every backward-capable scan request unchanged.<span data-glossary-navigation="true"> ([compile glossary](../../../../glossary.md#compile); [fixture glossary](../../../../glossary.md#term-fixture); [Tensor glossary](../../../../glossary.md#tensor); [PARAMETER glossary](../../../../glossary.md#term-parameter); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [Compiler glossary](../../../../glossary.md#term-compiler))</span>

## [Validation](../../../../glossary.md#term-validation)

Recorded [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26 worker commands (successful evidence reused at closure):

```bash
./gradlew :backends:cpu:test --tests '*CpuRecurrentScan*' --tests '*CpuCapabilityProviderTest' --tests '*CpuComposedValueValidationTest' --tests '*CpuPartitionDagResourceTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
```

Focused integration command:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuRecurrentScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest' :testing:integration-tests:test --tests '*EngineCpuRecurrentScanIntegrationTest' --tests '*FluentModelBuilderIntegrationTest'
```

Main's final accepted root command (JDK 26.0.1):

```bash
JAVA_HOME=/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home SYNAPTIK_METAL_TEST_LIBRARY=/Users/phujka/IdeaProjects/Synaptik/modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib ./gradlew test javadoc
```

Existing [Metal](../../../../glossary.md#term-metal) library [SHA-256](../../../../glossary.md#term-sha-256): `100d194c2273a657ad0d5d03bfe2280a0acea4f5d9f8260cee79e7cfa44562f3`,
unchanged. Closure runs static documentation checks only; no tests/builds are replayed.

## Documentation and review impact

Affected [CPU](../../../../glossary.md#term-cpu)/[Model](../../../../glossary.md#term-model) Javadocs, CPU [backend](../../../../glossary.md#backend) guide, `docs/architecture/runtime-prepare-backend-boundary.md`,
`docs/user-guide/running-models.md`, `docs/api/graph-checkpoint-api.md` and fixed-[scan](../../../../glossary.md#term-cumulative-scan) glossary status
now describe implemented execution. Independent code/docs review approved the final [snapshot](../../../../glossary.md#term-snapshot);
final [Javadoc](../../../../glossary.md#term-javadoc) generation passed. Existing glossary meanings, architecture, ADRs, public [API](../../../../glossary.md#term-application-programming-interface-api),
[persistence](../../../../glossary.md#term-persistence) [admission](../../../../glossary.md#term-admission) and [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) workflows are unchanged. Closure uses general and [planning](../../../../glossary.md#term-planning) profiles.

## Result

Accepted 2026-10-20: all nine criteria fulfilled for direct Java [scalar](../../../../glossary.md#term-scalar) [FLOAT32](../../../../glossary.md#term-float32) [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../../../glossary.md#term-long-short-term-memory-lstm),
all six [signatures](../../../../glossary.md#term-signature), both directions, output-flag OR, [canonical](../../../../glossary.md#term-canonical-form) [heap](../../../../glossary.md#term-heap)/native/mixed storage,
[producer](../../../../glossary.md#term-producer)-ordered [INT64](../../../../glossary.md#data-type--datatype) [preflight](../../../../glossary.md#term-preflight) before any [scan](../../../../glossary.md#term-cumulative-scan) write and isolated [RunState](../../../../glossary.md#run-state--runstate) scratch.
Changed areas: [CPU](../../../../glossary.md#term-cpu) [admission](../../../../glossary.md#term-admission)/[lowering](../../../../glossary.md#lowering)/preparation/execution, focused CPU/[conformance](../../../../glossary.md#term-conformance)/integration
tests, v2 ledger and the documentation/Javadocs listed above; three new internal production types.
Worker commands above: focused CPU 55 passed; full CPU 1,101 total/1,071 passed/30 existing opt-in
skips; conformance/ledger 7, [Engine](../../../../glossary.md#term-engine) 5 and Fluent 5 passed. No failures/errors.
After test-only rounded-FLOAT32 LSTM cell [oracle](../../../../glossary.md#term-test-oracle) and lowering/finalization [Javadoc](../../../../glossary.md#term-javadoc) corrections,
7 CPU execution + 1 conformance tests passed; final `:backends:cpu:javadoc :modules:model:javadoc`
passed (94 existing CPU warnings). Review reused executable evidence (419 links); closure static
checks passed: 346 local links/anchors, whitespace, status/[metadata](../../../../glossary.md#term-metadata), sizes and protected [hashes](../../../../glossary.md#term-hash).
Dewey final independent code/docs APPROVE, zero findings, on the unchanged 32-file [manifest](../../../../glossary.md#term-manifest)
`f8b00fbb3f32cd3d0be3e0fa78eded76d1cc40ddc74925304ccc62ec63ba33e1`.
Main's [JDK](../../../../glossary.md#term-java-development-kit-jdk) 26.0.1 root command used `SYNAPTIK_METAL_TEST_LIBRARY` at
`modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib`: exit 0 in 5m44s,
107 tasks, 35 executed/72 UP-TO-DATE; configuration [cache](../../../../glossary.md#term-cache) reused.
XML from current mapped project paths only: 4,328 tests/647 suites, 41 skips, no failures/errors;
includes reused UP-TO-DATE evidence. Skips: CPU 30, [Metal](../../../../glossary.md#term-metal) 10, integration 1.
No architecture/[API](../../../../glossary.md#term-application-programming-interface-api)/[schema](../../../../glossary.md#term-schema)/[ABI](../../../../glossary.md#term-application-binary-interface-abi)/[dependency](../../../../glossary.md#term-dependency) change; static [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) unchanged. [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt), Data/NN migration,
other types/[layouts](../../../../glossary.md#layout)/strategies and performance claims excluded. No unresolved issues or follow-up.
Authorized CPU frontier: None; protected unrelated files unchanged; no [commit](../../../../glossary.md#term-commit)/push.
Status: Complete
