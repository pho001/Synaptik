# Task CPU0028 Fixed recurrent scan forward execution

## Status

Complete — main accepted the independently approved implementation and root gate on 2026-10-20.

## Change class

Class C — new backend execution, pre-write value validation, per-run scratch and hot-loop invariants.
Clean implementation and independent code/docs review required.

## Goal

Execute each existing flat recurrent occurrence once per run through the ordinary CPU partition
lifecycle, using a reusable backend-private Java kernel and explicit run-owned state and outputs.

## Scope and first support set

Every admitted occurrence must satisfy all rows below; support is not inferred across types/routes.

| Dimension | Admitted set |
|---|---|
| Meaning | RNN_TANH, GRU_RESET_AFTER, LSTM; bias/no bias; FORWARD/REVERSE; exact Model port order. |
| Types and flags | FLOAT32 floating roles with arbitrary valid gradient flags; every output flag is their exact OR. Lengths INT64[B], `requiresGrad == false`. Flags do not request backward execution. |
| Geometry | Fully static `[T,B,I]`, states `[B,H]`, exact packed weights/bias and outputs; I/H positive, T/B may be zero. Each dimension, packed gate extent and each boundary element count fits `int`; all products/byte spans/workspace sizes checked. |
| Layout | Every descriptor equals `LayoutDescriptor.contiguous(shape)`, including zero offset; no unresolved, strided, broadcast or view layout. Segment slices may supply a different physical base with this same descriptor. |
| Storage | Exact accessible aligned segment access for canonical heap float[]/long[], native segments and mixtures, including read-only inputs; run-owned writable outputs/scratch. No carrier copy. |
| Strategy | One sequential SCALAR scan range, species bits zero, on the invoking thread; vector preference/parallel budgets do not promote the scan. Other units retain their existing strategies. |

FLOAT32 avoids new low rounding/FLOAT64 qualification. Other types/layouts, backward execution,
scan SIMD/parallel/native, fusion/compaction, tuning and speed claims are excluded. Scan-containing
partitions use existing direct decomposition, without complete-plan/copy enumeration. Discovery
reports incomplete/unsupported as for direct long plans; ordinary untuned preparation works.

## Non-goals

No shared executable edits, public wrapper/session, new hierarchy, Data/NN migration, training/BPTT,
wire format or dependency change.
No static per-step graph, callback/body/region, hidden state, workspace pool, generated scan code,
benchmark campaign or cleanup of unrelated CPU code.

## Contracts

- Root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants),
  [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules),
  [Explicit non-goals](../../../../../ARCHITECTURE.md#explicit-non-goals),
  [Testing requirements](../../../../../ARCHITECTURE.md#testing-requirements).
- Read the complete [recurrent-scan contract](../../../../architecture/contracts/recurrent-scan.md),
  including `Fixed recurrent scan without graph regions`, `Fixed family and current Model surface`,
  `Static Shape and runtime-value boundary`, `Valid lengths, traversal, and outputs`,
  `Purity and lifecycle ownership`, `Performance and migration invariants`.
- Backend [Concrete backend modules](../../../../architecture/contracts/backend-execution.md#concrete-backend-modules)
  and [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).
- Lifecycle [modules/runtime](../../../../architecture/contracts/runtime-prepare-engine.md#modulesruntime),
  [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle),
  [modules/engine](../../../../architecture/contracts/runtime-prepare-engine.md#modulesengine),
  [Public reusable prepared execution](../../../../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution).
- Model [modules/model](../../../../architecture/contracts/foundational-modules.md#modulesmodel),
  [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)
  (family formulas, primitive/aggregate/composite sites); Compiler
  [Compile lifecycle](../../../../architecture/contracts/compiler-autograd.md#compile-lifecycle) and
  recurrent contract's explicit BPTT rejection. Stop on an actual contract gap, not a hypothetical one.

## Dependencies and integration

- Depends on: NN0021A; Model0025E/0025F; Compiler0006A; CPU0027; Engine0021; CP0010/CP0011 — Complete.
- Conflicts with: CPU capability, unit facts/decomposition/finalization and shared scan/lifecycle
  documentation; CPU0023/other route work must not overlap these scopes.
- Parallel group: None.
- Common base revision: `e9b00bafddb36c28b12a9bb9fa5add3f19978547`, `main == origin/main`.
- Integration order: main Ready verification → clean CPU worker → independent review → main gate
  → NN0021B acceptance; no dependency on NN0021B completion.
- Integration validation: focused conformance/Engine gate and one main-owned root `test javadoc`.
- Shared-document integration owner: main; assign explanatory docs to one writer during review.

Launch verified 2026-10-20: user authorization, Ready/frontier, Complete predecessors/reviews,
recorded base, unchanged authority, inactive conflicts and all seven integration fields.
Closure preserves unrelated THIRD_PARTY_NOTICES.md and CpuOrdinaryPointwiseCastCleanJavaOracle.java;
no commit/push. Authorized CPU frontier after acceptance: None.

## Files and symbols

CPU paths are relative to `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.

- `CpuCapabilityProvider.supports`: exact recurrent admission before one-output defaults.
- `internal/lowering/CpuPartitionLowering.LoweredPartition`, `CpuPartitionDagDecomposer.Unit`,
  `internal/prepare/CpuPartitionPreparer.analyze/analyzeUnit/analyzeDag` and
  `CpuPartitionPreparationPlan.ExecutionUnitPlan/UnitRuntimeFacts`: mutually exclusive direct
  recurrent versus generated portable facts; existing generated consumers remain guarded.
- `internal/prepare/CpuPartitionFinalizer` and `internal/executable/CpuPreparedPartitionExecutable`:
  direct scan children after slot assignment, without generated artifacts or weaker access checks.
- Added `internal/lowering/CpuRecurrentScanLowering` (immutable `Facts`/`Geometry`),
  `internal/executable/CpuRecurrentScanPreparedExecutable` (private `Invocation`) and
  `internal/route/portable/CpuRecurrentScanKernel`: checked geometry/roles, binding/preflight and
  direct FLOAT32 family loops using existing Runtime, segment and workspace ownership.
- CPU tests: `CpuRecurrentScanLoweringTest`, `CpuRecurrentScanExecutionTest`, capability,
  composed validation, partition DAG, preparation/finalization and internal package inventory.
- Testing projects: `CpuRecurrentScanConformanceTest`, `EngineCpuRecurrentScanIntegrationTest`,
  v2 low-precision capability ledger/test; v1 unchanged. Shared executable seams remain read-only.

## Acceptance criteria

1. Provider and lowerer agree on the table, all six signatures and two directions, exact shapes,
   flags/layouts, checked overflow and malformed negatives. Prepare reads no length values and
   retains no prototype storage/length specialization; repeated lengths use the same preparation.
2. One immutable prepared scan kernel per occurrence, independent of T. Deduplicate repeated input
   ValueIds but retain ordered roles; preserve every sibling output, subset/reordered/aliased roots,
   fan-out and composition. Dense output allocation remains O(T*B*H), not constant memory.
3. Declare buffers and per-run gate/state scratch before slots. Derive checked exact scratch for
   the chosen row-wise loop (O(gates*H), independent of T and lengths); omit unused zero-domain
   scratch. No hidden prepared mutable state, per-step allocations, graph growth or hot maps,
   boxing, reflection, operation/backend lookup. Finish all recurrent projections from prior h
   before overwriting h; explicitly carry LSTM c. No output/scratch overlap with inputs or each other.
4. Cold bind validates complete dtype/span/capacity/alignment, liveness/current-thread access,
   writability and overlap, without reading values. Bound execution, after any length producer,
   validates the entire INT64 vector in [0,T] and all remaining access preconditions before ANY
   result representation write, even padding/zero-domain initialization. Invalid last row leaves
   sentinel bytes of every scan output unchanged and returns no public result; earlier producers
   need not roll back. Repairing lengths permits a fresh run of the same preparation.
5. Use existing unchecked failures: invalid bounds throw `IllegalArgumentException` with
   row/value/T context; access errors follow existing CPU checks. Engine propagates through its
   existing cleanup convention; no new public recurrent exception/translation guarantee or type.
6. Independent test-only equations (not production kernels) cover H>1 and non-diagonal hidden
   weights, every contributor once, correct packing, exactly one optional input-side bias,
   GRU reset AFTER candidate recurrent projection and `n + z*(h-n)`, and LSTM explicit c/h updates.
   Qualify finite FLOAT32 arithmetic with a named test oracle/metric/threshold; test special classes,
   guards, signs and raw movement separately, without blanket final-state tolerance.
7. Test lengths 0,1,T and mixed rows; T=0/B=0; FORWARD/REVERSE valid-prefix traversal at original
   time positions. Padded NaN/Inf inputs must not enter dot products/gates. Store exact +0 padding;
   copy raw initial h/c for L=0, including negative zero/NaN bits. Include active NaN/Inf/extreme
   activation cases and stable sign-guarded sigmoid; do not add a finite-input restriction.
8. Exercise heap/native/mixed canonical segments, sliced physical bases, read-only inputs and
   rejected dead/undersized/misaligned/overlapping representations. Test produced INT64 lengths
   (ordinary FLOAT32→INT64 cast), poisoned pre-producer storage, repeated/fresh lengths, two
   scan occurrences and a >8-node composed partition. Shared validation stays recurrence-blind.
9. Public compile/prepare/run/runValues cover mixed dtypes, port orders, finalCell-only roots and
   legal distinct-root aliased publications (or an explicitly inward duplicate-occurrence fixture).
   Preserve public rejection of identical Tensor roots/duplicate logical ports. Cover downstream
   consumers, isolated concurrent scratch/results, repeated runs with earlier results open, true
   PARAMETER weights (CP0011 save/load), floating flags/output OR and FORWARD_ONLY success. Keep
   static NN compute/train and Compiler rejection of every backward-capable scan request unchanged.

## Validation

Recorded JDK 26 worker commands (successful evidence reused at closure):

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

Existing Metal library SHA-256: `100d194c2273a657ad0d5d03bfe2280a0acea4f5d9f8260cee79e7cfa44562f3`,
unchanged. Closure runs static documentation checks only; no tests/builds are replayed.

## Documentation and review impact

Affected CPU/Model Javadocs, CPU backend guide, `docs/architecture/runtime-prepare-backend-boundary.md`,
`docs/user-guide/running-models.md`, `docs/api/graph-checkpoint-api.md` and fixed-scan glossary status
now describe implemented execution. Independent code/docs review approved the final snapshot;
final Javadoc generation passed. Existing glossary meanings, architecture, ADRs, public API,
persistence admission and NN workflows are unchanged. Closure uses general and planning profiles.

## Result

Accepted 2026-10-20: all nine criteria fulfilled for direct Java scalar FLOAT32 RNN/GRU/LSTM,
all six signatures, both directions, output-flag OR, canonical heap/native/mixed storage,
producer-ordered INT64 preflight before any scan write and isolated RunState scratch.
Changed areas: CPU admission/lowering/preparation/execution, focused CPU/conformance/integration
tests, v2 ledger and the documentation/Javadocs listed above; three new internal production types.
Worker commands above: focused CPU 55 passed; full CPU 1,101 total/1,071 passed/30 existing opt-in
skips; conformance/ledger 7, Engine 5 and Fluent 5 passed. No failures/errors.
After test-only rounded-FLOAT32 LSTM cell oracle and lowering/finalization Javadoc corrections,
7 CPU execution + 1 conformance tests passed; final `:backends:cpu:javadoc :modules:model:javadoc`
passed (94 existing CPU warnings). Review reused executable evidence (419 links); closure static
checks passed: 346 local links/anchors, whitespace, status/metadata, sizes and protected hashes.
Dewey final independent code/docs APPROVE, zero findings, on the unchanged 32-file manifest
`f8b00fbb3f32cd3d0be3e0fa78eded76d1cc40ddc74925304ccc62ec63ba33e1`.
Main's JDK 26.0.1 root command used `SYNAPTIK_METAL_TEST_LIBRARY` at
`modules/backend/metal/macos-arm64/build/libsynaptik_metal_foundation.dylib`: exit 0 in 5m44s,
107 tasks, 35 executed/72 UP-TO-DATE; configuration cache reused.
XML from current mapped project paths only: 4,328 tests/647 suites, 41 skips, no failures/errors;
includes reused UP-TO-DATE evidence. Skips: CPU 30, Metal 10, integration 1.
No architecture/API/schema/ABI/dependency change; static NN unchanged. BPTT, Data/NN migration,
other types/layouts/strategies and performance claims excluded. No unresolved issues or follow-up.
Authorized CPU frontier: None; protected unrelated files unchanged; no commit/push.
Status: Complete
