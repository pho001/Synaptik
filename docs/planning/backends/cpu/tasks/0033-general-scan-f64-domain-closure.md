# Task CPU0033: General-scan FLOAT64 domain closure

## Status

Complete — main accepted 2026-10-22; CPU frontier None. ImplementationAuthorized=true.

## Change class

Class C — typed generated hot loops, double-width resources and run binding; clean implementation
and independent targeted code/documentation review are mandatory before main acceptance.

## Goal

Complete homogeneous FLOAT64 (F64) forward and actual Compiler-generated first-order automatic
differentiation (AD) reverse
execution across the entire currently declared CPU scan algebra, not a pointwise-only slice.
Reuse the emitted time loop: one qualified body → exact prepared resources → run-owned bindings.

## Scope

Static T≥0, canonical boundaries plus existing proved all-zero-stride scalar reads; fixed body
shapes, at least one carry/sequence, optional invariants/emissions, both traversal directions.
Admit F64 tensor/scalar ADD/SUB/MUL/DIV, NEG/TANH/SIGMOID, rank-two MATMUL and `[1,0]` PERMUTE;
qualify actual reverse closure through SUM/sumToShape, right-aligned broadcast/EXPAND, compatible
RESHAPE, EXPAND_DIMS/SQUEEZE, CONTIGUOUS, closed constants and same-type identity casts.

## Non-goals

No BFLOAT16/FLOAT16, mixed dtype, arbitrary all-operation admission, wider boundary layouts,
nested/control-flow scans, lengths/padding, NN/training/persistence API widening, vector/parallel/
OpenBLAS scan route, tuning, new AD intermediate representation or execution owner. Preserve fixed
four-dtype/layout and static routes. Do not reopen or re-approve completed F32 tasks.

## Contracts

- [Root authority/index](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)
  and [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts).
- [Recurrent-scan — General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan), all subheadings: source/scopes, residual demand, exact primal bindings, traversal/empties and qualification gates.
- [Backend execution — CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes) and [Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity).
- [Foundational modules — Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries) and [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics), including arithmetic-site DAZ/FTZ, exact/discrete, primitive, aggregate and composite rules.
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation).
- [Runtime/Prepare — `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare), [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle) and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).

If a contract or required abstraction is missing/ambiguous, stop and request clarification.

## Dependencies and integration

- Depends on: Model0034/0035, Planning0008, Compiler0011, Prepare0010, CPU0030–0032; all Complete.
- Conflicts with: scan admission/lowering/IR/emission/binding/resources, generated compatibility,
  numerics/status/ledger and shared docs; serialized.
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` + dirty authority/
  Model0034/0035, Planning0008, Compiler0011, Prepare0010, CPU0030–0032; accepted, uncommitted, preserved.
- Integration order: serial clean implementation → independent Class C code/docs review and
  corrections → main checkpoint → main acceptance; completed serially, no successor launch.
- Integration validation: `CPU0033-F64-DOMAIN-JOIN` below; main's source-matched native-enabled
  root test/Javadoc at F64/domain closure.
- Shared-document integration owner: main.

User “ok, pust se do prace”; 2026-10-22 launch verified Ready/deps/reviews/base/index/dirty-source/conflicts.

## Files and symbols

Production root: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.
Change only needed existing seams below; no new top-level production type/package or F64 executor.

- `CpuCapabilityProvider`; `internal/lowering/CpuGeneralScanLowering` (type, span/stride bytes,
  scratch/liveness and slot/method budgets); `internal/ir/CpuGeneralScanIr` (typed boundary/stages,
  canonical versus baked identity). Carry dtype through existing facts rather than duplicating them.
- `internal/codegen/emit/{CpuGeneralScanEmitter,CpuGeneralScanReductionEmitter,CpuGeneralScanMatmulEmitter}`;
  reuse `CpuScalarEmitter`, `CpuMatmulEmitter`, `CpuCarrierEmitter`, `CpuClassFileKernelGenerator`.
- `internal/executable/CpuGeneralScanPreparedExecutable`; `internal/prepare/{CpuPartitionPreparer,
  CpuPartitionPreparationPlan,CpuPartitionFinalizer}`; `internal/cache/{CpuKernelSpecialization,CpuGeneratorSchema}`.
  Review existing memory/binding recipes; preserve segment-only scan entry and ownership.
- Read-only Compiler `ScanGradientRules`/`LinearAlgebraGradientRules`: ordinary derivative rules
  already cover F64; no Compiler production change expected. Stop if closure requires new algebra.
- CPU tests use the matching `src/test/java` packages: `CpuGeneralScanCapabilityTest`, lowering/
  `CpuGeneralScanLoweringTest`, prepare/`CpuGeneralScanPreparationTest`, executable/
  `CpuGeneralScanExecutionTest`, emit/`CpuGeneralScanGeneratedKernelTest`,
  `CpuGeneralScanCleanJavaOracle`, `CpuGeneralScanTimingTest`, cache/`CpuKernelSpecializationTest`;
  retain ordinary typed/carrier/fixed regressions.
- Main owns backend-conformance `testing.conformance.{CpuGeneralScanConformanceTest,
  LowPrecisionCapabilityLedgerTest}` and its `src/test/resources/low-precision-capability-ledger-v2.tsv`,
  plus integration-tests `testing.integration.EngineCpuGeneralScanIntegrationTest` (package prefix
  `io.github.pho001.synaptik.`); use existing architecture suites below.

## Acceptance criteria

1. Provider, lowering and binder independently qualify homogeneous F64 across the scoped algebra
   and each actual reverse body. Reject missing/malformed attachments, unsupported dead/T=0 body
   occurrences, mixed/low types, noncanonical unproved boundaries and supported children with an
   unsupported binder; no forward-answer shortcut or capability inferred from carriers.
2. Arithmetic, scalar-attribute raw bits, temporaries and SUM/MATMUL accumulators remain double;
   no F32 narrowing. Generated dload/dstore and two-slot JVM locals verify; preflight conservatively
   budgets entry parameters, double/long locals, nested loops and method bytes before generation.
3. Effective addresses/strides/spans, history/output frames, old/new carry banks and reusable cuts
   use eight-byte elements/alignment. Declare checked exact individual AND aggregate resources
   before slots; test overflow where four-byte geometry fits but eight-byte geometry does not,
   including affine stride products and stacked histories. Fail before allocation/generation/write.
4. Compare actual Compiler AD to ordinary unrolled Tensor AD and represented direct-double sites:
   T≥3, both directions/optimization settings, non-gradient initial carry with trainable sequence/
   parameters, differently shaped carries/parameter accumulators, missing and joint final/emission
   seeds, tied roles/roots, reused templates and cross-occurrence sums. Test every scoped form.
5. Non-square rank-two products and both operand transpose views retain both matrix inputs through
   full K and cross-coordinate consumers. Prove old/new carry lifetime, transitive forwarding,
   saved-frame backing, disjoint inclusive cut reuse and necessary invariant-product hoisting only
   after the nonempty-time guard; no overwritten operand, stale bank or needless product copy.
6. Demanded previous-carry and produced TANH/SIGMOID histories retain exact stored values through
   views/copies/identity casts and later derivatives; save prior carry before its step, publish
   distinct canonical wrappers and map emissions/histories to ORIGINAL time in either direction.
7. F64 arithmetic and generated gradients never apply denormals-are-zero/flush-to-zero (DAZ/FTZ).
   Probe signed subnormal inputs and representable subnormal results in primitive, SUM and MATMUL
   sites; distinguish legitimate mathematical/rounded underflow to zero. Check NaN class, infinity,
   zero-divisor, overflow and family zero signs independently of finite tolerances; arithmetic NaN
   payload/sign are unspecified, while raw movement/saved-site copies preserve actual stored bits.
8. T=0 preserves raw initial carry and dFinalCarry seed as dInitialCarry, empty emissions/histories,
   exact typed-zero invariant sums. Identity/no-axis SUM and no-op paths preserve raw subnormals,
   signed zeros and NaN payloads. Zero K publishes nonempty positive-zero products; zero M/empty
   `[0,H]` steps never skip nonempty `[H]` reductions, parameter sums or another nonempty carry.
9. Existing heap/native/mixed representations bind without new public carrier capability. Check
   complete capacities/tail overlap, eight-byte alignment, access/writability and legal read/read
   aliases before writes; malformed bindings leave outputs unchanged. Prove cleanup, retry, fresh
   inputs, retained open results and concurrent-run isolation; no shared mutable bound state.
10. Source/captured/prepared objects and code stay O(body) for T=0/3/large T; runtime has no graph,
    per-op lookup/dispatch/callback, boxing or per-step allocation. Inference dead histories,
    including reverse residuals, have ZERO bytes/stores, not zero total demanded output storage.
11. Preserve accepted F32 pointwise/shaped/MATMUL keys AND generated bytes. Retain envelope/family
    71, shaped-v1/MATMUL-v1 projections and `scan-shaped-v1`/`scan-matmul-v1` payloads; add typed F64
    specializations, not an incompatible global schema. Key every baked dtype/attribute/geometry/
    resource fact in both existing specialization streams; canonical IR stays geometry-free.
    Exclude compatible T, slots and run addresses; reject stale/mismatched F64 artifacts, dtype,
    width or binding facts. Document append-only facts; stop if old compatibility cannot survive.
12. Extend the optimal specialized clean Java oracle with direct-double forward/actual-reverse
    pointwise, shaped and MATMUL cases. Automated Class-File inspection verifies double loops,
    matched dataflow and no hidden arithmetic helpers/dispatch/allocation/copies. Short warmed
    matched-work measurements use identical dtype/carrier/layout/numerics and comparable validation;
    separate cold generation/load/JIT, kernel and entry/preflight end-to-end costs. Fix or explicitly
    justify diagnosed avoidable overhead; no five-fork/C2 campaign or universal parity claim.

## Validation

Worker focused tests:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScan*Test' --tests '*CpuKernelSpecializationTest'
```

Bounded Float64 diagnostic; existing switch, retained evidence; no old F32 campaign:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScanTimingTest.*Float64*' -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.evidenceRoot=build/cpu0033-f64-evidence
```

Worker module/Javadoc once: `./gradlew :backends:cpu:test :backends:cpu:javadoc`.
Main's reviewed `CPU0033-F64-DOMAIN-JOIN` reuses worker evidence:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'
./gradlew :testing:architecture-tests:test --tests '*RuntimeDependencyAndHotPathContractTest' --tests '*BackendLayoutContractTest'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/modules/backend/metal/macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew test javadoc
```

Native source/package authentication precedes root gate for binding/RunState/hot-path isolation.
Reuse UP-TO-DATE evidence; report executed/reused/skipped counts, optional observers and warnings/limits;
No root before F64/domain closure; no blanket reruns.

## Documentation and review impact

Root/owner status, CPU/scan guides, tensor/compile APIs, Javadocs/glossary finalized; no new semantics,
term/ADR/dependency. Main owns shared integration; independent Class C code/docs APPROVE and final
Javadoc PASS. Architecture/dependency decisions still require stop/approval and owning tests.

## Result

- Full F64 algebra/actual reverse, double/eight-byte/raw/empty/isolation; F32 keys/bytes preserved,
  schema/family 71; no new production type/owner/ABI.
- Changed: 25 code/test/ledger paths + identity test; docs 16 MD/11 comment-only Java + two inline
  field comments (executable/literal/Unicode-neutral). Code/identity APPROVE, 0 blockers; docs Complete.
- CPU test/Javadoc PASS 5m14s: 1193 cases, 1156 pass/37 opt-in skips/0 fail; 4 tasks ran/19 reused.
- Final Javadoc-only PASS 3s: 2 ran/9 reused, 94 baseline warnings (92 constructor/2 incubator), 0 new.
- Main conformance 14/14, Engine 17/17, architecture 4/4 PASS.
- Initial native root FAILED 26s: stale identity expected 70, active pre-F64 schema already 71;
  test-only correction (frozen 68 distinguished), independent APPROVE + focused identity 1/1 PASS.
- Final same native root PASS 2s: 119 tasks, 1 ran/118 reused; CPU/Javadoc UP-TO-DATE.
- Active-project XML: 680 suites/4724 cases, 4676 pass/48 skip/0 failure/error; 59 new conformance
  cases pass, 4665 reused cases (4617 pass/48 skip); inactive extensions/{training,checkpoint} excluded.
- Skips: 37 CPU opt-in, 10 Metal observers (no SYNAPTIK_METAL_TEST_OBSERVER), 1 host-tuning fixture.
- Native: 16 stable sources, fresh signed/verified package; final root source hashes unchanged.
- Base above unchanged, index empty/256 dirty paths before closure; accepted work preserved.
- R2 shaped-forward gap UNEXPLAINED: ~28–29us generated vs 0.32–0.35us Java; reverse nonstationary.
- No oracle-work mismatch/concrete avoidable extra work found; no JIT-cause/parity claim or extra campaign.
- No public end-to-end claim; runtime native-access/incubator warnings.
- Evidence: /private/tmp/synaptik-cpu0033-implementation.woFkCA/{main-root-native-final.log,root-xml-final.json}.
- Main accepted timing limit; CPU frontier None, low-level join Complete; no successor.
- NN/padding/persistence/low/vector/parallel/OpenBLAS separately gated Draft; no required follow-up.
- Status: Complete
