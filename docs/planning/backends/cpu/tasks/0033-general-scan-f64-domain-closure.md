# Task CPU0033: General-[scan](../../../../glossary.md#term-cumulative-scan) [FLOAT64](../../../../glossary.md#term-float64) domain closure

## Reader context — historical integration boundary

FLOAT64 closure here means support for the whole then-declared general-Scan algebra and the actual
Compiler-produced first-order reverse, not just retyping a pointwise kernel. Eight-byte values
affect addressing, resource sizes and JVM local slots as well as arithmetic. CPU derives those facts
before shared assignment; Runtime receives only the typed recipe and per-run resources. ‘Closure’
does not prove every mathematically valid body or derivative fits the independent budgets. The
retained timing gap and limits below remain unresolved historical observations, not universal
current performance claims or permission to weaken the oracle.

Current reading: [Current general Scan composition](../../../../backend-guide/cpu-backend.md#current-general-scan-composition); [Retained static Scan resources](../../../../backend-guide/cpu-backend.md#retained-static-scan-qualification-and-resources).

## Status

Complete — main accepted 2026-10-22; [CPU](../../../../glossary.md#term-cpu) frontier None. ImplementationAuthorized=true.

## Change class

Class C — typed generated hot loops, double-width resources and [run](../../../../glossary.md#term-run) [binding](../../../../glossary.md#term-binding); clean implementation
and independent targeted code/documentation review are mandatory before main acceptance.

## Goal

Complete [homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT64](../../../../glossary.md#term-float64) ([F64](../../../../glossary.md#floating-point-precision-and-rounding)) forward and actual [Compiler](../../../../glossary.md#term-compiler)-generated first-order [automatic
differentiation](../../../../glossary.md#autograd) ([AD](../../../../glossary.md#autograd)) reverse
execution across the entire currently declared [CPU](../../../../glossary.md#term-cpu) [scan](../../../../glossary.md#term-cumulative-scan) algebra, not a [pointwise](../../../../glossary.md#term-pointwise)-only [slice](../../../../glossary.md#slice).
Reuse the emitted time loop: one qualified body → exact prepared resources → [run](../../../../glossary.md#term-run)-owned [bindings](../../../../glossary.md#term-binding).

## [Scope](../../../../glossary.md#term-scope)

Static T≥0, [canonical](../../../../glossary.md#term-canonical-form) boundaries plus existing proved all-zero-[stride](../../../../glossary.md#term-stride) [scalar](../../../../glossary.md#term-scalar) reads; fixed body
[shapes](../../../../glossary.md#shape), at least one [carry](../../../../glossary.md#term-carry)/sequence, optional invariants/emissions, both traversal directions.
Admit [F64](../../../../glossary.md#floating-point-precision-and-rounding) [tensor](../../../../glossary.md#tensor)/scalar ADD/SUB/MUL/DIV, NEG/[TANH](../../../../glossary.md#term-tanh)/[SIGMOID](../../../../glossary.md#term-sigmoid), [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) and `[1,0]` PERMUTE;
qualify actual reverse closure through SUM/sumToShape, right-aligned broadcast/EXPAND, compatible
RESHAPE, EXPAND_DIMS/SQUEEZE, [CONTIGUOUS](../../../../glossary.md#term-contiguous), closed [constants](../../../../glossary.md#term-constant) and same-type [identity](../../../../glossary.md#term-identity-meanings) casts.

## Non-goals

No [BFLOAT16](../../../../glossary.md#term-bfloat16)/[FLOAT16](../../../../glossary.md#term-float16), mixed [dtype](../../../../glossary.md#data-type--datatype), arbitrary all-[operation](../../../../glossary.md#operation) [admission](../../../../glossary.md#term-admission), wider boundary [layouts](../../../../glossary.md#layout),
nested/control-flow scans, lengths/[padding](../../../../glossary.md#padding), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/training/[persistence](../../../../glossary.md#term-persistence) [API](../../../../glossary.md#term-application-programming-interface-api) [widening](../../../../glossary.md#term-widening), [vector](../../../../glossary.md#term-vector)/parallel/
[OpenBLAS](../../../../glossary.md#term-openblas) [scan](../../../../glossary.md#term-cumulative-scan) route, tuning, new [AD](../../../../glossary.md#autograd) [intermediate representation](../../../../glossary.md#term-intermediate-representation-ir) or execution [owner](../../../../glossary.md#term-owner). Preserve fixed
four-dtype/layout and static routes. Do not reopen or re-approve completed [F32](../../../../glossary.md#floating-point-precision-and-rounding) tasks.

## Contracts

- [Root authority/index](../../../../../ARCHITECTURE.md#authority-incorporation-and-precedence),
  [Core invariants](../../../../../ARCHITECTURE.md#core-invariants), [Dependency rules](../../../../../ARCHITECTURE.md#dependency-rules)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
  and [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>.
- [Recurrent-scan — General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>, all subheadings: source/scopes, [residual](../../../../glossary.md#scan-residual-port-and-history-demand) demand, exact [primal](../../../../glossary.md#term-primal) [bindings](../../../../glossary.md#term-binding), traversal/empties and [qualification](../../../../glossary.md#term-qualification) gates.
- [Backend execution — CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend); [CPU glossary](../../../../glossary.md#term-cpu))</span> and [Profile-free backend capability and identity](../../../../architecture/contracts/backend-execution.md#profile-free-backend-capability-and-identity)<span data-glossary-navigation="true"> ([backend capability glossary](../../../../glossary.md#backend-capability); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>.
- [Foundational modules — Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span> and [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../../../../glossary.md#profile-free-numerical-semantics))</span>, including arithmetic-site [DAZ](../../../../glossary.md#term-daz)/[FTZ](../../../../glossary.md#term-ftz), exact/discrete, primitive, aggregate and composite rules.
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>.
- [Runtime/Prepare — `modules/prepare`](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare))</span>, [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span> and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>.

If a contract or required abstraction is missing/ambiguous, stop and request clarification.

## Dependencies and integration

- Depends on: Model0034/0035, Planning0008, Compiler0011, Prepare0010, CPU0030–0032; all Complete.
- Conflicts with: scan admission/lowering/IR/emission/binding/resources, generated compatibility,
  numerics/status/ledger and shared docs; serialized.<span data-glossary-navigation="true"> ([scan glossary](../../../../glossary.md#term-cumulative-scan); [admission glossary](../../../../glossary.md#term-admission); [lowering glossary](../../../../glossary.md#lowering); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [binding glossary](../../../../glossary.md#term-binding))</span>
- Parallel group: None.
- Common base revision: HEAD=origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` + dirty authority/
  Model0034/0035, Planning0008, Compiler0011, Prepare0010, CPU0030–0032; accepted, uncommitted, preserved.
- Integration order: serial clean implementation → independent Class C code/docs review and
  corrections → main checkpoint → main acceptance; completed serially, no successor launch.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Integration [validation](../../../../glossary.md#term-validation): `CPU0033-F64-DOMAIN-JOIN` below; main's source-matched native-enabled
  root test/Javadoc at F64/domain closure.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- Shared-document integration [owner](../../../../glossary.md#term-owner): main.

User “ok, pust se do prace”; 2026-10-22 launch verified [Ready](../../../../glossary.md#term-ready)/deps/reviews/base/index/dirty-source/conflicts.

## Files and symbols

Production root: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.
Change only needed existing seams below; no new top-level production type/package or [F64](../../../../glossary.md#floating-point-precision-and-rounding) executor.

- `CpuCapabilityProvider`; `internal/lowering/CpuGeneralScanLowering` (type, span/stride bytes,
  scratch/liveness and slot/method budgets); <span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [liveness glossary](../../../../glossary.md#term-liveness))</span>`internal/ir/CpuGeneralScanIr` (typed boundary/stages,
  canonical versus baked identity). Carry dtype through existing facts rather than duplicating them.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings); [Carry glossary](../../../../glossary.md#term-carry); [dtype glossary](../../../../glossary.md#data-type--datatype))</span>
- `internal/codegen/emit/{CpuGeneralScanEmitter,CpuGeneralScanReductionEmitter,CpuGeneralScanMatmulEmitter}`;
  reuse `CpuScalarEmitter`, `CpuMatmulEmitter`, `CpuCarrierEmitter`, `CpuClassFileKernelGenerator`.
- `internal/executable/CpuGeneralScanPreparedExecutable`; `internal/prepare/{CpuPartitionPreparer,
  CpuPartitionPreparationPlan,CpuPartitionFinalizer}`; `internal/cache/{CpuKernelSpecialization,CpuGeneratorSchema}`.
  Review existing memory/binding recipes; preserve segment-only scan entry and ownership.<span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [scan glossary](../../../../glossary.md#term-cumulative-scan); [ownership glossary](../../../../glossary.md#term-ownership-meanings))</span>
- Read-only [Compiler](../../../../glossary.md#term-compiler) `ScanGradientRules`/`LinearAlgebraGradientRules`: ordinary derivative rules
  already cover F64; no Compiler production change expected. Stop if closure requires new algebra.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative))</span>
- [CPU](../../../../glossary.md#term-cpu) tests use the matching `src/test/java` packages: `CpuGeneralScanCapabilityTest`, [lowering](../../../../glossary.md#lowering)/
  `CpuGeneralScanLoweringTest`, [prepare](../../../../glossary.md#prepare)/`CpuGeneralScanPreparationTest`, executable/
  `CpuGeneralScanExecutionTest`, emit/`CpuGeneralScanGeneratedKernelTest`,
  `CpuGeneralScanCleanJavaOracle`, `CpuGeneralScanTimingTest`, [cache](../../../../glossary.md#term-cache)/`CpuKernelSpecializationTest`;
  retain ordinary typed/carrier/fixed regressions.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier))</span>
- Main owns [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) `testing.conformance.{CpuGeneralScanConformanceTest,
  LowPrecisionCapabilityLedgerTest}` and its `src/test/resources/low-precision-capability-ledger-v2.tsv`,
  plus integration-tests `testing.integration.EngineCpuGeneralScanIntegrationTest` (package prefix
  `io.github.pho001.synaptik.`); use existing architecture suites below.

## Acceptance criteria

1. Provider, lowering and binder independently qualify homogeneous F64 across the scoped algebra
   and each actual reverse body. Reject missing/malformed attachments, unsupported dead/T=0 body
   occurrences, mixed/low types, noncanonical unproved boundaries and supported children with an
   unsupported binder; no forward-answer shortcut or capability inferred from carriers.<span data-glossary-navigation="true"> ([lowering glossary](../../../../glossary.md#lowering); [homogeneous glossary](../../../../glossary.md#term-homogeneous); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
2. Arithmetic, scalar-attribute raw bits, temporaries and SUM/MATMUL accumulators remain double;
   no F32 narrowing. Generated dload/dstore and two-slot JVM locals verify; preflight conservatively
   budgets entry parameters, double/long locals, nested loops and method bytes before generation.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [narrowing glossary](../../../../glossary.md#term-narrowing); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [preflight glossary](../../../../glossary.md#term-preflight))</span>
3. Effective addresses/strides/spans, history/output frames, old/new carry banks and reusable cuts
   use eight-byte elements/alignment. Declare checked exact individual AND aggregate resources
   before slots; test overflow where four-byte geometry fits but eight-byte geometry does not,
   including affine stride products and stacked histories. Fail before allocation/generation/write.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [elements glossary](../../../../glossary.md#term-element); [alignment glossary](../../../../glossary.md#term-alignment); [overflow glossary](../../../../glossary.md#term-overflow))</span>
4. Compare actual Compiler AD to ordinary unrolled Tensor AD and represented direct-double sites:
   T≥3, both directions/optimization settings, non-gradient initial carry with trainable sequence/
   parameters, differently shaped carries/parameter accumulators, missing and joint final/emission
   seeds, tied roles/roots, reused templates and cross-occurrence sums. Test every scoped form.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [AD glossary](../../../../glossary.md#autograd); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [Tensor glossary](../../../../glossary.md#tensor); [gradient glossary](../../../../glossary.md#term-gradient); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
5. Non-square rank-two products and both operand transpose views retain both matrix inputs through
   full K and cross-coordinate consumers. Prove old/new carry lifetime, transitive forwarding,
   saved-frame backing, disjoint inclusive cut reuse and necessary invariant-product hoisting only
   after the nonempty-time guard; no overwritten operand, stale bank or needless product copy.<span data-glossary-navigation="true"> ([rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [transpose glossary](../../../../glossary.md#term-transpose); [views glossary](../../../../glossary.md#term-view-meanings); [matrix glossary](../../../../glossary.md#term-matrix-meanings); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [lifetime glossary](../../../../glossary.md#term-lifetime); [invariant glossary](../../../../glossary.md#term-invariant); [guard glossary](../../../../glossary.md#term-guard))</span>
6. Demanded previous-carry and produced TANH/SIGMOID histories retain exact stored values through
   views/copies/identity casts and later derivatives; save prior carry before its step, publish
   distinct canonical wrappers and map emissions/histories to ORIGINAL time in either direction.<span data-glossary-navigation="true"> ([TANH glossary](../../../../glossary.md#term-tanh); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [identity glossary](../../../../glossary.md#term-identity-meanings); [derivatives glossary](../../../../glossary.md#term-derivative); [canonical glossary](../../../../glossary.md#term-canonical-form))</span>
7. F64 arithmetic and generated gradients never apply denormals-are-zero/flush-to-zero (DAZ/FTZ).
   Probe signed subnormal inputs and representable subnormal results in primitive, SUM and MATMUL
   sites; distinguish legitimate mathematical/rounded underflow to zero. Check NaN class, infinity,
   zero-divisor, overflow and family zero signs independently of finite tolerances; arithmetic NaN
   payload/sign are unspecified, while raw movement/saved-site copies preserve actual stored bits.<span data-glossary-navigation="true"> ([denormals-are-zero glossary](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow); [flush-to-zero glossary](../../../../glossary.md#subnormal-daz-ftz-overflow-and-underflow); [DAZ glossary](../../../../glossary.md#term-daz); [FTZ glossary](../../../../glossary.md#term-ftz); [subnormal glossary](../../../../glossary.md#term-subnormal); [underflow glossary](../../../../glossary.md#term-underflow); [NaN glossary](../../../../glossary.md#term-nan); [infinity glossary](../../../../glossary.md#term-infinity); [finite glossary](../../../../glossary.md#term-finite); [NaN payload glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>
8. T=0 preserves raw initial carry and dFinalCarry seed as dInitialCarry, empty emissions/histories,
   exact typed-zero invariant sums. Identity/no-axis SUM and no-op paths preserve raw subnormals,
   signed zeros and NaN payloads. Zero K publishes nonempty positive-zero products; zero M/empty<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [axis glossary](../../../../glossary.md#term-axis))</span>
   `[0,H]` steps never skip nonempty `[H]` reductions, [parameter](../../../../glossary.md#term-parameter) sums or another nonempty [carry](../../../../glossary.md#term-carry).
9. Existing heap/native/mixed representations bind without new public carrier capability. Check
   complete capacities/tail overlap, eight-byte alignment, access/writability and legal read/read
   aliases before writes; malformed bindings leave outputs unchanged. Prove cleanup, retry, fresh
   inputs, retained open results and concurrent-run isolation; no shared mutable bound state.<span data-glossary-navigation="true"> ([byte alignment glossary](../../../../glossary.md#term-byte-alignment))</span><span data-glossary-navigation="true"> ([heap glossary](../../../../glossary.md#term-heap); [carrier glossary](../../../../glossary.md#term-storage-carrier); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [bindings glossary](../../../../glossary.md#term-binding); [run glossary](../../../../glossary.md#term-run); [mutable glossary](../../../../glossary.md#term-mutable); [bound state glossary](../../../../glossary.md#term-bound-state))</span>
10. Source/captured/prepared objects and code stay O(body) for T=0/3/large T; runtime has no graph,
    per-op lookup/dispatch/callback, boxing or per-step allocation. Inference dead histories,
    including reverse residuals, have ZERO bytes/stores, not zero total demanded output storage.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [graph glossary](../../../../glossary.md#graph); [lookup glossary](../../../../glossary.md#term-lookup); [dispatch glossary](../../../../glossary.md#term-dispatch); [callback glossary](../../../../glossary.md#term-callback); [Inference glossary](../../../../glossary.md#term-inference))</span>
11. Preserve accepted F32 pointwise/shaped/MATMUL keys AND generated bytes. Retain envelope/family
    71, shaped-v1/MATMUL-v1 projections and <span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise))</span>`scan-shaped-v1`/`scan-matmul-v1` payloads; add typed F64
    specializations, not an incompatible global schema. Key every baked dtype/attribute/geometry/
    resource fact in both existing specialization streams; canonical IR stays geometry-free.
    Exclude compatible T, slots and run addresses; reject stale/mismatched F64 artifacts, dtype,
    width or binding facts. Document append-only facts; stop if old compatibility cannot survive.<span data-glossary-navigation="true"> ([schema glossary](../../../../glossary.md#term-schema); [dtype glossary](../../../../glossary.md#data-type--datatype); [specialization glossary](../../../../glossary.md#term-specialization); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [artifacts glossary](../../../../glossary.md#term-artifact))</span>
12. Extend the optimal specialized clean Java oracle with direct-double forward/actual-reverse
    pointwise, shaped and MATMUL cases. Automated Class-File inspection verifies double loops,
    matched dataflow and no hidden arithmetic helpers/dispatch/allocation/copies. Short warmed
    matched-work measurements use identical dtype/carrier/layout/numerics and comparable validation;
    separate cold generation/load/JIT, kernel and entry/preflight end-to-end costs. Fix or explicitly
    justify diagnosed avoidable overhead; no five-fork/C2 campaign or universal parity claim.<span data-glossary-navigation="true"> ([oracle glossary](../../../../glossary.md#term-test-oracle); [dataflow glossary](../../../../glossary.md#term-dataflow); [layout glossary](../../../../glossary.md#layout); [validation glossary](../../../../glossary.md#term-validation); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit); [kernel glossary](../../../../glossary.md#kernel); [fork glossary](../../../../glossary.md#term-fork-meanings); [C2 glossary](../../../../glossary.md#term-c2))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker focused tests:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScan*Test' --tests '*CpuKernelSpecializationTest'
```

Bounded Float64 diagnostic; existing switch, retained evidence; no old [F32](../../../../glossary.md#floating-point-precision-and-rounding) campaign:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScanTimingTest.*Float64*' -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.evidenceRoot=build/cpu0033-f64-evidence
```

Worker [module](../../../../glossary.md#term-module)/[Javadoc](../../../../glossary.md#term-javadoc) once: `./gradlew :backends:cpu:test :backends:cpu:javadoc`.
Main's reviewed `CPU0033-F64-DOMAIN-JOIN` reuses worker evidence:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest'
./gradlew :testing:architecture-tests:test --tests '*RuntimeDependencyAndHotPathContractTest' --tests '*BackendLayoutContractTest'
SYNAPTIK_METAL_TEST_LIBRARY="$PWD/modules/backend/metal/macos-arm64/build/package-v1/macos-arm64/libsynaptik_metal_foundation.dylib" ./gradlew test javadoc
```

Native source/package [authentication](../../../../glossary.md#term-authentication) precedes root gate for [binding](../../../../glossary.md#term-binding)/[RunState](../../../../glossary.md#run-state--runstate)/hot-path isolation.
Reuse UP-TO-DATE evidence; report executed/reused/skipped counts, optional observers and warnings/limits;
No root before [F64](../../../../glossary.md#floating-point-precision-and-rounding)/domain closure; no blanket reruns.

## Documentation and review impact

Root/[owner](../../../../glossary.md#term-owner) status, [CPU](../../../../glossary.md#term-cpu)/[scan](../../../../glossary.md#term-cumulative-scan) guides, [tensor](../../../../glossary.md#tensor)/[compile](../../../../glossary.md#compile) APIs, Javadocs/glossary finalized; no new semantics,
term/[ADR](../../../../glossary.md#term-architecture-decision-record-adr)/[dependency](../../../../glossary.md#term-dependency). Main owns shared integration; independent Class C code/docs APPROVE and final
[Javadoc](../../../../glossary.md#term-javadoc) PASS. Architecture/dependency decisions still require stop/approval and owning tests.

## Result

- Full F64 algebra/actual reverse, double/eight-byte/raw/empty/isolation; F32 keys/bytes preserved,
  schema/family 71; no new production type/owner/ABI.<span data-glossary-navigation="true"> ([F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [schema glossary](../../../../glossary.md#term-schema); [owner glossary](../../../../glossary.md#term-owner); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi))</span>
- Changed: 25 code/test/ledger paths + identity test; docs 16 MD/11 comment-only Java + two inline
  field comments (executable/literal/Unicode-neutral). Code/identity APPROVE, 0 blockers; docs Complete.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [Unicode glossary](../../../../glossary.md#term-unicode))</span>
- [CPU](../../../../glossary.md#term-cpu) test/[Javadoc](../../../../glossary.md#term-javadoc) PASS 5m14s: 1193 cases, 1156 pass/37 opt-in skips/0 fail; 4 tasks ran/19 reused.
- Final Javadoc-only PASS 3s: 2 ran/9 reused, 94 baseline warnings (92 constructor/2 incubator), 0 new.
- Main [conformance](../../../../glossary.md#term-conformance) 14/14, [Engine](../../../../glossary.md#term-engine) 17/17, architecture 4/4 PASS.
- Initial native root FAILED 26s: stale identity expected 70, active pre-F64 schema already 71;
  test-only correction (frozen 68 distinguished), independent APPROVE + focused identity 1/1 PASS.
- Final same native root PASS 2s: 119 tasks, 1 ran/118 reused; CPU/Javadoc UP-TO-DATE.
- Active-project XML: 680 suites/4724 cases, 4676 pass/48 skip/0 failure/error; 59 new conformance
  cases pass, 4665 reused cases (4617 pass/48 skip); inactive extensions/{training,checkpoint} excluded.<span data-glossary-navigation="true"> ([checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Skips: 37 CPU opt-in, 10 [Metal](../../../../glossary.md#term-metal) observers (no SYNAPTIK_METAL_TEST_OBSERVER), 1 [host](../../../../glossary.md#term-host)-tuning [fixture](../../../../glossary.md#term-fixture).
- Native: 16 stable sources, fresh signed/verified package; final root source [hashes](../../../../glossary.md#term-hash) unchanged.
- Base above unchanged, index empty/256 dirty paths before closure; accepted work preserved.
- R2 shaped-forward gap UNEXPLAINED: ~28–29us generated vs 0.32–0.35us Java; reverse [nonstationary](../../../../glossary.md#term-nonstationary).
- No [oracle](../../../../glossary.md#term-test-oracle)-work mismatch/concrete avoidable extra work found; no [JIT](../../../../glossary.md#term-just-in-time-compilation-jit)-cause/parity claim or extra campaign.
- No public end-to-end claim; [runtime](../../../../glossary.md#term-runtime) native-access/incubator warnings.
- Evidence: /private/tmp/synaptik-cpu0033-implementation.woFkCA/{main-root-native-final.[log](../../../../glossary.md#term-log),root-xml-final.json}.
- Main accepted timing limit; CPU frontier None, low-level join Complete; no successor.
- [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/[padding](../../../../glossary.md#padding)/[persistence](../../../../glossary.md#term-persistence)/low/[vector](../../../../glossary.md#term-vector)/parallel/[OpenBLAS](../../../../glossary.md#term-openblas) separately gated Draft; no required follow-up.
- Status: Complete
