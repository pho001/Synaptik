# Task CPU0032: [F32](../../../../glossary.md#floating-point-precision-and-rounding) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) and actual reverse inside emitted [general scan](../../../../glossary.md#general-differentiable-scan)

## Reader context — historical integration boundary

This milestone places rank-two MATMUL inside the generated general-Scan body; it does not call the
OpenBLAS route once per step. Compiler constructs transpose/contraction operations for the actual
reverse body, and CPU plans both operands' lifetimes before emitting the time loop. A local cut
preserves a value across body stages; a demanded history preserves selected values across time for a
later consumer. Confusing the two can overwrite an operand or old carry too early. The bounded F32
admission and identity statements below remain historical; the current guide also describes the
later common contraction owner.

Current reading: [Current general Scan composition](../../../../backend-guide/cpu-backend.md#current-general-scan-composition); [Retained static Scan resources](../../../../backend-guide/cpu-backend.md#retained-static-scan-qualification-and-resources).

## Status

Complete — accepted by main 2026-10-22, 17:09+ UTC; [CPU](../../../../glossary.md#term-cpu) frontier None.

## Change class

Class C — generated hot loops, whole-body capability, temporal storage and [artifact](../../../../glossary.md#term-artifact) compatibility;
clean implementation and independent code/documentation review completed.

## Goal and [scope](../../../../glossary.md#term-scope)

Static [homogeneous](../../../../glossary.md#term-homogeneous) [FLOAT32](../../../../glossary.md#term-float32) ([F32](../../../../glossary.md#floating-point-precision-and-rounding)) [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) and qualified actual [Compiler](../../../../glossary.md#term-compiler) reverse compose
inside the existing emitted time loop, using only existing [automatic differentiation](../../../../glossary.md#autograd) ([AD](../../../../glossary.md#autograd)) rules.
Preserve CPU0031 arithmetic, broadcast/SUM/[layout](../../../../glossary.md#layout), demand/[liveness](../../../../glossary.md#term-liveness), carries, resources, [identity](../../../../glossary.md#term-identity-meanings),
[canonical](../../../../glossary.md#term-canonical-form) boundaries and [scalar](../../../../glossary.md#term-scalar)-zero reads; internal [views](../../../../glossary.md#term-view-meanings) do not widen boundary [admission](../../../../glossary.md#term-admission).

## Non-goals

No [F64](../../../../glossary.md#floating-point-precision-and-rounding)/low-type execution, [rank-one](../../../../glossary.md#axis-rank-scalar-vector-and-matrix)/batched [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) [widening](../../../../glossary.md#term-widening), new [AD](../../../../glossary.md#autograd) rules, mixed-[owner](../../../../glossary.md#term-owner) bodies,
executor/interpreter, [callback](../../../../glossary.md#term-callback) fallback, T-[unrolling](../../../../glossary.md#term-unrolling), arbitrary control flow, public [API](../../../../glossary.md#term-application-programming-interface-api) or edge change.
[NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/training [admission](../../../../glossary.md#term-admission), per-row lengths/[padding](../../../../glossary.md#padding), [persistence](../../../../glossary.md#term-persistence), [Vector API](../../../../glossary.md#term-vector-api)/parallel/[OpenBLAS](../../../../glossary.md#term-openblas) [scan](../../../../glossary.md#term-cumulative-scan) routes
and tuning remain separately gated. F64 full algebra/closure is the next Draft gate, not a launch.

## Contracts

- Root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([Scope glossary](../../../../glossary.md#term-scope); [normative glossary](../../../../glossary.md#term-normative))</span>,
  plus exact headings `Authority, incorporation, and precedence`, `Core invariants`, `Dependency rules`.
- [Scan](../../../../glossary.md#term-cumulative-scan) [owner](../../../../glossary.md#term-owner) [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([General differentiable scan glossary](../../../../glossary.md#general-differentiable-scan))</span>,
  especially [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand)<span data-glossary-navigation="true"> ([Canonical glossary](../../../../glossary.md#term-canonical-form); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>,
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)<span data-glossary-navigation="true"> ([primal glossary](../../../../glossary.md#term-primal); [bindings glossary](../../../../glossary.md#term-binding))</span>,
  [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)<span data-glossary-navigation="true"> ([qualification glossary](../../../../glossary.md#term-qualification))</span>
  and [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates).
- [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [backend glossary](../../../../glossary.md#backend))</span>.
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [automatic differentiation glossary](../../../../glossary.md#autograd))</span>.
- [Runtime](../../../../glossary.md#term-runtime)/[Prepare](../../../../glossary.md#prepare) [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare)<span data-glossary-navigation="true"> ([modules glossary](../../../../glossary.md#term-module))</span>,
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../../../../glossary.md#lifecycle))</span>,
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle)<span data-glossary-navigation="true"> ([Run glossary](../../../../glossary.md#term-run))</span>.
- Foundational [Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries)
  and [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics)<span data-glossary-navigation="true"> ([Profile-free numerical semantics glossary](../../../../glossary.md#profile-free-numerical-semantics))</span>,
  especially `Floating arithmetic primitives`, [`Aggregate floor`](../../../../glossary.md#term-aggregate-floor), `Composite inheritance`.

Stop on contract/seam uncertainty before inventing architecture; this plan changes no authority.

## Dependencies and integration

- Depends on: CPU0031 directly; accepted Model0034/0035, Planning0008, Compiler0011, Prepare0010
  and CPU0022/0027/0028/0029/0030 baseline, all Complete.
- Conflicts with: CPU scan admission/lowering/IR/emission/carriers/resources, ordinary MATMUL
  generation seams, generated identity/cache, shared capability facts and scan documentation.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [scan glossary](../../../../glossary.md#term-cumulative-scan); [admission glossary](../../../../glossary.md#term-admission); [lowering glossary](../../../../glossary.md#lowering); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [identity glossary](../../../../glossary.md#term-identity-meanings); [cache glossary](../../../../glossary.md#term-cache))</span>
- Parallel group: None; sole serial writer, no concurrent implementation or shared-document writer.
- Common base revision: HEAD = origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` PLUS accepted
  uncommitted foundations and CPU0030/0031; 250 dirty paths, empty index before this packet. Do not reset,
  stage, commit or push; existing changes belong to the user, not to this task's diff.<span data-glossary-navigation="true"> ([commit glossary](../../../../glossary.md#term-commit))</span>
- Integration order: independent plan APPROVE → main launch verification → clean CPU implementation
  → independent Class C code/docs APPROVE and corrections → main checkpoint → explicit acceptance.
- Integration [validation](../../../../glossary.md#term-validation): **CPU0032 [F32](../../../../glossary.md#floating-point-precision-and-rounding) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/actual-reverse [checkpoint](../../../../glossary.md#term-checkpoint)**, exact commands below;
  architecture only for an identified boundary risk; root/native gate stays at the later F64 low-level join.<span data-glossary-navigation="true"> ([F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding))</span>
- Shared-document integration owner: main; serialize Javadoc/explanation drafting and independent
  finalization; main owns shared planning/status and acceptance facts.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>

Plan review: Gauss the 3rd **APPROVE**, zero P1/P2; four frozen [hashes](../../../../glossary.md#term-hash) matched (tmp progress record).
Main launch verification — 2026-10-22 16:18+ UTC: user authorized F32 [rank-two](../../../../glossary.md#axis-rank-scalar-vector-and-matrix) MATMUL + actual reverse;
reviewed [Ready](../../../../glossary.md#term-ready) brief accepted; CPU0031/listed foundations Complete with final review/[module](../../../../glossary.md#term-module)/checkpoint
evidence, directly verified from current masters. HEAD/origin as above, index empty, 251 dirty paths
(250 accepted + new brief): exactly four [planning](../../../../glossary.md#term-planning) paths differ from accepted launch baseline;
contracts/source/protected files unchanged. Conflicts inactive; sole serial planning writer, Gauss
read-only, no implementation active. All seven fields and CPU0032 sole [authorized frontier](../../../../glossary.md#term-authorized-frontier) verified.
Preimplementation task-delta/old-byte archive:
`/private/tmp/synaptik-cpu0032-baseline.ykrQUK/preimplementation.tar.gz`.

## Files, symbols and production type inventory

[CPU](../../../../glossary.md#term-cpu) paths: `modules/backend/cpu/src/{main,test}/java/io/github/pho001/synaptik/backend/cpu/`.

- Changed production: `internal/{lowering/CpuGeneralScanLowering,ir/CpuGeneralScanIr}` and
  `internal/codegen/emit/{CpuGeneralScanEmitter,CpuMatmulEmitter,CpuClassFileKernelGenerator}.java`
  (last file Javadoc-only). Only new type: package-private generation-only<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
  `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanMatmulEmitter`.
- Changed tests: `CpuGeneralScan{Capability,Lowering,Preparation,Execution,GeneratedKernel}Test`,
  `CpuKernelSpecializationTest`, `CpuInternalPackageInventoryTest`, `CpuGeneralScanCleanJavaOracle`
  and `CpuGeneralScanTimingTest` in their existing packages.
- [Checkpoint](../../../../glossary.md#term-checkpoint) files: `CpuGeneralScanConformanceTest`, `LowPrecisionCapabilityLedgerTest` and
  `src/test/resources/low-precision-capability-ledger-v2.tsv` under `testing/backend-conformance`;
  `EngineCpuGeneralScanIntegrationTest` under `testing/integration-tests`.
- [Compiler](../../../../glossary.md#term-compiler) `LinearAlgebraGradientRules.apply`/`ScanGradientRules` unchanged; no new [AD](../../../../glossary.md#autograd)/executor.

## Acceptance criteria

1. Qualify complete original and actual generated bodies/binders before liveness pruning: exact
   F32 rank-two <span data-glossary-navigation="true"> ([liveness glossary](../../../../glossary.md#term-liveness); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>`[M,K] @ [K,N] -> [M,N]`, attrs, descriptors, layout/view proof, both operands,
   roles/demand, checked spans/bytes and method/local/entry budgets. Unsupported interiors/generated
   occurrences, missing body, invalid contraction or unsupported binder reject even dead/T=0/empty.
   Forward true is not a reverse promise; retain budget negatives.
   Append a current v2 MATMUL-scan seed/rows: independently query CPU F32 true, F64/BF16/F16 false,
   and Metal false. Preserve immutable v1; emitter availability is not ledger evidence.<span data-glossary-navigation="true"> ([descriptors glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout); [view glossary](../../../../glossary.md#term-view-meanings); [contraction glossary](../../../../glossary.md#contraction-dimension); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [scan glossary](../../../../glossary.md#term-cumulative-scan); [seed glossary](../../../../glossary.md#term-seed); [CPU glossary](../../../../glossary.md#term-cpu); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [BF16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [Metal glossary](../../../../glossary.md#term-metal); [immutable glossary](../../../../glossary.md#term-immutable))</span>
2. Emit MATMUL stages within the single existing time loop. Explicitly bind invocation versus step
   carriers/typed locals, byte bases, ranges, M/N/K and affine strides; select time-dependent sequence,
   emission/residual and old/new carry addresses without graph/operation dispatch. Hoist invariant
   carrier/layout/geometry setup and proved invariant work. No per-step allocation/callback/dispatch,
   hidden MATMUL fallback, per-K geometry lookup or T-fold growth. Necessary direct static generated
   helpers need clean-Java rationale/inspection; no giant-method or blanket no-temporary mandate.<span data-glossary-navigation="true"> ([strides glossary](../../../../glossary.md#term-stride); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand); [carry glossary](../../../../glossary.md#term-carry); [graph glossary](../../../../glossary.md#graph); [operation glossary](../../../../glossary.md#operation); [dispatch glossary](../../../../glossary.md#term-dispatch); [invariant glossary](../../../../glossary.md#term-invariant); [carrier glossary](../../../../glossary.md#term-storage-carrier); [callback glossary](../../../../glossary.md#term-callback); [lookup glossary](../../../../glossary.md#term-lookup))</span>
3. Both operands participate in dependency closure and inclusive stage lifetimes. Materialize only
   required cuts, reuse disjoint-lifetime MATMUL temporaries, declare checked pre-slot bytes and justify
   copies. Prefer demanded exact frames/next banks when legal;
   never alias a live operand, saved site or old carry. All old carries survive the entire step;
   cross-coordinate MATMUL, transposes, swaps/forwarding and differently shaped parameter carries
   commit simultaneous new values. Mutable arrays/storage stay in the sole RunState.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [lifetime glossary](../../../../glossary.md#term-lifetime); [alias glossary](../../../../glossary.md#aliasing-and-arena); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [parameter glossary](../../../../glossary.md#term-parameter); [commit glossary](../../../../glossary.md#term-commit); [Mutable glossary](../../../../glossary.md#term-mutable); [RunState glossary](../../../../glossary.md#run-state--runstate))</span>
4. Execute actual Compiler AD for multi-stage/non-square recurrence, shared weights/bias,
   broadcast/SUM/layout and TANH/SIGMOID. Compare forward/all requested gradients with ordinary
   unrolled Tensor AD (≥3 steps), optimized/disabled, both directions and final/emission/joint seeds.
   Pin non-gradient h0 with trainable sequence/weights and internal carry cotangents; cover eligible
   h0 gradients, two carries, repeated roles, tied roots/templates and cross-occurrence sums.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [AD glossary](../../../../glossary.md#autograd); [weights glossary](../../../../glossary.md#term-weight); [bias glossary](../../../../glossary.md#term-bias); [TANH glossary](../../../../glossary.md#term-tanh); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [gradients glossary](../../../../glossary.md#term-gradient); [unrolled glossary](../../../../glossary.md#backpropagation-through-time-and-recurrent-state); [Tensor glossary](../../../../glossary.md#tensor); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
5. Preserve exact saved values/original time indices and pre-step carry history, without replay.
   Partial/public demand and dead forward AND reverse residuals require zero undemanded history,
   not zero MATMUL scratch/emission memory. Test left/right/both transposes, composed proved views,
   contiguous cuts and unsupported layouts; retain heap/native/mixed cold binding/overlap rules.<span data-glossary-navigation="true"> ([contiguous glossary](../../../../glossary.md#term-contiguous); [heap glossary](../../../../glossary.md#term-heap); [binding glossary](../../../../glossary.md#term-binding))</span>
6. Cover T=0/1/multiple and M/N/K=0: K=0 may produce NONEMPTY typed zeros; empty products can feed
   nonempty reductions/parameter sums. No global empty-domain return. T=0 preserves initial raw bits,
   dInitialCarry=dFinalCarry and typed-zero invariant sums without body access. Qualify rounding,
   cancellation, NaN/Inf, signed zero, zero divisors, overflow/underflow/subnormals and exact saved/raw
   sites under family rules. Retain overlap-before-write, overflow, cleanup/retry and concurrency tests.<span data-glossary-navigation="true"> ([raw bits glossary](../../../../glossary.md#storage-mapping-contiguity-and-injectivity); [rounding glossary](../../../../glossary.md#term-rounding); [cancellation glossary](../../../../glossary.md#term-cancellation); [NaN glossary](../../../../glossary.md#term-nan); [signed zero glossary](../../../../glossary.md#term-signed-zero); [overflow glossary](../../../../glossary.md#term-overflow); [underflow glossary](../../../../glossary.md#term-underflow); [subnormals glossary](../../../../glossary.md#term-subnormal); [concurrency glossary](../../../../glossary.md#term-concurrency))</span>
7. Preserve CPU0030 pointwise and CPU0031 shaped keys/bytes. Add a versioned MATMUL-only projection
   covering both operands, order, accesses/roles/demand and every baked geometry/resource/numerical
   fact in BOTH class and compatibility bytes. Canonical CpuKernelIr stays geometry-free; T, assigned
   slots and addresses remain runtime facts. Test independent baked mutations/stale-cache rejection
   and compatible T reuse. If preserving existing identities is impossible, stop before changing them.<span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([pointwise glossary](../../../../glossary.md#term-pointwise); [Canonical glossary](../../../../glossary.md#term-canonical-form); [runtime glossary](../../../../glossary.md#term-runtime); [cache glossary](../../../../glossary.md#term-cache))</span>
8. Inspect actual timed forward/reverse Class-Files against optimal specialized clean Java:
   O(body) object/code shape as T varies, matching algorithm/hot-loop/dataflow, no hidden fallback,
   allocation/boxing/reflection/dispatch or avoidable copies/address work. Short warmed small and
   larger/non-square diagnostics match dtype/carrier/layout/numerics/work and validation. Separate
   cold generation/class-load/JIT, entry/preflight and kernel/step costs; fix/justify diagnosed overhead.
   No five-fork/C2 campaign, timing assertion/parity claim or inherited CPU0029 timing proof.<span data-glossary-navigation="true"> ([shape glossary](../../../../glossary.md#shape); [dataflow glossary](../../../../glossary.md#term-dataflow); [dtype glossary](../../../../glossary.md#data-type--datatype); [validation glossary](../../../../glossary.md#term-validation); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit); [preflight glossary](../../../../glossary.md#term-preflight); [kernel glossary](../../../../glossary.md#kernel); [fork glossary](../../../../glossary.md#term-fork-meanings); [C2 glossary](../../../../glossary.md#term-c2))</span>

## [Validation](../../../../glossary.md#term-validation)

Worker commands (results below):

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScan*' --tests '*CpuKernelSpecializationTest' --tests '*CpuInternalPackageInventoryTest'
./gradlew :backends:cpu:test --tests '*CpuGeneralScanTimingTest.matchedMatmulForwardAndActualReverse' -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0032-evidence
./gradlew :backends:cpu:test :backends:cpu:javadoc
```

Main's **CPU0032 [F32](../../../../glossary.md#floating-point-precision-and-rounding) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/actual-reverse [checkpoint](../../../../glossary.md#term-checkpoint)**, once after ordered review/integration:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*CpuRecurrentScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuRecurrentScanIntegrationTest'
git diff --check
```

Architecture risk/command/outcome recorded below; no edge/build change. Root/native test/[Javadoc](../../../../glossary.md#term-javadoc)
stays at the [F64](../../../../glossary.md#floating-point-precision-and-rounding)/domain join. [Planning](../../../../glossary.md#term-planning) checks links/size/[metadata](../../../../glossary.md#term-metadata)/diff only.

## Documentation and review impact

Independent Class C code/docs passes followed documentation-rules and general/[backend](../../../../glossary.md#backend)/architecture/
[API](../../../../glossary.md#term-application-programming-interface-api)-[Javadoc](../../../../glossary.md#term-javadoc) profiles. [Scope](../../../../glossary.md#term-scope): root/four scoped facts, three architecture explanations, [CPU](../../../../glossary.md#term-cpu) guide,
three API pages, glossary and ADR0045 status; six Java comment-only files. No semantic change.
Four existing glossary entries updated, no new term; unaffected Javadoc remains accurate.
Inventory/checks: `/private/tmp/synaptik-cpu0032-docs-progress.md`.

## Result

- Main accepted 2026-10-22, 17:09+ UTC: files/types above implement F32 rank-two MATMUL/actual reverse,
  both-operand liveness and one loop. Schema 71/old keys/bytes preserved; canonical identity geometry-free.<span data-glossary-navigation="true"> ([F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [rank-two glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [liveness glossary](../../../../glossary.md#term-liveness); [Schema glossary](../../../../glossary.md#term-schema); [canonical glossary](../../../../glossary.md#term-canonical-form); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
- Boyle code APPROVE, zero P1/P2, R1/R2 closed; 20-path [manifest](../../../../glossary.md#term-manifest) [SHA-256](../../../../glossary.md#term-sha-256)
  `26dbd14848d2c60f20347ed1a9ffe1f1c15d4826a04b697fefef43293be16fb6`.
- Lagrange docs Complete: 14 Markdown/six comment-only Java, 18 token streams identical/12 byte-identical;
  561 links PASS, final Javadoc reconciled. Reports: <span data-glossary-navigation="true"> ([token glossary](../../../../glossary.md#term-token); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>`/private/tmp/synaptik-cpu0032-{code-review,docs-progress}.md`.
- Focused command above PASS: 63 methods/three opt-in skips; additional tests separately PASS.
- `./gradlew :backends:cpu:test :backends:cpu:javadoc` PASS (25206, 5m11s; 4 executed/19 up-to-date tasks):
  216 fresh suites, 1163 tests/1129 passed/34 skipped/zero failures/errors, 16:54:57–17:00:02Z.
  Javadoc: 92 baseline missing-@param warnings in two untouched types plus two Vector warnings; none in changed types.<span data-glossary-navigation="true"> ([Vector glossary](../../../../glossary.md#term-vector))</span>
- Post-module only ledger census 119→120 and two Engine M=0 actual-AD tests changed; both directions/T=1/3
  passed exact nonempty dW zeros/NaN dead weights and nonempty auxiliary carry/nonzero parameter sums.
  Boyle addendum APPROVE/zero findings; corrections manifest SHA-256 <span data-glossary-navigation="true"> ([module glossary](../../../../glossary.md#term-module); [Engine glossary](../../../../glossary.md#term-engine); [AD glossary](../../../../glossary.md#autograd); [NaN glossary](../../../../glossary.md#term-nan); [weights glossary](../../../../glossary.md#term-weight); [carry glossary](../../../../glossary.md#term-carry); [parameter glossary](../../../../glossary.md#term-parameter))</span>`a2dc03d742ccde8463f475dd1735cd6b5b1c1a87c4f53510cc18fe9e99263db0`.
- Exact conformance command above: corrected rerun PASS, three fresh suites/13 tests (17:07:55Z).
  Exact Engine command above: PASS, two fresh suites/15 tests (17:08:12–14Z); no checkpoint skips/failures/errors.<span data-glossary-navigation="true"> ([conformance glossary](../../../../glossary.md#term-conformance); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- `./gradlew :testing:architecture-tests:test --console=plain` PASS: 11 fresh suites/25 tests (17:08:41–42Z).
  Risk: composer/body-binding resources, imports/dependencies and Runtime graph exclusion; no edges changed.
  Checkpoint 53 passed; <span data-glossary-navigation="true"> ([binding glossary](../../../../glossary.md#term-binding); [Runtime glossary](../../../../glossary.md#term-runtime); [graph glossary](../../../../glossary.md#graph))</span>`git diff --check` PASS, [CPU](../../../../glossary.md#term-cpu) [hashes](../../../../glossary.md#term-hash) stable/index empty; no CPU rerun after addendum.
- Four native-FFM r4 classes match r3; clean-Java/Class-File inspection, 1200 capped warmups/three samples.
  Larger forward 22.3–22.7µs generated vs 21.8–22.7 Java, reverse 67.2–69.2 vs 73.1–75.2; small near 1×.
  Reverse scratch 160/10240 bytes, 96 redundant bytes removed; assertion preserved. Warmup gap retained;
  bound pipeline separate, no universal/E2E/JIT-tier parity. <span data-glossary-navigation="true"> ([FFM glossary](../../../../glossary.md#term-ffm); [samples glossary](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters); [Warmup glossary](../../../../glossary.md#term-warmup); [JIT glossary](../../../../glossary.md#term-just-in-time-compilation-jit))</span>`/private/tmp/synaptik-cpu0032-evidence-r4/`,
  `/private/tmp/synaptik-cpu0032-inspection.md` retain [artifacts](../../../../glossary.md#term-artifact)/details.
- No unresolved issues/required follow-up. Full homogeneous F64 closure next mandatory Draft, then one
  root/native join; NN/padding/persistence/low types/vector/parallel/OpenBLAS gated. No successor launch.<span data-glossary-navigation="true"> ([homogeneous glossary](../../../../glossary.md#term-homogeneous); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [padding glossary](../../../../glossary.md#padding); [persistence glossary](../../../../glossary.md#term-persistence); [OpenBLAS glossary](../../../../glossary.md#term-openblas))</span>

Status: Complete
