# Task CPU0032: F32 MATMUL and actual reverse inside emitted general scan

## Status

Complete — accepted by main 2026-10-22, 17:09+ UTC; CPU frontier None.

## Change class

Class C — generated hot loops, whole-body capability, temporal storage and artifact compatibility;
clean implementation and independent code/documentation review completed.

## Goal and scope

Static homogeneous FLOAT32 (F32) rank-two MATMUL and qualified actual Compiler reverse compose
inside the existing emitted time loop, using only existing automatic differentiation (AD) rules.
Preserve CPU0031 arithmetic, broadcast/SUM/layout, demand/liveness, carries, resources, identity,
canonical boundaries and scalar-zero reads; internal views do not widen boundary admission.

## Non-goals

No F64/low-type execution, rank-one/batched MATMUL widening, new AD rules, mixed-owner bodies,
executor/interpreter, callback fallback, T-unrolling, arbitrary control flow, public API or edge change.
NN/training admission, per-row lengths/padding, persistence, Vector API/parallel/OpenBLAS scan routes
and tuning remain separately gated. F64 full algebra/closure is the next Draft gate, not a launch.

## Contracts

- Root [Scope-indexed normative contracts](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
  plus exact headings `Authority, incorporation, and precedence`, `Core invariants`, `Dependency rules`.
- Scan owner [General differentiable scan](../../../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
  especially [Canonical residual ports and demand](../../../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand),
  [Reverse scan and exact primal bindings](../../../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings),
  [Traversal, empty domains and declared qualification](../../../../architecture/contracts/recurrent-scan.md#traversal-empty-domains-and-declared-qualification)
  and [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates).
- [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes).
- [Compiler-owned automatic differentiation](../../../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation).
- Runtime/Prepare [modules/prepare](../../../../architecture/contracts/runtime-prepare-engine.md#modulesprepare),
  [Prepare lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#prepare-lifecycle),
  and [Run lifecycle](../../../../architecture/contracts/runtime-prepare-engine.md#run-lifecycle).
- Foundational [Profile-free capability queries](../../../../architecture/contracts/foundational-modules.md#profile-free-capability-queries)
  and [Profile-free numerical semantics](../../../../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics),
  especially `Floating arithmetic primitives`, `Aggregate floor`, `Composite inheritance`.

Stop on contract/seam uncertainty before inventing architecture; this plan changes no authority.

## Dependencies and integration

- Depends on: CPU0031 directly; accepted Model0034/0035, Planning0008, Compiler0011, Prepare0010
  and CPU0022/0027/0028/0029/0030 baseline, all Complete.
- Conflicts with: CPU scan admission/lowering/IR/emission/carriers/resources, ordinary MATMUL
  generation seams, generated identity/cache, shared capability facts and scan documentation.
- Parallel group: None; sole serial writer, no concurrent implementation or shared-document writer.
- Common base revision: HEAD = origin/main `6da23a0b42f015979a88f5e84416799f5ada3f40` PLUS accepted
  uncommitted foundations and CPU0030/0031; 250 dirty paths, empty index before this packet. Do not reset,
  stage, commit or push; existing changes belong to the user, not to this task's diff.
- Integration order: independent plan APPROVE → main launch verification → clean CPU implementation
  → independent Class C code/docs APPROVE and corrections → main checkpoint → explicit acceptance.
- Integration validation: **CPU0032 F32 MATMUL/actual-reverse checkpoint**, exact commands below;
  architecture only for an identified boundary risk; root/native gate stays at the later F64 low-level join.
- Shared-document integration owner: main; serialize Javadoc/explanation drafting and independent
  finalization; main owns shared planning/status and acceptance facts.

Plan review: Gauss the 3rd **APPROVE**, zero P1/P2; four frozen hashes matched (tmp progress record).
Main launch verification — 2026-10-22 16:18+ UTC: user authorized F32 rank-two MATMUL + actual reverse;
reviewed Ready brief accepted; CPU0031/listed foundations Complete with final review/module/checkpoint
evidence, directly verified from current masters. HEAD/origin as above, index empty, 251 dirty paths
(250 accepted + new brief): exactly four planning paths differ from accepted launch baseline;
contracts/source/protected files unchanged. Conflicts inactive; sole serial planning writer, Gauss
read-only, no implementation active. All seven fields and CPU0032 sole authorized frontier verified.
Preimplementation task-delta/old-byte archive:
`/private/tmp/synaptik-cpu0032-baseline.ykrQUK/preimplementation.tar.gz`.

## Files, symbols and production type inventory

CPU paths: `modules/backend/cpu/src/{main,test}/java/io/github/pho001/synaptik/backend/cpu/`.

- Changed production: `internal/{lowering/CpuGeneralScanLowering,ir/CpuGeneralScanIr}` and
  `internal/codegen/emit/{CpuGeneralScanEmitter,CpuMatmulEmitter,CpuClassFileKernelGenerator}.java`
  (last file Javadoc-only). Only new type: package-private generation-only
  `io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneralScanMatmulEmitter`.
- Changed tests: `CpuGeneralScan{Capability,Lowering,Preparation,Execution,GeneratedKernel}Test`,
  `CpuKernelSpecializationTest`, `CpuInternalPackageInventoryTest`, `CpuGeneralScanCleanJavaOracle`
  and `CpuGeneralScanTimingTest` in their existing packages.
- Checkpoint files: `CpuGeneralScanConformanceTest`, `LowPrecisionCapabilityLedgerTest` and
  `src/test/resources/low-precision-capability-ledger-v2.tsv` under `testing/backend-conformance`;
  `EngineCpuGeneralScanIntegrationTest` under `testing/integration-tests`.
- Compiler `LinearAlgebraGradientRules.apply`/`ScanGradientRules` unchanged; no new AD/executor.

## Acceptance criteria

1. Qualify complete original and actual generated bodies/binders before liveness pruning: exact
   F32 rank-two `[M,K] @ [K,N] -> [M,N]`, attrs, descriptors, layout/view proof, both operands,
   roles/demand, checked spans/bytes and method/local/entry budgets. Unsupported interiors/generated
   occurrences, missing body, invalid contraction or unsupported binder reject even dead/T=0/empty.
   Forward true is not a reverse promise; retain budget negatives.
   Append a current v2 MATMUL-scan seed/rows: independently query CPU F32 true, F64/BF16/F16 false,
   and Metal false. Preserve immutable v1; emitter availability is not ledger evidence.
2. Emit MATMUL stages within the single existing time loop. Explicitly bind invocation versus step
   carriers/typed locals, byte bases, ranges, M/N/K and affine strides; select time-dependent sequence,
   emission/residual and old/new carry addresses without graph/operation dispatch. Hoist invariant
   carrier/layout/geometry setup and proved invariant work. No per-step allocation/callback/dispatch,
   hidden MATMUL fallback, per-K geometry lookup or T-fold growth. Necessary direct static generated
   helpers need clean-Java rationale/inspection; no giant-method or blanket no-temporary mandate.
3. Both operands participate in dependency closure and inclusive stage lifetimes. Materialize only
   required cuts, reuse disjoint-lifetime MATMUL temporaries, declare checked pre-slot bytes and justify
   copies. Prefer demanded exact frames/next banks when legal;
   never alias a live operand, saved site or old carry. All old carries survive the entire step;
   cross-coordinate MATMUL, transposes, swaps/forwarding and differently shaped parameter carries
   commit simultaneous new values. Mutable arrays/storage stay in the sole RunState.
4. Execute actual Compiler AD for multi-stage/non-square recurrence, shared weights/bias,
   broadcast/SUM/layout and TANH/SIGMOID. Compare forward/all requested gradients with ordinary
   unrolled Tensor AD (≥3 steps), optimized/disabled, both directions and final/emission/joint seeds.
   Pin non-gradient h0 with trainable sequence/weights and internal carry cotangents; cover eligible
   h0 gradients, two carries, repeated roles, tied roots/templates and cross-occurrence sums.
5. Preserve exact saved values/original time indices and pre-step carry history, without replay.
   Partial/public demand and dead forward AND reverse residuals require zero undemanded history,
   not zero MATMUL scratch/emission memory. Test left/right/both transposes, composed proved views,
   contiguous cuts and unsupported layouts; retain heap/native/mixed cold binding/overlap rules.
6. Cover T=0/1/multiple and M/N/K=0: K=0 may produce NONEMPTY typed zeros; empty products can feed
   nonempty reductions/parameter sums. No global empty-domain return. T=0 preserves initial raw bits,
   dInitialCarry=dFinalCarry and typed-zero invariant sums without body access. Qualify rounding,
   cancellation, NaN/Inf, signed zero, zero divisors, overflow/underflow/subnormals and exact saved/raw
   sites under family rules. Retain overlap-before-write, overflow, cleanup/retry and concurrency tests.
7. Preserve CPU0030 pointwise and CPU0031 shaped keys/bytes. Add a versioned MATMUL-only projection
   covering both operands, order, accesses/roles/demand and every baked geometry/resource/numerical
   fact in BOTH class and compatibility bytes. Canonical CpuKernelIr stays geometry-free; T, assigned
   slots and addresses remain runtime facts. Test independent baked mutations/stale-cache rejection
   and compatible T reuse. If preserving existing identities is impossible, stop before changing them.
8. Inspect actual timed forward/reverse Class-Files against optimal specialized clean Java:
   O(body) object/code shape as T varies, matching algorithm/hot-loop/dataflow, no hidden fallback,
   allocation/boxing/reflection/dispatch or avoidable copies/address work. Short warmed small and
   larger/non-square diagnostics match dtype/carrier/layout/numerics/work and validation. Separate
   cold generation/class-load/JIT, entry/preflight and kernel/step costs; fix/justify diagnosed overhead.
   No five-fork/C2 campaign, timing assertion/parity claim or inherited CPU0029 timing proof.

## Validation

Worker commands (results below):

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScan*' --tests '*CpuKernelSpecializationTest' --tests '*CpuInternalPackageInventoryTest'
./gradlew :backends:cpu:test --tests '*CpuGeneralScanTimingTest.matchedMatmulForwardAndActualReverse' -Dsynaptik.cpu.generalScan.performance=true -Dsynaptik.cpu.generalScan.evidenceRoot=/private/tmp/synaptik-cpu0032-evidence
./gradlew :backends:cpu:test :backends:cpu:javadoc
```

Main's **CPU0032 F32 MATMUL/actual-reverse checkpoint**, once after ordered review/integration:

```bash
./gradlew :testing:backend-conformance:test --tests '*CpuGeneralScanConformanceTest' --tests '*CpuRecurrentScanConformanceTest' --tests '*LowPrecisionCapabilityLedgerTest'
./gradlew :testing:integration-tests:test --tests '*EngineCpuGeneralScanIntegrationTest' --tests '*EngineCpuRecurrentScanIntegrationTest'
git diff --check
```

Architecture risk/command/outcome recorded below; no edge/build change. Root/native test/Javadoc
stays at the F64/domain join. Planning checks links/size/metadata/diff only.

## Documentation and review impact

Independent Class C code/docs passes followed documentation-rules and general/backend/architecture/
API-Javadoc profiles. Scope: root/four scoped facts, three architecture explanations, CPU guide,
three API pages, glossary and ADR0045 status; six Java comment-only files. No semantic change.
Four existing glossary entries updated, no new term; unaffected Javadoc remains accurate.
Inventory/checks: `/private/tmp/synaptik-cpu0032-docs-progress.md`.

## Result

- Main accepted 2026-10-22, 17:09+ UTC: files/types above implement F32 rank-two MATMUL/actual reverse,
  both-operand liveness and one loop. Schema 71/old keys/bytes preserved; canonical identity geometry-free.
- Boyle code APPROVE, zero P1/P2, R1/R2 closed; 20-path manifest SHA-256
  `26dbd14848d2c60f20347ed1a9ffe1f1c15d4826a04b697fefef43293be16fb6`.
- Lagrange docs Complete: 14 Markdown/six comment-only Java, 18 token streams identical/12 byte-identical;
  561 links PASS, final Javadoc reconciled. Reports: `/private/tmp/synaptik-cpu0032-{code-review,docs-progress}.md`.
- Focused command above PASS: 63 methods/three opt-in skips; additional tests separately PASS.
- `./gradlew :backends:cpu:test :backends:cpu:javadoc` PASS (25206, 5m11s; 4 executed/19 up-to-date tasks):
  216 fresh suites, 1163 tests/1129 passed/34 skipped/zero failures/errors, 16:54:57–17:00:02Z.
  Javadoc: 92 baseline missing-@param warnings in two untouched types plus two Vector warnings; none in changed types.
- Post-module only ledger census 119→120 and two Engine M=0 actual-AD tests changed; both directions/T=1/3
  passed exact nonempty dW zeros/NaN dead weights and nonempty auxiliary carry/nonzero parameter sums.
  Boyle addendum APPROVE/zero findings; corrections manifest SHA-256 `a2dc03d742ccde8463f475dd1735cd6b5b1c1a87c4f53510cc18fe9e99263db0`.
- Exact conformance command above: corrected rerun PASS, three fresh suites/13 tests (17:07:55Z).
  Exact Engine command above: PASS, two fresh suites/15 tests (17:08:12–14Z); no checkpoint skips/failures/errors.
- `./gradlew :testing:architecture-tests:test --console=plain` PASS: 11 fresh suites/25 tests (17:08:41–42Z).
  Risk: composer/body-binding resources, imports/dependencies and Runtime graph exclusion; no edges changed.
  Checkpoint 53 passed; `git diff --check` PASS, CPU hashes stable/index empty; no CPU rerun after addendum.
- Four native-FFM r4 classes match r3; clean-Java/Class-File inspection, 1200 capped warmups/three samples.
  Larger forward 22.3–22.7µs generated vs 21.8–22.7 Java, reverse 67.2–69.2 vs 73.1–75.2; small near 1×.
  Reverse scratch 160/10240 bytes, 96 redundant bytes removed; assertion preserved. Warmup gap retained;
  bound pipeline separate, no universal/E2E/JIT-tier parity. `/private/tmp/synaptik-cpu0032-evidence-r4/`,
  `/private/tmp/synaptik-cpu0032-inspection.md` retain artifacts/details.
- No unresolved issues/required follow-up. Full homogeneous F64 closure next mandatory Draft, then one
  root/native join; NN/padding/persistence/low types/vector/parallel/OpenBLAS gated. No successor launch.

Status: Complete
