# CPU0068: Qualify shared [feature](../../../../glossary.md#term-feature)-[vector](../../../../glossary.md#term-vector) body emission

## Reader context

This superseded Draft proposed qualifying feature-lane execution, not simply enabling a
vector opcode. Preparation must prove that every lane may read its inputs and publish its
saved values under the enclosing Scan's activity rules. A masked result cannot repair an
invalid read that happened first. Scalar tails, full-K contributor order and exact saved
forward bits are separate obligations; vector TANH need not equal scalar TANH bit for bit.
None of those generation choices by itself submits workers or creates parallel run ownership.

For the existing ordinary realization, [portable MATMUL](../../../../backend-guide/cpu-backend.md#current-portable-matmul-family)
explains vector interiors, scalar tails and range ownership. The [current Scan overview](../../../../backend-guide/cpu-backend.md#current-general-scan-composition)
is the present support entry point; the archived type names and budgets below are not current
capability or launch authority.

> Superseded by CPU0070's [replacement plan](../existing-emitter-composition-plan.md#simd-admission-and-parallel-composition-gates).
> The original Draft below is archived, not executable. Shared existing SIMD mathematics is
> retained; Scan SIMD admission still requires bounded authority and whole-binder qualification.
> No diagnostic budget or scalar-then-vector reimplementation sequence carries forward.<span data-glossary-navigation="true"> ([SIMD glossary](../../../../glossary.md#simd); [admission glossary](../../../../glossary.md#term-admission); [qualification glossary](../../../../glossary.md#term-qualification); [scalar glossary](../../../../glossary.md#term-scalar))</span>

## Status and change class

Superseded. The following pre-supersession status and remaining brief are archived:

Draft; no implementation frontier or performance grant. Class C: generated [vector](../../../../glossary.md#term-vector) hot paths,
represented saved/state boundaries and bounded [backend capability](../../../../glossary.md#backend-capability). Before [Ready](../../../../glossary.md#term-ready), CPU0067 must be
accepted and MAIN/user must approve the exact vector predicates and coordinated authority update.

## Goal, [scope](../../../../glossary.md#term-scope) and exclusions

Qualify single-thread [feature](../../../../glossary.md#term-feature)-[lane](../../../../glossary.md#term-lane) [pointwise](../../../../glossary.md#term-pointwise) and full-K/N-[vector](../../../../glossary.md#term-vector) [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) through CPU0067's common
body seams for existing Scan occurrences. Ordinary vector bodies use those same seams; no second
Scan vector emitter. Reuse CPU0065 direct/tiled [TANH](../../../../glossary.md#term-tanh) where each required saved/state site qualifies.
The [proposal](../shared-body-emission-proposal.md#vector-realization-then-worker-integration)
defines boundaries. Preserve [scalar](../../../../glossary.md#term-scalar) execution for unproved cases, including scalar reductions
and transposed [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) contractions without the required [contiguous](../../../../glossary.md#term-contiguous) feature accesses.

No parallel submission/phases, K splitting, vector-reduction [reassociation](../../../../glossary.md#term-reassociation), new body operations,
[layout](../../../../glossary.md#layout)/[dtype](../../../../glossary.md#data-type--datatype) [widening](../../../../glossary.md#term-widening), ordinary symbolic [F64](../../../../glossary.md#floating-point-precision-and-rounding) MATMUL/epilogues, low/native/[Metal](../../../../glossary.md#term-metal)/[OpenBLAS](../../../../glossary.md#term-openblas) Scan,
[Compiler](../../../../glossary.md#term-compiler)-body rewrites, [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) packing/migration, tuning or public [API](../../../../glossary.md#term-application-programming-interface-api)/[Runtime](../../../../glossary.md#term-runtime) change.

## Contracts and approval gate

Read AGENTS; root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
[scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>; [backend](../../../../glossary.md#backend)
[CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu))</span> and
[Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract)<span data-glossary-navigation="true"> ([batch glossary](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters))</span>;
Scan [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates)<span data-glossary-navigation="true"> ([Qualification glossary](../../../../glossary.md#term-qualification))</span>,
[Qualification and mechanism gates](../../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates),
[Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
[Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>
and [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates)<span data-glossary-navigation="true"> ([Runtime glossary](../../../../glossary.md#term-runtime); [preflight glossary](../../../../glossary.md#term-preflight))</span>.
Before Java, approve exact positive/negative [occurrence](../../../../glossary.md#term-occurrence)/route predicates, changed [artifact](../../../../glossary.md#term-artifact) list
and bounded root/backend/Scan authority wording with independent review/MAIN acceptance.
Current [vector](../../../../glossary.md#term-vector) prohibitions are not waived by this Draft. Missing/ambiguous contracts stop work.

## Dependencies and integration

- Depends on: CPU0067 Complete/accepted; CPU0065 retained; approved bounded [vector](../../../../glossary.md#term-vector) authority.
- Conflicts with: common-body/Scan/vector/[MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/[cache](../../../../glossary.md#term-cache) and CPU0048–0052 writers.
- Parallel group: None; clean implementation plus independent read-only review.
- Common base revision: CPU0067 accepted source on `07a281edd3086efde2853211b71b48288db027e1`
  plus accepted dirty tree; MAIN records fresh exact base/preservation/contracts before Ready.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready))</span>
- Integration order: accepted CPU0067 → approved authority/predicates → vector vertical/review
  → MAIN capability checkpoint → consider CPU0069. No automatic launch.
- Integration [validation](../../../../glossary.md#term-validation): CPU0068 [checkpoint](../../../../glossary.md#term-checkpoint) below, docs/links/whitespace/preservation.
- Shared-document integration [owner](../../../../glossary.md#term-owner): MAIN; exclusive delegation for owned explanation/[Javadoc](../../../../glossary.md#term-javadoc).

## Files and symbols

Prefix: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.
Existing `CpuCapabilityProvider`; `internal/lowering/CpuGeneralScanLowering`, `CpuMatmulLowering`;
`internal/ir/CpuGeneralScanIr`, `CpuMatmulIr`; `internal/prepare/CpuPartitionPreparer.analyzeGeneralScan`,
`CpuPartitionFinalizer`; `internal/codegen/emit/CpuBodyBindings`, `CpuPointwiseBodyEmitter`,
`CpuLoopEmitter`, `CpuCarrierEmitter`, `CpuVectorInstructionEmitter`, `CpuVectorMath`,
`CpuMatmulEmitter`, `CpuGeneralScanEmitter`; `internal/cache/CpuKernelSpecialization`.
No new production type/package proposed beyond accepted CPU0067 types. If one is necessary,
stop for its exact name/package/purpose/[signature](../../../../glossary.md#term-signature) approval. Execution remains one inline invocation.

## Acceptance criteria

1. Preparation selects typed per-stage vector/scalar realization from exact whole-body/binder
   qualification. Initial vector predicates require contiguous feature lanes, exact supported
   F32/F64 math, complete active lanes and injective writes; tails use shared scalar emission.
   No inactive operand is loaded/evaluated then masked. Preserve ROW(0)/ROW(1)/SHARED scopes.<span data-glossary-navigation="true"> ([vector glossary](../../../../glossary.md#term-vector); [qualification glossary](../../../../glossary.md#term-qualification); [contiguous glossary](../../../../glossary.md#term-contiguous); [feature glossary](../../../../glossary.md#term-feature); [lanes glossary](../../../../glossary.md#term-lane); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [injective glossary](../../../../glossary.md#term-injective))</span>
2. Independently qualify existing static F32/F64, symbolic F32 and combined length F32/F64
   forward and actual Compiler siblings against ordinary unrolling, accounting for exact versus
   represented-sequential SUM and using matching-algorithm oracles for exact checks. Admission is unchanged
   outside the explicitly approved routes; all-valid symbolic F64 remains rejected. Ordinary<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [unrolling glossary](../../../../glossary.md#term-unrolling); [Admission glossary](../../../../glossary.md#term-admission))</span>
   `CpuMatmulIr.batchGeometry` remains [F32](../../../../glossary.md#floating-point-precision-and-rounding)/direct-[scalar](../../../../glossary.md#term-scalar)/no-[epilogue](../../../../glossary.md#term-epilogue); Scan numeric [binding](../../../../glossary.md#term-binding) is separate.
3. Retain ordered full-K math, distinct per-time ADD, sequential SUM and exact outer SUM/MEAN.
   Saved/state values preserve exactly the bits produced by the selected forward realization
   and consumed by reverse; finite output tolerances alone do not prove this. Vector TANH/SIGMOID
   need not match scalar last bits. Retain scalar sites if their retained-value contract cannot
   be satisfied. Preserve fan-out,
   demanded histories, aliases/carry versions, raw singleton/empty/padding and zero-sign rules.<span data-glossary-navigation="true"> ([MEAN glossary](../../../../glossary.md#term-mean); [finite glossary](../../../../glossary.md#term-finite); [TANH glossary](../../../../glossary.md#term-tanh); [SIGMOID glossary](../../../../glossary.md#term-sigmoid); [fan-out glossary](../../../../glossary.md#term-fan-out); [aliases glossary](../../../../glossary.md#aliasing-and-arena); [carry glossary](../../../../glossary.md#term-carry); [singleton glossary](../../../../glossary.md#term-singleton); [padding glossary](../../../../glossary.md#padding))</span>
4. Prove common vector load/instruction/store and MATMUL bodies are used by ordinary and Scan
   callers. Species/setup is invocation-level; temporal addresses do not cause entry decoding,
   per-step allocation or hidden helpers. Preserve v9 scalar fallbacks and code-shape gains.<span data-glossary-navigation="true"> ([MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
5. Test laneCount−1/laneCount/laneCount+1 and tiled tails, empty K/T/B/features, transposed reads,
   mixed length activity and poisoned padding, both seeds/traversals, tied parameters and reusable
   64→40→1→64. Exact scratch/history declarations remain complete before slots. No worker use.<span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [parameters glossary](../../../../glossary.md#term-parameter))</span>
6. Both identity streams distinguish changed vector/scalar/algorithm/binding facts, reject stale
   artifacts and reuse compatible code. Preserve unaffected bytes/keys per the accepted gate.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [artifacts glossary](../../../../glossary.md#term-artifact))</span>

## [Validation](../../../../glossary.md#term-validation)

Future focused development selection followed by one stabilized affected-[module](../../../../glossary.md#term-module) [run](../../../../glossary.md#term-run):

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScanCapabilityTest' --tests '*CpuGeneralScanGeneratedKernelTest' --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuGeneralScanExecutionTest' --tests '*CpuMatmulTanhEpilogueTest' --tests '*CpuKernelSpecializationTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
```

MAIN runs the named CPU0067 [conformance](../../../../glossary.md#term-conformance)/[Engine](../../../../glossary.md#term-engine) [checkpoint](../../../../glossary.md#term-checkpoint) commands once at this capability
cutover, with [vector](../../../../glossary.md#term-vector)-positive/negative and actual-[AD](../../../../glossary.md#autograd) fixtures added to those existing tests.
Inspect generated ClassFiles against specialized clean vector Java, with exact saved sites and
[scalar](../../../../glossary.md#term-scalar) tails. Proposed fresh diagnostic: add `sharedBodyVectorDiagnostic` to existing
`CpuGeneralScanTimingTest`, comparing common scalar/vector and specialized clean Java for the
proposal's repeated [MLP](../../../../glossary.md#term-multilayer-perceptron-mlp) and combined [RNN](../../../../glossary.md#term-recurrent-neural-network-rnn) forward/actual reverse, T=4/B=7/F=3/H=17, both [dtypes](../../../../glossary.md#data-type--datatype):

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScanTimingTest.sharedBodyVectorDiagnostic' -Dsynaptik.cpu.generalScan.performance=true
```

Budget: one test [JVM](../../../../glossary.md#term-java-virtual-machine-jvm), no retry/[forks](../../../../glossary.md#term-fork-meanings), at most 120 seconds of diagnostic body; precheck exact
saved sites/specials and permitted [finite](../../../../glossary.md#term-finite) results, then warmed interleaved matched-work [samples](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters).
Freeze/approve protocol before [Ready](../../../../glossary.md#term-ready). Disclose cold/[JIT](../../../../glossary.md#term-just-in-time-compilation-jit)/[admission](../../../../glossary.md#term-admission)/entry/[kernel](../../../../glossary.md#kernel) costs and [variance](../../../../glossary.md#term-variance);
no universal speed threshold or old measurement-budget reuse. Review reuses executable evidence.

## Documentation, review and result

Update affected Javadocs, [CPU](../../../../glossary.md#term-cpu) explanation and approved root/owning contracts/[ADR](../../../../glossary.md#term-architecture-decision-record-adr) in the same
change; targeted glossary review and applicable profiles. Independent Class C code/docs review
must verify saved-site eligibility and truthful capability, not infer it from [opcode](../../../../glossary.md#term-opcode) coverage.
Result empty until execution; record changed files, checks/skips/limits and follow-up, ending
`Status: Complete` or `Status: Incomplete` plus `Follow-up required: ...`.
