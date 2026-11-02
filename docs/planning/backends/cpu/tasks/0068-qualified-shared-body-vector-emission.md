# CPU0068: Qualify shared feature-vector body emission

> Superseded by CPU0070's [replacement plan](../existing-emitter-composition-plan.md#simd-admission-and-parallel-composition-gates).
> The original Draft below is archived, not executable. Shared existing SIMD mathematics is
> retained; Scan SIMD admission still requires bounded authority and whole-binder qualification.
> No diagnostic budget or scalar-then-vector reimplementation sequence carries forward.

## Status and change class

Superseded. The following pre-supersession status and remaining brief are archived:

Draft; no implementation frontier or performance grant. Class C: generated vector hot paths,
represented saved/state boundaries and bounded backend capability. Before Ready, CPU0067 must be
accepted and MAIN/user must approve the exact vector predicates and coordinated authority update.

## Goal, scope and exclusions

Qualify single-thread feature-lane pointwise and full-K/N-vector MATMUL through CPU0067's common
body seams for existing Scan occurrences. Ordinary vector bodies use those same seams; no second
Scan vector emitter. Reuse CPU0065 direct/tiled TANH where each required saved/state site qualifies.
The [proposal](../shared-body-emission-proposal.md#vector-realization-then-worker-integration)
defines boundaries. Preserve scalar execution for unproved cases, including scalar reductions
and transposed batch contractions without the required contiguous feature accesses.

No parallel submission/phases, K splitting, vector-reduction reassociation, new body operations,
layout/dtype widening, ordinary symbolic F64 MATMUL/epilogues, low/native/Metal/OpenBLAS Scan,
Compiler-body rewrites, NN packing/migration, tuning or public API/Runtime change.

## Contracts and approval gate

Read AGENTS; root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and
[scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts); backend
[CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes) and
[Accepted ordinary symbolic-batch CPU contract](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-symbolic-batch-cpu-contract);
Scan [Qualification, performance and migration gates](../../../../architecture/contracts/recurrent-scan.md#qualification-performance-and-migration-gates),
[Qualification and mechanism gates](../../../../architecture/contracts/recurrent-scan.md#qualification-and-mechanism-gates),
[Actual first-order closure and arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic),
[Activity, raw identity and retained histories](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories)
and [Runtime preflight and implementation gates](../../../../architecture/contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates).
Before Java, approve exact positive/negative occurrence/route predicates, changed artifact list
and bounded root/backend/Scan authority wording with independent review/MAIN acceptance.
Current vector prohibitions are not waived by this Draft. Missing/ambiguous contracts stop work.

## Dependencies and integration

- Depends on: CPU0067 Complete/accepted; CPU0065 retained; approved bounded vector authority.
- Conflicts with: common-body/Scan/vector/MATMUL/cache and CPU0048–0052 writers.
- Parallel group: None; clean implementation plus independent read-only review.
- Common base revision: CPU0067 accepted source on `07a281edd3086efde2853211b71b48288db027e1`
  plus accepted dirty tree; MAIN records fresh exact base/preservation/contracts before Ready.
- Integration order: accepted CPU0067 → approved authority/predicates → vector vertical/review
  → MAIN capability checkpoint → consider CPU0069. No automatic launch.
- Integration validation: CPU0068 checkpoint below, docs/links/whitespace/preservation.
- Shared-document integration owner: MAIN; exclusive delegation for owned explanation/Javadoc.

## Files and symbols

Prefix: `modules/backend/cpu/src/main/java/io/github/pho001/synaptik/backend/cpu/`.
Existing `CpuCapabilityProvider`; `internal/lowering/CpuGeneralScanLowering`, `CpuMatmulLowering`;
`internal/ir/CpuGeneralScanIr`, `CpuMatmulIr`; `internal/prepare/CpuPartitionPreparer.analyzeGeneralScan`,
`CpuPartitionFinalizer`; `internal/codegen/emit/CpuBodyBindings`, `CpuPointwiseBodyEmitter`,
`CpuLoopEmitter`, `CpuCarrierEmitter`, `CpuVectorInstructionEmitter`, `CpuVectorMath`,
`CpuMatmulEmitter`, `CpuGeneralScanEmitter`; `internal/cache/CpuKernelSpecialization`.
No new production type/package proposed beyond accepted CPU0067 types. If one is necessary,
stop for its exact name/package/purpose/signature approval. Execution remains one inline invocation.

## Acceptance criteria

1. Preparation selects typed per-stage vector/scalar realization from exact whole-body/binder
   qualification. Initial vector predicates require contiguous feature lanes, exact supported
   F32/F64 math, complete active lanes and injective writes; tails use shared scalar emission.
   No inactive operand is loaded/evaluated then masked. Preserve ROW(0)/ROW(1)/SHARED scopes.
2. Independently qualify existing static F32/F64, symbolic F32 and combined length F32/F64
   forward and actual Compiler siblings against ordinary unrolling, accounting for exact versus
   represented-sequential SUM and using matching-algorithm oracles for exact checks. Admission is unchanged
   outside the explicitly approved routes; all-valid symbolic F64 remains rejected. Ordinary
   `CpuMatmulIr.batchGeometry` remains F32/direct-scalar/no-epilogue; Scan numeric binding is separate.
3. Retain ordered full-K math, distinct per-time ADD, sequential SUM and exact outer SUM/MEAN.
   Saved/state values preserve exactly the bits produced by the selected forward realization
   and consumed by reverse; finite output tolerances alone do not prove this. Vector TANH/SIGMOID
   need not match scalar last bits. Retain scalar sites if their retained-value contract cannot
   be satisfied. Preserve fan-out,
   demanded histories, aliases/carry versions, raw singleton/empty/padding and zero-sign rules.
4. Prove common vector load/instruction/store and MATMUL bodies are used by ordinary and Scan
   callers. Species/setup is invocation-level; temporal addresses do not cause entry decoding,
   per-step allocation or hidden helpers. Preserve v9 scalar fallbacks and code-shape gains.
5. Test laneCount−1/laneCount/laneCount+1 and tiled tails, empty K/T/B/features, transposed reads,
   mixed length activity and poisoned padding, both seeds/traversals, tied parameters and reusable
   64→40→1→64. Exact scratch/history declarations remain complete before slots. No worker use.
6. Both identity streams distinguish changed vector/scalar/algorithm/binding facts, reject stale
   artifacts and reuse compatible code. Preserve unaffected bytes/keys per the accepted gate.

## Validation

Future focused development selection followed by one stabilized affected-module run:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScanCapabilityTest' --tests '*CpuGeneralScanGeneratedKernelTest' --tests '*CpuSymbolicScanGeneratedKernelTest' --tests '*CpuGeneralScanExecutionTest' --tests '*CpuMatmulTanhEpilogueTest' --tests '*CpuKernelSpecializationTest'
./gradlew :backends:cpu:test :backends:cpu:javadoc
```

MAIN runs the named CPU0067 conformance/Engine checkpoint commands once at this capability
cutover, with vector-positive/negative and actual-AD fixtures added to those existing tests.
Inspect generated ClassFiles against specialized clean vector Java, with exact saved sites and
scalar tails. Proposed fresh diagnostic: add `sharedBodyVectorDiagnostic` to existing
`CpuGeneralScanTimingTest`, comparing common scalar/vector and specialized clean Java for the
proposal's repeated MLP and combined RNN forward/actual reverse, T=4/B=7/F=3/H=17, both dtypes:

```bash
./gradlew :backends:cpu:test --tests '*CpuGeneralScanTimingTest.sharedBodyVectorDiagnostic' -Dsynaptik.cpu.generalScan.performance=true
```

Budget: one test JVM, no retry/forks, at most 120 seconds of diagnostic body; precheck exact
saved sites/specials and permitted finite results, then warmed interleaved matched-work samples.
Freeze/approve protocol before Ready. Disclose cold/JIT/admission/entry/kernel costs and variance;
no universal speed threshold or old measurement-budget reuse. Review reuses executable evidence.

## Documentation, review and result

Update affected Javadocs, CPU explanation and approved root/owning contracts/ADR in the same
change; targeted glossary review and applicable profiles. Independent Class C code/docs review
must verify saved-site eligibility and truthful capability, not infer it from opcode coverage.
Result empty until execution; record changed files, checks/skips/limits and follow-up, ending
`Status: Complete` or `Status: Incomplete` plus `Follow-up required: ...`.
