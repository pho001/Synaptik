# CPU0065: Vector MATMUL TANH epilogue

## Status and scope

Complete; MAIN accepted on 2026-10-28 after clean implementation, independent code/documentation APPROVE and the final gates below. User explicitly authorized TANH as the additional vector activation epilogue. Before launch MAIN verified CPU0064 Complete, HEAD=origin `07a281edd3086efde2853211b71b48288db027e1`, no conflicting agent/writer and the accepted dirty tree. Class C generated-code/capability change.
Extend the ordinary CPU MATMUL path, not a Scan-specific implementation. TANH already exists for scalar epilogues and pointwise vectors; reuse the existing typed vector mathematics. Enable only homogeneous FLOAT32/FLOAT64 TANH for eligible direct-N-vector and vector-2x2 realizations, with existing bias ADD order, scalar N tails and parallel range execution. Other vector terminals, low types, Scan routes, defaults, architecture boundaries and public APIs remain unchanged.

## Contracts and execution packet

Read root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants) and [scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts); owning [CPU backend routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes); Model [Family formulas and unary sites](../../../../architecture/contracts/foundational-modules.md#family-formulas-and-unary-sites) and [Floating arithmetic primitives](../../../../architecture/contracts/foundational-modules.md#floating-arithmetic-primitives). CPU prepare owns target lowering/selection; preserve represented arithmetic and TANH class/sign/domain rules. Stop on an authority conflict rather than changing architecture silently.
Source: `CpuMatmulIr` validation; `CpuMatmulLowering` and package-private `CpuMatmulCandidateSelector` in the same source file; `CpuMatmulEmitter` vector stores and scalar tails; existing `CpuVectorMath.tanh` / `CpuVectorInstructionEmitter`; generated/IR/lowering/cache tests and `CpuMatmulStrategyConformanceTest`. No new production type is planned. Read affected source/tests, not global histories.

- Depends on: CPU0064 Complete as stable accepted base; existing ordinary MATMUL/vector TANH implementations.
- Conflicts with: CPU codegen, MATMUL lowering/IR/cache, conformance and performance writers; Scan composition refactor is not launched.
- Parallel group: None; one implementation writer, independent read-only review, then serial documentation finalization.
- Common base revision: HEAD above plus accepted dirty tree; preserve all unrelated changes, no staging/commit/push.
- Integration order: implementation/focused validation → independent review → documentation finalization → CPU module/Javadoc and named conformance once after source freeze.
- Integration validation: `:backends:cpu:test :backends:cpu:javadoc` and `:testing:backend-conformance:test --tests '*CpuMatmulStrategyConformanceTest'`; no root suite or historical five-fork campaign.
- Shared-document integration owner: MAIN; reviewer finalizes targeted guide text only after implementation freeze.

## Acceptance and validation

Use the same specialized clean-Java vector algorithm as design oracle: full-K represented accumulation, ordered bias, existing vector TANH, one output store; scalar tail uses existing scalar TANH. No intermediate full Tensor, operation dispatch, per-element allocation or new activation formula. Test both floating types, direct/tiled vector, vector interior and tails, optional bias/both ADD orders, heap and MemorySegment carriers, layout/range preconditions, finite tolerance versus independent scalar reference, and separate NaN/Inf/signed-zero cases where meaningful. Unsupported terminals/dtypes/layouts retain their prior behavior.
Confirm real preparer selection and existing parallel range ownership through focused conformance; do not claim that this enables vector/parallel Scan. Inspect ClassFiles for actual vector activation and absence of avoidable allocation/helper fallback. Verify cache identity for newly valid forms; change only affected identity facts if needed, never an unproved global schema bump.
Focused tests first. Add at most a small targeted generated-versus-equivalent-vector-Java diagnostic with matched carriers/geometry and semantic checks; at most two short serial JVMs, no five-fork campaign, new benchmark framework, performance threshold gate, or guaranteed speed claim. Preserve historical benchmark evidence. Final affected-module/Javadoc and named conformance gate runs once after review fixes and final comments.
JDK: `/Users/phujka/Library/Java/JavaVirtualMachines/openjdk-26.0.1/Contents/Home`; use `--add-modules jdk.incubator.vector` for standalone Java. MAIN owns Gradle escalations if needed. Implementation reports source freeze before final gates.

## Documentation and completion

Review affected implementation Javadoc and CPU guide [Current portable MATMUL family](../../../../backend-guide/cpu-backend.md#current-portable-matmul-family), currently saying vector forms have no terminal. Independent pass follows documentation rules/profiles and targeted glossary search. No root or authority changes unless a concrete conflict appears. Keep an exact changed-file list and concise validation/evidence handoff. Finish with `Status: Complete`, or `Status: Incomplete` plus completed/missing work, blocker and specific follow-up.

## Result

Ordinary homogeneous FLOAT32/FLOAT64 direct-N/vector-2x2 MATMUL now admits TANH, with absent or unit-stride bias in either ordered ADD form. Full-K accumulation precedes bias, existing vector TANH and the sole store; scalar tails retain scalar TANH. Unsupported terminals/layouts/dtypes and scalar defaults remain unchanged. Existing canonical identity distinguishes newly valid forms; no schema bump, Scan route or public API change.
Changed production: `CpuMatmulIr`, `CpuMatmulLowering`/its candidate selector and `CpuMatmulEmitter`. Tests: IR, selector, lowering, new `CpuMatmulTanhEpilogueTest`, and `CpuMatmulStrategyConformanceTest`. Independent reviewer approved production/tests, inspected all 24 emitted ClassFiles for actual vector/scalar-tail TANH and no allocation/hidden fallback, and finalized the CPU guide/Javadoc. Targeted glossary review required no terminology addition.
Validation: 31 focused tests plus the new conformance method passed before review. Final `JAVA_HOME=<JDK26> ./gradlew :backends:cpu:test :backends:cpu:javadoc :testing:backend-conformance:test --tests '*CpuMatmulStrategyConformanceTest' --console=plain` PASS: CPU 1,378 tests/53 skipped/zero failures or errors; named conformance 5/5; Javadoc succeeds with 94 existing warnings. Scoped whitespace and preservation audit PASS; HEAD/origin/index and protected prior evidence unchanged.
One short heap-FLOAT32 diagnostic JVM passed exact matched-vector semantics; median generated/direct ratios 1.276 direct-N and 1.080 tiled. This retains a measured performance gap, not parity or a universal causal/speed claim. Report and ClassFiles: `/private/tmp/synaptik-cpu0065-evidence`; no further measurement or historical five-fork campaign. No commit/push.
Status: Complete. Required follow-up: none for bounded epilogue capability; performance parity and shared Scan-body composition are not claimed or automatically launched.
