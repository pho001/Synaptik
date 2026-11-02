# CPU0064: Port the measured clean-Java Scan shape

## Status and execution packet

Complete; MAIN accepted on 2026-10-28 after clean implementation, independent code/documentation review and the gates below. User explicitly authorized the generator port after the reviewed J4 experiment; MAIN verified accepted v7 sources, dependencies, empty index/base and no conflicting writer. Class C generated-hot-path change.

- Depends on: CPU0060/Prepare0015 accepted; CPU0063 J4 experiment completed and reviewed (plans were intentionally not maintained during tuning).
- Conflicts with: other Scan/source/build/performance writers.
- Parallel group: None; implementation and read-only review only.
- Common base revision: `07a281edd3086efde2853211b71b48288db027e1` plus accepted dirty tree; preserve unrelated changes. No commit/push requested.
- Integration order: implement/compile → generated source inspection and short matched measurement → focused semantic/cache/structural tests → CPU module/Javadoc → review/docs.
- Integration validation: fresh generated ClassFiles versus G7/J4/E; focused combined F32/F64 cases and existing noncombined byte pins, then affected-module validation once. No full repository suite.
- Shared-document integration owner: MAIN; reviewer finalizes only affected implementation docs/Javadoc.

## Scope and contracts

CPU `CpuGeneralScanEmitter`, `CpuGeneralScanMatmulEmitter`, `CpuGeneralScanReductionEmitter`, `CpuGeneralScanIr` and focused generated tests. Add a focused generation-only type only if it has a concrete responsibility; no runtime helper, javac pipeline or benchmark-specific hardcoding.
Read root [Core invariants](../../../../../ARCHITECTURE.md#core-invariants)/[scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts), owning [CPU routes](../../../../architecture/contracts/backend-execution.md#cpu-backend-routes), and Scan [arithmetic](../../../../architecture/contracts/recurrent-scan.md#actual-first-order-closure-and-arithmetic)/[activity](../../../../architecture/contracts/recurrent-scan.md#activity-raw-identity-and-retained-histories). Existing geometry, memory ownership, precision and capability contracts do not change.

## Acceptance

Design oracle: `/private/tmp/synaptik-scan-java-abi-lab.K8KBvk/ScanJ4.java`; original E remains unchanged. Reproduce its loop/dataflow shape generically from proved IR: direct stable carriers/zero bases, actual runtime B/T/scalars, affine geometry, int-proved row/feature loops, direct indexed contractions, early activity guard and dependency/alias-safe scheduling. Preserve every represented float/double product/add, per-time carry ADD, raw singleton/empty/padding and saved histories. Do not merely repeat the smaller changes that failed in J0–J3; inspect the actual new generated body.
No unconditional stage reorder across scratch reuse/lifetimes. Prove applicability and retain generic fallback for unproved cases, without hardcoding F3/H2, RNN-TANH or benchmark slots. Keep noncombined byte streams unchanged. Advance only the affected baked identity, skipping withdrawn experimental v8; preserve canonical IR, numeric pack and schema 71 unless a concrete conflict is reported first.
Compile/generate and measure before running functional suites, as requested. Reuse off-tree benchmark mechanics; fresh generated/G7/J4/E in matched short serial JVMs, unchanged inputs and latest raw checks. No large validation framework. Worker must pause for MAIN measurement before semantic/module suites. Off-tree work root: `/private/tmp/synaptik-scan-generator-port.AVKbte`.
Update affected implementation Javadoc; independent review checks actual algorithm, bytes, cache invalidation and documentation. Keep root/owning contract changes limited to current internal baked-version facts if needed; no new architecture/ADR unless an actual decision is required. Report changed files, measurements/tests, remaining issues, and `Status: Complete` or `Status: Incomplete` with exact follow-up.

## Result

Combined baked v9 implements the proved carrier/base, affine geometry, indexed contraction and safe stage-scheduling changes; failed proofs retain existing paths. Added generation-only `CpuGeneralScanStageSchedule` and seven focused proof tests; updated emitter/IR/cache/structural tests and implementation documentation. Canonical IR, numeric pack, schema 71, API and scalar/single-range capability are unchanged; all 20 noncombined generation controls retain their bytes and identities. Independent review found no remaining executable-code blockers after conservative physical-destination and overflow proof fixes.
Two matched JVMs at T4/F3/H2 observed new reverse entry medians of 633/650 ns (B7) and 2167/2184 ns (B32), approximately 10–12%/25% below v7 and 1–3% above unchanged Java E. Entry ABIs differ from E; no pure-body, other-shape/dtype or Engine parity claim. Retained evidence: `/private/tmp/synaptik-scan-generator-port.AVKbte` (`analysis-results.json`, generated/post-review ClassFiles, `GNEW.javap.txt`, raw/JIT logs); post-review measured ClassFile bytes are unchanged.
Validation: CPU module 1,371 tests, 52 skipped, zero failures/errors; final CPU Javadoc PASS (94 existing warnings); named CPU symbolic-Scan conformance 2/2 and Engine integration 2/2 PASS. Documentation/Javadoc reviewed and updated in root, owning backend contract and CPU guide. Unrelated dirty files, HEAD/origin/index and protected prior evidence preserved. No commit/push, new timing campaign or automatic successor. Status: Complete. Required follow-up: none for this port; vector/parallel Scan remains outside its capability.
