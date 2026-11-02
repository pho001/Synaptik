# Task 0082: Native host-copy autorelease boundaries

## Status

Complete — the reviewed two-method fix is integrated in MAIN after source-matched native packaging,
unchanged actual before/after probes, full GPU gates and unchanged public MLP/CNN/raw movement.
Final independent source/evidence/documentation review APPROVE with zero P1/P2 findings.

## Class, owner and observed failure

Class C, within the existing [Metal native/lifecycle owner](../../../../architecture/contracts/backend-execution.md#metal-backend).
This corrects host-call resource lifetime, not program semantics, numerical kernels, supported
operations, ABI, wire, Java API, identities or cache keys.

The exact current38 library ed96fce02938c887ac4367cefd3bfa792f8d726788f9fa6af30dfd3883a59901
and native source5b6b8340fa13d56848412d8d137511efc9b19f11cd5b8e122817b196f293626f were used.
MetalLifetimePlateauSmoke executed six identical real public Engine windows with prepared reuse,
raw-zero gradients, BF16 custom work, MATMUL, reprepare, context churn, invalid-index recovery
and retained closed handles. Numerical/error paths passed; the memory gate did not. IOAccelerator
(graphics) dirty bytes grew54,345,728→321,339,392 and stayed321,306,624 after owner close. The
separate constant unmapped312,475,648-byte graphics allocation explains the overall close-time
footprint drop. Heap metadata grew only3.52→5.63MB. Do not call this a native plateau.

A different causal raw-FFM experiment uses one context and2,560explicitly created/released64KiB
buffers, with no Engine, Runtime, graphs, programs or GPU commands. Create/release-only stays at
81,920IOAccelerator bytes both with and without a caller-local Objective-C pool. Adding identical
upload/download calls without a pool grows33,636,352→167,854,080bytes; with a pool it stays81,920.
The difference is the caller-local pool, not reset, GC, hidden retry or omitted cleanup. That pool
exists only in the diagnostic; production must own its Foundation boundary.

Both native upload/download call Objective-C buffer.contents without a local pool. Preparation
and execution already use local pools. Preserve explicit bridge-retained buffer/context ownership.

## Exact correction

Wrap the existing synaptik_metal_buffer_upload and synaptik_metal_buffer_download implementations
in local @autoreleasepool scopes with their existing @try/@catch **inside** each pool. Clang does
not drain a pool on exceptional exit, so mapped exceptions must return normally through its boundary
([ARC specification](https://clang.llvm.org/docs/AutomaticReferenceCounting.html#autoreleasepool)).
Keep validation, zero-count/range behavior, memcpy ordering, exception/status mapping and opaque
handle ownership unchanged. Retained caller buffers must remain usable after the pool drains.

Only these two Foundation host-copy entrypoints change. Do not expand to context/create/release
when the causal create-only arms are flat; no Objective-C/runtime FFM helper or caller-pool API
ships in Java. No new export, source kernel, shader byte, schema field, identity or executor.

## Integration fields

- Depends on: integrated Complete Metal0080/0081 and Engine0023; actual controlled before evidence.
- Conflicts with: native .m writers and native source/package receipts. MAIN is the sole writer.
- Parallel group: RELEASE-METAL82, disjoint from isolated Model0047 descriptor factoring.
- Common base: exact dirty MAIN source snapshot captured after bounded review, not HEAD alone.
- Integration order: review/Ready → clean exact-source context → two-method fix → independent
  source/Javadoc review → rebuilt/signed/verified/clean-extracted package → unchanged raw/public
  before probes and complete affected GPU gates → final evidence/docs review.
- Integration validation: actual plain raw-copy isolation and unchanged mixed public lifetime
  workflow, including physical/IOAccelerator categories and heap; full Metal/Javadoc and affected
  unchanged typed/public integration against new source-matched native bytes. Keep old failures
  and local-signing/trust limits. A finite stable workload is not a universal allocation proof.

No worker builds/tests/lint/formatters/commands. MAIN owns implementation, source freeze, native
build/package/extract, actual GPU observations and final integration. No permanent source-text,
member/count or nondeterministic OS-footprint test. Reuse current behavioral suites and throwaway
actual consumers; remove scaffolding after accepted proof.

## Result

Only the existing upload/download Foundation boundaries changed. Their @try/@catch is inside
each @autoreleasepool so mapped exceptions return normally through the pool boundary. The first
outer-catch candidate was rejected before build and corrected; explicit opaque ownership,
validation, memcpy and status behavior remain unchanged. No Java pool helper or reset ships.

Native source SHA256 is d1c051e953d769386934d7ef0607db0fda487b16097482893221ec7910e04fbe.
The rebuilt, ad-hoc signed, verified, ZIP-extracted and reverified installed library SHA256 is
4c2d36d2a546e1db05000604a27fd06348fbd19b3aa946e3e0cee9415d3dd2d2.
ABI7/schema20/identity38 and all shader/kernel bytes remain unchanged. This is verified local
source-build installation, not an officially trusted public prebuilt.

The complete actual GPU gate passed 376 Metal tests with 10 opt-in observer skips and 50 public/typed
integration tests with zero skips, plus Javadocs/jars. The original 15 typed F32/BF16/F16 forward/
both-gradient graphs stayed unchanged. Unchanged public F32 MLP and CNN passed all-parameter
training/save/close/load/new training; exact raw materializing reshape passed all seven carriers.

The unchanged raw-copy probe without a caller-local pool now stays at 81,920 IOAccelerator dirty
bytes throughout all six observations instead of growing 33,636,352→167,854,080. The unchanged mixed
public probe stays at 999,424 bytes across six windows and 966,656 after close, instead of growing
54,345,728→321,339,392 and retaining 321,306,624 after close. It retained 3,288 closed handles, exercised
48 invalid-index recoveries, prepared reuse/reprepare, raw-zero gradients and context churn.
Full-process physical footprint includes warmup and intentionally retained Java metadata; only
the named graphics allocation category is claimed flat. Both before and after harnesses sample
quiescent OS footprint categories after the same explicit GC step; the production fix adds no GC.

Exact frozen source/install/first-failure/GPU/public/category evidence is retained in metal0082-*
and the original controlled four-arm receipts. This finite workload is not a universal allocation
proof; no native OOM or Objective-C exception was induced. Final independent source/evidence/docs
review APPROVE has zero P1/P2 findings. This completes Metal0082 only, not the production release
or its final source-matched root checkpoint.

