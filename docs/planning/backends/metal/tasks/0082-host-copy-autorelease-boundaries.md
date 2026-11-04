# Task 0082: Native [host](../../../../glossary.md#term-host)-copy autorelease boundaries

## Reader context

Upload/download crosses two ownership systems: Java/FFM borrows a live native handle, while Objective-C may create temporary Foundation objects when obtaining buffer contents. A local `@autoreleasepool` drains temporaries, not the explicitly retained buffer; the application need not own an Objective-C pool. Keeping `@try/@catch` inside permits mapped failures to return normally through its drain boundary.

Finite footprint receipts below isolate host-copy lifetime from graph execution. They are not a universal allocation proof, an exercised out-of-memory path or a trusted public release. This reader-context edit reruns none of those probes.

- [Current native buffer and run ownership](../../../../backend-guide/metal-backend.md#cold-run-setup-and-binding) — Separates retained handles from temporary host-call objects.
- [Current context and completed-result lifetime](../../../../backend-guide/metal-backend.md#completion-and-device-topology) — Explains why pool draining preserves owned buffers.

## Status

Complete — the reviewed two-method fix is integrated in MAIN after source-matched native packaging,
unchanged actual before/after probes, full [GPU](../../../../glossary.md#term-gpu) gates and unchanged public [MLP](../../../../glossary.md#term-multilayer-perceptron-mlp)/[CNN](../../../../glossary.md#term-convolutional-neural-network-cnn)/raw movement.
Final independent source/evidence/documentation review APPROVE with zero P1/P2 findings.

## Class, [owner](../../../../glossary.md#term-owner) and observed failure

Class C, within the existing [Metal native/lifecycle owner](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>.
This corrects [host](../../../../glossary.md#term-host)-call [resource lifetime](../../../../glossary.md#ownership-borrowing-and-lease), not program semantics, numerical [kernels](../../../../glossary.md#kernel), supported
operations, [ABI](../../../../glossary.md#term-application-binary-interface-abi), wire, Java [API](../../../../glossary.md#term-application-programming-interface-api), [identities](../../../../glossary.md#term-identity-meanings) or [cache](../../../../glossary.md#term-cache) keys.

The exact current38 library ed96fce02938c887ac4367cefd3bfa792f8d726788f9fa6af30dfd3883a59901
and native source5b6b8340fa13d56848412d8d137511efc9b19f11cd5b8e122817b196f293626f were used.
MetalLifetimePlateauSmoke executed six identical real public [Engine](../../../../glossary.md#term-engine) windows with prepared reuse,
raw-zero [gradients](../../../../glossary.md#term-gradient), [BF16](../../../../glossary.md#floating-point-precision-and-rounding) custom work, [MATMUL](../../../../glossary.md#matrix-multiplication--matmul), reprepare, context churn, invalid-index recovery
and retained closed handles. Numerical/error paths passed; the memory gate did not. IOAccelerator
(graphics) dirty bytes grew54,345,728→321,339,392 and stayed321,306,624 after owner close. The
separate [constant](../../../../glossary.md#term-constant) unmapped312,475,648-byte graphics allocation explains the overall close-time
footprint drop. [Heap](../../../../glossary.md#term-heap) [metadata](../../../../glossary.md#term-metadata) grew only3.52→5.63MB. Do not call this a native plateau.

A different causal raw-[FFM](../../../../glossary.md#term-ffm) experiment uses one context and2,560explicitly created/released64KiB
[buffers](../../../../glossary.md#term-buffer), with no Engine, [Runtime](../../../../glossary.md#term-runtime), [graphs](../../../../glossary.md#graph), programs or [GPU](../../../../glossary.md#term-gpu) commands. Create/[release](../../../../glossary.md#term-release)-only stays at
81,920IOAccelerator bytes both with and without a caller-local [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) pool. Adding identical
upload/download calls without a pool grows33,636,352→167,854,080bytes; with a pool it stays81,920.
The difference is the caller-local pool, not reset, [GC](../../../../glossary.md#term-garbage-collection-gc), hidden retry or omitted cleanup. That pool
exists only in the diagnostic; production must own its Foundation boundary.

Both native upload/download call Objective-C buffer.contents without a local pool. Preparation
and execution already use local pools. Preserve explicit bridge-retained buffer/context [ownership](../../../../glossary.md#term-ownership-meanings).

## Exact correction

Wrap the existing synaptik_metal_buffer_upload and synaptik_metal_buffer_download implementations
in local @autoreleasepool scopes with their existing @try/@catch **inside** each pool. Clang does
not drain a pool on exceptional exit, so mapped exceptions must return normally through its boundary
([ARC specification](https://clang.llvm.org/docs/AutomaticReferenceCounting.html#autoreleasepool)<span data-glossary-navigation="true"> ([ARC glossary](../../../../glossary.md#term-arc))</span>).
Keep [validation](../../../../glossary.md#term-validation), zero-count/range behavior, memcpy ordering, exception/status mapping and [opaque](../../../../glossary.md#term-opaque)
handle [ownership](../../../../glossary.md#term-ownership-meanings) unchanged. Retained caller [buffers](../../../../glossary.md#term-buffer) must remain usable after the pool drains.

Only these two Foundation [host](../../../../glossary.md#term-host)-copy entrypoints change. Do not expand to context/create/[release](../../../../glossary.md#term-release)
when the causal create-only arms are flat; no [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary)/runtime [FFM](../../../../glossary.md#term-ffm) helper or caller-pool [API](../../../../glossary.md#term-application-programming-interface-api)
ships in Java. No new export, source [kernel](../../../../glossary.md#kernel), [shader](../../../../glossary.md#term-shader) byte, [schema](../../../../glossary.md#term-schema) field, [identity](../../../../glossary.md#term-identity-meanings) or executor.

## Integration fields

- Depends on: integrated Complete Metal0080/0081 and Engine0023; actual controlled before evidence.
- Conflicts with: native .m writers and native source/package receipts. MAIN is the sole writer.
- Parallel group: [RELEASE](../../../../glossary.md#term-release)-METAL82, disjoint from isolated Model0047 [descriptor](../../../../glossary.md#term-descriptor) factoring.
- Common base: exact dirty MAIN source [snapshot](../../../../glossary.md#term-snapshot) captured after bounded review, not HEAD alone.
- Integration order: review/Ready → clean exact-source context → two-method fix → independent
  source/Javadoc review → rebuilt/signed/verified/clean-extracted package → unchanged raw/public
  before probes and complete affected GPU gates → final evidence/docs review.<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [Javadoc glossary](../../../../glossary.md#term-javadoc))</span>
- Integration validation: actual plain raw-copy isolation and unchanged mixed public lifetime
  workflow, including physical/IOAccelerator categories and heap; full Metal/Javadoc and affected
  unchanged typed/public integration against new source-matched native bytes. Keep old failures
  and local-signing/trust limits. A finite stable workload is not a universal allocation proof.<span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [lifetime glossary](../../../../glossary.md#term-lifetime); [heap glossary](../../../../glossary.md#term-heap); [Metal glossary](../../../../glossary.md#term-metal); [finite glossary](../../../../glossary.md#term-finite))</span>

No worker builds/tests/lint/formatters/commands. MAIN owns implementation, source freeze, native
build/package/extract, actual [GPU](../../../../glossary.md#term-gpu) observations and final integration. No permanent source-text,
member/count or nondeterministic OS-footprint test. Reuse current behavioral suites and throwaway
actual consumers; remove scaffolding after accepted proof.

## Result

Only the existing upload/download Foundation boundaries changed. Their @try/@catch is inside
each @autoreleasepool so mapped exceptions return normally through the pool boundary. The first
outer-catch candidate was rejected before build and corrected; explicit [opaque](../../../../glossary.md#term-opaque) [ownership](../../../../glossary.md#term-ownership-meanings),
[validation](../../../../glossary.md#term-validation), memcpy and status behavior remain unchanged. No Java pool helper or reset ships.

Native source SHA256 is d1c051e953d769386934d7ef0607db0fda487b16097482893221ec7910e04fbe.
The rebuilt, ad-hoc signed, verified, ZIP-extracted and reverified installed library SHA256 is
4c2d36d2a546e1db05000604a27fd06348fbd19b3aa946e3e0cee9415d3dd2d2.
ABI7/schema20/identity38 and all [shader](../../../../glossary.md#term-shader)/[kernel](../../../../glossary.md#kernel) bytes remain unchanged. This is verified local
source-build installation, not an officially trusted public prebuilt.

The complete actual [GPU](../../../../glossary.md#term-gpu) gate passed 376 [Metal](../../../../glossary.md#term-metal) tests with 10 opt-in [observer](../../../../glossary.md#term-observer) skips and 50 public/typed
integration tests with zero skips, plus Javadocs/jars. The original 15 typed [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[BF16](../../../../glossary.md#floating-point-precision-and-rounding)/[F16](../../../../glossary.md#floating-point-precision-and-rounding) forward/
both-[gradient](../../../../glossary.md#term-gradient) [graphs](../../../../glossary.md#graph) stayed unchanged. Unchanged public F32 [MLP](../../../../glossary.md#term-multilayer-perceptron-mlp) and [CNN](../../../../glossary.md#term-convolutional-neural-network-cnn) passed all-[parameter](../../../../glossary.md#term-parameter)
training/save/close/load/new training; exact raw materializing reshape passed all seven carriers.

The unchanged raw-copy probe without a caller-local pool now stays at 81,920 IOAccelerator dirty
bytes throughout all six observations instead of growing 33,636,352→167,854,080. The unchanged mixed
public probe stays at 999,424 bytes across six windows and 966,656 after close, instead of growing
54,345,728→321,339,392 and retaining 321,306,624 after close. It retained 3,288 closed handles, exercised
48 invalid-index recoveries, prepared reuse/reprepare, raw-zero gradients and context churn.
Full-process physical footprint includes [warmup](../../../../glossary.md#term-warmup) and intentionally retained Java [metadata](../../../../glossary.md#term-metadata); only
the named graphics allocation category is claimed flat. Both before and after harnesses [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters)
[quiescent](../../../../glossary.md#term-quiescent) OS footprint categories after the same explicit [GC](../../../../glossary.md#term-garbage-collection-gc) step; the production fix adds no GC.

Exact frozen source/install/first-failure/GPU/public/category evidence is retained in metal0082-*
and the original controlled four-arm receipts. This [finite](../../../../glossary.md#term-finite) workload is not a universal allocation
proof; no native [OOM](../../../../glossary.md#term-out-of-memory-oom) or [Objective-C](../../../../glossary.md#native-packaging-and-platform-vocabulary) exception was induced. Final independent source/evidence/docs
review APPROVE has zero P1/P2 findings. This completes Metal0082 only, not the production [release](../../../../glossary.md#term-release)
or its final source-matched root checkpoint.

