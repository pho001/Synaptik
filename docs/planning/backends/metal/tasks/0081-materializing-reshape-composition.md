# Task 0081: Materializing RESHAPE composition and backward closure

## Reader context

RESHAPE is a view when new coordinates correctly address existing storage, or materialization when logical order requires copying. Materialization needs a distinct canonical destination and raw movement; it cannot pretend to alias or let MATMUL reinterpret an unrelated ancestor. Compiler supplies final geometry; Metal independently checks physical source/destination.

Unchanged original-typed derivative graphs and leaf targets matter: handwritten derivatives or consumer copies would test a different problem. Here a leaf target is an original input/parameter Tensor selected for differentiation, not a node inside the copy implementation. Receipts qualify this component only, with their stated release limits.

- [Current materialization and view authentication](../../../../backend-guide/metal-backend.md#physical-authentication-and-fixed-routes) — Separates canonical destinations from source aliases.
- [Current generated MATMUL consumers](../../../../backend-guide/metal-backend.md#general-static-matmul-and-generated-gradients) — Connects the repaired primitive to unchanged backward graphs.

## Status

Complete — exact independently reviewed cumulative twenty-file final38 source is integrated.
Source-matched native install, full [Metal](../../../../glossary.md#term-metal) gate, unchanged real [GPU](../../../../glossary.md#term-gpu) consumers, current306-row
conformance78 and independent final source/evidence/documentation review passed.

## Goal and change class

Class C. Complete the ordinary materializing RESHAPE primitive required by Metal0080's unchanged
original-typed forward/both-[gradient](../../../../glossary.md#term-gradient) consumer. This is a separate explicit extension, not a silent
relaxation of Metal0080's [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)-only authority or a consumer [CONTIGUOUS](../../../../glossary.md#term-contiguous) workaround.

## Authority and exact contract

Read [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts)<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>,
[Metal contract](../../../../architecture/contracts/backend-execution.md#metal-backend)<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal))</span>, and
[ADR0062](../../../../design/decisions/0062-ordinary-materializing-reshape-closure.md).
The owning [Compiler final-layout heading](../../../../architecture/contracts/compiler-autograd.md#accepted-ordinary-materializing-reshape-final-layout)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [layout glossary](../../../../glossary.md#layout))</span>
and [Backend extension heading](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-materializing-reshape-extension)<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>
are [normative](../../../../glossary.md#term-normative) under the root. ADR0062 supplies coordinated decision detail and explanatory evidence,
not a competing authority. Retain their exact [view](../../../../glossary.md#term-view-meanings)/state/physical-source/[kernel](../../../../glossary.md#kernel) and domain bounds.

Preserve genuine affine RESHAPE views. New selected [materialization](../../../../glossary.md#materialization) has [canonical](../../../../glossary.md#term-canonical-form) non-view
output/CANONICAL state, densePhysical=false and an ordinary distinct destination; never pretend
it is an [alias](../../../../glossary.md#aliasing-and-arena). Validate independently in original-typed provider, preparation, Java/native
[preflight](../../../../glossary.md#term-preflight) and actual execution. Use exact_move_unary_0066 op6, not a new [shader](../../../../glossary.md#term-shader) or old0059 kernel.

Advance the six coordinated [identities](../../../../glossary.md#term-identity-meanings) 37 -> 38 and reject valid stale37 handoffs/decisions.
Keep schema20/header124/ABI7/thirteen exports and fixed corpus unchanged. Preserve existing
route/[dtype](../../../../glossary.md#data-type--datatype)/[gradient](../../../../glossary.md#term-gradient) bounds, CUSTOM_PROGRAM-only low partitions, physical-versus-logical facts,
[transpose](../../../../glossary.md#term-transpose) [fan-out](../../../../glossary.md#term-fan-out)/[publication](../../../../glossary.md#term-publication) and all previous [producer](../../../../glossary.md#term-producer)-forgery regressions. No new view elision,
source substitution, generic canonicalStorage [widening](../../../../glossary.md#term-widening), numeric math or dtype promotion.

## Dependencies, [owners](../../../../glossary.md#term-owner) and integration

- Depends on accepted reviewed Metal0080 candidate bytes plus Compiler0019 final geometry for
  actual execution. Source slices may proceed against their frozen shared interface.
- CPU0079 is not a [Metal](../../../../glossary.md#term-metal) source prerequisite. No [CPU](../../../../glossary.md#term-cpu) fallback is allowed.
- Parallel group RELEASE-RESHAPE. Java slice owns existing capability/preparer/program/NativeApi/
  fusion/identity owners, JVM tests and truthful MetalTestProgram helper. Native slice owns .m,
  existing RawAbi and AffineNative tests. Integration gradient test remains unchanged.<span data-glossary-navigation="true"> ([RELEASE glossary](../../../../glossary.md#term-release); [slice glossary](../../../../glossary.md#slice); [fusion glossary](../../../../glossary.md#term-fusion); [identity glossary](../../../../glossary.md#term-identity-meanings); [JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [gradient glossary](../../../../glossary.md#term-gradient))</span>
- Both isolated slices share one complete frozen candidate base: current MAIN plus all exact
  reviewed Metal0080 bytes. Record layered Main36 -> reviewed37 -> final38 preimages; never
  require MAIN to have unintegrated37 bytes or overwrite concurrent CPU source/docs.
- MAIN owns Compiler0019, serial CPU0079, shared docs and LowPrecisionCapabilityLedgerTest/basis/
  v2 rendering. Backend workers MUST NOT edit those shared files or snapshots. Preserve v1 and
  the accepted CPU lock; combined current rendering occurs only after applicable cutovers.<span data-glossary-navigation="true"> ([Backend glossary](../../../../glossary.md#backend))</span>
- No worker builds/tests/native loads/formatters. After independent source approval, MAIN combines
  exact bytes, builds/signs/packages/clean-installs the changed native source and runs GPU proof.
  Previous ad-hoc local package success does not certify the new source or public trust.<span data-glossary-navigation="true"> ([GPU glossary](../../../../glossary.md#term-gpu))</span>

## Acceptance and verification

1. Unchanged all-15-graph F32/BF16/F16 five-form forward/both-generated-gradient integration passes
   with nonuniform seeds/fresh inputs and original leaf targets. Keep before failure FAILED.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [BF16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F16 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [gradient glossary](../../../../glossary.md#term-gradient); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [leaf glossary](../../../../glossary.md#term-leaf); [targets glossary](../../../../glossary.md#term-target))</span>
2. Original-typed raw movement covers external strided and local materialized PERMUTE sources,
   offsets/holes, expected logical sequences, adversarial carrier bits, reuse, source preservation
   and target canaries. All admitted carriers preserve existing roles; no integral/BOOL adjoints.<span data-glossary-navigation="true"> ([carrier glossary](../../../../glossary.md#term-storage-carrier); [BOOL glossary](../../../../glossary.md#data-type--datatype))</span>
3. Publication/PERMUTE fan-out to MATMUL and RESHAPE works in both consumer orders, applying
   effective source addressing exactly once. No changed consumer graph or handwritten gradient.<span data-glossary-navigation="true"> ([Publication glossary](../../../../glossary.md#term-publication); [fan-out glossary](../../../../glossary.md#term-fan-out); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul))</span>
4. Independent Java and raw native forgery negatives cover state/offset/stride/span/count/target/
   type/gradient/provenance/plan membership and runtime alias/size/context. Keep manifests/digests
   valid for the altered claim so outer-integrity rejection cannot substitute for semantic proof.<span data-glossary-navigation="true"> ([stride glossary](../../../../glossary.md#term-stride); [provenance glossary](../../../../glossary.md#provenance); [runtime glossary](../../../../glossary.md#term-runtime); [alias glossary](../../../../glossary.md#aliasing-and-arena); [integrity glossary](../../../../glossary.md#term-integrity))</span>
5. MAIN runs full Metal/Javadoc/jar, affected current conformance, unchanged real MLP/CNN public
   compute/all-parameter SGD/save/close/load/new train and source-matched clean install. Report
   observer/other skips precisely. One final production repository checkpoint joins all changes.<span data-glossary-navigation="true"> ([Metal glossary](../../../../glossary.md#term-metal); [Javadoc glossary](../../../../glossary.md#term-javadoc); [conformance glossary](../../../../glossary.md#term-conformance); [MLP glossary](../../../../glossary.md#term-multilayer-perceptron-mlp); [CNN glossary](../../../../glossary.md#term-convolutional-neural-network-cnn); [parameter glossary](../../../../glossary.md#term-parameter); [SGD glossary](../../../../glossary.md#term-stochastic-gradient-descent-sgd); [observer glossary](../../../../glossary.md#term-observer))</span>
6. Update actual support matrix/guide/master/changelog and owning contract statuses after proof.
   No claim of release readiness from source approval or the earlier 362-pass native gate.<span data-glossary-navigation="true"> ([support matrix glossary](../../../../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([matrix glossary](../../../../glossary.md#term-review-matrix); [release glossary](../../../../glossary.md#term-release))</span>

## Result

MAIN integrated the exact reviewed37 prerequisites plus final38 bytes after the combined source
APPROVE0 and sole [fixture](../../../../glossary.md#term-fixture) P2 repair. No production P1/P2 remained. The frozen all-fifteen original-
typed five-form input [graphs](../../../../glossary.md#graph) and MatmulNative tests are byte-identical to their reviewed inputs.

The exact reviewed native source was built, ad-hoc signed, packaged, ZIP-extracted outside the
checkout and reverified. Native source SHA is 5b6b8340fa13d56848412d8d137511efc9b19f11cd5b8e122817b196f293626f;
installed library SHA is ed96fce02938c887ac4367cefd3bfa792f8d726788f9fa6af30dfd3883a59901.
This is a local supported source-build package, not an official trusted prebuilt [release](../../../../glossary.md#term-release).

The actual full [Metal](../../../../glossary.md#term-metal)/[Javadoc](../../../../glossary.md#term-javadoc)/jar gate passed 376 tests with ten opt-in [observer](../../../../glossary.md#term-observer) skips; all sixteen
DtypeLayoutGradientMetalIntegrationTest cases passed, including the unchanged aggregate fifteen-
graph [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[BF16](../../../../glossary.md#floating-point-precision-and-rounding)/[F16](../../../../glossary.md#floating-point-precision-and-rounding) forward/both-generated-[gradient](../../../../glossary.md#term-gradient) corpus. RawAbi 37 and AffineNative seven cases
executed with no skips. The older approved37 aggregate failure remains FAILED, not relabeled.

Unchanged Metal-only public F32 [MLP](../../../../glossary.md#term-multilayer-perceptron-mlp)/[CNN](../../../../glossary.md#term-convolutional-neural-network-cnn) consumers passed [scalar](../../../../glossary.md#term-scalar)-[oracle](../../../../glossary.md#term-test-oracle) forward, all [parameter](../../../../glossary.md#term-parameter)
gradients/three [SGD](../../../../glossary.md#term-stochastic-gradient-descent-sgd) steps, save/close/load and fresh configured training. Public ordinary
PERMUTE -> RESHAPE passed exact logical order/[raw bits](../../../../glossary.md#storage-mapping-contiguity-and-injectivity) for all seven carriers using Metal-only
[Engine](../../../../glossary.md#term-engine) composition and the new installed library; only the existing driver's engine selector changed.

metal0081-actual-evidence.json, source-matched-install and exact integrated-files receipts retain
commands, skip [identities](../../../../glossary.md#term-identity-meanings) and guards. Actual sole-[owner](../../../../glossary.md#term-owner) v2 rendering adds sixteen distinct rows
and preserves all290original rows literally, historical v1 and [CPU](../../../../glossary.md#term-cpu) inventory unchanged. Full
[conformance](../../../../glossary.md#term-conformance) passed78tests after deleting the obsolete reflection/copy [identity](../../../../glossary.md#term-identity-meanings) pin rather than
repinning it; its first failed79-test checkpoint and XML stay FAILED. New-header-dependent
Model0046 formatter work is staged separately and will land atomically with its source record.

Independent final source/evidence/docs review returned APPROVE0, closing the current-identity
and qualified [MATMUL](../../../../glossary.md#matrix-multiplication--matmul) state descriptions. The joined repository/release gate remains open;
native-memory plateau and official prebuilt trust are not established by this component.

## Launch freeze

MAIN verified [Ready](../../../../glossary.md#term-ready)/owning authority, both independent coordinated APPROVE0 receipts and the
closed authority-hierarchy P2. HEAD is `07a281edd3086efde2853211b71b48288db027e1`; HEAD alone
is not the base. `reshape0062-launch-base.json` records 2,647 full dirty MAIN/candidate [hashes](../../../../glossary.md#term-hash)
and the complete exact 19-file reviewed37 overlay, including MAIN36 preimages for final cutover.

The clean contexts are `reshape-metal-java-worktree` and `reshape-metal-native-worktree`.
Java owns the listed fifteen existing Java/[JVM](../../../../glossary.md#term-java-virtual-machine-jvm)/helper files; native owns .m, RawAbiNativeTest
and AffineNativeTest. The original DtypeLayoutGradientMetalIntegrationTest and approved
MetalMpsGraphMatmulNativeTest bytes are frozen acceptance inputs, owned by neither new [slice](../../../../glossary.md#slice).
[Schema](../../../../glossary.md#term-schema) fields/flags, operation6 attributes and constructors remain the shared interface;
native fixtures declare [canonical](../../../../glossary.md#term-canonical-form) [materialization](../../../../glossary.md#materialization) explicitly without requiring new helper APIs.

MAIN's Compiler0019 context is source-disjoint. CPU0078 is Complete; CPU0079 remains serialized
behind final [Compiler](../../../../glossary.md#term-compiler) geometry. Workers [run](../../../../glossary.md#term-run) no checks; MAIN integrates only approved exact
source, then rebuilds and qualifies the changed native package and unchanged consumers.

