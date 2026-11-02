# Task 0081: Materializing RESHAPE composition and backward closure

## Status

Complete — exact independently reviewed cumulative twenty-file final38 source is integrated.
Source-matched native install, full Metal gate, unchanged real GPU consumers, current306-row
conformance78 and independent final source/evidence/documentation review passed.

## Goal and change class

Class C. Complete the ordinary materializing RESHAPE primitive required by Metal0080's unchanged
original-typed forward/both-gradient consumer. This is a separate explicit extension, not a silent
relaxation of Metal0080's MATMUL-only authority or a consumer CONTIGUOUS workaround.

## Authority and exact contract

Read [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
[Metal contract](../../../../architecture/contracts/backend-execution.md#metal-backend), and
[ADR0062](../../../../design/decisions/0062-ordinary-materializing-reshape-closure.md).
The owning [Compiler final-layout heading](../../../../architecture/contracts/compiler-autograd.md#accepted-ordinary-materializing-reshape-final-layout)
and [Backend extension heading](../../../../architecture/contracts/backend-execution.md#accepted-ordinary-materializing-reshape-extension)
are normative under the root. ADR0062 supplies coordinated decision detail and explanatory evidence,
not a competing authority. Retain their exact view/state/physical-source/kernel and domain bounds.

Preserve genuine affine RESHAPE views. New selected materialization has canonical non-view
output/CANONICAL state, densePhysical=false and an ordinary distinct destination; never pretend
it is an alias. Validate independently in original-typed provider, preparation, Java/native
preflight and actual execution. Use exact_move_unary_0066 op6, not a new shader or old0059 kernel.

Advance the six coordinated identities 37 -> 38 and reject valid stale37 handoffs/decisions.
Keep schema20/header124/ABI7/thirteen exports and fixed corpus unchanged. Preserve existing
route/dtype/gradient bounds, CUSTOM_PROGRAM-only low partitions, physical-versus-logical facts,
transpose fan-out/publication and all previous producer-forgery regressions. No new view elision,
source substitution, generic canonicalStorage widening, numeric math or dtype promotion.

## Dependencies, owners and integration

- Depends on accepted reviewed Metal0080 candidate bytes plus Compiler0019 final geometry for
  actual execution. Source slices may proceed against their frozen shared interface.
- CPU0079 is not a Metal source prerequisite. No CPU fallback is allowed.
- Parallel group RELEASE-RESHAPE. Java slice owns existing capability/preparer/program/NativeApi/
  fusion/identity owners, JVM tests and truthful MetalTestProgram helper. Native slice owns .m,
  existing RawAbi and AffineNative tests. Integration gradient test remains unchanged.
- Both isolated slices share one complete frozen candidate base: current MAIN plus all exact
  reviewed Metal0080 bytes. Record layered Main36 -> reviewed37 -> final38 preimages; never
  require MAIN to have unintegrated37 bytes or overwrite concurrent CPU source/docs.
- MAIN owns Compiler0019, serial CPU0079, shared docs and LowPrecisionCapabilityLedgerTest/basis/
  v2 rendering. Backend workers MUST NOT edit those shared files or snapshots. Preserve v1 and
  the accepted CPU lock; combined current rendering occurs only after applicable cutovers.
- No worker builds/tests/native loads/formatters. After independent source approval, MAIN combines
  exact bytes, builds/signs/packages/clean-installs the changed native source and runs GPU proof.
  Previous ad-hoc local package success does not certify the new source or public trust.

## Acceptance and verification

1. Unchanged all-15-graph F32/BF16/F16 five-form forward/both-generated-gradient integration passes
   with nonuniform seeds/fresh inputs and original leaf targets. Keep before failure FAILED.
2. Original-typed raw movement covers external strided and local materialized PERMUTE sources,
   offsets/holes, expected logical sequences, adversarial carrier bits, reuse, source preservation
   and target canaries. All admitted carriers preserve existing roles; no integral/BOOL adjoints.
3. Publication/PERMUTE fan-out to MATMUL and RESHAPE works in both consumer orders, applying
   effective source addressing exactly once. No changed consumer graph or handwritten gradient.
4. Independent Java and raw native forgery negatives cover state/offset/stride/span/count/target/
   type/gradient/provenance/plan membership and runtime alias/size/context. Keep manifests/digests
   valid for the altered claim so outer-integrity rejection cannot substitute for semantic proof.
5. MAIN runs full Metal/Javadoc/jar, affected current conformance, unchanged real MLP/CNN public
   compute/all-parameter SGD/save/close/load/new train and source-matched clean install. Report
   observer/other skips precisely. One final production repository checkpoint joins all changes.
6. Update actual support matrix/guide/master/changelog and owning contract statuses after proof.
   No claim of release readiness from source approval or the earlier 362-pass native gate.

## Result

MAIN integrated the exact reviewed37 prerequisites plus final38 bytes after the combined source
APPROVE0 and sole fixture P2 repair. No production P1/P2 remained. The frozen all-fifteen original-
typed five-form input graphs and MatmulNative tests are byte-identical to their reviewed inputs.

The exact reviewed native source was built, ad-hoc signed, packaged, ZIP-extracted outside the
checkout and reverified. Native source SHA is 5b6b8340fa13d56848412d8d137511efc9b19f11cd5b8e122817b196f293626f;
installed library SHA is ed96fce02938c887ac4367cefd3bfa792f8d726788f9fa6af30dfd3883a59901.
This is a local supported source-build package, not an official trusted prebuilt release.

The actual full Metal/Javadoc/jar gate passed 376 tests with ten opt-in observer skips; all sixteen
DtypeLayoutGradientMetalIntegrationTest cases passed, including the unchanged aggregate fifteen-
graph F32/BF16/F16 forward/both-generated-gradient corpus. RawAbi 37 and AffineNative seven cases
executed with no skips. The older approved37 aggregate failure remains FAILED, not relabeled.

Unchanged Metal-only public F32 MLP/CNN consumers passed scalar-oracle forward, all parameter
gradients/three SGD steps, save/close/load and fresh configured training. Public ordinary
PERMUTE -> RESHAPE passed exact logical order/raw bits for all seven carriers using Metal-only
Engine composition and the new installed library; only the existing driver's engine selector changed.

metal0081-actual-evidence.json, source-matched-install and exact integrated-files receipts retain
commands, skip identities and guards. Actual sole-owner v2 rendering adds sixteen distinct rows
and preserves all290original rows literally, historical v1 and CPU inventory unchanged. Full
conformance passed78tests after deleting the obsolete reflection/copy identity pin rather than
repinning it; its first failed79-test checkpoint and XML stay FAILED. New-header-dependent
Model0046 formatter work is staged separately and will land atomically with its source record.

Independent final source/evidence/docs review returned APPROVE0, closing the current-identity
and qualified MATMUL state descriptions. The joined repository/release gate remains open;
native-memory plateau and official prebuilt trust are not established by this component.

## Launch freeze

MAIN verified Ready/owning authority, both independent coordinated APPROVE0 receipts and the
closed authority-hierarchy P2. HEAD is `07a281edd3086efde2853211b71b48288db027e1`; HEAD alone
is not the base. `reshape0062-launch-base.json` records 2,647 full dirty MAIN/candidate hashes
and the complete exact 19-file reviewed37 overlay, including MAIN36 preimages for final cutover.

The clean contexts are `reshape-metal-java-worktree` and `reshape-metal-native-worktree`.
Java owns the listed fifteen existing Java/JVM/helper files; native owns .m, RawAbiNativeTest
and AffineNativeTest. The original DtypeLayoutGradientMetalIntegrationTest and approved
MetalMpsGraphMatmulNativeTest bytes are frozen acceptance inputs, owned by neither new slice.
Schema fields/flags, operation6 attributes and constructors remain the shared interface;
native fixtures declare canonical materialization explicitly without requiring new helper APIs.

MAIN's Compiler0019 context is source-disjoint. CPU0078 is Complete; CPU0079 remains serialized
behind final Compiler geometry. Workers run no checks; MAIN integrates only approved exact
source, then rebuilds and qualifies the changed native package and unchanged consumers.

