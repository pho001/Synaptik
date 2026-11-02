# Task 0019: Static materializing RESHAPE logical layout closure

## Status

Complete — independent final source/Javadoc/evidence/documentation APPROVE0; exact source is
integrated and full Compiler/actual captured-surface proof passed. Backend execution is separate.

## Goal and change class

Class C. Close final logical geometry for valid ordinary materializing RESHAPE so its existing
value semantics can be consumed without NN/user copies or a derivative workaround. The actual
Metal generated-gradient and CPU before consumers reject unresolved noncontiguous RESHAPE.

## Authority and bounded scope

Read [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
[Compiler contract](../../../../architecture/contracts/compiler-autograd.md), and
[ADR0062](../../../../design/decisions/0062-ordinary-materializing-reshape-closure.md).
The ADR defines the exact view-first/materializing-result distinction and all exclusions.

Extend only StaticResultLogicalLayoutClosure's second topological phase and its Javadoc/tests.
Reuse existing LayoutInference and closeValue. Preserve inferred dtype/Shape, existing exact
views, gradient flags, source/caller descriptors, IDs, topology, constraints and sidecars. Do not
change raw inference, Model, VJP math, source reconstruction, wire, scoped Scan admission, the
family-wide canonical classifier or backend capability. CPU0079 and Metal0081 own consumers.

## Ownership and integration

- Depends on completed Compiler0009 and accepted ADR0062; CPU0078 is not a source prerequisite.
- MAIN owns implementation in a clean frozen context, docs, integration and executable checks.
- Production owner: modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/StaticResultLogicalLayoutClosure.java.
- Test owner: corresponding existing StaticResultLogicalLayoutClosureTest; use existing inference/
  graph tests for regression. No copying, incidental wording/schema/count-pinning tests.
- Other Compiler source writers serialize. CPU/Metal consumer slices are source-disjoint.
- Freeze HEAD plus complete dirty preimages before launch, independently review source, then
  integrate exact approved bytes. No commits/pushes or child checks.

## Acceptance and verification

1. Actual captured `[3,2]/[1,3] -> [6]` and `[5,3]/[1,5] -> [15]` obtain canonical non-view
   outputs only after ordinary validated cardinality; downstream views close in the same pass.
2. Existing resolved/offset views remain literal; matching family inference wins. Missing source
   layouts, dynamic Shapes and unrelated unresolved operations do not acquire invented geometry.
3. Static scalar/empty and checked overflow follow ordinary rules; no Metal extent restriction is
   imported. Preserve raw source reconstruction and checkpoint v1-v5 bytes.
4. MAIN runs the focused/complete Compiler tests and Javadoc/jar, plus the actual captured-graph
   descriptor probe. Then CPU0079/Metal0081 run real movement and unchanged generated-gradient
   consumers; Compiler-only success is not backend execution qualification.
5. Preserve qualified old CPU inventory keys/classes where descriptors do not change. Update final
   compile API/lifecycle explanation, task/master/changelog and relevant authority after proof.

## Result

Complete after independent final source, evidence and documentation acceptance. Only the two
scoped source/test files changed. ScanSemanticDesignReview's source/Javadoc APPROVE0 confirms
the validated static-cardinality handoff, exact view-first path, preserved metadata and absence of
a second shape algorithm or changed raw inference.

MAIN's isolated accepted-source command passed (artifact575/bg101, 6.02s): complete Compiler
418 tests, zero failures/errors/skips, Javadoc/jar; the approved37 Metal jar was built only for
the pure descriptor probe classpath. The actual unchanged captured-graph probe constructs all
15 original-typed MATMUL/view graphs and ordinary joint VJPs. Both transpose-of-dense leaf-gradient
outputs now have canonical non-view geometry in F32/BF16/F16. The old Metal37 provider still rejects
that occurrence, explicitly confirming that Compiler geometry is not backend execution proof.

The five permanent regressions also pass all four floating gradient types and cover offset/
holed source, same-pass downstream view, raw unresolved inference, scalar/empty selection,
missing/dynamic input and checked canonical-stride overflow even at zero element count.
MAIN verified final child hashes and frozen MAIN preimages, then integrated the exact two files.
See compiler0019-source-files.json, compiler0019-actual-evidence.json, copied XML,
compiler0019-captured-descriptor-after.log and compiler0019-integrated-files.json in the retained
source-snapshot run. CPU0079/Metal0081 movement/alias/physical-order/device gates remain open;
no fallback, broader wire/cold domain or release-readiness claim follows.

## Launch freeze

MAIN verified Ready/owning authority, both independent coordinated APPROVE0 receipts and the
closed authority-hierarchy finding. HEAD is `07a281edd3086efde2853211b71b48288db027e1`; the full
dirty source is the base, not HEAD alone. `reshape0062-launch-base.json` in MAIN's retained run
directory records 2,647 MAIN/candidate file hashes and all 19 exact reviewed Metal37 overlay
preimages. No Metal candidate is silently integrated into MAIN.

MAIN's clean context is `reshape-compiler-worktree`; production/test ownership is this packet's
two existing closure files. Disjoint Metal source contexts use the same frozen candidate and
unchanged schema interface. CPU0078 is accepted Complete; CPU0079 still waits for this final
geometry. MAIN alone owns integration, docs, all checks and actual consumer/package evidence.

