# Task 0019: Static materializing RESHAPE logical [layout](../../../../glossary.md#layout) closure

## Status

Complete — independent final source/[Javadoc](../../../../glossary.md#term-javadoc)/evidence/documentation APPROVE0; exact source is
integrated and full [Compiler](../../../../glossary.md#term-compiler)/actual captured-surface proof passed. [Backend](../../../../glossary.md#backend) execution is separate.

## Goal and change class

Class C. Close final logical geometry for valid ordinary materializing RESHAPE so its existing
value semantics can be consumed without [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/user copies or a [derivative](../../../../glossary.md#term-derivative) workaround. The actual
[Metal](../../../../glossary.md#term-metal) generated-[gradient](../../../../glossary.md#term-gradient) and [CPU](../../../../glossary.md#term-cpu) before consumers reject unresolved noncontiguous RESHAPE.

## Authority and bounded [scope](../../../../glossary.md#term-scope)

Read [root scope index](../../../../../ARCHITECTURE.md#scope-indexed-normative-contracts),
[Compiler contract](../../../../architecture/contracts/compiler-autograd.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>, and
[ADR0062](../../../../design/decisions/0062-ordinary-materializing-reshape-closure.md).
The [ADR](../../../../glossary.md#term-architecture-decision-record-adr) defines the exact [view](../../../../glossary.md#term-view-meanings)-first/materializing-result distinction and all exclusions.

Extend only StaticResultLogicalLayoutClosure's second [topological](../../../../glossary.md#term-topological-order) phase and its [Javadoc](../../../../glossary.md#term-javadoc)/tests.
Reuse existing LayoutInference and closeValue. Preserve inferred [dtype](../../../../glossary.md#data-type--datatype)/[Shape](../../../../glossary.md#shape), existing exact
views, [gradient](../../../../glossary.md#term-gradient) flags, source/caller [descriptors](../../../../glossary.md#term-descriptor), IDs, [topology](../../../../glossary.md#term-topology), constraints and sidecars. Do not
change raw [inference](../../../../glossary.md#term-inference), [Model](../../../../glossary.md#term-model), [VJP](../../../../glossary.md#term-vector-jacobian-product-vjp) math, source reconstruction, wire, scoped [Scan](../../../../glossary.md#term-cumulative-scan) [admission](../../../../glossary.md#term-admission), the
family-wide [canonical](../../../../glossary.md#term-canonical-form) classifier or [backend capability](../../../../glossary.md#backend-capability). CPU0079 and Metal0081 own consumers.

## [Ownership](../../../../glossary.md#term-ownership-meanings) and integration

- Depends on completed Compiler0009 and accepted ADR0062; CPU0078 is not a source prerequisite.
- MAIN owns implementation in a clean frozen context, docs, integration and executable checks.
- Production [owner](../../../../glossary.md#term-owner): [modules](../../../../glossary.md#term-module)/[compiler](../../../../glossary.md#term-compiler)/src/main/java/io/github/pho001/synaptik/compiler/StaticResultLogicalLayoutClosure.java.
- Test owner: corresponding existing StaticResultLogicalLayoutClosureTest; use existing inference/
  graph tests for regression. No copying, incidental wording/schema/count-pinning tests.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [graph glossary](../../../../glossary.md#graph); [regression glossary](../../../../glossary.md#term-regression); [schema glossary](../../../../glossary.md#term-schema))</span>
- Other Compiler source writers serialize. [CPU](../../../../glossary.md#term-cpu)/[Metal](../../../../glossary.md#term-metal) consumer slices are source-disjoint.
- Freeze HEAD plus complete dirty preimages before launch, independently review source, then
  integrate exact approved bytes. No commits/pushes or child checks.<span data-glossary-navigation="true"> ([commits glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings))</span>

## Acceptance and verification

1. Actual captured `[3,2]/[1,3] -> [6]` and `[5,3]/[1,5] -> [15]` obtain canonical non-view
   outputs only after ordinary validated cardinality; downstream views close in the same pass.<span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [view glossary](../../../../glossary.md#term-view-meanings); [cardinality glossary](../../../../glossary.md#term-cardinality))</span>
2. Existing resolved/offset views remain literal; matching family inference wins. Missing source
   layouts, dynamic Shapes and unrelated unresolved operations do not acquire invented geometry.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [layouts glossary](../../../../glossary.md#layout); [Shapes glossary](../../../../glossary.md#shape))</span>
3. Static scalar/empty and checked overflow follow ordinary rules; no Metal extent restriction is
   imported. Preserve raw source reconstruction and checkpoint v1-v5 bytes.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [overflow glossary](../../../../glossary.md#term-overflow); [Metal glossary](../../../../glossary.md#term-metal); [extent glossary](../../../../glossary.md#term-extent); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
4. MAIN runs the focused/complete Compiler tests and Javadoc/jar, plus the actual captured-graph
   descriptor probe. Then CPU0079/Metal0081 run real movement and unchanged generated-gradient
   consumers; Compiler-only success is not backend execution qualification.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Javadoc glossary](../../../../glossary.md#term-javadoc); [graph glossary](../../../../glossary.md#graph); [descriptor glossary](../../../../glossary.md#term-descriptor); [run glossary](../../../../glossary.md#term-run); [gradient glossary](../../../../glossary.md#term-gradient); [backend glossary](../../../../glossary.md#backend); [qualification glossary](../../../../glossary.md#term-qualification))</span>
5. Preserve qualified old CPU inventory keys/classes where descriptors do not change. Update final
   compile API/lifecycle explanation, task/master/changelog and relevant authority after proof.<span data-glossary-navigation="true"> ([CPU glossary](../../../../glossary.md#term-cpu); [compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>

## Result

Complete after independent final source, evidence and documentation acceptance. Only the two
scoped source/test files changed. ScanSemanticDesignReview's source/[Javadoc](../../../../glossary.md#term-javadoc) APPROVE0 confirms
the validated static-[cardinality](../../../../glossary.md#term-cardinality) [handoff](../../../../glossary.md#term-handoff), exact [view](../../../../glossary.md#term-view-meanings)-first path, preserved [metadata](../../../../glossary.md#term-metadata) and absence of
a second [shape](../../../../glossary.md#shape) algorithm or changed raw [inference](../../../../glossary.md#term-inference).

MAIN's isolated accepted-source command passed (artifact575/bg101, 6.02s): complete [Compiler](../../../../glossary.md#term-compiler)
418 tests, zero failures/errors/skips, Javadoc/jar; the approved37 [Metal](../../../../glossary.md#term-metal) jar was built only for
the pure [descriptor](../../../../glossary.md#term-descriptor) probe [classpath](../../../../glossary.md#term-classpath). The actual unchanged captured-[graph](../../../../glossary.md#graph) probe constructs all
15 original-typed [MATMUL](../../../../glossary.md#matrix-multiplication--matmul)/view graphs and ordinary joint VJPs. Both [transpose](../../../../glossary.md#term-transpose)-of-dense [leaf](../../../../glossary.md#term-leaf)-[gradient](../../../../glossary.md#term-gradient)
outputs now have [canonical](../../../../glossary.md#term-canonical-form) non-view geometry in [F32](../../../../glossary.md#floating-point-precision-and-rounding)/[BF16](../../../../glossary.md#floating-point-precision-and-rounding)/[F16](../../../../glossary.md#floating-point-precision-and-rounding). The old Metal37 provider still rejects
that [occurrence](../../../../glossary.md#term-occurrence), explicitly confirming that Compiler geometry is not [backend](../../../../glossary.md#backend) execution proof.

The five permanent regressions also pass all four floating gradient types and cover offset/
holed source, same-pass downstream view, raw unresolved inference, [scalar](../../../../glossary.md#term-scalar)/empty selection,
missing/dynamic input and checked canonical-[stride](../../../../glossary.md#term-stride) [overflow](../../../../glossary.md#term-overflow) even at zero [element](../../../../glossary.md#term-element) count.
MAIN verified final child [hashes](../../../../glossary.md#term-hash) and frozen MAIN preimages, then integrated the exact two files.
See compiler0019-source-files.json, compiler0019-actual-evidence.json, copied XML,
compiler0019-captured-descriptor-after.[log](../../../../glossary.md#term-log) and compiler0019-integrated-files.json in the retained
source-[snapshot](../../../../glossary.md#term-snapshot) [run](../../../../glossary.md#term-run). CPU0079/Metal0081 movement/[alias](../../../../glossary.md#aliasing-and-arena)/physical-order/[device](../../../../glossary.md#term-device) gates remain open;
no fallback, broader wire/cold domain or [release](../../../../glossary.md#term-release)-readiness claim follows.

## Launch freeze

MAIN verified [Ready](../../../../glossary.md#term-ready)/owning authority, both independent coordinated APPROVE0 receipts and the
closed authority-hierarchy finding. HEAD is `07a281edd3086efde2853211b71b48288db027e1`; the full
dirty source is the base, not HEAD alone. `reshape0062-launch-base.json` in MAIN's retained [run](../../../../glossary.md#term-run)
directory records 2,647 MAIN/candidate file [hashes](../../../../glossary.md#term-hash) and all 19 exact reviewed Metal37 overlay
preimages. No [Metal](../../../../glossary.md#term-metal) candidate is silently integrated into MAIN.

MAIN's clean context is `reshape-compiler-worktree`; production/test [ownership](../../../../glossary.md#term-ownership-meanings) is this packet's
two existing closure files. Disjoint Metal source contexts use the same frozen candidate and
unchanged [schema](../../../../glossary.md#term-schema) interface. CPU0078 is accepted Complete; CPU0079 still waits for this final
geometry. MAIN alone owns integration, docs, all checks and actual consumer/package evidence.

