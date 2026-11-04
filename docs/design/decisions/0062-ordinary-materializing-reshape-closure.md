# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0062: Ordinary materializing RESHAPE closure


## Editorial reading context

A reshape changes logical geometry. If existing strides cannot represent it as a view, the qualified CPU path materializes a new canonical result instead of inventing contiguous metadata for the input. Compiler closes proven result descriptors; backend lowering realizes the copy. That is distinct from making every layout or reshape source persistable.

Use the [logical result closure versus physical materialization](../../architecture/lifecycle.md#compile-lifecycle) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted source direction — independent [Compiler](../../glossary.md#term-compiler)/[Metal](../../glossary.md#term-metal) design reviews APPROVE0. This decision
is not an execution or [release](../../glossary.md#term-release)-readiness [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions). Compiler0019, CPU0079 and Metal0081 own
implementation and [qualification](../../glossary.md#term-qualification). CPU0079 waits for CPU0078's completed cutover.

## Context

Ordinary [Model](../../glossary.md#term-model) RESHAPE accepts equal-[cardinality](../../glossary.md#term-cardinality) value semantics. Model construction and raw
[Compiler](../../glossary.md#term-compiler) [inference](../../glossary.md#term-inference) prove an exact [view](../../glossary.md#term-view-meanings) only for a present [contiguous](../../glossary.md#term-contiguous) source [layout](../../glossary.md#layout); otherwise
the result remains unresolved. That does not prove a view or reject [materialization](../../glossary.md#materialization).

The source-matched Metal0080 candidate passes the unchanged full [F32](../../glossary.md#floating-point-precision-and-rounding) [MLP](../../glossary.md#term-multilayer-perceptron-mlp) and [CNN](../../glossary.md#term-convolutional-neural-network-cnn) public
compute/three-[SGD](../../glossary.md#term-stochastic-gradient-descent-sgd)/save/close/load/new-train consumers. Its new original-typed five-form [MATMUL](../../glossary.md#matrix-multiplication--matmul)
backward test still fails: the [transpose](../../glossary.md#term-transpose)-of-dense form produces PERMUTE -> RESHAPE [gradients](../../glossary.md#term-gradient)
`[3,2]/[1,3] -> [6]` and `[5,3]/[1,5] -> [15]`. Final closure leaves these results unresolved;
[Metal](../../glossary.md#term-metal) rejects them. A separate public [CPU](../../glossary.md#term-cpu) PERMUTE -> RESHAPE before consumer also stops at
its first F32 [ownership](../../glossary.md#term-ownership-meanings) query. No CPU seven-type execution follows from that attempt.

Retain the before failures and separate successful public journeys. Do not reduce gradient
[targets](../../glossary.md#term-target), insert consumer CONTIGUOUS operations, replace ordinary [derivatives](../../glossary.md#term-derivative), or infer [GPU](../../glossary.md#term-gpu)
closure from a handwritten gradient or CPU result.

## Decision

### Final [Compiler](../../glossary.md#term-compiler) geometry, not raw source [inference](../../glossary.md#term-inference)

In the second [topological](../../glossary.md#term-topological-order) phase of StaticResultLogicalLayoutClosure, retain resolved outputs,
retry existing LayoutInference, check inferred [dtype](../../glossary.md#data-type--datatype)/[Shape](../../glossary.md#shape) agreement, and prefer any proved
[view](../../glossary.md#term-view-meanings) exactly. Only ordinary RESHAPE with fully static input/output Shapes, a present
noncontiguous input [layout](../../glossary.md#layout), proven ordinary [cardinality](../../glossary.md#term-cardinality) and still-unresolved inferred output
may select [canonical](../../glossary.md#term-canonical-form) non-view result geometry through existing closeValue.

Noncontiguity does not prove that every conceivable reshape [alias](../../glossary.md#aliasing-and-arena) is impossible. This authority
selects [materialization](../../glossary.md#materialization) where current inference does not establish its supported view mapping.
Do not add RESHAPE to the family-wide first-phase classifier, invent another Shape algorithm,
change [graph](../../glossary.md#graph) [topology](../../glossary.md#term-topology)/IDs/attributes/sidecars, or override existing offset/strided views.
Static [scalar](../../glossary.md#term-scalar)/empty geometry follows checked ordinary arithmetic; dynamic/unresolved inputs
remain unresolved. A Compiler layout choice does not qualify a [backend](../../glossary.md#backend)'s empty domain.

[Model](../../glossary.md#term-model) constructors, raw CapturedGraphInference/LayoutInference, SourceGraphReconstruction,
source [descriptors](../../glossary.md#term-descriptor) and [checkpoint](../../glossary.md#term-checkpoint) v1-v5 remain unchanged. Final geometry is not wire authority.

### [CPU](../../glossary.md#term-cpu): retain [views](../../glossary.md#term-view-meanings) and complete the materializing [occurrence](../../glossary.md#term-occurrence)

Retain the exact [contiguous](../../glossary.md#term-contiguous)-input/same-offset view branch. Independently admit the selected
noncontiguous-input occurrence only with matching [dtype](../../glossary.md#data-type--datatype)/[target](../../glossary.md#term-target)/[cardinality](../../glossary.md#term-cardinality) and exact [canonical](../../glossary.md#term-canonical-form)
non-view output. Reuse CpuAffineLayoutLowering's reverse ordinal mapping and existing affine
copy emission; no new copy mathematics, numerical [operation](../../glossary.md#operation) or low-arithmetic unlock.

Actual source addressing, [alias](../../glossary.md#aliasing-and-arena) safety, [lifetime](../../glossary.md#term-lifetime), empty-domain behavior and [raw bits](../../glossary.md#storage-mapping-contiguity-and-injectivity) require
execution proof. Existing [lowering](../../glossary.md#lowering) code is not evidence that the provider admits this case.
Serialize the provider change after CPU0078; preserve qualified old [IR](../../glossary.md#term-intermediate-representation-ir)/keys/class bytes.

### [Metal](../../glossary.md#term-metal): [occurrence](../../glossary.md#term-occurrence)-specific state and the existing exact movement [kernel](../../glossary.md#kernel)

Keep the existing positive-static [rank](../../glossary.md#term-rank) 0..16 movement types/roles and physical-storage bounds.
Prove two distinct relationships independently in provider/preparation/Java and [native code](../../glossary.md#native-numerical-library-and-instruction-architecture):

|Occurrence|Logical output and declared state|
|---|---|
|Proved [view](../../glossary.md#term-view-meanings)|Exact inferred view; source offset/span preserved; AFFINE_VIEW.|
|Selected [materialization](../../glossary.md#materialization)|[Canonical](../../glossary.md#term-canonical-form) [strides](../../glossary.md#term-stride), offset 0, isView=false, span=count, densePhysical=false; CANONICAL.|

Do not apply source/output offset or span equality to materialization. Keep it for actual views.
Do not globally change every RESHAPE's state or erase source view [provenance](../../glossary.md#provenance). Preparation must
allocate the ordinary destination/[lifetime](../../glossary.md#term-lifetime); materialization is not a new [alias](../../glossary.md#aliasing-and-arena) or elision.

Reuse the selected exact_move_unary_0066 [operation](../../glossary.md#operation)-6 kernel and existing copy_word [metadata](../../glossary.md#term-metadata)/
helpers. Its output ordinal maps through the input [dimensions](../../glossary.md#dimension) and effective physical strides/
offset, then through output geometry, copying the exact [carrier](../../glossary.md#term-storage-carrier) width. Do not substitute the
older Task0059 movement kernel, add a [shader](../../glossary.md#term-shader), perform a floating conversion, or duplicate math.

Use the representation actually bound: an executed PERMUTE may own a dense physical [buffer](../../glossary.md#term-buffer);
an authenticated existing elision still uses its source. Do not reorder twice, drop a required
[permutation](../../glossary.md#permutation), follow arbitrary ancestry, add source substitution, or change [fan-out](../../glossary.md#term-fan-out)/[publication](../../glossary.md#term-publication).
Retain distinct-buffer, membership, size, context and [runtime](../../glossary.md#term-runtime) [target](../../glossary.md#term-target) [authentication](../../glossary.md#term-authentication).

All seven admitted movement carriers retain their existing [gradient](../../glossary.md#term-gradient) roles; no integral/[BOOL](../../glossary.md#data-type--datatype)
adjoints, [widening](../../glossary.md#term-widening), [F64](../../glossary.md#floating-point-precision-and-rounding) [MATMUL](../../glossary.md#matrix-multiplication--matmul) or other movement/operand domain follows. Low-containing
partitions remain exclusively CUSTOM_PROGRAM, without generated/anchor [fusion](../../glossary.md#term-fusion).

Schema20/header124/ABI7/thirteen exports already represent both results and remain unchanged.
Advance all six coordinated Metal [identities](../../glossary.md#term-identity-meanings) 37 -> 38; reject [checksum](../../glossary.md#term-checksum)-valid stale 37. The
unpublished profile37 candidate is not redefined. No blanket [Compiler](../../glossary.md#term-compiler)/[CPU](../../glossary.md#term-cpu) [identity](../../glossary.md#term-identity-meanings) bump.

## Integration and evidence

Compiler0019 and Metal0081 may implement against a frozen coordinated candidate; CPU0079's
provider cutover waits for CPU0078 acceptance. MAIN owns exact layered preimages/approved
bytes, shared ledgers, native build/sign/package/clean install and every executable check.
Workers [run](../../glossary.md#term-run) no builds/tests/formatters. Rebuild the changed native source; the earlier ad-hoc
profile37 package cannot certify the final source and is not a trusted public [binary](../../glossary.md#term-binary-descriptor-and-metadata).

Keep the unchanged all-15-[graph](../../glossary.md#graph) original [F32](../../glossary.md#floating-point-precision-and-rounding)/[BF16](../../glossary.md#floating-point-precision-and-rounding)/[F16](../../glossary.md#floating-point-precision-and-rounding) forward/both-generated-[gradient](../../glossary.md#term-gradient) consumer.
Exercise external strided and local materialized sources, offsets/holes, raw adversarial bits,
fresh-input reuse, canaries/source preservation, [publication](../../glossary.md#term-publication)/[fan-out](../../glossary.md#term-fan-out) in both orders, and valid-
[digest](../../glossary.md#term-digest) Java/native forgery/[alias](../../glossary.md#aliasing-and-arena)/size/context negatives. [Compiler](../../glossary.md#term-compiler) checks cover [view](../../glossary.md#term-view-meanings) preservation,
same-pass downstream closure, static empty/[scalar](../../glossary.md#term-scalar)/dynamic boundaries and unchanged raw cold
source. [CPU](../../glossary.md#term-cpu) checks cover actual admitted seven-type movement, not an inferred [lowering](../../glossary.md#lowering) promise.

Current capability rendering is MAIN-only after all applicable cutovers, preserving historical
v1, the CPU lock and every original query. New original-typed basis rows are not F32 projections.
