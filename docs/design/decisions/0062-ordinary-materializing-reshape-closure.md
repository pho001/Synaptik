# ADR 0062: Ordinary materializing RESHAPE closure

## Status

Accepted source direction — independent Compiler/Metal design reviews APPROVE0. This decision
is not an execution or release-readiness certificate. Compiler0019, CPU0079 and Metal0081 own
implementation and qualification. CPU0079 waits for CPU0078's completed cutover.

## Context

Ordinary Model RESHAPE accepts equal-cardinality value semantics. Model construction and raw
Compiler inference prove an exact view only for a present contiguous source layout; otherwise
the result remains unresolved. That does not prove a view or reject materialization.

The source-matched Metal0080 candidate passes the unchanged full F32 MLP and CNN public
compute/three-SGD/save/close/load/new-train consumers. Its new original-typed five-form MATMUL
backward test still fails: the transpose-of-dense form produces PERMUTE -> RESHAPE gradients
`[3,2]/[1,3] -> [6]` and `[5,3]/[1,5] -> [15]`. Final closure leaves these results unresolved;
Metal rejects them. A separate public CPU PERMUTE -> RESHAPE before consumer also stops at
its first F32 ownership query. No CPU seven-type execution follows from that attempt.

Retain the before failures and separate successful public journeys. Do not reduce gradient
targets, insert consumer CONTIGUOUS operations, replace ordinary derivatives, or infer GPU
closure from a handwritten gradient or CPU result.

## Decision

### Final Compiler geometry, not raw source inference

In the second topological phase of StaticResultLogicalLayoutClosure, retain resolved outputs,
retry existing LayoutInference, check inferred dtype/Shape agreement, and prefer any proved
view exactly. Only ordinary RESHAPE with fully static input/output Shapes, a present
noncontiguous input layout, proven ordinary cardinality and still-unresolved inferred output
may select canonical non-view result geometry through existing closeValue.

Noncontiguity does not prove that every conceivable reshape alias is impossible. This authority
selects materialization where current inference does not establish its supported view mapping.
Do not add RESHAPE to the family-wide first-phase classifier, invent another Shape algorithm,
change graph topology/IDs/attributes/sidecars, or override existing offset/strided views.
Static scalar/empty geometry follows checked ordinary arithmetic; dynamic/unresolved inputs
remain unresolved. A Compiler layout choice does not qualify a backend's empty domain.

Model constructors, raw CapturedGraphInference/LayoutInference, SourceGraphReconstruction,
source descriptors and checkpoint v1-v5 remain unchanged. Final geometry is not wire authority.

### CPU: retain views and complete the materializing occurrence

Retain the exact contiguous-input/same-offset view branch. Independently admit the selected
noncontiguous-input occurrence only with matching dtype/target/cardinality and exact canonical
non-view output. Reuse CpuAffineLayoutLowering's reverse ordinal mapping and existing affine
copy emission; no new copy mathematics, numerical operation or low-arithmetic unlock.

Actual source addressing, alias safety, lifetime, empty-domain behavior and raw bits require
execution proof. Existing lowering code is not evidence that the provider admits this case.
Serialize the provider change after CPU0078; preserve qualified old IR/keys/class bytes.

### Metal: occurrence-specific state and the existing exact movement kernel

Keep the existing positive-static rank 0..16 movement types/roles and physical-storage bounds.
Prove two distinct relationships independently in provider/preparation/Java and native code:

|Occurrence|Logical output and declared state|
|---|---|
|Proved view|Exact inferred view; source offset/span preserved; AFFINE_VIEW.|
|Selected materialization|Canonical strides, offset 0, isView=false, span=count, densePhysical=false; CANONICAL.|

Do not apply source/output offset or span equality to materialization. Keep it for actual views.
Do not globally change every RESHAPE's state or erase source view provenance. Preparation must
allocate the ordinary destination/lifetime; materialization is not a new alias or elision.

Reuse the selected exact_move_unary_0066 operation-6 kernel and existing copy_word metadata/
helpers. Its output ordinal maps through the input dimensions and effective physical strides/
offset, then through output geometry, copying the exact carrier width. Do not substitute the
older Task0059 movement kernel, add a shader, perform a floating conversion, or duplicate math.

Use the representation actually bound: an executed PERMUTE may own a dense physical buffer;
an authenticated existing elision still uses its source. Do not reorder twice, drop a required
permutation, follow arbitrary ancestry, add source substitution, or change fan-out/publication.
Retain distinct-buffer, membership, size, context and runtime target authentication.

All seven admitted movement carriers retain their existing gradient roles; no integral/BOOL
adjoints, widening, F64 MATMUL or other movement/operand domain follows. Low-containing
partitions remain exclusively CUSTOM_PROGRAM, without generated/anchor fusion.

Schema20/header124/ABI7/thirteen exports already represent both results and remain unchanged.
Advance all six coordinated Metal identities 37 -> 38; reject checksum-valid stale 37. The
unpublished profile37 candidate is not redefined. No blanket Compiler/CPU identity bump.

## Integration and evidence

Compiler0019 and Metal0081 may implement against a frozen coordinated candidate; CPU0079's
provider cutover waits for CPU0078 acceptance. MAIN owns exact layered preimages/approved
bytes, shared ledgers, native build/sign/package/clean install and every executable check.
Workers run no builds/tests/formatters. Rebuild the changed native source; the earlier ad-hoc
profile37 package cannot certify the final source and is not a trusted public binary.

Keep the unchanged all-15-graph original F32/BF16/F16 forward/both-generated-gradient consumer.
Exercise external strided and local materialized sources, offsets/holes, raw adversarial bits,
fresh-input reuse, canaries/source preservation, publication/fan-out in both orders, and valid-
digest Java/native forgery/alias/size/context negatives. Compiler checks cover view preservation,
same-pass downstream closure, static empty/scalar/dynamic boundaries and unchanged raw cold
source. CPU checks cover actual admitted seven-type movement, not an inferred lowering promise.

Current capability rendering is MAIN-only after all applicable cutovers, preserving historical
v1, the CPU lock and every original query. New original-typed basis rows are not F32 projections.
