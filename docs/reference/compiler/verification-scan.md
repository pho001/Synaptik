# Verification inference and scoped Scan

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="current-package-private-verification-inference"></a>
### Current package-private verification inference

Package-private `CapturedGraphInference.inferAndValidate(CompiledGraphModel)` is the current
semantic boundary after structural capture. It visits stored nodes in deterministic topological
order, resolves ordered input and output descriptors, selects a typed operation-family rule, and
independently derives the complete expected `TensorDescriptor` list. It checks data-type domains
and promotion, operand roles, ranks, axes, Shape and attribute relationships, output roles,
resolved-or-unresolved layout, and gradient-eligibility metadata. The first unequal expected and
stored descriptor fails with node, operation-kind, output-position, and value context.

The pass covers every current forward operation-kind enum and accepted attributes variant in six
cohesive families: elementwise, reduction/normalization/ordering, indexing,
layout/composition/window, structured numeric/loss/RNG-state operations, and fixed recurrent
scan, plus independently scoped general scan bodies. Dispatch is closed and typed. An unrecognized
custom `OperationKind` fails rather than bypassing validation.

Some model operations determine an exact output descriptor while leaving a Shape relationship
undecidable. The compiler represents such a relationship as a package-private
`DeferredGraphConstraint`: one owning `NodeId`, one semantic subject, and one typed immutable
predicate. A conservative three-valued proof returns proven, disproven, or deferred from static,
structural, and available bounded dimension facts. Proven predicates disappear; disproven
predicates fail in deterministic rule order; deferred predicates remain ordered in the successful
result. The evaluator never binds or unifies symbols, and checked-arithmetic overflow makes a
proof unavailable rather than false.

On success, package-private `ValidatedGraph` retains the exact accepted compiler sidecar and
snapshots only the unresolved constraints. Its graph-only compatibility path wraps the accepted
`CompiledGraphModel` with no constant facts. The semantic inference rules continue to inspect the
graph alone; sidecar construction independently checks that every fact names a non-gradient graph
input of the same data type. For example, a reshape from Shape
`[N]` to Shape `[M]`, where `N` and `M` are distinct unresolved dimensions, retains an
element-count-equality constraint because the output descriptor is already exact but equality is
not locally decidable. This proves neither a concrete binding nor executability. Value-dependent
index bounds, duplicate scatter targets, target contents, numerical results, and storage validity
remain outside descriptor-only verification.

This pass is reusable compiler-internal state, not a public artifact. It performs no graph
transformation, concrete dimension binding, autograd, publication or planning orchestration,
diagnostic aggregation, preparation, backend lowering, runtime work, or execution. The current
canonicalization and optional optimization pipeline reuses it after mandatory canonicalization
and after every changed graph candidate.

<a id="current-scoped-scan-capture-and-first-order-ad"></a>
### Current scoped scan capture and first-order AD

Ordinary compilation accepts valid all-valid pure `ScanBody` source independently of backend support or
the narrower AD inventory. Each scan occurrence receives its own `CompiledScanBody.graph()`, even
when occurrences reuse one immutable source template. Local inputs contain all carry, step and
invariant declarations in order, including unused ones. Ordered next-carry/emission/residual
bindings may repeat a unique local graph output. Numeric local IDs never refer to outer values;
producer/formal escape and sharing across different templates reject.

Compiler0013 additionally supports Model's bounded
[length-aware source subset](../tensor/general-scan.md#length-aware-source-only-construction). Independent
inference validates the complete local-ID body, explicit row roles and terminal shared carries,
including unused declarations and empty domains. The final non-gradient INT64 `[B]` lengths actual
has no body formal and contributes nothing to outer gradient eligibility. ROW(0) identifies the
leading batch coordinate; SHARED has no batch coordinate. Original-source AD
requires every carry and every original sequence to be ROW(0), even unused/non-gradient positions;
broader forward inference does not relax this restriction. Generated reverse histories/seeds
may mix explicit ROW(0), rank-two ROW(1) and SHARED roles.

Compiler0015 combines that length protocol with one named B and fixed T/features for homogeneous
FLOAT32/FLOAT64 inference, actual first-order sibling AD and fresh scoped all-port binding. Every
bind checks all actuals (including unused lengths), local obligations and latent ports; it reads
descriptors only, never length storage or values. For h `[Rows,2]`, X `[4,Rows,3]` and INT64 L
`[Rows]`, Rows=7 requires matching `[7,2]`, `[4,7,3]` and `[7]` descriptors, not seven valid-length
values at compile/bind time. Compiler0015 is Complete. Its metadata-only fixtures establish
Compiler admission, not runtime execution; Prepare/CPU/Engine qualify their actual combined
execution domains independently. No new API, representation or execution owner is added.

Cold capture/reconstruction additionally admits the fully static homogeneous FLOAT32/FLOAT64
intersection of the existing constructor inventory and length row rules. Complete preflight
checks every declaration/body value, role and exact descriptor, including late/dead/T=0 scans,
before formal or producer allocation. Compatible repeated/opaque substitutions remain valid;
new ROW/SHARED or differently mapped ties reject before IDs. Homogeneity and aliases are checked
per occurrence. The final lengths actual is separate from c+s+i formals and invariant slicing;
neither its storage nor values are read. Symbolic or wider per-value typed cold bodies remain
unsupported. Checkpoint copy/rebind and inherited NN acquisition/cold forward use this same
boundary. Checkpoint0014 implements separate static length-aware v5 encoding; NN0033 implements
and qualifies the bounded public recurrent-composition workflow.

Compiler0014 extends all-valid scoped inference to one named B, fixed T
and fixed features, independently checking known PURE families and applicable dtypes. It retains
numeric-domain predicates such as BatchNorm's contributor minimum for fresh binding, while rejecting
unproved named-coordinate equality or mixing. Independent review and MAIN's named compatibility
checkpoint passed. The initial Prepare/CPU/Engine symbolic layer qualified bounded FLOAT32;
current ordinary CPU composition below supersedes that dtype-only execution summary.

First-order Scan AD selects existing ordinary VJP owners for the exact requested output/input
pairs and finite execution paths; it is not a fixed body-kind whitelist or captured-IR AD.
Complete lexical source validation precedes derivative construction, including unused work and T=0.
Primal-only paths may contain ordinary non-differentiable operations. Intentional ND roles cut
cotangent paths; a required unsupported derivative route fails closed rather than losing a
contribution. Ordinary dtype, attribute, layout and derivative guards remain authoritative.
Residual differentiation and every Scan higher-order/createGraph request remain unsupported;
non-Scan higher-order and fixed-recurrent-scan rejection boundaries are unchanged.

Selected final-carry and emission seeds jointly produce one opposite-traversal reverse sibling
per original occurrence before the single combined capture. Structural absence is distinct from
a present all-zero seed; activity plans preserve that distinction through shared carries and
contribution merges, with explicit materialization across Scan boundaries.
Public explicit/implicit seed guards are unchanged.
Cotangents flow through false-flag internal carry values without changing their flags. Every primal
operand binds to its exact residual, step or invariant reverse formal; saved TANH/SIGMOID outputs
are never replayed. Scalar constants close through non-gradient invariant formals backed by
registered outer splats. Length-mode row-shaped literals and disconnected zeros instead bind
explicit non-gradient ROW sequences, without reading a primal or using `0*x`. Each incoming
cotangent for an original SHARED value is normalized before contribution append/accumulation:
ROW terms are batch-reduced, including intermediates at B=1; SHARED terms are not reduced again. For example, a
direct ROW term for shared `q[1,H]` must explicitly SUM the singleton batch axis before adding a
terminal-SHARED MATMUL term. Equal-shape sumToShape cannot erase that ROW identity.
Each invariant gradient has one invariant-shaped shared carry and its own distinct
`old.add(contribution)` site, not a B-by-parameter bank. Tied actual inputs, repeated roots and shared parameters accumulate once
per position and occurrence using ordinary Tensor addition.

Every outer graph rebuild preserves body attachments and local saved sites. Final
`demandedOutputIndices` comes from outer publications and ordinary uses after optimization/layout
closure, not retained siblings or internal body uses. Queries carry the complete validated body
to one owner, not child-only or mixed-owner planning. This is metadata and first-order expression
construction only: logical history records allocate no storage. Prepare separately validates
projection and demand; CPU0030 executes the bounded FLOAT32 same-shape pointwise slice only when
the actual forward and Compiler-generated reverse bodies independently qualify. CPU0031 adds
homogeneous FLOAT32 broadcast, SUM/sumToShape and bounded shape/layout closure under that same
independent qualification. CPU0032 adds static homogeneous FLOAT32 rank-two MATMUL under the
same independent actual-forward/reverse body and binder qualification. CPU0033 implements
homogeneous FLOAT64 across the complete bounded declared algebra and actual first-order reverse
closure, without changing Compiler derivative rules or exact saved-value bindings. Low-type scan
AD remains unsupported; CPU0042 separately qualifies fixed-geometry F32/F64 length execution.
Static length cold support is distinct from symbolic cold and numerical qualification.

The CPU0030–CPU0042 inventories above record the historical qualification layers, not a current
finite operator whitelist. Current general Scan composes every ordinary CPU-supported PURE unit
through the existing ordinary emitters with the same dtype, layout, attribute and derivative
guards. All-valid values use WholeTensor semantics; length-aware execution proves ROW(axis)
and SHARED roles, without limiting SHARED tensor rank. Original length-AD carries/sequences
and emissions stay ROW(0); intermediate ROW axes follow the proved coordinate.
Row-mixing requires materialized support with the same lengths, not Tensor identity.
Forward and selected generated reverse bodies qualify independently.
CPU BFLOAT16, FLOAT16 and both FP8 formats use the qualified low-precision forward/AD closure;
storage, casts and exact paths retain their own semantics.
This widens neither cold/wire inventories nor NN workflow admission and grants no Metal Scan.
Checkpoint v5 is implemented separately; NN0033 retains its own consumer acceptance. See the
[scan explanation](../../architecture/general-differentiable-scan.md) for a joint-seed walkthrough.


---

[Previous topic](cold-source.md) · [Next topic](optimization.md)
