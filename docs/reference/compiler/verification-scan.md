# Verification [inference](../../glossary.md#term-inference) and scoped [Scan](../../glossary.md#term-cumulative-scan)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

Verification inference independently derives the [metadata](../../glossary.md#term-metadata) each [operation](../../glossary.md#operation) should produce and
compares it with the captured description. A **deferred constraint** is a relationship that cannot
yet be proved and must be checked when actual [dimensions](../../glossary.md#dimension) are supplied. A Scan **body** is the
repeated step's [graph](../../glossary.md#graph), with its own local identifiers; equal numeric IDs in outer and body graphs
never establish a connection by themselves.

## What is checked without values?

The compiler pipeline calls verification with captured graph metadata, not current input storage.
Verification derives expected descriptors and retains relationships that a future descriptor bind
must resolve. It cannot inspect an index, valid-length count, NaN or actual parameter value.
For a recurrent Scan, it validates the independently scoped body and its explicit connections
to outer inputs; body-local numeric IDs alone never identify outer values. The accepted result
remains a description, with no allocated history or runtime tape. See
[batch metadata binding](batch-binding.md) for deferred checks and
[runtime-length NN composition](../nn/recurrent.md) for where actual length values eventually enter.

<a id="current-package-private-verification-inference"></a>
### Current package-private verification [inference](../../glossary.md#term-inference)

Package-private `CapturedGraphInference.inferAndValidate(CompiledGraphModel)` is the current
semantic boundary after [structural capture](../../glossary.md#term-structural-capture). It visits stored [nodes](../../glossary.md#node) in deterministic [topological
order](../../glossary.md#term-topological-order), resolves ordered input and output [descriptors](../../glossary.md#term-descriptor), selects a typed [operation](../../glossary.md#operation)-family rule, and
independently derives the complete expected `TensorDescriptor` list. It checks data-type domains
and promotion, operand roles, ranks, [axes](../../glossary.md#term-axis), [Shape](../../glossary.md#shape) and attribute relationships, output roles,
resolved-or-unresolved [layout](../../glossary.md#layout), and [gradient](../../glossary.md#term-gradient)-eligibility [metadata](../../glossary.md#term-metadata). The first unequal expected and
stored descriptor fails with node, operation-kind, output-position, and value context.

The pass covers every current forward operation-kind enum and accepted attributes variant in six
cohesive families: [elementwise](../../glossary.md#term-elementwise), reduction/[normalization](../../glossary.md#term-normalization)/ordering, indexing,
layout/composition/window, structured numeric/[loss](../../glossary.md#loss)/[RNG](../../glossary.md#graph-rng-state--graphrngstate)-state operations, and fixed recurrent
[scan](../../glossary.md#term-cumulative-scan), plus independently scoped [general scan](../../glossary.md#general-differentiable-scan) bodies. [Dispatch](../../glossary.md#term-dispatch) is closed and typed. An unrecognized
custom [`OperationKind`](../../glossary.md#operationkind) fails rather than bypassing [validation](../../glossary.md#term-validation).

Some [model](../../glossary.md#term-model) operations determine an exact output descriptor while leaving a Shape relationship
undecidable. The [compiler](../../glossary.md#term-compiler) represents such a relationship as a package-private
`DeferredGraphConstraint`: one owning [`NodeId`](../../glossary.md#nodeid), one semantic subject, and one typed [immutable](../../glossary.md#term-immutable)
predicate. A conservative three-valued proof returns proven, disproven, or deferred from static,
structural, and available bounded [dimension](../../glossary.md#dimension) facts. Proven predicates disappear; disproven
predicates fail in deterministic rule order; deferred predicates remain ordered in the successful
result. The evaluator never binds or unifies symbols, and checked-arithmetic [overflow](../../glossary.md#term-overflow) makes a
proof unavailable rather than false.

On success, package-private `ValidatedGraph` retains the exact accepted compiler sidecar and
snapshots only the unresolved constraints. Its [graph](../../glossary.md#graph)-only compatibility path wraps the accepted
[`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel) with no [constant](../../glossary.md#term-constant) facts. The semantic inference rules continue to inspect the
graph alone; sidecar construction independently checks that every fact names a non-gradient graph
input of the same [data type](../../glossary.md#data-type--datatype). For example, a reshape from Shape
`[N]` to Shape `[M]`, where `N` and `M` are distinct unresolved dimensions, retains an
[element](../../glossary.md#term-element)-count-equality constraint because the output descriptor is already exact but equality is
not locally decidable. This proves neither a concrete [binding](../../glossary.md#term-binding) nor executability. Value-dependent
index bounds, duplicate scatter [targets](../../glossary.md#term-target), target contents, numerical results, and storage validity
remain outside descriptor-only verification.

This pass is reusable compiler-internal state, not a public [artifact](../../glossary.md#term-artifact). It performs no graph
transformation, concrete dimension binding, [autograd](../../glossary.md#autograd), [publication](../../glossary.md#term-publication) or [planning](../../glossary.md#term-planning) orchestration,
diagnostic aggregation, preparation, [backend](../../glossary.md#backend) [lowering](../../glossary.md#lowering), [runtime](../../glossary.md#term-runtime) work, or execution. The current
[canonicalization](../../glossary.md#occurrence-identity-and-canonical-form) and optional optimization pipeline reuses it after mandatory canonicalization
and after every changed graph candidate.

<a id="current-scoped-scan-capture-and-first-order-ad"></a>
### Current scoped [scan](../../glossary.md#term-cumulative-scan) [capture](../../glossary.md#term-capture-meanings) and first-order [AD](../../glossary.md#autograd)

Ordinary [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) accepts valid all-valid pure `ScanBody` source independently of [backend](../../glossary.md#backend) support or
the narrower AD inventory. Each scan [occurrence](../../glossary.md#term-occurrence) receives its own `CompiledScanBody.graph()`, even
when occurrences reuse one [immutable](../../glossary.md#term-immutable) source template. Local inputs contain all [carry](../../glossary.md#term-carry), step and
[invariant](../../glossary.md#term-invariant) declarations in order, including unused ones. Ordered next-carry/emission/[residual](../../glossary.md#scan-residual-port-and-history-demand)
[bindings](../../glossary.md#term-binding) may repeat a unique local [graph](../../glossary.md#graph) output. Numeric local IDs never refer to outer values;
[producer](../../glossary.md#term-producer)/formal escape and sharing across different templates reject.

Compiler0013 additionally supports [Model](../../glossary.md#term-model)'s bounded
[length-aware source subset](../tensor/general-scan.md#length-aware-source-only-construction). Independent
[inference](../../glossary.md#term-inference) validates the complete local-ID body, explicit row roles and terminal shared carries,
including unused declarations and empty domains. The final non-[gradient](../../glossary.md#term-gradient) [INT64](../../glossary.md#data-type--datatype) `[B]` lengths actual
has no body formal and contributes nothing to outer gradient eligibility. ROW(0) identifies the
leading [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix); SHARED has no batch coordinate. Original-source AD
requires every carry and every original sequence to be ROW(0), even unused/non-gradient positions;
broader forward inference does not relax this restriction. Generated reverse histories/[seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings)
may mix explicit ROW(0), [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) ROW(1) and SHARED roles.

Compiler0015 combines that length protocol with one named B and fixed T/[features](../../glossary.md#term-software-feature) for [homogeneous](../../glossary.md#term-homogeneous)
[FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) inference, actual first-order sibling AD and fresh scoped all-port binding. Every
bind checks all actuals (including unused lengths), local obligations and latent ports; it reads
[descriptors](../../glossary.md#term-descriptor) only, never length storage or values. For h `[Rows,2]`, X `[4,Rows,3]` and INT64 L
`[Rows]`, Rows=7 requires matching `[7,2]`, `[4,7,3]` and `[7]` descriptors, not seven valid-length
values at [compile](../../glossary.md#compile)/bind time. Compiler0015 is Complete. Its [metadata](../../glossary.md#term-metadata)-only fixtures establish
[Compiler](../../glossary.md#term-compiler) [admission](../../glossary.md#term-admission), not [runtime](../../glossary.md#term-runtime) execution; [Prepare](../../glossary.md#prepare)/[CPU](../../glossary.md#term-cpu)/[Engine](../../glossary.md#term-engine) qualify their actual combined
execution domains independently. No new [API](../../glossary.md#term-application-programming-interface-api), representation or execution [owner](../../glossary.md#term-owner) is added.

Cold capture/reconstruction additionally admits the bounded fully static length-mode
intersection, with per-role floating [dtypes](../../glossary.md#data-type--datatype) (including low types and explicit floating casts),
exact positional actual/formal and next/initial matching, ROW/SHARED rules, and no deferred
obligations. It does not admit arbitrary [PURE](../../glossary.md#term-pure) bodies, symbolic cold geometry, or a [backend
route](../../glossary.md#backend-route) merely because reconstruction succeeds. See the
[cold constructor contract](../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction).

Compiler0014 extends all-valid scoped inference to one named B, fixed T
and fixed [features](../../glossary.md#term-feature), independently checking known PURE families and applicable dtypes. It retains
numeric-domain predicates such as BatchNorm's contributor minimum for fresh binding, while rejecting
unproved named-coordinate equality or mixing. Independent review and MAIN's named compatibility
[checkpoint](../../glossary.md#term-checkpoint) passed. The initial Prepare/CPU/Engine symbolic layer qualified bounded FLOAT32;
current ordinary CPU composition below supersedes that dtype-only execution summary.

First-order Scan AD selects existing ordinary [VJP](../../glossary.md#term-vector-jacobian-product-vjp) owners for the exact requested output/input
pairs and [finite](../../glossary.md#term-finite) execution paths; it is not a fixed body-kind whitelist or captured-[IR](../../glossary.md#term-intermediate-representation-ir) AD.
Complete lexical source [validation](../../glossary.md#term-validation) precedes [derivative](../../glossary.md#term-derivative) construction, including unused work and T=0.
[Primal](../../glossary.md#term-primal)-only paths may contain ordinary non-[differentiable](../../glossary.md#derivative-and-chain-rule) operations. Intentional [ND](../../glossary.md#term-nd) roles cut
[cotangent](../../glossary.md#cotangent) paths; a required unsupported derivative route fails closed rather than losing a
contribution. Ordinary dtype, attribute, [layout](../../glossary.md#layout) and derivative guards remain authoritative.
Residual differentiation and every Scan higher-order/createGraph request remain unsupported;
non-Scan higher-order limits remain separate. Fixed [RNN](../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm) first-order [BPTT](../../glossary.md#term-backpropagation-through-time-bptt) is now
implemented through RecurrentGradientRules and the shared length-aware Scan derivative owner;
this does not authorize recurrence higher orders.

Selected final-carry and emission seeds jointly produce one opposite-traversal reverse sibling
per original occurrence before the single combined capture. Structural absence is distinct from
a present all-zero [seed](../../glossary.md#term-seed); activity plans preserve that distinction through shared carries and
contribution merges, with explicit [materialization](../../glossary.md#materialization) across Scan boundaries.
Public explicit/implicit seed guards are unchanged.
Cotangents flow through false-flag internal carry values without changing their flags. Every primal
operand binds to its exact residual, step or invariant reverse formal; saved [TANH](../../glossary.md#term-tanh)/[SIGMOID](../../glossary.md#term-sigmoid) outputs
are never replayed. [Scalar](../../glossary.md#term-scalar) [constants](../../glossary.md#term-constant) close through non-gradient invariant formals backed by
registered outer splats. Length-mode row-shaped literals and disconnected zeros instead bind
explicit non-gradient ROW sequences, without reading a primal or using `0*x`. Each incoming
cotangent for an original SHARED value is normalized before contribution append/accumulation:
ROW terms are batch-reduced, including intermediates at B=1; SHARED terms are not reduced again. For example, a
direct ROW term for shared `q[1,H]` must explicitly SUM the [singleton](../../glossary.md#term-singleton) batch [axis](../../glossary.md#term-axis) before adding a
terminal-SHARED [MATMUL](../../glossary.md#matrix-multiplication--matmul) term. Equal-[shape](../../glossary.md#shape) sumToShape cannot erase that ROW [identity](../../glossary.md#term-identity-meanings).
Each invariant gradient has one invariant-shaped shared carry and its own distinct
`old.add(contribution)` site, not a B-by-[parameter](../../glossary.md#term-parameter) bank. Tied actual inputs, repeated roots and shared parameters accumulate once
per position and occurrence using ordinary [Tensor](../../glossary.md#tensor) addition.

Every outer graph rebuild preserves body attachments and local saved sites. Final
`demandedOutputIndices` comes from outer publications and ordinary uses after optimization/layout
closure, not retained siblings or internal body uses. Queries carry the complete validated body
to one owner, not child-only or mixed-owner [planning](../../glossary.md#term-planning). This is metadata and first-order expression
construction only: logical history records allocate no storage. Prepare separately validates
projection and demand; CPU0030 executes the bounded FLOAT32 same-shape [pointwise](../../glossary.md#term-pointwise) [slice](../../glossary.md#slice) only when
the actual forward and Compiler-generated reverse bodies independently qualify. CPU0031 adds
homogeneous FLOAT32 broadcast, SUM/sumToShape and bounded shape/layout closure under that same
independent [qualification](../../glossary.md#term-qualification). CPU0032 adds static homogeneous FLOAT32 rank-two MATMUL under the
same independent actual-forward/reverse body and binder qualification. CPU0033 implements
homogeneous FLOAT64 across the complete bounded declared algebra and actual first-order reverse
closure, without changing Compiler derivative rules or exact saved-value bindings. Those task
receipts are historical increments: low-type Scan execution and actual first-order AD now use
the ordinary CPU/shared Compiler owners under the same occurrence and derivative guards.
CPU0042 records the earlier fixed-geometry [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) length qualification, not a current dtype ceiling.
Static length cold support is distinct from symbolic cold and numerical qualification.

The CPU0030–CPU0042 inventories above record the historical qualification layers, not a current
finite operator whitelist. Current [general Scan](../../glossary.md#general-differentiable-scan) composes every ordinary CPU-supported PURE unit
through the existing ordinary emitters with the same dtype, layout, attribute and derivative
guards. All-valid values use WholeTensor semantics; length-aware execution proves ROW(axis)
and SHARED roles, without limiting SHARED tensor [rank](../../glossary.md#term-rank). Original length-AD carries/sequences
and emissions stay ROW(0); intermediate ROW axes follow the proved coordinate.
Row-mixing requires materialized support with the same lengths, not Tensor identity.
Forward and selected generated reverse bodies qualify independently.
CPU [BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16) and both [FP8](../../glossary.md#term-fp8) formats use the qualified low-[precision](../../glossary.md#term-precision) forward/AD closure;
storage, casts and exact paths retain their own semantics.
This widens neither cold/wire inventories nor [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) workflow admission and grants no [Metal](../../glossary.md#term-metal) Scan.
Checkpoint v5 is implemented separately; NN0033 retains its own consumer acceptance. See the
[scan explanation](../../architecture/general-differentiable-scan.md) for a joint-seed walkthrough.


---

[Previous topic](cold-source.md) · [Next topic](optimization.md)
