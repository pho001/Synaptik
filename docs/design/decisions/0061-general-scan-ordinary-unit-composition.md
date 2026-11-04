# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0061: [General Scan](../../glossary.md#general-differentiable-scan) composition through ordinary semantic and [CPU](../../glossary.md#term-cpu) [owners](../../glossary.md#term-owner)


## Editorial reading context

Model checks ordinary operation meaning and row dependencies; Compiler builds derivatives; Prepare transports qualified facts; CPU lowers body-local units into fixed generated calls inside the existing loop. Structural cotangent absence means no contribution arrived, unlike an explicitly supplied numeric zero. A final mask cannot repair invalid arithmetic that should never have executed. This record describes the accepted design, not blanket all-PURE support.

Use the [source, activity, AD and execution boundaries](../../architecture/general-differentiable-scan.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted implementation design after both independent full-proof and physical authority/interface
APPROVE0 reviews and MAIN's coordinated root/scoped acceptance. CPU0078, Model0046/0047/0048,
Compiler0017A/0017B and CPU0080 are Complete prerequisites. Exact [Ready](../../glossary.md#term-ready) and implementation/actual
typed-demand/activity/physical [qualification](../../glossary.md#term-qualification) remain separate; this [ADR](../../glossary.md#term-architecture-decision-record-adr) is not authority or an
implemented capability claim. No narrower [ReLU](../../glossary.md#term-relu)-only milestone replaces the mandatory [target](../../glossary.md#term-target).

## Problem and constraints

The existing [general Scan](../../glossary.md#general-differentiable-scan) contains ordinary [PURE](../../glossary.md#term-pure) [Tensor](../../glossary.md#tensor) expressions, but separate length-mode
row, [Compiler](../../glossary.md#term-compiler) [derivative](../../glossary.md#term-derivative) and [CPU](../../glossary.md#term-cpu) emitter whitelists prevent ordinary composition. The actual
unchanged public baseline rejected all 24 [F32](../../glossary.md#floating-point-precision-and-rounding)/[F64](../../glossary.md#floating-point-precision-and-rounding) RELU/[SOFTMAX](../../glossary.md#softmax--log-softmax)/LayerNorm cases across all-valid/
length-aware and both traversal directions. The required [target](../../glossary.md#term-target) is every ordinary CPU-supported
PURE family within genuine [descriptor](../../glossary.md#term-descriptor), [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix), activity and ordinary derivative constraints.
Effects, unknown operations, nested bodies, arbitrary while and higher-order [AD](../../glossary.md#autograd) remain excluded.

Do not copy ordinary [shape](../../glossary.md#shape) algorithms, derivatives or [kernels](../../glossary.md#kernel) into [Scan](../../glossary.md#general-differentiable-scan). Do not route each time
step through prepared executables, an interpreter, [graph](../../glossary.md#graph) [lookup](../../glossary.md#term-lookup), source replay, callbacks or a
new scheduler. Existing [Model](../../glossary.md#term-model)/Compiler/[Prepare](../../glossary.md#prepare)/CPU [owners](../../glossary.md#term-owner) remain responsible for their phases.
CPU low numerical [admission](../../glossary.md#term-admission) stays locked; [Metal](../../glossary.md#term-metal)'s separately qualified production work remains
mandatory and is not changed by this CPU design.

## Accepted implementation decisions

### Shared per-output [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) and [dependency](../../glossary.md#term-dependency) facts

Ordinary [descriptor](../../glossary.md#term-descriptor) validity remains with [Model](../../glossary.md#term-model)'s checked family constructors and independent
[Compiler](../../glossary.md#term-compiler) `CapturedGraphInference`. Factor [Scan](../../glossary.md#general-differentiable-scan)'s existing row-transfer logic into one descriptor-
only semantic seam on the existing Model Scan [owner](../../glossary.md#term-owner), callable by Model construction and raw
Compiler [validation](../../glossary.md#term-validation) after their own ordinary descriptor proof. It must not reconstruct validation
[Tensors](../../glossary.md#tensor), consume IDs, consult [CPU](../../glossary.md#term-cpu) geometry or trust a serialized/source-attached proof [artifact](../../glossary.md#term-artifact).

A single [immutable](../../glossary.md#term-immutable) nested advanced [metadata](../../glossary.md#term-metadata) carrier is justified to replace the present duplicated
one-role-per-[producer](../../glossary.md#term-producer) assumption: per [canonical](../../glossary.md#term-canonical-form) output, identified optional row [axis](../../glossary.md#term-axis), terminal
row-dependency taint and ordered direct input-dependency positions. This is semantic metadata,
not another [graph](../../glossary.md#graph)/[registry](../../glossary.md#term-registry)/type hierarchy. [Model0048](../../planning/modules/model/tasks/0048-shared-ordinary-scan-value-semantics.md)
records the implemented Scan.ValueSemantics/declaredSemantics/ordinarySemantics [API](../../glossary.md#term-application-programming-interface-api); no second
[operation](../../glossary.md#operation) namespace is introduced.

- Include conditions, indices and masks independently of gradient flags. Infer coordinates from
  normalized axes and ordinary contributor mappings, never equal extents, F=B or sample B.<span data-glossary-navigation="true"> ([gradient glossary](../../glossary.md#term-gradient); [sample glossary](../../glossary.md#term-sample-machine-learning-and-hyperparameters))</span>
- Validate all canonical siblings, including undemanded auxiliaries. BatchNorm normalized/statistic
  outputs and attention output/weights do not share one dependency set. In particular, attention
  weights do not depend on V; training BatchNorm outputs have distinct data/running-state routes.<span data-glossary-navigation="true"> ([statistic glossary](../../glossary.md#term-statistic); [weights glossary](../../glossary.md#term-weight))</span>
- A known entirely SHARED PURE operation has no row-coordinate reason to reject by mathematical
  kind. Ordinary guards, dependency/terminal checks and shared-carry isolation still apply.<span data-glossary-navigation="true"> ([PURE operation glossary](../../glossary.md#admission-preflight-and-fail-closed-validation))</span>
- Permit proved ROW(axis) intermediates at any valid rank/axis, retaining exactly one identified B.
  Carry/emission roots retain their required ROW(0) boundary meaning. Original length-source AD's
  all-carries/all-sequences ROW(0) rule may remain; it is separate from derived intermediate roles.
  Grouped Conv2d's existing derivative legitimately produces <span data-glossary-navigation="true"> ([rank glossary](../../glossary.md#term-rank); [AD glossary](../../glossary.md#autograd); [Conv2d glossary](../../glossary.md#convolution--conv2d); [derivative glossary](../../glossary.md#term-derivative))</span>`[groups,B,...]`, so rank-two-only
  ROW(1) cannot remain a hidden backward restriction.<span data-glossary-navigation="true"> ([rank-two glossary](../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Preserve coordinate boundaries through reshape and movement. Crossing B in feature softmax,
  cumulative/order/gather/normalization is not row-local merely because output shape is unchanged.
  No blanket active MEAN/variance/softmax-across-B meaning is invented.<span data-glossary-navigation="true"> ([feature glossary](../../glossary.md#term-feature); [softmax glossary](../../glossary.md#softmax--log-softmax); [gather glossary](../../glossary.md#gather); [normalization glossary](../../glossary.md#term-normalization); [MEAN glossary](../../glossary.md#term-mean); [variance glossary](../../glossary.md#term-variance))</span>
- Preserve represented active SUM and admitted terminal MATMUL semantics. Ordinary embedding
  lookup's parameter derivative additionally requires functional shared-target ADD-scatter over
  active update contributors, with original update order and pre-write bounds/uniqueness rules.
  Qualify that explicit terminal closure rather than rejecting embedding gradients or using a
  dense one-hot replacement. Other cross-row indexed/aggregate meanings need their own exact
  existing-operation contributor proof; they are not admitted by an output-extent heuristic.<span data-glossary-navigation="true"> ([MATMUL glossary](../../glossary.md#matrix-multiplication--matmul); [embedding glossary](../../glossary.md#embedding); [lookup glossary](../../glossary.md#term-lookup); [parameter glossary](../../glossary.md#term-parameter); [target glossary](../../glossary.md#term-target); [extent glossary](../../glossary.md#term-extent); [heuristic glossary](../../glossary.md#term-heuristic))</span>

Terminal taint and shared-[carry](../../glossary.md#term-carry) dependence propagate through the actual per-output input routes.
A terminal result cannot feed row work. A SHARED [accumulator](../../glossary.md#accumulator-type) still appears only as the exact old
left operand of its own distinct per-time `old.add(contribution)`, with exact [shape](../../glossary.md#shape)/[dtype](../../glossary.md#data-type--datatype) and no
shared-carry dependency in the contribution. No per-row accumulator update or [mask](../../glossary.md#term-boolean-mask)-after-math.

### Ordinary typed [inference](../../glossary.md#term-inference-meanings) and lexical first-order [AD](../../glossary.md#autograd)

Reuse ordinary [descriptor](../../glossary.md#term-descriptor) inference for every local value, including [BOOL](../../glossary.md#data-type--datatype)/[INT32](../../glossary.md#data-type--datatype)/[INT64](../../glossary.md#data-type--datatype) outputs,
explicit casts, saved auxiliaries and [canonical](../../glossary.md#term-canonical-form) siblings. Remove body-wide floating [dtype](../../glossary.md#data-type--datatype) coupling,
not ordinary type/promotion/[target](../../glossary.md#term-target) rules. Keep scoped local IDs and fresh all-port obligations.

Compiler0017A completes the lexical resolver through all ordinary [PURE](../../glossary.md#term-pure) rule [owners](../../glossary.md#term-owner), including
nondifferentiable indices/[masks](../../glossary.md#term-boolean-mask) and saved canonical outputs. The existing expansion seam remains
`FirstOrderAutograd.expandBody(plan, seeds, primal, constants, normalize)`. Original [identity](../../glossary.md#term-identity-meanings)
contribution keys, canonical output positions, selected-input [normalization](../../glossary.md#term-normalization) and no-replay remain.
Compiler0017B factors the SAME actual formula bodies over thin real-[Tensor](../../glossary.md#tensor)/checked-[metadata](../../glossary.md#term-metadata)
construction adapters; it is not a companion [derivative](../../glossary.md#term-derivative) switch or a new numerical language.

[Planning](../../glossary.md#term-planning) must be selected-demand-aware, not just filter all declarations by floating dtype:

- Only genuine floating cotangent roots/targets get slots. Discrete values remain exact primals;
  no integer/BOOL zero gradients. Preserve necessary false-flag floating temporal flow.<span data-glossary-navigation="true"> ([cotangent glossary](../../glossary.md#cotangent); [gradients glossary](../../glossary.md#term-gradient))</span>
- An unseeded saved statistic is not an independent derivative root; an unrequested floating
  value used only for a condition/index is not automatically a cotangent target. A genuinely
  requested ND-only route still rejects under the ordinary policy, including ZERO mode.<span data-glossary-navigation="true"> ([saved statistic glossary](../../glossary.md#saved-statistic); [ND glossary](../../glossary.md#term-nd))</span>
- Compute required temporal carry flow using existing compile-time coverage/planning owners,
  without unrolling a T-sized derivative graph. Preserve tied positional contributions and
  original-input-aligned nullable derivative results. T=0 identity and finite traversal demand
  must not be replaced with a blanket all-input differentiable classification.<span data-glossary-navigation="true"> ([carry glossary](../../glossary.md#term-carry); [compile glossary](../../glossary.md#compile); [unrolling glossary](../../glossary.md#term-unrolling); [graph glossary](../../glossary.md#graph); [differentiable glossary](../../glossary.md#derivative-and-chain-rule))</span>
- Model0046 represents a no-ROW reverse with only required SHARED accumulators, or true empty
  carry when no accumulator is needed. Do not restore a synthetic floating carry restriction.<span data-glossary-navigation="true"> ([accumulator glossary](../../glossary.md#accumulator-type))</span>
- Full lexical original inventory must receive ordinary forward-only exclusions as well as
  selected derivative eligibility. Outer inventory traversal alone does not visit Scan bodies.
  Existing Conv3d/fixed-recurrent and other ordinary gates do not disappear with the whitelist.<span data-glossary-navigation="true"> ([Scan glossary](../../glossary.md#general-differentiable-scan); [Conv3d glossary](../../glossary.md#conv3d--ncdhw))</span>

A substantive preconstruction gap remains to close in the [Compiler](../../glossary.md#term-compiler) task: reuse/factor ordinary
metadata/constructibility proof beside the actual ordinary rule owners, and prove the relevant
typed/[coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) closure before derivative IDs. Coverage eligibility alone is insufficient; late
[validation](../../glossary.md#term-validation) after `expandBody` is not this proof. Do not solve the gap by duplicating derivative
formulas, allocating fake Tensor identities, [capture](../../glossary.md#term-capture-meanings)-as-[preflight](../../glossary.md#term-preflight) or a new shadow derivative
algebra. Exact reusable proof APIs and selected-demand [handoff](../../glossary.md#term-handoff) require review before [Ready](../../glossary.md#term-ready).
Independent combined inference still validates every constructed canonical output afterward.

### Structural [cotangent](../../glossary.md#cotangent) presence and neutral compiled transport

Absence is not a numeric zero. An [owner](../../glossary.md#term-owner) with no incoming contribution must not execute its
generated arithmetic, [primal](../../glossary.md#term-primal)-only prerequisites or value validators. Explicit zero [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings),
[cancellation](../../glossary.md#term-cancellation), FLOOR/SIGN zeros and numerical WHERE/attention selectors remain present. A final
WHERE cannot repair an earlier 0/0, 0*Inf, [parameter](../../glossary.md#term-parameter) SUM contamination or generated [softmax](../../glossary.md#softmax--log-softmax) error.

Use [compile](../../glossary.md#compile)-local presence root sets, not a general [Boolean](../../glossary.md#data-type--datatype)-expression language. At one reverse
step the roots are required carried cotangent channels followed by selected emission-[seed](../../glossary.md#term-seed) channels.
Each owner [occurrence](../../glossary.md#term-occurrence)'s presence is the ordered union of its incoming roots. All operations
created by that selected [canonical](../../glossary.md#term-canonical-form)-output [VJP](../../glossary.md#term-vector-jacobian-product-vjp) inherit that [scope](../../glossary.md#term-scope), including [constants](../../glossary.md#term-constant) and
primal-only prerequisites. Resolving a primal handle or consulting a [descriptor](../../glossary.md#term-descriptor) is not a load.
The existing same-body [metadata](../../glossary.md#term-metadata) construction observes actual operand use and proves closure before
[derivative](../../glossary.md#term-derivative) IDs; it must also cover enclosing [normalization](../../glossary.md#term-normalization), cast-back and contribution merges.

Ordinary contribution-list merging preserves source order: neither present means no ADD; one
present means exact raw [identity](../../glossary.md#term-identity-meanings) of that term; both present execute the existing ordinary ADD.
The represented ADD remains in the numeric [graph](../../glossary.md#graph) with a two-scope merge fact, not zero operands
or a different formula. A ROW merge uses original-b presence, not [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix) zero. Presence roots
propagate through the finite selected [carry](../../glossary.md#term-carry) routes; inactive length rows retain old numeric/control
carry, while active absent cotangents have false presence and publish typed [positive zero](../../glossary.md#nan-infinity-and-signed-zero) if needed.

All-valid empty [Tensor](../../glossary.md#tensor) occurrences remain structurally present even with B=0 or zero [feature](../../glossary.md#term-feature)
[extent](../../glossary.md#term-extent); do not replace presence with an any-numerical-contributor test. Length mode separately
retains its no-body-work rule at inactive times/all-zero lengths/B=0. Terminal ROW work uses the
same structural eligible-row domain as its existing contributors, without an ordinary ANY or a
new active [MEAN](../../glossary.md#term-mean)/divisor interpretation. Across time, keep the existing distinct old.add(contribution)
for every present completed parameter contribution; absent time contributions leave old state raw.
Do not remove the initial +0 ADD for a first present temporal contribution or add a has-ever-term
shortcut. Optional ordinary list merging and the retained temporal ADD protocol are distinct.

Neutral transport is one [immutable](../../glossary.md#term-immutable) [Planning](../../glossary.md#term-planning)-owned ScanActivityPlan beside the exact compiled body,
not a [Model](../../glossary.md#term-model) Tensor/attrs attachment, second numeric graph or [Runtime](../../glossary.md#term-runtime) protocol:

```text
graph: exact independently scoped CompiledGraphModel
carries: (numeric carry position, initial presence, next scope)
seedFormals: selected emission-seed formal positions
scopes: immutable sorted unique root-index lists
nodeScopes: local NodeId -> scope index
merges: local ADD NodeId -> (left scope, right scope)
accumulators: (numeric shared-carry position, completed contribution scope)
```

Root numbering is carries followed by seedFormals; the structural records have no Tensor,
[producer](../../glossary.md#term-producer), numerical cotangent values, runtime address or derivative formula. [Capture](../../glossary.md#term-capture-meanings) remaps the
ephemeral source identities into local IDs. [Compiler](../../glossary.md#term-compiler)'s existing DerivativeGraphMetadata retains
the scoped plans per containing [Scan](../../glossary.md#general-differentiable-scan) [node](../../glossary.md#node) through its existing eighth CompileArtifacts component;
Planning capability queries carry the neutral optional plan and [Prepare](../../glossary.md#prepare) projects it without
Compiler aggregate objects. No ninth source sidecar or Model [gradient](../../glossary.md#term-gradient) state is introduced.

Every captured generated node needs a scope, including primal-only nodes; actual unreachable
eager temporaries are checked before IDs but need no runtime node. Rebuild/demand/optimization
must remap control metadata with the exact body. [CSE](../../glossary.md#term-common-subexpression-elimination-cse)/[fusion](../../glossary.md#term-fusion)/hoisting cannot cross incompatible
activity or merge semantics; equal readiness must be proved, not inferred from equal numeric
inputs. Providers/Prepare independently validate memberships, [bindings](../../glossary.md#term-binding), root positions, complete
canonical descriptors and ordinary predicates; metadata is not a trusted [admission](../../glossary.md#term-admission) [certificate](../../glossary.md#formal-proof-certificate-and-trusted-assumptions).
Absent optional plans preserve old query/projection/[IR](../../glossary.md#term-intermediate-representation-ir) identity streams without new serialized
empty fields. New controlled/composite entries use a selective namespace only.


### Scoped ordinary [CPU](../../glossary.md#term-cpu) unit [lowering](../../glossary.md#lowering)

Construct a body-local analysis/[PrepareContext](../../glossary.md#prepare-context-preparecontext) from the actual `CompiledScanBody.graph()` [nodes](../../glossary.md#node),
values, requirements and next/emission/[residual](../../glossary.md#scan-residual-port-and-history-demand) demand. Then reuse
`CpuPartitionDagDecomposer.decompose/unitContext` and `CpuPartitionLowering.lower` for ordinary
physical units. A subset context built against the outer one-node [Scan](../../glossary.md#general-differentiable-scan) [DAG](../../glossary.md#term-directed-acyclic-graph-dag) is incorrect.

Honor real logical [layouts](../../glossary.md#layout) and selected physical mappings. Do not mark [views](../../glossary.md#term-view-meanings)/destinations
[contiguous](../../glossary.md#term-contiguous) just to make a lowerer accept them. Translate ordinary unit boundaries to existing
Scan temporal origins, retained cuts, ports and lifetimes using per-value [dtype](../../glossary.md#data-type--datatype) widths/[alignment](../../glossary.md#term-alignment).
The contained compiled body remains the only semantic [graph](../../glossary.md#graph); derived physical unit/[binding](../../glossary.md#term-binding) facts
are not a second semantic executor or another [Runtime](../../glossary.md#term-runtime) protocol.

### Same-class static methods, one existing emitted time loop

Factor the central `CpuClassFileKernelGenerator` method-emission dispatch so a focused ordinary
emitter can populate a named static method in the same [hidden class](../../glossary.md#term-hidden-class). Each helper retains its
ordinary [carrier](../../glossary.md#term-storage-carrier)/scratch/long-array/start/end [descriptor](../../glossary.md#term-descriptor) and local-slot assumptions; existing
`CpuCarrierEmitter` [layout](../../glossary.md#layout) setup stays first. Fixed direct calls connect methods from the existing
emitted [Scan](../../glossary.md#general-differentiable-scan) control loop. No per-time [method-handle lookup](../../glossary.md#term-method-handle-lookup), interface dispatch or unit loop.

Prefix subsidiary method names deterministically per unit. `CpuLossEmitter`'s dispatcher,
`lossContiguousInt` and `lossGenericAffine` must all use that namespace. Standalone names/method
order remain unchanged. Authenticate the exact composite method inventory, descriptors/access,
code/local/class/[constant](../../glossary.md#term-constant)-pool budgets and verified bytes in the existing verifier, not its old
one-or-three-method assumption. Method limits are real resource bounds, not an [opcode](../../glossary.md#term-opcode) whitelist.

Composite-only Scan entries may carry numeric long[][] geometry packs and one [run](../../glossary.md#term-run)-owned control
segment; retained Scan and standalone ordinary entry descriptors remain unchanged. Controlled
ordinary helpers append that control segment after their existing carrier/scratch/geometry/range
[parameters](../../glossary.md#term-parameter). Allocate packs/control/cursors and bind disjoint scratch once per run, not per T.
Reuse each geometry [owner](../../glossary.md#term-owner)'s pack/reset/base-location logic: indexing/scatter/aggregate/movement
mutate cursor cells, so temporal bases alone are insufficient. Emit nonallocating fixed-callsite
resets. Per-value byte/[element](../../glossary.md#term-element) conversions must never use the first carry's [dtype](../../glossary.md#data-type--datatype).

Retain existing represented Scan SUM emitters and `CpuMatmulEmitter` active-contributor seams for
qualified terminal contractions. Substituting ordinary exact SUM or zero-masked [MATMUL](../../glossary.md#matrix-multiplication--matmul) would
change arithmetic or read inactive NaNs. The distinct completed per-time carry ADD remains.

### Existing value guards and activity

Ordinary numerical entry points do not include all executable-owned validators. Factor temporal/
range/activity entry points in existing [Softmax](../../glossary.md#softmax--log-softmax)/[Loss](../../glossary.md#loss)/[BOOL](../../glossary.md#data-type--datatype)/attention-[mask](../../glossary.md#term-boolean-mask) and Index/Scatter validator
[owners](../../glossary.md#term-owner); no new hierarchy, per-step closure construction or generic validator [registry](../../glossary.md#term-registry). A fixed
generated call to an existing numerical-domain [guard](../../glossary.md#term-guard) is not a prepared-executable invocation.
Only numeric current bases/ranges/roles survive; no [runtime](../../glossary.md#term-runtime) [graph](../../glossary.md#graph) is supplied to a validator.

For each unit: inputs ready, validate its complete active [validation](../../glossary.md#term-validation) domain, then perform its
active numerical ranges. Do not validate row zero, write it, and discover a later bad row.
Earlier units need no [rollback](../../glossary.md#term-rollback). Activity must dominate loads and dependent arithmetic, not just
published stores. Scatter's output range does not restrict its update/index traversal; active
update and duplicate-candidate scopes must be correct independently. Preserve ignored-index,
empty-domain, nonempty-index/empty-result and source-role freshness semantics.

No new [finite](../../glossary.md#term-finite)-input validator for [normalization](../../glossary.md#term-normalization): ordinary Layer/RMS/BatchNorm numerical special
values retain their existing meanings. Softmax keeps CPU0077 finite-extreme [admission](../../glossary.md#term-admission). Every
nested ordinary [IR](../../glossary.md#term-intermediate-representation-ir)/[specialization](../../glossary.md#term-specialization) also receives CPU0078 low-role admission before [cache](../../glossary.md#term-cache)/resource/
generation/definition, not merely an outer [FLOAT32](../../glossary.md#term-float32) envelope check.

### Compatibility and selective [identity](../../glossary.md#term-identity-meanings)

New code-shaping composite facts belong in a selective [Scan](../../glossary.md#general-differentiable-scan) namespace/baked suffix: ordered
ordinary [IR](../../glossary.md#term-intermediate-representation-ir)/specializations, exact method inventory, typed temporal mapping, activity/[validation](../../glossary.md#term-validation)/
reset protocol, [provenance](../../glossary.md#provenance) and scratch/[lifetime](../../glossary.md#term-lifetime) facts. Exclude invocation B/L/T-varying bases,
physical slots and addresses. No blanket global/family bump, dormant profile or [runtime](../../glossary.md#term-runtime) unlock.
Retained standalone and existing qualified Scan recipes keep their keys/bytes when unchanged.

The source and execution work does not silently widen [cold reconstruction](../../glossary.md#term-cold-reconstruction) or v1-v5 encoding.
Keep their independent constructor/type/role domains and byte selection until explicitly
qualified. Newly relaxed row/header constructors must not accidentally expand raw-[codec](../../glossary.md#term-codec) [admission](../../glossary.md#term-admission).
3D forward remains distinct from the explicit v1 1D/2D [CNN](../../glossary.md#term-convolutional-neural-network-cnn) training decision.

## Implementation order and acceptance

CPU0078 accepted source/resources -> Model0046 source/callers/independent negative guards ->
shared per-output row/[dependency](../../glossary.md#term-dependency) source and raw-consumer boundaries -> ordinary typed/demand-aware
lexical [AD](../../glossary.md#autograd) and proof -> scoped [Prepare](../../glossary.md#prepare)/[CPU](../../glossary.md#term-cpu) ordinary units -> integrated actual [qualification](../../glossary.md#term-qualification).
Independent design/authority/[Ready](../../glossary.md#term-ready) review precedes each coherent source [owner](../../glossary.md#term-owner); no speculative stub
or competing [runtime](../../glossary.md#term-runtime) path is an acceptable intermediate deliverable.

Required proof includes the unchanged 24-case [scalar](../../glossary.md#cpu-portable-execution-strategy)-forward/actual joint-backward program;
representative supported families and their genuine ordinary backward closure; typed predicates/
indices/casts/discrete histories; all-SHARED families; per-output BatchNorm/attention routes;
grouped [Conv2d](../../glossary.md#convolution--conv2d) with nonleading row intermediates; [embedding](../../glossary.md#embedding) shared-[parameter](../../glossary.md#term-parameter) ADD-scatter [gradients](../../glossary.md#term-gradient);
true empty/no-ROW reverse; tied positional inputs; static/named B and T/B=0/1 boundaries; mixed/
all-zero lengths and poisoned inactive inputs/[seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings); later-invalid-active-domain pre-write failure;
failure/retry, scratch/[lifetime](../../glossary.md#term-lifetime)/[concurrency](../../glossary.md#term-concurrency) checks and selective [artifact](../../glossary.md#term-artifact)/byte-retention controls.
Record exclusions as actual semantic/ordinary capability constraints, not a smaller hidden math
list. Full-root [release](../../glossary.md#term-release)/[Metal](../../glossary.md#term-metal)/native/install/legal/performance gates remain separate requirements.
