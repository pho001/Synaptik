# ADR 0061: General Scan composition through ordinary semantic and CPU owners

## Status

Accepted implementation design after both independent full-proof and physical authority/interface
APPROVE0 reviews and MAIN's coordinated root/scoped acceptance. CPU0078, Model0046/0047/0048,
Compiler0017A/0017B and CPU0080 are Complete prerequisites. Exact Ready and implementation/actual
typed-demand/activity/physical qualification remain separate; this ADR is not authority or an
implemented capability claim. No narrower ReLU-only milestone replaces the mandatory target.

## Problem and constraints

The existing general Scan contains ordinary PURE Tensor expressions, but separate length-mode
row, Compiler derivative and CPU emitter whitelists prevent ordinary composition. The actual
unchanged public baseline rejected all 24 F32/F64 RELU/SOFTMAX/LayerNorm cases across all-valid/
length-aware and both traversal directions. The required target is every ordinary CPU-supported
PURE family within genuine descriptor, coordinate, activity and ordinary derivative constraints.
Effects, unknown operations, nested bodies, arbitrary while and higher-order AD remain excluded.

Do not copy ordinary shape algorithms, derivatives or kernels into Scan. Do not route each time
step through prepared executables, an interpreter, graph lookup, source replay, callbacks or a
new scheduler. Existing Model/Compiler/Prepare/CPU owners remain responsible for their phases.
CPU low numerical admission stays locked; Metal's separately qualified production work remains
mandatory and is not changed by this CPU design.

## Accepted implementation decisions

### Shared per-output coordinate and dependency facts

Ordinary descriptor validity remains with Model's checked family constructors and independent
Compiler `CapturedGraphInference`. Factor Scan's existing row-transfer logic into one descriptor-
only semantic seam on the existing Model Scan owner, callable by Model construction and raw
Compiler validation after their own ordinary descriptor proof. It must not reconstruct validation
Tensors, consume IDs, consult CPU geometry or trust a serialized/source-attached proof artifact.

A single immutable nested advanced metadata carrier is justified to replace the present duplicated
one-role-per-producer assumption: per canonical output, identified optional row axis, terminal
row-dependency taint and ordered direct input-dependency positions. This is semantic metadata,
not another graph/registry/type hierarchy. [Model0048](../../planning/modules/model/tasks/0048-shared-ordinary-scan-value-semantics.md)
records the implemented Scan.ValueSemantics/declaredSemantics/ordinarySemantics API; no second
operation namespace is introduced.

- Include conditions, indices and masks independently of gradient flags. Infer coordinates from
  normalized axes and ordinary contributor mappings, never equal extents, F=B or sample B.
- Validate all canonical siblings, including undemanded auxiliaries. BatchNorm normalized/statistic
  outputs and attention output/weights do not share one dependency set. In particular, attention
  weights do not depend on V; training BatchNorm outputs have distinct data/running-state routes.
- A known entirely SHARED PURE operation has no row-coordinate reason to reject by mathematical
  kind. Ordinary guards, dependency/terminal checks and shared-carry isolation still apply.
- Permit proved ROW(axis) intermediates at any valid rank/axis, retaining exactly one identified B.
  Carry/emission roots retain their required ROW(0) boundary meaning. Original length-source AD's
  all-carries/all-sequences ROW(0) rule may remain; it is separate from derived intermediate roles.
  Grouped Conv2d's existing derivative legitimately produces `[groups,B,...]`, so rank-two-only
  ROW(1) cannot remain a hidden backward restriction.
- Preserve coordinate boundaries through reshape and movement. Crossing B in feature softmax,
  cumulative/order/gather/normalization is not row-local merely because output shape is unchanged.
  No blanket active MEAN/variance/softmax-across-B meaning is invented.
- Preserve represented active SUM and admitted terminal MATMUL semantics. Ordinary embedding
  lookup's parameter derivative additionally requires functional shared-target ADD-scatter over
  active update contributors, with original update order and pre-write bounds/uniqueness rules.
  Qualify that explicit terminal closure rather than rejecting embedding gradients or using a
  dense one-hot replacement. Other cross-row indexed/aggregate meanings need their own exact
  existing-operation contributor proof; they are not admitted by an output-extent heuristic.

Terminal taint and shared-carry dependence propagate through the actual per-output input routes.
A terminal result cannot feed row work. A SHARED accumulator still appears only as the exact old
left operand of its own distinct per-time `old.add(contribution)`, with exact shape/dtype and no
shared-carry dependency in the contribution. No per-row accumulator update or mask-after-math.

### Ordinary typed inference and lexical first-order AD

Reuse ordinary descriptor inference for every local value, including BOOL/INT32/INT64 outputs,
explicit casts, saved auxiliaries and canonical siblings. Remove body-wide floating dtype coupling,
not ordinary type/promotion/target rules. Keep scoped local IDs and fresh all-port obligations.

Compiler0017A completes the lexical resolver through all ordinary PURE rule owners, including
nondifferentiable indices/masks and saved canonical outputs. The existing expansion seam remains
`FirstOrderAutograd.expandBody(plan, seeds, primal, constants, normalize)`. Original identity
contribution keys, canonical output positions, selected-input normalization and no-replay remain.
Compiler0017B factors the SAME actual formula bodies over thin real-Tensor/checked-metadata
construction adapters; it is not a companion derivative switch or a new numerical language.

Planning must be selected-demand-aware, not just filter all declarations by floating dtype:

- Only genuine floating cotangent roots/targets get slots. Discrete values remain exact primals;
  no integer/BOOL zero gradients. Preserve necessary false-flag floating temporal flow.
- An unseeded saved statistic is not an independent derivative root; an unrequested floating
  value used only for a condition/index is not automatically a cotangent target. A genuinely
  requested ND-only route still rejects under the ordinary policy, including ZERO mode.
- Compute required temporal carry flow using existing compile-time coverage/planning owners,
  without unrolling a T-sized derivative graph. Preserve tied positional contributions and
  original-input-aligned nullable derivative results. T=0 identity and finite traversal demand
  must not be replaced with a blanket all-input differentiable classification.
- Model0046 represents a no-ROW reverse with only required SHARED accumulators, or true empty
  carry when no accumulator is needed. Do not restore a synthetic floating carry restriction.
- Full lexical original inventory must receive ordinary forward-only exclusions as well as
  selected derivative eligibility. Outer inventory traversal alone does not visit Scan bodies.
  Existing Conv3d/fixed-recurrent and other ordinary gates do not disappear with the whitelist.

A substantive preconstruction gap remains to close in the Compiler task: reuse/factor ordinary
metadata/constructibility proof beside the actual ordinary rule owners, and prove the relevant
typed/coordinate closure before derivative IDs. Coverage eligibility alone is insufficient; late
validation after `expandBody` is not this proof. Do not solve the gap by duplicating derivative
formulas, allocating fake Tensor identities, capture-as-preflight or a new shadow derivative
algebra. Exact reusable proof APIs and selected-demand handoff require review before Ready.
Independent combined inference still validates every constructed canonical output afterward.

### Structural cotangent presence and neutral compiled transport

Absence is not a numeric zero. An owner with no incoming contribution must not execute its
generated arithmetic, primal-only prerequisites or value validators. Explicit zero seeds,
cancellation, FLOOR/SIGN zeros and numerical WHERE/attention selectors remain present. A final
WHERE cannot repair an earlier 0/0, 0*Inf, parameter SUM contamination or generated softmax error.

Use compile-local presence root sets, not a general Boolean-expression language. At one reverse
step the roots are required carried cotangent channels followed by selected emission-seed channels.
Each owner occurrence's presence is the ordered union of its incoming roots. All operations
created by that selected canonical-output VJP inherit that scope, including constants and
primal-only prerequisites. Resolving a primal handle or consulting a descriptor is not a load.
The existing same-body metadata construction observes actual operand use and proves closure before
derivative IDs; it must also cover enclosing normalization, cast-back and contribution merges.

Ordinary contribution-list merging preserves source order: neither present means no ADD; one
present means exact raw identity of that term; both present execute the existing ordinary ADD.
The represented ADD remains in the numeric graph with a two-scope merge fact, not zero operands
or a different formula. A ROW merge uses original-b presence, not coordinate zero. Presence roots
propagate through the finite selected carry routes; inactive length rows retain old numeric/control
carry, while active absent cotangents have false presence and publish typed positive zero if needed.

All-valid empty Tensor occurrences remain structurally present even with B=0 or zero feature
extent; do not replace presence with an any-numerical-contributor test. Length mode separately
retains its no-body-work rule at inactive times/all-zero lengths/B=0. Terminal ROW work uses the
same structural eligible-row domain as its existing contributors, without an ordinary ANY or a
new active MEAN/divisor interpretation. Across time, keep the existing distinct old.add(contribution)
for every present completed parameter contribution; absent time contributions leave old state raw.
Do not remove the initial +0 ADD for a first present temporal contribution or add a has-ever-term
shortcut. Optional ordinary list merging and the retained temporal ADD protocol are distinct.

Neutral transport is one immutable Planning-owned ScanActivityPlan beside the exact compiled body,
not a Model Tensor/attrs attachment, second numeric graph or Runtime protocol:

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
producer, numerical cotangent values, runtime address or derivative formula. Capture remaps the
ephemeral source identities into local IDs. Compiler's existing DerivativeGraphMetadata retains
the scoped plans per containing Scan node through its existing eighth CompileArtifacts component;
Planning capability queries carry the neutral optional plan and Prepare projects it without
Compiler aggregate objects. No ninth source sidecar or Model gradient state is introduced.

Every captured generated node needs a scope, including primal-only nodes; actual unreachable
eager temporaries are checked before IDs but need no runtime node. Rebuild/demand/optimization
must remap control metadata with the exact body. CSE/fusion/hoisting cannot cross incompatible
activity or merge semantics; equal readiness must be proved, not inferred from equal numeric
inputs. Providers/Prepare independently validate memberships, bindings, root positions, complete
canonical descriptors and ordinary predicates; metadata is not a trusted admission certificate.
Absent optional plans preserve old query/projection/IR identity streams without new serialized
empty fields. New controlled/composite entries use a selective namespace only.


### Scoped ordinary CPU unit lowering

Construct a body-local analysis/PrepareContext from the actual `CompiledScanBody.graph()` nodes,
values, requirements and next/emission/residual demand. Then reuse
`CpuPartitionDagDecomposer.decompose/unitContext` and `CpuPartitionLowering.lower` for ordinary
physical units. A subset context built against the outer one-node Scan DAG is incorrect.

Honor real logical layouts and selected physical mappings. Do not mark views/destinations
contiguous just to make a lowerer accept them. Translate ordinary unit boundaries to existing
Scan temporal origins, retained cuts, ports and lifetimes using per-value dtype widths/alignment.
The contained compiled body remains the only semantic graph; derived physical unit/binding facts
are not a second semantic executor or another Runtime protocol.

### Same-class static methods, one existing emitted time loop

Factor the central `CpuClassFileKernelGenerator` method-emission dispatch so a focused ordinary
emitter can populate a named static method in the same hidden class. Each helper retains its
ordinary carrier/scratch/long-array/start/end descriptor and local-slot assumptions; existing
`CpuCarrierEmitter` layout setup stays first. Fixed direct calls connect methods from the existing
emitted Scan control loop. No per-time method-handle lookup, interface dispatch or unit loop.

Prefix subsidiary method names deterministically per unit. `CpuLossEmitter`'s dispatcher,
`lossContiguousInt` and `lossGenericAffine` must all use that namespace. Standalone names/method
order remain unchanged. Authenticate the exact composite method inventory, descriptors/access,
code/local/class/constant-pool budgets and verified bytes in the existing verifier, not its old
one-or-three-method assumption. Method limits are real resource bounds, not an opcode whitelist.

Composite-only Scan entries may carry numeric long[][] geometry packs and one run-owned control
segment; retained Scan and standalone ordinary entry descriptors remain unchanged. Controlled
ordinary helpers append that control segment after their existing carrier/scratch/geometry/range
parameters. Allocate packs/control/cursors and bind disjoint scratch once per run, not per T.
Reuse each geometry owner's pack/reset/base-location logic: indexing/scatter/aggregate/movement
mutate cursor cells, so temporal bases alone are insufficient. Emit nonallocating fixed-callsite
resets. Per-value byte/element conversions must never use the first carry's dtype.

Retain existing represented Scan SUM emitters and `CpuMatmulEmitter` active-contributor seams for
qualified terminal contractions. Substituting ordinary exact SUM or zero-masked MATMUL would
change arithmetic or read inactive NaNs. The distinct completed per-time carry ADD remains.

### Existing value guards and activity

Ordinary numerical entry points do not include all executable-owned validators. Factor temporal/
range/activity entry points in existing Softmax/Loss/BOOL/attention-mask and Index/Scatter validator
owners; no new hierarchy, per-step closure construction or generic validator registry. A fixed
generated call to an existing numerical-domain guard is not a prepared-executable invocation.
Only numeric current bases/ranges/roles survive; no runtime graph is supplied to a validator.

For each unit: inputs ready, validate its complete active validation domain, then perform its
active numerical ranges. Do not validate row zero, write it, and discover a later bad row.
Earlier units need no rollback. Activity must dominate loads and dependent arithmetic, not just
published stores. Scatter's output range does not restrict its update/index traversal; active
update and duplicate-candidate scopes must be correct independently. Preserve ignored-index,
empty-domain, nonempty-index/empty-result and source-role freshness semantics.

No new finite-input validator for normalization: ordinary Layer/RMS/BatchNorm numerical special
values retain their existing meanings. Softmax keeps CPU0077 finite-extreme admission. Every
nested ordinary IR/specialization also receives CPU0078 low-role admission before cache/resource/
generation/definition, not merely an outer FLOAT32 envelope check.

### Compatibility and selective identity

New code-shaping composite facts belong in a selective Scan namespace/baked suffix: ordered
ordinary IR/specializations, exact method inventory, typed temporal mapping, activity/validation/
reset protocol, provenance and scratch/lifetime facts. Exclude invocation B/L/T-varying bases,
physical slots and addresses. No blanket global/family bump, dormant profile or runtime unlock.
Retained standalone and existing qualified Scan recipes keep their keys/bytes when unchanged.

The source and execution work does not silently widen cold reconstruction or v1-v5 encoding.
Keep their independent constructor/type/role domains and byte selection until explicitly
qualified. Newly relaxed row/header constructors must not accidentally expand raw-codec admission.
3D forward remains distinct from the explicit v1 1D/2D CNN training decision.

## Implementation order and acceptance

CPU0078 accepted source/resources -> Model0046 source/callers/independent negative guards ->
shared per-output row/dependency source and raw-consumer boundaries -> ordinary typed/demand-aware
lexical AD and proof -> scoped Prepare/CPU ordinary units -> integrated actual qualification.
Independent design/authority/Ready review precedes each coherent source owner; no speculative stub
or competing runtime path is an acceptable intermediate deliverable.

Required proof includes the unchanged 24-case scalar-forward/actual joint-backward program;
representative supported families and their genuine ordinary backward closure; typed predicates/
indices/casts/discrete histories; all-SHARED families; per-output BatchNorm/attention routes;
grouped Conv2d with nonleading row intermediates; embedding shared-parameter ADD-scatter gradients;
true empty/no-ROW reverse; tied positional inputs; static/named B and T/B=0/1 boundaries; mixed/
all-zero lengths and poisoned inactive inputs/seeds; later-invalid-active-domain pre-write failure;
failure/retry, scratch/lifetime/concurrency checks and selective artifact/byte-retention controls.
Record exclusions as actual semantic/ordinary capability constraints, not a smaller hidden math
list. Full-root release/Metal/native/install/legal/performance gates remain separate requirements.
