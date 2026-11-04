# CPU0081: [General Scan](../../../../glossary.md#general-differentiable-scan) through existing ordinary physical units

## Reader context

This record connects three distinct boundaries:

- **Source construction:** application code supplies `ScanBody` formals and actual tensors.
  A formal is a leaf with no producing expression, standing for one body input; it need not
  contain stored values. [Scan source roles](../../../../reference/tensor/general-scan.md#formals-actual-positions-and-source-closure)
  explain positional carry, sequence and invariant binding.
- **Compilation and preparation:** Compiler passes a captured body and activity facts;
  Prepare checks the bindings and CPU decomposes that body using ordinary physical units.
  Lowerers and emitters are internal callers here, not application callbacks. Analysis declares
  cuts, scratch and control before shared slot assignment; generation emits fixed helper calls.
- **Invocation:** Runtime binds an invocation's actual storage and executes the generated time
  loop. Mutable packs, cursors, control and scratch belong to that run, not shared class state.
  Old carry values stay valid until every required old-state read has finished.

“Caller seed” in the zero-time result means the actual seed supplied for the selected derivative
request. With no body step producing a replacement, the source-resource path borrows that seed;
it does not invent a producer, initialize a substitute, or take over the supplier's storage
lifetime. “Structurally absent” means a derivative channel was not supplied by the selected
graph; it differs from a present tensor whose numerical cells happen to be zero, because the
latter still participates in validation and arithmetic.

Read [current general Scan composition](../../../../backend-guide/cpu-backend.md#current-general-scan-composition)
for present qualification boundaries. The frozen interfaces, original failures and finite
ownership/allocation observations below remain this checkpoint's evidence, not a universal
zero-copy, zero-allocation, concurrency or whole-release guarantee.

## Status

Complete — MAIN integrated the coordinated Compiler0017/CPU0081 cutover after exact [Ready](../../../../glossary.md#term-ready)/source
freezes, independent authority/interface and final implementation APPROVE0 reviews, actual
ordinary composition/control/guards/[binding](../../../../glossary.md#term-binding) consumers and the unfiltered affected test/[Javadoc](../../../../glossary.md#term-javadoc)
gate. This is component [qualification](../../../../glossary.md#term-qualification), not whole-[release](../../../../glossary.md#term-release) or wider cold/wire/[Metal](../../../../glossary.md#term-metal) Scan support.

## MAIN [Ready](../../../../glossary.md#term-ready) and integration

- **Depends on:** Complete CPU0078/0079/0080, Model0046/0047/0048 and Compiler0017A/0017B; accepted
  frozen Compiler0017 neutral interface, NOT completion of its independently writable source slice.
- **Conflicts with:** Compiler0017 exclusively owns Compiler/Planning/Model/Checkpoint semantic/
  source API; this slice exclusively owns Prepare/CPU and non-Compiler context/query migrations.
  MAIN alone owns authority/shared docs/build/release inventory/integration and every command.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [Planning glossary](../../../../glossary.md#term-planning); [Model glossary](../../../../glossary.md#term-model); [Checkpoint glossary](../../../../glossary.md#term-checkpoint); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [Prepare glossary](../../../../glossary.md#prepare); [CPU glossary](../../../../glossary.md#term-cpu); [release glossary](../../../../glossary.md#term-release))</span>
- **Parallel group:** general-scan-full-ordinary-2026-11-01.
- **Common base revision:** HEAD07a281edd3086efde2853211b71b48288db027e1 PLUS every accepted dirty
  file in `/var/folders/z2/kdt1_wq1213732bs4wkhfrh40000gq/T/synaptik-production-run.gloed_6d/fullscan-ready-source.json`.
- **Integration order:** reviewed Compiler0017 neutral/source cutover, then this reviewed physical
  source; MAIN compiles/verifies only the combined context, never a private substitute API.
- **Integration [validation](../../../../glossary.md#term-validation):** GeneralScanFullJoin as defined in Compiler0017: affected suites/
  Javadocs, unchanged24actual cases, full typed/family/presence/guard/scratch/lifetime/identity
  consumers, then the final full source/artifact/CPU+Metal neural/native release checkpoint.<span data-glossary-navigation="true"> ([guard glossary](../../../../glossary.md#term-guard); [lifetime glossary](../../../../glossary.md#term-lifetime); [identity glossary](../../../../glossary.md#term-identity-meanings); [artifact glossary](../../../../glossary.md#term-artifact); [Metal glossary](../../../../glossary.md#term-metal))</span>
- **Shared-document integration [owner](../../../../glossary.md#term-owner):** MAIN.

MAIN verifies satisfied accepted prerequisites, frozen cross-[slice](../../../../glossary.md#slice) APIs and disjoint contexts before
launch. Physical source context is
`/var/folders/z2/kdt1_wq1213732bs4wkhfrh40000gq/T/synaptik-production-run.gloed_6d/fullscan-cpu0081-worktree`.
Exact final Ready source/context [authentication](../../../../glossary.md#term-authentication) and independent review receipts are retained in
the named [manifest](../../../../glossary.md#term-manifest) before source launch. Workers do source/tests/[Javadoc](../../../../glossary.md#term-javadoc) only, NO commands.

## Goal and boundaries

Class C. All ordinary [CPU](../../../../glossary.md#term-cpu)-supported known [PURE](../../../../glossary.md#term-pure) body operations compose in both all-valid and
length-aware Scan using the existing ordinary [lowering](../../../../glossary.md#lowering), focused arithmetic emitters and Scan time
loop. Actual selected backward is mandatory. Do not substitute a [ReLU](../../../../glossary.md#term-relu)-only cut, copied mathematics,
a permanent Scan kind list, interpreter, prepared-body executor, scheduler, callbacks or per-step
allocation. Preserve declared ordinary [descriptor](../../../../glossary.md#term-descriptor)/type/purity and global first-order exclusions.
Java low numerical arithmetic stays locked; [Metal](../../../../glossary.md#term-metal) remains supported production [scope](../../../../glossary.md#term-scope) independently.
This task grants no Metal Scan or per-step [OpenBLAS](../../../../glossary.md#term-openblas)/native route.

MAIN owns coordinated authority, interface freeze, source contexts, integration and every command.
Workers perform source/[Javadoc](../../../../glossary.md#term-javadoc) edits only, NO builds/tests/formatters/native/git. Independent source
review precedes MAIN verification. Full [target](../../../../glossary.md#term-target) remains actionable until qualified end to end.

## Frozen semantic [handoff](../../../../glossary.md#term-handoff) and source [ownership](../../../../glossary.md#term-ownership-meanings)

Compiler0017 owns the exact [Planning](../../../../glossary.md#term-planning) `ScanActivityPlan` definition and Query fifth component,
DerivativeGraphMetadata component-eight map, typed/source/selected [AD](../../../../glossary.md#autograd), [capture](../../../../glossary.md#term-capture-meanings) and [Model](../../../../glossary.md#term-model)/history
header changes. The incorporated Planning/[Compiler](../../../../glossary.md#term-compiler) contracts freeze those public names and record
components; this brief records the handoff, not independent authority. No ninth component.

CPU0081 owns [PrepareContext](../../../../glossary.md#prepare-context-preparecontext) sixth [canonical](../../../../glossary.md#term-canonical-form) component
`Map<NodeId, ScanActivityPlan> scanActivityPlans` after backendInputs, [GraphPreparation](../../../../glossary.md#graph-preparation--graphpreparation) projection
from `artifacts.derivatives().scanActivityByNode()`, all [CPU](../../../../glossary.md#term-cpu) query/copy/control/physical/codegen/
validator/[binding](../../../../glossary.md#term-binding) changes and non-Compiler context-copy migrations, including [Metal](../../../../glossary.md#term-metal) tuning.
Associations use containing projected Scan [node](../../../../glossary.md#node) IDs; the plan references the exact local [graph](../../../../glossary.md#graph).
CPU never imports Compiler. Keep genuine ordinary empty-plan construction; copying annotated
contexts must preserve/filter the plan map explicitly, never silently drop it through a convenience
constructor. Migrate all body-bearing Query callers to the new fifth optional component.
The incorporated CPU amendment conditionally selects only the additional focused top-level
CpuGeneralScanCompositionLowering, CpuGeneralScanBodyEmitter and CpuScanControl [owners](../../../../glossary.md#term-owner); private
physical records otherwise stay in existing owners. No second executor/validator framework.

The reviewed [layout](../../../../glossary.md#layout) prerequisite is supplied by Compiler0017 through existing Model ScanBody:
`closeOrdinaryLayouts(Operation,List<TensorDescriptor>,List<TensorDescriptor>)`, plus the exact
generic `hasCanonicalContiguousResults(Operation)` classification shared with flat Compiler and
symbolic binding. CPU consumes this [owner](../../../../glossary.md#term-owner), never copies its decisions or depends on Compiler.
Centralize computed-[descriptor](../../../../glossary.md#term-descriptor) completion in `bodyContext`, which both [decomposition](../../../../glossary.md#term-decomposition)/analysis and
helper resource preparation use. Keep captured graph/activity [identity](../../../../glossary.md#term-identity-meanings), strict actual/formal
guards, named layouts, exact [views](../../../../glossary.md#term-view-meanings) and all per-value physical spans.

The reviewed producerless-value prerequisite is a clean cutover of the existing complete-graph
source contribution path, not another Scan executor. Consume the renamed five-component
[ProducerlessPublishedValueResource](../../../../glossary.md#producerless-published-value-resource--producerlesspublishedvalueresource) with existing [Runtime](../../../../glossary.md#term-runtime) ByteSize and the migrated optional-[scalar](../../../../glossary.md#term-scalar)
geometry hook. Reuse existing CPU static span/access/named binding/representation owners; the
sole possible additional nested numeric recipe is CpuRepresentationRecipes.BatchLayoutByteSize.
Caller roots retain leading [occurrence](../../../../glossary.md#term-occurrence) order, [constants](../../../../glossary.md#term-constant) stay last and complete source facts
participate in explicit [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters) mode. Cold compatibility on every affected fixed CPU unit/container
authenticates exact fixed declarations/resolved geometry inside that complete batch plan, without
relaxing Runtime guards or changing code identities. Preserve nonempty-[partition](../../../../glossary.md#partition) policy.
Acceptance includes actual T0 caller-[seed](../../../../glossary.md#term-seed) [VJP](../../../../glossary.md#term-vector-jacobian-product-vjp), source-only B plus fixed work across B=0/reentry,
raw affine source publications/ownership and unchanged authentic retained old recipes.

## Independent [validation](../../../../glossary.md#term-validation) and real scoped [decomposition](../../../../glossary.md#term-decomposition)

- Query applicability is not capability. Prepare independently validates complete typed outer/
  local declarations, exact actual/formal/result descriptors, known purity, ordered role/root/
  demand bindings and activity membership. No first-carry dtype/nonempty/positive-ROW assumption.
  Arbitrary proved history ROW axes/ranks qualify together; lengths are separate INT64[B], no formal.<span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [descriptors glossary](../../../../glossary.md#term-descriptor); [bindings glossary](../../../../glossary.md#term-binding); [carry glossary](../../../../glossary.md#term-carry); [dtype glossary](../../../../glossary.md#data-type--datatype); [axes glossary](../../../../glossary.md#term-axis); [INT64 glossary](../../../../glossary.md#data-type--datatype))</span>
- All-valid retains ordinary whole-Tensor semantics, not length-mode row-locality. Do not
  fabricate B/ROW provenance or reject all-valid cross-row operations through length-only proof.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [provenance glossary](../../../../glossary.md#provenance))</span>
- Build the local PrepareContext from exact captured nodes/IDs/topology and real local producer/
  consumer membership. Only computed descriptors may use the shared logical-layout projection
  overlay, with matching logical requirements; never mutate the captured body/activity graph.
  Local constants stay empty unless actual local ingress exists, never copied by outer ValueId.<span data-glossary-navigation="true"> ([PrepareContext glossary](../../../../glossary.md#prepare-context-preparecontext); [nodes glossary](../../../../glossary.md#node); [topology glossary](../../../../glossary.md#term-topology); [producer glossary](../../../../glossary.md#term-producer); [layout glossary](../../../../glossary.md#layout); [graph glossary](../../../../glossary.md#graph); [constants glossary](../../../../glossary.md#term-constant); [ValueId glossary](../../../../glossary.md#logical-valueid-versus-physical-memory-slot))</span>
- Use existing CpuPartitionDagDecomposer.decompose/unitContext and CpuPartitionLowering.lower.
  The outer one-node Scan DAG is not a body context. Protect complete next-state temporal closure,
  declared publication/residual roots and all exact ordinary physical boundaries/cuts.<span data-glossary-navigation="true"> ([DAG glossary](../../../../glossary.md#term-directed-acyclic-graph-dag); [publication glossary](../../../../glossary.md#term-publication); [residual glossary](../../../../glossary.md#scan-residual-port-and-history-demand))</span>
- A zero-node passthrough body is legitimate: bind declared roots directly; do not forge a node,
  disable partition invariants or allocate a dummy carry/ROW value. Genuinely empty carry and
  generated no-ROW/empty reverse remain supported with explicit length mode.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition))</span>
- Per-output Model Scan.ValueSemantics supplies coordinate/terminal/direct-input facts only after
  independent ordinary descriptors. Do not copy its transfer algorithm into CPU or trust a neutral
  certificate instead of checking body membership/bindings/predicates and ordinary feasibility.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix); [CPU glossary](../../../../glossary.md#term-cpu); [certificate glossary](../../../../glossary.md#formal-proof-certificate-and-trusted-assumptions))</span>
- Fusion needs compatible root scopes AND proper row/SHARED geometry, complete value-validation
  boundary, optional-merge/temporal-ADD semantics and publication lifetime. No cross-scope CSE,
  hoist or fused eager operand load. Reuse existing ordinary eligibility rather than disable all
  fusion or add a second permanent family registry.<span data-glossary-navigation="true"> ([Fusion glossary](../../../../glossary.md#term-fusion); [lifetime glossary](../../../../glossary.md#term-lifetime); [scope glossary](../../../../glossary.md#term-scope); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [fused glossary](../../../../glossary.md#specialization-and-fusion); [eager glossary](../../../../glossary.md#term-eager-evaluation); [registry glossary](../../../../glossary.md#term-registry))</span>

## Typed physical facts and [resource lifetime](../../../../glossary.md#ownership-borrowing-and-lease)

- Every value/boundary/cut retains its real dtype, layout, offset, strides, carrier and alignment.
  Use its own byte width; real movement/copy units materialize required transformations. Never
  fabricate contiguous layouts, a homogeneous float envelope or first-carry workspace width.<span data-glossary-navigation="true"> ([dtype glossary](../../../../glossary.md#data-type--datatype); [layout glossary](../../../../glossary.md#layout); [strides glossary](../../../../glossary.md#term-stride); [carrier glossary](../../../../glossary.md#term-storage-carrier); [alignment glossary](../../../../glossary.md#term-alignment); [contiguous glossary](../../../../glossary.md#term-contiguous); [homogeneous glossary](../../../../glossary.md#term-homogeneous); [carry glossary](../../../../glossary.md#term-carry))</span>
- Preserve old/new carry versions until all old readers finish, every demanded history and live
  ordinary producing cut, and scratch for simultaneous ranges. Local IDs never become outer
  buffer declarations. Declare body cuts, helper scratch, packs and control through existing
  resource analysis before slot assignment; all native control storage is RunState-owned workspace.<span data-glossary-navigation="true"> ([buffer glossary](../../../../glossary.md#term-buffer); [RunState glossary](../../../../glossary.md#run-state--runstate))</span>
- No hidden Arena, undeclared storage or shared mutable class/static control. Each bound invocation
  exclusively owns its mutable packs/cursors. Cleanup and run/close failure use existing owners.<span data-glossary-navigation="true"> ([owners glossary](../../../../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([Arena glossary](../../../../glossary.md#term-arena); [mutable glossary](../../../../glossary.md#term-mutable); [bound invocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [run glossary](../../../../glossary.md#term-run))</span>
- Qualify each nested ordinary specialization AND IR independently against CPU0078 before cache,
  generation/definition or resource realization. Storage/casts/discrete primals are not low math.<span data-glossary-navigation="true"> ([specialization glossary](../../../../glossary.md#term-specialization); [IR glossary](../../../../glossary.md#term-intermediate-representation-ir); [cache glossary](../../../../glossary.md#term-cache))</span>

## Symbolic geometry without representative B

Static and one-[canonical](../../../../glossary.md#term-canonical-form)-named-B/fixed-T/[features](../../../../glossary.md#term-feature) Scan remain the declared geometry [target](../../../../glossary.md#term-target).
Use existing [batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/recipe [owners](../../../../glossary.md#term-owner) and per-family ordinary pack [layouts](../../../../glossary.md#layout). A higher-[rank](../../../../glossary.md#term-rank) proved
intermediate ROW or mixed typed boundary must not fail merely at an obsolete rank-three/first-
float projection assumption. B-dependent geometry is symbolic/checked numeric recipe data, never
an invented B=1 [compile](../../../../glossary.md#compile) [graph](../../../../glossary.md#graph), [sample](../../../../glossary.md#term-sample-machine-learning-and-hyperparameters) [specialization](../../../../glossary.md#term-specialization) or actual-B-baked [identity](../../../../glossary.md#term-identity-meanings). Qualify complete
original [descriptors](../../../../glossary.md#term-descriptor) and every [binding](../../../../glossary.md#term-binding) before numerical execution. Do not silently substitute the
narrower existing ordinary batch family list for the mandatory general composition domain.
Unproved/unknown/independently named geometry still fails closed; ordinary numerical rules stay.

## Same-class direct helper execution and packs

- Extend existing CpuClassFileKernelGenerator.emitMethods with qualified controlled emission,
  retaining its empty-prefix old declarations and returned ACTUAL method inventory. One ordinary
  numeric helper appends exactly one invocation-owned control MemorySegment after its old ABI.
  Internal loss/row/auxiliary helper calls forward it when controlled. No numerical Tensor slot.<span data-glossary-navigation="true"> ([MemorySegment glossary](../../../../glossary.md#term-memorysegment); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [loss glossary](../../../../glossary.md#loss); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Existing Scan entry owns the time loop and emits a FIXED sequence of invokestatic helper calls.
  Generation-time iteration over units is fine; runtime unit-object iteration/lookup, MethodHandle
  dispatch per unit/time, BoundInvocation body children, callbacks and graph traversal are not.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [lookup glossary](../../../../glossary.md#term-lookup); [MethodHandle glossary](../../../../glossary.md#native-numerical-library-and-instruction-architecture); [dispatch glossary](../../../../glossary.md#term-dispatch); [BoundInvocation glossary](../../../../glossary.md#bound-invocation--boundinvocation); [graph glossary](../../../../glossary.md#graph))</span>
- Bind typed segment wrappers, numeric long[][] packs and validator cursors once per invocation.
  Rebind temporal numeric bases and reset mutable owner-defined cursor cells before each helper;
  retain per-family pack/reset ownership. Do not pack/clone/reallocate/slice/MemorySegment.ofArray
  per step. Existing length full-capacity validation arrays belong in bind, not executeBound.<span data-glossary-navigation="true"> ([Rebind glossary](../../../../glossary.md#capture-graph-ports-and-reconstruction); [mutable glossary](../../../../glossary.md#term-mutable); [owner glossary](../../../../glossary.md#term-owner); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [slice glossary](../../../../glossary.md#slice); [validation glossary](../../../../glossary.md#term-validation))</span>
- Fix descriptor-derived generated-local reservation for a trailing reference; current two-slot
  last-long assumption cannot survive the trailer. Existing layout locals remain first.<span data-glossary-navigation="true"> ([descriptor glossary](../../../../glossary.md#term-descriptor); [layout glossary](../../../../glossary.md#layout))</span>
- Authenticate the complete composite declaration inventory, names/types/access, JVM parameter,
  code/local/class/constant-pool limits on fresh AND cached artifacts. Derive verification from
  the same method declaration owner, not a second hardcoded one-or-three-method registry.<span data-glossary-navigation="true"> ([JVM glossary](../../../../glossary.md#term-java-virtual-machine-jvm); [parameter glossary](../../../../glossary.md#term-parameter); [constant glossary](../../../../glossary.md#term-constant); [artifacts glossary](../../../../glossary.md#term-artifact); [registry glossary](../../../../glossary.md#term-registry))</span>

## Structural presence and contributor execution

The neutral scopes are [finite](../../../../glossary.md#term-finite) unions of fixed root indices, not numeric-zero tests or a [Boolean](../../../../glossary.md#data-type--datatype)
language. Roots are required carried-[cotangent](../../../../glossary.md#cotangent) channels then selected emission-[seed](../../../../glossary.md#term-seed) formals;
[parameter](../../../../glossary.md#term-parameter) accumulators are NOT carried-cotangent roots. Bind numeric [run](../../../../glossary.md#term-run) control without [runtime](../../../../glossary.md#term-runtime)
semantic plan traversal. Reinitialize it per invocation and preserve inactive [carry](../../../../glossary.md#term-carry) raw.
External [Tensor](../../../../glossary.md#tensor) [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) are present under the reviewed local materialized-boundary clarification;
ordinary [publication](../../../../glossary.md#term-publication) does not export activity. No new physical/neutral [API](../../../../glossary.md#term-application-programming-interface-api) or storage follows.
The accepted scoped four-[argument](../../../../glossary.md#term-argument) [Model](../../../../glossary.md#term-model) ordinary [guard](../../../../glossary.md#term-guard) proves deferred named-[shape](../../../../glossary.md#shape) relations
before analyzers using the exact sole optional declared [symbol](../../../../glossary.md#term-symbol). Preserve its independent
ordinary three-argument API; no copied family proof, sampled B or fake all-valid ROW is permitted.

- Whole-owner absence dominates ALL validators, operand loads, eager/primal-only work and math.
  Explicit zero, cancellation, FLOOR/SIGN zeros and numeric WHERE/attention selection stay present.<span data-glossary-navigation="true"> ([owner glossary](../../../../glossary.md#term-owner); [eager glossary](../../../../glossary.md#term-eager-evaluation); [primal glossary](../../../../glossary.md#term-primal); [cancellation glossary](../../../../glossary.md#term-cancellation))</span>
- Optional ordinary merge ADD has neither/left/right/both behavior: absent, raw sole operand or
  SAME existing ordered ADD. Union-scope ADD with zero substituted for a missing operand is wrong.<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>
- Every present completed shared parameter contribution retains a distinct old.add(contribution),
  including initial +0; absent completed contribution retains old raw. No has-ever-term shortcut,
  scatter-cell sparsity presence or per-row parameter update.
- Maintain declaration/occurrence presence independently of whether a scalar loop had any
  iterations. All-valid B=0/zero-feature occurrences remain present. Length B=0/all-zero-length
  no-body-work is separate. Control needs both complete-occurrence/empty-domain meaning and the
  proved row-indexed root presence; do not derive Shared presence from a numeric reduction or
  assume an empty row-bit array makes an all-valid terminal absent.<span data-glossary-navigation="true"> ([occurrence glossary](../../../../glossary.md#term-occurrence); [scalar glossary](../../../../glossary.md#term-scalar); [feature glossary](../../../../glossary.md#term-feature))</span>
- ROW activity uses the PROVED original row at every valid axis/rank. Guard contributor traversal
  before either operand/index load for terminal SUM, row-contracting MATMUL and shared-target
  ADD-scatter. Reuse existing represented SUM/MM hooks; factor scatter's existing copy/update,
  coordinate and reduction owner with generation-time active-contributor hooks. No zero-masked
  full reduction, reordered updates or duplicate arithmetic implementation.<span data-glossary-navigation="true"> ([axis glossary](../../../../glossary.md#term-axis); [rank glossary](../../../../glossary.md#term-rank); [MATMUL glossary](../../../../glossary.md#matrix-multiplication--matmul); [target glossary](../../../../glossary.md#term-target); [coordinate glossary](../../../../glossary.md#axis-rank-scalar-vector-and-matrix))</span>
- Preserve increasing represented contributor order and ordinary rounding sites. Inactive step
  sequence/seed/history is never loaded; inactive emissions/sequence gradients are positive zero.
  Actual parameter contribution completion precedes its single per-time ADD.<span data-glossary-navigation="true"> ([rounding glossary](../../../../glossary.md#term-rounding); [gradients glossary](../../../../glossary.md#term-gradient); [positive zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero))</span>

## Value guards: existing [owners](../../../../glossary.md#term-owner), complete active domain before writes

Factor existing SoftmaxInputValidator, LossInputValidator, BooleanInputValidator,
AttentionMaskValidator and executable-owned index/scatter loops into shared nonallocating temporal/
range/activity entry points. Existing ordinary bind paths must use the SAME loops/predicates.
Generated helper-call [targets](../../../../glossary.md#term-target) must be accessible, cold-bound storage/cursors retained once.
No additional validator hierarchy or new [finite](../../../../glossary.md#term-finite) policy on Layer/RMS/BatchNorm/categorical [loss](../../../../glossary.md#loss).

For each physical unit: inputs ready -> structural/activity check -> validate ENTIRE eligible
active domain -> all numerical ranges -> completed [publication](../../../../glossary.md#term-publication). A later bad active row/index must
reject before the unit writes any output/history/numeric scratch. Earlier completed units need no
[rollback](../../../../glossary.md#term-rollback). Absent/inactive values are not inspected. [Canonical](../../../../glossary.md#term-canonical-form) [BOOL](../../../../glossary.md#data-type--datatype) 0/1, ignored categorical targets,
GATHER_ND component bounds and empty-output/nonempty-index guards retain ordinary behavior.
Scatter checks bounds before replacement uniqueness; inactive earlier/later update candidates
participate in neither. Do not accidentally restrict its [validation](../../../../glossary.md#term-validation) to output range.

All B lengths validate after their [producers](../../../../glossary.md#term-producer) and before THIS Scan's first [padding](../../../../glossary.md#padding)/output/history/
scratch/control write, including T=0. [Metadata](../../../../glossary.md#term-metadata) geometry [admission](../../../../glossary.md#term-admission) reads no [host](../../../../glossary.md#term-host) values.

## [Identity](../../../../glossary.md#term-identity-meanings), old recipes and independent cold boundaries

No appended empty activity/query/[IR](../../../../glossary.md#term-intermediate-representation-ir) field, blanket class/[schema](../../../../glossary.md#term-schema) bump or old-route namespace churn.
Retain qualified existing recipes when their actual numeric/control facts are unchanged; ordinary
composite bodies and genuine new control facts use a selective namespace authenticated by all
code-shaping typed geometry/roles/demand/scopes/merges/packs/inventory. Exclude actual B/L, addresses,
slots and storage identities. Preserve exact old defaults/fixed/raw/cast identities.
Fresh higher-level [AD](../../../../glossary.md#autograd) may now produce a different correct compact controlled [graph](../../../../glossary.md#graph): distinguish
that from replaying an unchanged frozen qualified recipe. Never compare a new recipe to old bytes
then relabel mismatch as an unchanged old [artifact](../../../../glossary.md#term-artifact), or claim byte retention without running it.
Source/[checkpoint](../../../../glossary.md#term-checkpoint) v1-v5 remain independently old constructor/header/type/role intersections;
activity plans are [compile](../../../../glossary.md#compile) output, not wire data. No broader [persistence](../../../../glossary.md#term-persistence) claim from a [Model](../../../../glossary.md#term-model) change.

## Required actual acceptance

- Unchanged 24-case all-valid/length RELU/SOFTMAX/LayerNorm F32/F64 both-traversal actual forward+
  backward smoke passes; genuine ordinary-family coverage includes normalization/BN saved siblings,
  attention/masks, losses, pooling/windows/grouped Conv2d with nonleading high-rank rows, ordering,
  gather/scatter/embedding, casts/predicates/discrete typed histories and all-SHARED bodies.<span data-glossary-navigation="true"> ([SOFTMAX glossary](../../../../glossary.md#softmax--log-softmax); [F32 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [F64 glossary](../../../../glossary.md#floating-point-precision-and-rounding); [smoke glossary](../../../../glossary.md#term-smoke-test); [normalization glossary](../../../../glossary.md#term-normalization); [masks glossary](../../../../glossary.md#term-boolean-mask); [losses glossary](../../../../glossary.md#loss); [pooling glossary](../../../../glossary.md#term-pooling); [Conv2d glossary](../../../../glossary.md#convolution--conv2d); [rank glossary](../../../../glossary.md#term-rank); [gather glossary](../../../../glossary.md#gather); [embedding glossary](../../../../glossary.md#embedding))</span>
- Compare actual selected backward with independent ordinary active-prefix unrolling, nonuniform
  seeds and T/B=0/1/static/named, tied declarations, empty/no-ROW reverse, mixed/all-zero lengths,
  poison padding/seed/history and same run changed buffers. No handwritten substitute VJP.<span data-glossary-navigation="true"> ([unrolling glossary](../../../../glossary.md#term-unrolling); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [padding glossary](../../../../glossary.md#padding); [seed glossary](../../../../glossary.md#term-seed); [run glossary](../../../../glossary.md#term-run); [VJP glossary](../../../../glossary.md#term-vector-jacobian-product-vjp))</span>
- Prove absence-versus-zero singular/validator behavior, raw signed-zero sole merges, separate
  temporal ADDs, active contributor SUM/MM/scatter and all-valid empty occurrence presence.<span data-glossary-navigation="true"> ([signed-zero glossary](../../../../glossary.md#nan-infinity-and-signed-zero); [occurrence glossary](../../../../glossary.md#term-occurrence))</span>
- Verify that a published zero seed and an explicit zero seed initialize the same present local
  root, while omitted/internal channels remain absent. Compiler's uniform route omission must
  remove the channel, not require a runtime cross-occurrence control lookup.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [runtime glossary](../../../../glossary.md#term-runtime); [lookup glossary](../../../../glossary.md#term-lookup))</span>
- Malformed symbolic `[B,2] -> [B,3]` fails Prepare before backend analysis, while legal all-valid
  whole-Tensor <span data-glossary-navigation="true"> ([Prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [Tensor glossary](../../../../glossary.md#tensor))</span>`[2,B] -> [B,2]` executes across changed B/[buffers](../../../../glossary.md#term-buffer) without a length-locality shortcut.
- Later invalid active values reject before current-unit writes; inactive bad values do not.
  Exercise softmax/noncanonical BOOL/loss indices/gather/scatter bounds+duplicates with real outputs,
  failure retry, mutable-pack resets over multiple steps, alias/alignment and zero output indices.<span data-glossary-navigation="true"> ([BOOL glossary](../../../../glossary.md#data-type--datatype); [mutable glossary](../../../../glossary.md#term-mutable); [alias glossary](../../../../glossary.md#aliasing-and-arena); [alignment glossary](../../../../glossary.md#term-alignment))</span>
- Inspect real generated composite bytecode/inventory and helper ABI; actual runtime has no per-
  step dispatch/packing/allocation. Verify declared resource cleanup, reuse and concurrent runs.<span data-glossary-navigation="true"> ([bytecode glossary](../../../../glossary.md#term-bytecode); [ABI glossary](../../../../glossary.md#term-application-binary-interface-abi); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Run unchanged old-recipe identity/byte capture, affected suites/Javadocs and actual final CPU/
  Metal neural/persistence/installed-consumer workflows, finite relevant performance observations
  and one final root/native checkpoint, truthfully recording skips/unsupported scopes.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [capture glossary](../../../../glossary.md#term-capture-meanings); [CPU glossary](../../../../glossary.md#term-cpu); [Metal glossary](../../../../glossary.md#term-metal); [persistence glossary](../../../../glossary.md#term-persistence); [finite glossary](../../../../glossary.md#term-finite); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>

## Result

The existing local [graph](../../../../glossary.md#graph)/[DAG](../../../../glossary.md#term-directed-acyclic-graph-dag) [decomposition](../../../../glossary.md#term-decomposition), ordinary lowerers, focused emitters and same-class
method declaration [owner](../../../../glossary.md#term-owner) supply body execution. One generated Scan entry retains time traversal;
[run](../../../../glossary.md#term-run)-private typed packs, cursor/control storage, scratch and cleanup use existing [Prepare](../../../../glossary.md#prepare)/[Runtime](../../../../glossary.md#term-runtime)
owners. There is no second executor, runtime body-unit [lookup](../../../../glossary.md#term-lookup) or copied mathematical inventory.
The genuine T0 initial-[carry](../../../../glossary.md#term-carry) [derivative](../../../../glossary.md#term-derivative) borrows its actual selected caller [seed](../../../../glossary.md#term-seed) through the
complete authenticated source-resource path, rather than a fake [producer](../../../../glossary.md#term-producer) or initializer.

MAIN retained the exact 2,673-path joined source in
`fullscan-final-legacy-inspection-removal-source.json`. The recovered unfiltered twelve-project
test/[Javadoc](../../../../glossary.md#term-javadoc) gate passed: 4,942 tests, 64 skips, zero failures/errors; 71 tasks, 21 executed and
50 from [cache](../../../../glossary.md#term-cache). Earlier full-gate and targeted failures remain retained as FAIL. The final
test-only corrections remove legacy-stages/[layout](../../../../glossary.md#layout)/[opcode](../../../../glossary.md#term-opcode) assumptions, not numerical cases,
oracles, tolerances, [target](../../../../glossary.md#term-target) roles or the real [Engine](../../../../glossary.md#term-engine) [lifecycle](../../../../glossary.md#lifecycle).

Actual unchanged numerical consumers passed ordinary84, 208 all-SHARED + 24 ROW [convolution](../../../../glossary.md#term-channels-first-convolution-layers)/
[pooling](../../../../glossary.md#term-pooling) + 36 empty-carry composition cases, the original24 and 144 named-[batch](../../../../glossary.md#term-batch-machine-learning-and-hyperparameters)/prefix cases.
Separate consumers passed structural absence versus present zero, materialized [ND](../../../../glossary.md#term-nd)/FC boundaries,
all-valid whole-[Tensor](../../../../glossary.md#tensor) symbolic reshape, invalid-input/retry controls, sixteen direct T0
caller-seed cases and 91 all-seven-[carrier](../../../../glossary.md#term-storage-carrier) raw/static/named/empty source publications.

Genuine old-recipe replay preserved fourteen captured graph families, 28 persisted [kernel](../../../../glossary.md#kernel) hits,
28 explicit byte definitions and all 85 historical files, with exact keys/[ABI](../../../../glossary.md#term-application-binary-interface-abi)/[IR](../../../../glossary.md#term-intermediate-representation-ir)/class bytes,
forward/[gradient](../../../../glossary.md#term-gradient)/raw observations and freshAD=0. Composite-only v2 fingerprints reject the
authentic old invalid interface-static-call [artifact](../../../../glossary.md#term-artifact) without changing qualified legacy recipes.

[Ownership](../../../../glossary.md#term-ownership-meanings) consumers passed concurrent public callers and barrier-started exact-executable calls.
The public [CPU](../../../../glossary.md#term-cpu) budget was one; generated-entry overlap was not measured. Caller-thread Java-[heap](../../../../glossary.md#term-heap)
observations cover only the recorded [finite](../../../../glossary.md#term-finite) cases: length entries retained fixed overhead with
no observed prefix-dependent slope. They exclude worker/native allocation and do not establish
universal zero allocation, simultaneous generated-entry execution or profitability.

Receipts retained with the source [snapshot](../../../../glossary.md#term-snapshot): `fullscan-producerless-joined-numerical-corpus-result.json`,
`fullscan-producerless-joined-boundary-and-frozen-recipe-result.json`,
`fullscan-producerless-joined-ownership-and-composite-cache-result.json`,
`fullscan-producerless-raw91-all-seven-types-recovery-result.json`,
`fullscan-final-legacy-inspection-removal-integration-gate-result.json` and
`fullscan-final-production-code-review-result.json`. Final artifact/install/root [qualification](../../../../glossary.md#term-qualification)
remains a separate [release](../../../../glossary.md#term-release) step. Cold/wire intersections, [Metal](../../../../glossary.md#term-metal) exclusions and the Java
low-arithmetic lock are unchanged.

The subsequent whole-repository candidate check exposed a genuine [rank](../../../../glossary.md#term-rank)-boundary [regression](../../../../glossary.md#term-regression):
the composite [guard](../../../../glossary.md#term-guard) incorrectly applied the ordinary rank-16 limit to stacked outer containers.
The correction keeps that limit on every local value and reuses exact role [binding](../../../../glossary.md#term-binding) for the
additional time [axis](../../../../glossary.md#term-axis). The existing rank-16/rank-17 test now checks independent forward/initial/
sequence [BPTT](../../../../glossary.md#term-backpropagation-through-time-bptt) values in both directions, not just finiteness. Thirteen additional integration
fixtures explicitly request ZERO only for source-proved T0 or unused targets; connected positive
requests retain strict Engine APIs. The existing Engine test-access owner adapts actual staged
[compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names); production APIs and visibility remain unchanged.

The first whole-root fourteen-failure result and the first [fixture](../../../../glossary.md#term-fixture)-access compilation failure
remain FAIL. The separate corrected three-class numerical recovery passed all 36 tests with no
skips. Exact source and observations are retained in
`private-production-existing-engine-seam-recovery-source.json` and
`private-production-existing-engine-seam-targeted-result.json`; final unfiltered root/artifact
acceptance is separate.
