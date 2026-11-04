# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0045: Bounded [general differentiable scan](../../glossary.md#general-differentiable-scan)

> [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) is authoritative. This ADR records rationale for
> the incorporated [scan semantic contract](../../architecture/contracts/recurrent-scan.md#general-differentiable-scan)<span data-glossary-navigation="true"> ([scan glossary](../../glossary.md#term-cumulative-scan))</span>,
> not an independent contract or implemented capability.

## Status

Accepted — 2026-10-21. After the concrete ScanBody/source-expression, [Compiler](../../glossary.md#term-compiler)-generated reverse
[scan](../../glossary.md#term-cumulative-scan) and construction-time [canonical](../../glossary.md#term-canonical-form) [residual](../../glossary.md#scan-residual-port-and-history-demand) ports with demand-only history proposal, and a
source-checked framework comparison, the user said “ok, tak tedy pokracuj”. The coordinator recorded
this as approval of that specific proposal and authority cutover, not arbitrary control flow or
all implementation slices. The [completed decision brief](../../planning/modules/model/tasks/0034-general-differentiable-scan-decision.md)
records verification and approval; earlier [planning](../../glossary.md#term-planning)-only approval is distinct evidence.

This documentation-only authority enactment is complete: Ramanujan the 2nd returned final
independent Class C APPROVE with zero blocking findings, and main accepted the frozen cutover on
2026-10-21. On 2026-10-22 the user approved the two narrow source-[admission](../../glossary.md#term-admission)/outer-flag
clarifications recorded below; independent authority review and main acceptance preceded code.
The [Model0035 source/IR brief](../../planning/modules/model/tasks/0035-general-scan-source-and-compiled-body.md)<span data-glossary-navigation="true"> ([IR glossary](../../glossary.md#term-intermediate-representation-ir))</span>
records the implemented and validated [Model](../../glossary.md#term-model) foundation: source construction, canonical ports and
structural compiled-body attachments. Planning0008 implements optional compiled-body query transport
and provider contract tests. As of 2026-10-22, Compiler0011 implements scoped [capture](../../glossary.md#term-capture-meanings)/[inference](../../glossary.md#term-inference),
bounded [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) reverse-scan first-order [AD](../../glossary.md#autograd) and final body-bearing query/demand emission.
Prepare0010 now implements scoped projection, static [binding](../../glossary.md#term-binding) checks and complete-[graph](../../glossary.md#graph) demand
[validation](../../glossary.md#term-validation) through the existing resource [handoff](../../glossary.md#term-handoff). CPU0030 subsequently implements bounded
FLOAT32 same-[shape](../../glossary.md#shape) [pointwise](../../glossary.md#term-pointwise) execution and its independently qualified actual Compiler-generated
reverse sibling, with focused executable/resource evidence; its named main [checkpoint](../../glossary.md#term-checkpoint) is separate.
CPU0031 additionally implements qualified [homogeneous](../../glossary.md#term-homogeneous) FLOAT32 broadcast, SUM/sumToShape and bounded
shape/[layout](../../glossary.md#layout) closure, including actual independently qualified reverse bodies. CPU0032 additionally
implements static homogeneous FLOAT32 [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../glossary.md#matrix-multiplication--matmul) in the emitted scan and independently
qualified actual Compiler reverse bodies. CPU0033 implements homogeneous FLOAT64 across the
complete bounded declared algebra and actual first-order reverse closure, with the existing
static/canonical and proved [scalar](../../glossary.md#term-scalar)-read limits. [Padding](../../glossary.md#padding), [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/training, [persistence](../../glossary.md#term-persistence) and low-type
execution remain separately gated. These implementation
facts do not themselves establish [module](../../glossary.md#term-module)/checkpoint acceptance. [Metadata](../../glossary.md#term-metadata)/resource tests
alone do not prove execution. This factual implementation-status update does not reopen the
accepted decision or replace [occurrence](../../glossary.md#term-occurrence)-specific provider [qualification](../../glossary.md#term-qualification).
The no-code/no-measurement statements below describe the original authority-only cutover.

## Context

Existing fixed [RNN](../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm) scans keep source size independent of time and execute on [CPU](../../glossary.md#term-cpu), but
reject backward-capable [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) before [derivative](../../glossary.md#term-derivative) allocation. [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) static recurrence has
ordinary [differentiable](../../glossary.md#derivative-and-chain-rule) [unrolling](../../glossary.md#term-unrolling) but grows source [topology](../../glossary.md#term-topology) with time. We need a bounded loop over
ordinary [Tensor](../../glossary.md#tensor) algebra without moving global [AD](../../glossary.md#autograd) into [backends](../../glossary.md#backend), building a second [gradient](../../glossary.md#term-gradient) algebra
or executor, changing dependencies, or losing exact saved forward values.

Current [producers](../../glossary.md#term-producer) retain [immutable](../../glossary.md#term-immutable) [canonical](../../glossary.md#term-canonical-form) wrappers for every output. Derivative rules such as
[TANH](../../glossary.md#term-tanh) use the original stored output; replay or a replacement [producer](../../glossary.md#term-producer) cannot casually substitute
that [identity](../../glossary.md#term-identity-meanings)/[rounding site](../../glossary.md#term-rounding-site). Existing queries/projection and typed CPU emitters do not already
support generic bodies. Acceptance must therefore include [Model](../../glossary.md#term-model), rule-[binding](../../glossary.md#term-binding), [liveness](../../glossary.md#term-liveness), query,
projection and generated-binding obligations, not just a public loop constructor.

## Decision drivers

- Users compose ordinary [Tensor](../../glossary.md#tensor) algebra, without manual saved-value recipes or [derivative](../../glossary.md#term-derivative) rules.
- Preserve original [producer](../../glossary.md#term-producer)/[publication](../../glossary.md#term-publication) [identities](../../glossary.md#term-identity-meanings) and exact [primal](../../glossary.md#term-primal) saved sites.
- Keep [Compiler](../../glossary.md#term-compiler) [AD](../../glossary.md#autograd) before one combined [capture](../../glossary.md#term-capture-meanings) and [backend](../../glossary.md#backend) execution behind staged [Prepare](../../glossary.md#prepare).
- Keep code/object growth independent of T, live memory truthful and all mutation [run](../../glossary.md#term-run)-owned.
- Preserve existing fixed/static APIs, routes, numerics, [padding](../../glossary.md#padding) and [persistence](../../glossary.md#term-persistence) domains.

## Options considered

### Keep static [unrolling](../../glossary.md#term-unrolling) or add a fixed public [RNN](../../glossary.md#term-recurrent-neural-network-rnn) [adjoint](../../glossary.md#term-adjoint)

Static unrolling retains current algebra but grows [topology](../../glossary.md#term-topology) with time. A fixed RNN adjoint could
address one family but would not provide ordinary-body composition. Neither is selected as the
general solution; existing fixed/static paths remain intact. Superseded Compiler0010 remains
unchanged, not revived by this decision.

### Captured-[graph](../../glossary.md#graph) [AD](../../glossary.md#autograd), postcapture [producer](../../glossary.md#term-producer) replacement or replay

These avoid reserving source [residual](../../glossary.md#scan-residual-port-and-history-demand) wrappers but break the selected pre-[capture](../../glossary.md#term-capture-meanings) boundary or
require a different graph-to-[Tensor](../../glossary.md#tensor) bridge. Replay can change exact saved [rounding](../../glossary.md#term-rounding) sites;
replacement risks original [canonical](../../glossary.md#term-canonical-form) [identity](../../glossary.md#term-identity-meanings). Neither is selected. No [ValueId](../../glossary.md#logical-valueid-versus-physical-memory-slot)-to-Tensor AD map
is allowed; the existing bounded [cold reconstruction](../../glossary.md#term-cold-reconstruction) table remains unrelated.

### Manual saved-[tensor](../../glossary.md#tensor) recipes, attrs/[registry](../../glossary.md#term-registry) bodies or a second executor

Manual recipes burden ordinary users with [derivative](../../glossary.md#term-derivative) internals. Tensor/[graph](../../glossary.md#graph) attrs or registry
keys hide source containment. Another [AD](../../glossary.md#autograd) algebra/executor or [runtime](../../glossary.md#term-runtime) tape moves responsibilities
away from their existing [owners](../../glossary.md#term-owner). These are rejected rather than introduced as convenience layers.

### Source [Tensor](../../glossary.md#tensor) body with [canonical](../../glossary.md#term-canonical-form) [residual](../../glossary.md#scan-residual-port-and-history-demand) [metadata](../../glossary.md#term-metadata) and ordinary reverse [scan](../../glossary.md#term-cumulative-scan)

Selected. Original [producer](../../glossary.md#term-producer) construction reserves canonical residual ports. [Compiler](../../glossary.md#term-compiler) binds exact
original primals to reverse formals and uses ordinary [derivative](../../glossary.md#term-derivative) algebra to build a sibling scan.
Only live downstream/public residual demand allocates histories; [CPU](../../glossary.md#term-cpu) finalization emits the loop
after exact resource assignment. This serves the same composition need without arbitrary regions.

## Decision

Broaden the existing recurrent-[scan](../../glossary.md#term-cumulative-scan) [owner](../../glossary.md#term-owner) in place to own complete fixed/[general scan](../../glossary.md#general-differentiable-scan) semantics;
retain exactly six incorporated contracts. Root and other scoped contracts establish only their
own boundaries and link to that owner. The new inventory is exactly the approved [Model](../../glossary.md#term-model) ScanBody,
Scan/ScanResult, ScanKind/ScanAttrs and CompiledScanBody, existing [producer](../../glossary.md#term-producer)/[node](../../glossary.md#node)/query extensions,
and package-private [Compiler](../../glossary.md#term-compiler) ScanGradientRules/ScanBodyInference. No extra layer, [module](../../glossary.md#term-module), [dependency
edge](../../glossary.md#term-dependency-edge) or [CPU](../../glossary.md#term-cpu) type/[schema](../../glossary.md#term-schema) is selected here.

The [source/scoping protocol](../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities),
[residual-demand protocol](../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand)
and [exact primal-binding/reverse protocol](../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)
are enacted together. Reusable templates [capture](../../glossary.md#term-capture-meanings) independently without producer escape or
cross-template sharing. All [derivative](../../glossary.md#term-derivative) [primal](../../glossary.md#term-primal) operands resolve through explicit [bindings](../../glossary.md#term-binding); [constants](../../glossary.md#term-constant)
close through declared formals/registered outer splats. Joint output [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings) combine once, while
tied positional and cross-[occurrence](../../glossary.md#term-occurrence) terms each contribute once. Dead residuals, including reverse
residuals, allocate/store no history and impose no [residual](../../glossary.md#scan-residual-port-and-history-demand) [fusion](../../glossary.md#term-fusion) restriction.

The 2026-10-22 clarification separates valid Model source construction from the first CPU
execution [qualification](../../glossary.md#term-qualification), and selects conservative outer result flags rather than copying local
body flags. The complete rules live only in the
[owning source section](../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities).
Existing known pure semantics do not become custom-[operation](../../glossary.md#operation) [admission](../../glossary.md#term-admission) or a [backend](../../glossary.md#backend) promise.

Providers qualify the whole body and loop binder with [Tensor](../../glossary.md#tensor)-free facts; one owner prepares the
body once. Exact checked staging/[carry](../../glossary.md#term-carry)/temporary/output/live-history requirements precede slots;
only finalization emits the time loop plus typed body code. There is no [runtime](../../glossary.md#term-runtime) [graph](../../glossary.md#graph) interpreter,
per-op [lookup](../../glossary.md#term-lookup), per-step [callback](../../glossary.md#term-callback)/allocation, T-fold code growth or nested [RunState](../../glossary.md#run-state--runstate). Needed staging
and [MATMUL](../../glossary.md#matrix-multiplication--matmul) storage remain, with [mutable](../../glossary.md#term-mutable) carry/addresses/geometry owned by the [run](../../glossary.md#term-run).

## Rationale

[Canonical](../../glossary.md#term-canonical-form) [residual](../../glossary.md#scan-residual-port-and-history-demand) ports fit Synaptik's [immutable](../../glossary.md#term-immutable)-[producer](../../glossary.md#term-producer)/[publication](../../glossary.md#term-publication) [model](../../glossary.md#term-model) and permit exact
forward-to-reverse edges without replay. This is our design choice, not an industry requirement.
Separating [metadata](../../glossary.md#term-metadata) from demanded history preserves [inference](../../glossary.md#term-inference) [fusion](../../glossary.md#term-fusion) freedom and zero dead-[history
storage](../../glossary.md#scan-residual-port-and-history-demand). Explicit [primal](../../glossary.md#term-primal) [bindings](../../glossary.md#term-binding) let existing [Tensor](../../glossary.md#tensor) formulas retain their semantic algebra
without pretending today's rule entry points already support reverse formals.

Restricting source construction to the first [CPU](../../glossary.md#term-cpu) algebra/[dtypes](../../glossary.md#data-type--datatype) would confuse Model meaning with
provider capability: valid [FLOAT16](../../glossary.md#term-float16) [TANH](../../glossary.md#term-tanh) and [FLOAT32](../../glossary.md#term-float32) RELU source should not need a new Model design
when a later execution [slice](../../glossary.md#slice) qualifies them. Conversely, copying a source emission's local flag
can lose a temporal [dependency](../../glossary.md#term-dependency): with non-[gradient](../../glossary.md#term-gradient) h0, gradient-bearing x, `nextCarry=h+xStep`
and `emit=h`, later emissions depend on earlier x steps. The selected outer-input OR avoids that
false negative without changing formal flags. Its cost is intentional over-approximation for
unused inputs and T=0; it proves neither a nonzero [derivative](../../glossary.md#term-derivative) nor [AD](../../glossary.md#autograd)/[backend](../../glossary.md#backend) support. The
[conceptual temporal example](../../architecture/general-differentiable-scan.md#why-outer-construction-flags-are-conservative)
shows the distinction without claiming implemented [scan](../../glossary.md#term-cumulative-scan) execution.

The bounded exception supersedes only the blanket prohibition on this specific source/captured
[scan body](../../glossary.md#backpropagation-through-time-and-recurrent-state). [ADR 0012](0012-fixed-recurrent-scan-without-regions.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>'s fixed-family design, public
surface, routes and fixed [BPTT](../../glossary.md#term-backpropagation-through-time-bptt) rejection remain intact. Arbitrary [graph](../../glossary.md#graph) regions, if/while,
recursion/nested scans, parallel/associative general scans and mixed-[owner](../../glossary.md#term-owner) bodies remain excluded.

## Consequences

### Positive

Users can eventually express bounded recurrent algebra without time-expanded source or manual
saved-value policies. [Compiler](../../glossary.md#term-compiler) remains the single [AD](../../glossary.md#autograd) [owner](../../glossary.md#term-owner); existing preparation/[runtime](../../glossary.md#term-runtime) [lifecycle](../../glossary.md#lifecycle)
and family numerical rules remain the foundation. Independent scopes allow [immutable](../../glossary.md#term-immutable) template reuse.

### Costs and risks

Construction reserves [metadata](../../glossary.md#term-metadata)/wrappers even when history stays dead. Demanded conservative saved
history costs O(T * retained [elements](../../glossary.md#term-element)) memory and inhibits [fusion](../../glossary.md#term-fusion); selective saving/recomputation
needs further review. Query/projection/demand and [primal](../../glossary.md#term-primal)-[binding](../../glossary.md#term-binding) protocols are real implementation
work. Existing [pointwise](../../glossary.md#term-pointwise)/[carrier](../../glossary.md#term-storage-carrier)/[MATMUL](../../glossary.md#matrix-multiplication--matmul) emitter ABIs require explicit step bindings, not blind
composition. No universal no-copy, [JIT](../../glossary.md#term-just-in-time-compilation-jit) equivalence, speed, zero-penalty or CPU0029 parity is proven.
Generated compatibility must include every baked body/role/demand fact and reject stale [artifacts](../../glossary.md#term-artifact),
without putting extents/slots/[run](../../glossary.md#term-run) bindings into CpuKernelIr [identity](../../glossary.md#term-identity-meanings).

### Compatibility, [validation](../../glossary.md#term-validation) and follow-up

First execution is all-valid/static/[canonical](../../glossary.md#term-canonical-form). A complete bounded [F32](../../glossary.md#floating-point-precision-and-rounding) [pointwise](../../glossary.md#term-pointwise) forward/automatic-
reverse milestone precedes remaining declared algebra/generated closure and [homogeneous](../../glossary.md#term-homogeneous) F32/[F64](../../glossary.md#floating-point-precision-and-rounding)
completion, including DIV. Low types follow under existing [rounding](../../glossary.md#term-rounding)/saved-state semantics. T=0,
empties, exact saved values, tied roles, both [seed](../../glossary.md#term-seed) classes, [scope](../../glossary.md#term-scope)/escape rejection, demand/resource
[overflow](../../glossary.md#term-overflow), overlap, cleanup/retry/[concurrency](../../glossary.md#term-concurrency) and unsupported binder/interior/generated cases need
new focused evidence. [Residual](../../glossary.md#scan-residual-port-and-history-demand)-root and every [scan](../../glossary.md#term-cumulative-scan) higher-order/createGraph request fail before
[derivative](../../glossary.md#term-derivative) allocation. Current fixed tests prove none of this generic capability.

[Padding](../../glossary.md#padding) needs separate row-separability and branch-before-padded-[primal](../../glossary.md#term-primal)/[cotangent](../../glossary.md#cotangent)-read/arithmetic
semantics; masking is not skipping. [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) acquisition/training and [persistence](../../glossary.md#term-persistence) stay independently
gated, including early rejection before setup effects. Low-level [AD](../../glossary.md#autograd) grants no [Model](../../glossary.md#term-model).train/save/load
[admission](../../glossary.md#term-admission) or new v1/v2/v3 wire field. Existing four-[dtype](../../glossary.md#data-type--datatype)/read-[layout](../../glossary.md#layout) fixed [CPU](../../glossary.md#term-cpu) and static NN routes
remain unchanged.

Future CPU evidence uses an optimal specialized clean Java algorithm/hot-loop/[dataflow](../../glossary.md#term-dataflow) [oracle](../../glossary.md#term-test-oracle),
Class-File inspection and short warmed matched-work generated/direct checks; separate cold
generation/class-load/[JIT](../../glossary.md#term-just-in-time-compilation-jit), entry/[preflight](../../glossary.md#term-preflight) and [kernel](../../glossary.md#kernel)/step costs. Fix or justify diagnosed avoidable
overhead. No measurements or Java/build/test-source changes occur now. Future slices own focused/
[module](../../glossary.md#term-module) validation; main owns new cross-boundary coverage and one source-matched native-enabled
root test/[Javadoc](../../glossary.md#term-javadoc) gate at the low-level join. Independent Class C review/main acceptance precede
a separately [Ready](../../glossary.md#term-ready) Model source/[IR](../../glossary.md#term-intermediate-representation-ir) successor; no task becomes Ready automatically.

## Related documentation

- [Focused explanation and conceptual example](../../architecture/general-differentiable-scan.md)
- [Compiler AD boundary](../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../../glossary.md#term-compiler); [AD glossary](../../glossary.md#autograd))</span>
- [Planning query boundary](../../architecture/contracts/foundational-modules.md#profile-free-capability-queries)<span data-glossary-navigation="true"> ([Planning glossary](../../glossary.md#term-planning))</span>
- [Prepare handoff](../../architecture/runtime-prepare-backend-boundary.md#general-differentiable-scan-handoff)<span data-glossary-navigation="true"> ([Prepare glossary](../../glossary.md#prepare); [handoff glossary](../../glossary.md#term-handoff))</span>
- [CPU emission boundary](../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu))</span>
- [NN admission and persistence boundary](../../architecture/contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([NN glossary](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [admission glossary](../../glossary.md#term-admission); [persistence glossary](../../glossary.md#term-persistence))</span>
- [Approved program and serial gates](../../planning/general-differentiable-scan.md)
- [Current fixed capture/early rejection tests](../../../modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/RecurrentScanCompilerTest.java)<span data-glossary-navigation="true"> ([capture glossary](../../glossary.md#term-capture-meanings))</span>
  and [fixed CPU conformance](../../../testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuRecurrentScanConformanceTest.java)<span data-glossary-navigation="true"> ([conformance glossary](../../glossary.md#term-conformance))</span>
  are baseline evidence only, not generic-body qualification.<span data-glossary-navigation="true"> ([qualification glossary](../../glossary.md#term-qualification))</span>
