# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0051: [Symbolic Scan](../../glossary.md#symbolic-scan) and bounded geometry reuse

## Status

Accepted — 2026-10-25. At decision time, these were accepted-but-unimplemented semantics. The coordinated authority
delivery [Model0041](../../planning/modules/model/tasks/0041-symbolic-scan-authority-cutover.md)
is Complete after independent Class C APPROVE, zero open P1/P2; MAIN accepted 2026-10-25. That delivery
itself authorized no source successor. This [ADR](../../glossary.md#term-architecture-decision-record-adr) records rationale; the
[architecture root](../../../ARCHITECTURE.md#scope-indexed-normative-contracts) and incorporated
contracts alone are authoritative. ADR0050's original evidence and decision remain unchanged.
Bounded status amendment — 2026-10-26: Model0042/Compiler0014/Prepare0014 implement the all-valid
[F32](../../glossary.md#floating-point-precision-and-rounding) vertical; Model0044/Compiler0015 implement symbolic-length source/[inference](../../glossary.md#term-inference)/actual [AD](../../glossary.md#autograd)/[binding](../../glossary.md#term-binding).
Prepare0015's combined F32/[F64](../../glossary.md#floating-point-precision-and-rounding) execution and bounded outer/control/ROW(1) [target](../../glossary.md#term-target) passed independent
pre-Java review and MAIN acceptance. Implementation amendment — 2026-10-26: the bounded combined
path is implemented, with owning [validation](../../glossary.md#term-validation) and independent code/structural review passed. Root validation and
the sole bounded probe completed (evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending. This selects no [Engine](../../glossary.md#term-engine) [memo](../../glossary.md#cache-and-memoization) or new type/[API](../../glossary.md#term-application-programming-interface-api)/protocol.

## Context

Ordinary [homogeneous](../../glossary.md#term-homogeneous) [FLOAT32](../../glossary.md#term-float32) [CPU](../../glossary.md#term-cpu) symbolic-[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) forward and actual first-order backward already
reuse one [compilation](../../glossary.md#compiler-engine-and-execution-stage-names)/preparation. At decision time, [general Scan](../../glossary.md#general-differentiable-scan) still rejected symbolic [shapes](../../glossary.md#shape) in [Model](../../glossary.md#term-model), [Compiler](../../glossary.md#term-compiler)
and [Prepare](../../glossary.md#prepare); CPU staged [lowering](../../glossary.md#lowering)/emission bakes extents, [singleton](../../glossary.md#term-singleton) decisions and resource offsets.
Changing only invocation [binding](../../glossary.md#term-binding) cannot implement [symbolic Scan](../../glossary.md#symbolic-scan). Static FLOAT32/[FLOAT64](../../glossary.md#term-float64) [Scan](../../glossary.md#term-cumulative-scan),
its qualified actual-read [views](../../glossary.md#term-view-meanings) and fixed recurrent operations remain the retained baseline.

The user confirmed the original one-B design plus the Model0040 same-geometry amendment and
explicitly authorized contract-first work. The desired [mental model](../../glossary.md#term-mental-model) is unchanged meaning and code,
with actual-size resources for each invocation. Repeated B should permit reuse of pure geometry
facts, not input [validation](../../glossary.md#term-validation) or a prior invocation. Current private-array/prepared-immutability
wording therefore needs a narrow explicit exception, not an inferred [cache](../../glossary.md#term-cache) permission from tests.

## Decision drivers

- Keep one [Scan](../../glossary.md#term-cumulative-scan) semantic [owner](../../glossary.md#term-owner) and existing [compile](../../glossary.md#compile)/[prepare](../../glossary.md#prepare)/[run](../../glossary.md#term-run) boundaries, with numeric-only [Runtime](../../glossary.md#term-runtime).
- Retain complete actual generated first-order algebra, exact saved values and constituent numerics.
- Reuse code/routes/slots across B without maximum banks, hidden [topology](../../glossary.md#term-topology) changes or per-B preparation.
- Avoid repeated covered [Engine](../../glossary.md#term-engine) geometry work while admitting every current [descriptor](../../glossary.md#term-descriptor) and storage.
- Bound [metadata](../../glossary.md#term-metadata) retention and preserve failure, close precedence, sole Runtime [lease](../../glossary.md#term-lease) and run isolation.
- Select no new production type/public [signature](../../glossary.md#term-signature) or [concurrency](../../glossary.md#term-concurrency) mechanism without exact later approval.

## Options considered

The alternatives address the same fixed-T/[features](../../glossary.md#term-software-feature) recurrence; the table separates reuse of
executable meaning from reuse of numeric facts.

| Option | Across changed B | Repeated same B | Main cost or limitation |
|---|---|---|---|
| Status quo: static [Scan](../../glossary.md#term-cumulative-scan) only | New static [compilation](../../glossary.md#compiler-engine-and-execution-stage-names)/preparation | Existing prepared static [run](../../glossary.md#term-run) | No genuine [symbolic Scan](../../glossary.md#symbolic-scan); static execution remains supported |
| Per-B static respecialization | Another specialized [graph](../../glossary.md#graph)/preparation/[artifact](../../glossary.md#term-artifact) | Per-B [cache](../../glossary.md#term-cache) could hit | Multiplies plans/code/cache [ownership](../../glossary.md#term-ownership-meanings) and evades the selected one-preparation goal |
| Selected symbolic numeric [binding](../../glossary.md#term-binding) plus one [metadata](../../glossary.md#term-metadata) entry | Same graph/code/routes/slots, exact actual bytes | Reuse covered pure [Engine](../../glossary.md#term-engine) facts after full fresh [admission](../../glossary.md#term-admission) | Requires scoped obligations, all-port [layout](../../glossary.md#layout) recipes and careful [publication](../../glossary.md#term-publication)/close coordination |

A mandatory B_max bank would waste capacity and still require actual metadata/count correctness;
it is not selected. A [Runtime](../../glossary.md#term-runtime) cache would move semantic admission into the wrong [owner](../../glossary.md#term-owner). Retaining
Bindings or supplied [descriptors](../../glossary.md#term-descriptor) would let same-B callers inherit stale [view](../../glossary.md#term-view-meanings) metadata or storage.

## Decision

Accept the bounded [symbolic Scan semantics](../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan)<span data-glossary-navigation="true"> ([symbolic Scan glossary](../../glossary.md#symbolic-scan))</span>:
one named B with fixed T/[features](../../glossary.md#term-software-feature)/state widths, existing all-valid surface and exact scoped ports.
[Model](../../glossary.md#term-model) [PURE](../../glossary.md#term-pure)/[dtype](../../glossary.md#data-type--datatype) validity, bounded [FLOAT32](../../glossary.md#term-float32)/[FLOAT64](../../glossary.md#term-float64) [Compiler](../../glossary.md#term-compiler) [AD](../../glossary.md#autograd) and initial [homogeneous](../../glossary.md#term-homogeneous) FLOAT32
generated [scalar](../../glossary.md#term-scalar) single-range [CPU](../../glossary.md#term-cpu) execution remain distinct admissions. Forward and actual
generated reverse must qualify independently; this decision implements none of them.

Accept one [metadata](../../glossary.md#term-metadata)-only replacement slot on the exact [Engine](../../glossary.md#term-engine) prepared [owner](../../glossary.md#term-owner) under the
[memo contract](../../architecture/contracts/runtime-prepare-engine.md#accepted-engine-geometry-memo).
It retains only [immutable](../../glossary.md#term-immutable) same-plan numeric facts and template-derived non-input metadata, keyed
by exact semantic/inward/memory-plan [identities](../../glossary.md#term-identity-meanings) and B/mode. Every call retains complete fresh
Compiler bind/private obligations, current input-backed [descriptors](../../glossary.md#term-descriptor), storage/physical checks and
fresh resources/[mutable](../../glossary.md#term-mutable) working arrays. [Runtime](../../glossary.md#term-runtime) receives numeric geometry, not the [memo](../../glossary.md#cache-and-memoization).

[Atomic](../../glossary.md#resource-transaction-and-rollback) complete-entry [publication](../../glossary.md#term-publication) uses local immutable snapshots and active-[admission](../../glossary.md#term-admission)-bounded
candidates. Failed admission cannot replace a good entry. Close clears/prevents late publication;
after inward admission it suppresses memo publication, not the [run](../../glossary.md#term-run) via a new late outward-close
gate. Runtime's existing sole [lease](../../glossary.md#term-lease) decides the later [race](../../glossary.md#term-race). Exact fields/methods and coordination
mechanism remain later scoped-brief decisions, not a selected lock/[API](../../glossary.md#term-application-programming-interface-api)/protocol here.

## Rationale

Symbolic [axis](../../glossary.md#term-axis) [identity](../../glossary.md#term-identity-meanings) carries meaning that a [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) B cannot: [parameter](../../glossary.md#term-parameter) adjoints contract K=B,
[MEAN](../../glossary.md#term-mean) counts actual B*H, shared work can remain nonempty at B=0 and every stacked port needs its
own actual [layout](../../glossary.md#layout). Exact numeric recipes preserve these facts without [Runtime](../../glossary.md#term-runtime) [graph](../../glossary.md#graph) access or
per-[element](../../glossary.md#term-element) [lookup](../../glossary.md#term-lookup). Fixed [topology](../../glossary.md#term-topology) also keeps generation and compatibility reasoning independent
of invocation B. A one-entry [memo](../../glossary.md#cache-and-memoization) bounds retention and helps consecutive equal geometries without
becoming an executable [cache](../../glossary.md#term-cache). Fresh [admission](../../glossary.md#term-admission) prevents a hit from authorizing stale callers.

## Consequences

### Positive

The selected direction preserves existing [owners](../../glossary.md#term-owner) and one prepared meaning while allowing actual
resource sizing and narrowly reusable [immutable](../../glossary.md#term-immutable) facts. Static [identities](../../glossary.md#term-identity-meanings)/read [admission](../../glossary.md#term-admission) and
numerics stay unchanged. The [future workflow example](../../architecture/runtime-prepare-backend-boundary.md#future-symbolic-scan-and-geometry-reuse)
shows both actual bytes and miss/hit behavior without presenting a runnable [API](../../glossary.md#term-application-programming-interface-api).

### Negative and risks

[Compiler](../../glossary.md#term-compiler) scoped obligations/all-output [binding](../../glossary.md#term-binding) and [CPU](../../glossary.md#term-cpu) staged recipes/emission required substantive
implementation; Status above distinguishes completed all-valid work and the accepted combined [target](../../glossary.md#term-target).
Piecewise [alignment](../../glossary.md#term-alignment) and empty scratch need checked arithmetic, not a
blanket affine assumption. [Concurrency](../../glossary.md#term-concurrency)/close correctness and copying facts into [mutable](../../glossary.md#term-mutable) working
arrays add entry complexity; a [memo](../../glossary.md#cache-and-memoization) does not eliminate bind/storage/allocation costs or promise
speed. Alternating B values miss by design. Existing static and ordinary performance evidence is
retained baseline only, not [symbolic Scan](../../glossary.md#symbolic-scan) or memo [qualification](../../glossary.md#term-qualification).

### Migration, testing, and follow-up

Authority review and MAIN acceptance precede separately [Ready](../../glossary.md#term-ready) [Model](../../glossary.md#term-model) foundation, [Compiler](../../glossary.md#term-compiler)
[inference](../../glossary.md#term-inference)/actual [AD](../../glossary.md#autograd)/all-port [binding](../../glossary.md#term-binding) and coordinated [Prepare](../../glossary.md#prepare)/[CPU](../../glossary.md#term-cpu)/[Engine](../../glossary.md#term-engine) vertical gates. No new
type/[signature](../../glossary.md#term-signature) is selected; any necessary new type or [Runtime](../../glossary.md#term-runtime) [handoff](../../glossary.md#term-handoff) must stop for exact
package/name/purpose/signature approval. Fixed length execution is now Complete and Prepare0015's
combined execution is implemented, not a [memo](../../glossary.md#cache-and-memoization) [qualification](../../glossary.md#term-qualification). [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training state reuse,
cold/copy/wire, variable T, wider [dtypes](../../glossary.md#data-type--datatype), native/mixed and [vector](../../glossary.md#term-vector)/parallel retain separate gates.

Later matching [module](../../glossary.md#term-module), [conformance](../../glossary.md#term-conformance) and integration tests must cover nonuniform forward/actual
reverse versus ordinary [unrolling](../../glossary.md#term-unrolling), both traversals/[seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings), exact saved sites, fixed [parameter](../../glossary.md#term-parameter)
[gradients](../../glossary.md#term-gradient), all scoped ports, B=0/1/shared work and a separate fixed-T=0 plan. Validate exact bytes,
[overflow](../../glossary.md#term-overflow) before borrow, [aliases](../../glossary.md#aliasing-and-arena), failure/retry/close/concurrent isolation, current input-backed
[views](../../glossary.md#term-view-meanings), foreign-plan rejection and the bounded memo hit witness. Generated Class-Files must match
specialized clean-Java algorithm/hot-loop/[dataflow](../../glossary.md#term-dataflow) with no memo access or avoidable hidden
allocation/[dispatch](../../glossary.md#term-dispatch); bounded warmed matched-work evidence separates setup/[admission](../../glossary.md#term-admission)/[kernel](../../glossary.md#kernel) costs.
Retain static [identity](../../glossary.md#term-identity-meanings)/byte/read and unsupported length-boundary rejection regressions. No [dependency](../../glossary.md#term-dependency), [API](../../glossary.md#term-application-programming-interface-api) or executable
change occurs here, so no architecture-test rerun or performance [run](../../glossary.md#term-run) is required now.

## Related documentation

- [Compiler binding boundary](../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)<span data-glossary-navigation="true"> ([Compiler glossary](../../glossary.md#term-compiler); [binding glossary](../../glossary.md#term-binding))</span>
- [Numeric sizing and Runtime handoff](../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [handoff glossary](../../glossary.md#term-handoff))</span>
- [CPU route/emission boundary](../../architecture/contracts/backend-execution.md#cpu-backend-routes)<span data-glossary-navigation="true"> ([CPU glossary](../../glossary.md#term-cpu))</span>
- [Reviewed design and owner map](../../planning/modules/model/general-scan-symbolic-batch-proposal.md)<span data-glossary-navigation="true"> ([owner glossary](../../glossary.md#term-owner))</span>
- [Prior ordinary geometry decision](0050-dynamic-batch-geometry.md)
