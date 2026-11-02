# ADR 0051: Symbolic Scan and bounded geometry reuse

## Status

Accepted — 2026-10-25. At decision time, these were accepted-but-unimplemented semantics. The coordinated authority
delivery [Model0041](../../planning/modules/model/tasks/0041-symbolic-scan-authority-cutover.md)
is Complete after independent Class C APPROVE, zero open P1/P2; MAIN accepted 2026-10-25. That delivery
itself authorized no source successor. This ADR records rationale; the
[architecture root](../../../ARCHITECTURE.md#scope-indexed-normative-contracts) and incorporated
contracts alone are authoritative. ADR0050's original evidence and decision remain unchanged.
Bounded status amendment — 2026-10-26: Model0042/Compiler0014/Prepare0014 implement the all-valid
F32 vertical; Model0044/Compiler0015 implement symbolic-length source/inference/actual AD/binding.
Prepare0015's combined F32/F64 execution and bounded outer/control/ROW(1) target passed independent
pre-Java review and MAIN acceptance. Implementation amendment — 2026-10-26: the bounded combined
path is implemented, with owning validation and independent code/structural review passed. Root validation and
the sole bounded probe completed (evidence and limits:
[Prepare0015 Result](../../planning/modules/prepare/tasks/0015-symbolic-length-scan-cpu-vertical.md#result) );
final MAIN acceptance remains pending. This selects no Engine memo or new type/API/protocol.

## Context

Ordinary homogeneous FLOAT32 CPU symbolic-batch forward and actual first-order backward already
reuse one compilation/preparation. At decision time, general Scan still rejected symbolic shapes in Model, Compiler
and Prepare; CPU staged lowering/emission bakes extents, singleton decisions and resource offsets.
Changing only invocation binding cannot implement symbolic Scan. Static FLOAT32/FLOAT64 Scan,
its qualified actual-read views and fixed recurrent operations remain the retained baseline.

The user confirmed the original one-B design plus the Model0040 same-geometry amendment and
explicitly authorized contract-first work. The desired mental model is unchanged meaning and code,
with actual-size resources for each invocation. Repeated B should permit reuse of pure geometry
facts, not input validation or a prior invocation. Current private-array/prepared-immutability
wording therefore needs a narrow explicit exception, not an inferred cache permission from tests.

## Decision drivers

- Keep one Scan semantic owner and existing compile/prepare/run boundaries, with numeric-only Runtime.
- Retain complete actual generated first-order algebra, exact saved values and constituent numerics.
- Reuse code/routes/slots across B without maximum banks, hidden topology changes or per-B preparation.
- Avoid repeated covered Engine geometry work while admitting every current descriptor and storage.
- Bound metadata retention and preserve failure, close precedence, sole Runtime lease and run isolation.
- Select no new production type/public signature or concurrency mechanism without exact later approval.

## Options considered

The alternatives address the same fixed-T/features recurrence; the table separates reuse of
executable meaning from reuse of numeric facts.

| Option | Across changed B | Repeated same B | Main cost or limitation |
|---|---|---|---|
| Status quo: static Scan only | New static compilation/preparation | Existing prepared static run | No genuine symbolic Scan; static execution remains supported |
| Per-B static respecialization | Another specialized graph/preparation/artifact | Per-B cache could hit | Multiplies plans/code/cache ownership and evades the selected one-preparation goal |
| Selected symbolic numeric binding plus one metadata entry | Same graph/code/routes/slots, exact actual bytes | Reuse covered pure Engine facts after full fresh admission | Requires scoped obligations, all-port layout recipes and careful publication/close coordination |

A mandatory B_max bank would waste capacity and still require actual metadata/count correctness;
it is not selected. A Runtime cache would move semantic admission into the wrong owner. Retaining
Bindings or supplied descriptors would let same-B callers inherit stale view metadata or storage.

## Decision

Accept the bounded [symbolic Scan semantics](../../architecture/contracts/recurrent-scan.md#accepted-symbolic-batch-scan):
one named B with fixed T/features/state widths, existing all-valid surface and exact scoped ports.
Model PURE/dtype validity, bounded FLOAT32/FLOAT64 Compiler AD and initial homogeneous FLOAT32
generated scalar single-range CPU execution remain distinct admissions. Forward and actual
generated reverse must qualify independently; this decision implements none of them.

Accept one metadata-only replacement slot on the exact Engine prepared owner under the
[memo contract](../../architecture/contracts/runtime-prepare-engine.md#accepted-engine-geometry-memo).
It retains only immutable same-plan numeric facts and template-derived non-input metadata, keyed
by exact semantic/inward/memory-plan identities and B/mode. Every call retains complete fresh
Compiler bind/private obligations, current input-backed descriptors, storage/physical checks and
fresh resources/mutable working arrays. Runtime receives numeric geometry, not the memo.

Atomic complete-entry publication uses local immutable snapshots and active-admission-bounded
candidates. Failed admission cannot replace a good entry. Close clears/prevents late publication;
after inward admission it suppresses memo publication, not the run via a new late outward-close
gate. Runtime's existing sole lease decides the later race. Exact fields/methods and coordination
mechanism remain later scoped-brief decisions, not a selected lock/API/protocol here.

## Rationale

Symbolic axis identity carries meaning that a sample B cannot: parameter adjoints contract K=B,
MEAN counts actual B*H, shared work can remain nonempty at B=0 and every stacked port needs its
own actual layout. Exact numeric recipes preserve these facts without Runtime graph access or
per-element lookup. Fixed topology also keeps generation and compatibility reasoning independent
of invocation B. A one-entry memo bounds retention and helps consecutive equal geometries without
becoming an executable cache. Fresh admission prevents a hit from authorizing stale callers.

## Consequences

### Positive

The selected direction preserves existing owners and one prepared meaning while allowing actual
resource sizing and narrowly reusable immutable facts. Static identities/read admission and
numerics stay unchanged. The [future workflow example](../../architecture/runtime-prepare-backend-boundary.md#future-symbolic-scan-and-geometry-reuse)
shows both actual bytes and miss/hit behavior without presenting a runnable API.

### Negative and risks

Compiler scoped obligations/all-output binding and CPU staged recipes/emission required substantive
implementation; Status above distinguishes completed all-valid work and the accepted combined target.
Piecewise alignment and empty scratch need checked arithmetic, not a
blanket affine assumption. Concurrency/close correctness and copying facts into mutable working
arrays add entry complexity; a memo does not eliminate bind/storage/allocation costs or promise
speed. Alternating B values miss by design. Existing static and ordinary performance evidence is
retained baseline only, not symbolic Scan or memo qualification.

### Migration, testing, and follow-up

Authority review and MAIN acceptance precede separately Ready Model foundation, Compiler
inference/actual AD/all-port binding and coordinated Prepare/CPU/Engine vertical gates. No new
type/signature is selected; any necessary new type or Runtime handoff must stop for exact
package/name/purpose/signature approval. Fixed length execution is now Complete and Prepare0015's
combined execution is implemented, not a memo qualification. NN/Training state reuse,
cold/copy/wire, variable T, wider dtypes, native/mixed and vector/parallel retain separate gates.

Later matching module, conformance and integration tests must cover nonuniform forward/actual
reverse versus ordinary unrolling, both traversals/seeds, exact saved sites, fixed parameter
gradients, all scoped ports, B=0/1/shared work and a separate fixed-T=0 plan. Validate exact bytes,
overflow before borrow, aliases, failure/retry/close/concurrent isolation, current input-backed
views, foreign-plan rejection and the bounded memo hit witness. Generated Class-Files must match
specialized clean-Java algorithm/hot-loop/dataflow with no memo access or avoidable hidden
allocation/dispatch; bounded warmed matched-work evidence separates setup/admission/kernel costs.
Retain static identity/byte/read and unsupported length-boundary rejection regressions. No dependency, API or executable
change occurs here, so no architecture-test rerun or performance run is required now.

## Related documentation

- [Compiler binding boundary](../../architecture/contracts/compiler-autograd.md#accepted-symbolic-batch-metadata-binding)
- [Numeric sizing and Runtime handoff](../../architecture/contracts/runtime-prepare-engine.md#accepted-symbolic-batch-resource-contract)
- [CPU route/emission boundary](../../architecture/contracts/backend-execution.md#cpu-backend-routes)
- [Reviewed design and owner map](../../planning/modules/model/general-scan-symbolic-batch-proposal.md)
- [Prior ordinary geometry decision](0050-dynamic-batch-geometry.md)
