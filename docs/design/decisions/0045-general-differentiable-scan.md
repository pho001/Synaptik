# ADR 0045: Bounded general differentiable scan

> [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) is authoritative. This ADR records rationale for
> the incorporated [scan semantic contract](../../architecture/contracts/recurrent-scan.md#general-differentiable-scan),
> not an independent contract or implemented capability.

## Status

Accepted — 2026-10-21. After the concrete ScanBody/source-expression, Compiler-generated reverse
scan and construction-time canonical residual ports with demand-only history proposal, and a
source-checked framework comparison, the user said “ok, tak tedy pokracuj”. The coordinator recorded
this as approval of that specific proposal and authority cutover, not arbitrary control flow or
all implementation slices. The [completed decision brief](../../planning/modules/model/tasks/0034-general-differentiable-scan-decision.md)
records verification and approval; earlier planning-only approval is distinct evidence.

This documentation-only authority enactment is complete: Ramanujan the 2nd returned final
independent Class C APPROVE with zero blocking findings, and main accepted the frozen cutover on
2026-10-21. On 2026-10-22 the user approved the two narrow source-admission/outer-flag
clarifications recorded below; independent authority review and main acceptance preceded code.
The [Model0035 source/IR brief](../../planning/modules/model/tasks/0035-general-scan-source-and-compiled-body.md)
records the implemented and validated Model foundation: source construction, canonical ports and
structural compiled-body attachments. Planning0008 implements optional compiled-body query transport
and provider contract tests. As of 2026-10-22, Compiler0011 implements scoped capture/inference,
bounded FLOAT32/FLOAT64 reverse-scan first-order AD and final body-bearing query/demand emission.
Prepare0010 now implements scoped projection, static binding checks and complete-graph demand
validation through the existing resource handoff. CPU0030 subsequently implements bounded
FLOAT32 same-shape pointwise execution and its independently qualified actual Compiler-generated
reverse sibling, with focused executable/resource evidence; its named main checkpoint is separate.
CPU0031 additionally implements qualified homogeneous FLOAT32 broadcast, SUM/sumToShape and bounded
shape/layout closure, including actual independently qualified reverse bodies. CPU0032 additionally
implements static homogeneous FLOAT32 rank-two MATMUL in the emitted scan and independently
qualified actual Compiler reverse bodies. CPU0033 implements homogeneous FLOAT64 across the
complete bounded declared algebra and actual first-order reverse closure, with the existing
static/canonical and proved scalar-read limits. Padding, NN/training, persistence and low-type
execution remain separately gated. These implementation
facts do not themselves establish module/checkpoint acceptance. Metadata/resource tests
alone do not prove execution. This factual implementation-status update does not reopen the
accepted decision or replace occurrence-specific provider qualification.
The no-code/no-measurement statements below describe the original authority-only cutover.

## Context

Existing fixed RNN/GRU/LSTM scans keep source size independent of time and execute on CPU, but
reject backward-capable compilation before derivative allocation. NN static recurrence has
ordinary differentiable unrolling but grows source topology with time. We need a bounded loop over
ordinary Tensor algebra without moving global AD into backends, building a second gradient algebra
or executor, changing dependencies, or losing exact saved forward values.

Current producers retain immutable canonical wrappers for every output. Derivative rules such as
TANH use the original stored output; replay or a replacement producer cannot casually substitute
that identity/rounding site. Existing queries/projection and typed CPU emitters do not already
support generic bodies. Acceptance must therefore include Model, rule-binding, liveness, query,
projection and generated-binding obligations, not just a public loop constructor.

## Decision drivers

- Users compose ordinary Tensor algebra, without manual saved-value recipes or derivative rules.
- Preserve original producer/publication identities and exact primal saved sites.
- Keep Compiler AD before one combined capture and backend execution behind staged Prepare.
- Keep code/object growth independent of T, live memory truthful and all mutation run-owned.
- Preserve existing fixed/static APIs, routes, numerics, padding and persistence domains.

## Options considered

### Keep static unrolling or add a fixed public RNN adjoint

Static unrolling retains current algebra but grows topology with time. A fixed RNN adjoint could
address one family but would not provide ordinary-body composition. Neither is selected as the
general solution; existing fixed/static paths remain intact. Superseded Compiler0010 remains
unchanged, not revived by this decision.

### Captured-graph AD, postcapture producer replacement or replay

These avoid reserving source residual wrappers but break the selected pre-capture boundary or
require a different graph-to-Tensor bridge. Replay can change exact saved rounding sites;
replacement risks original canonical identity. Neither is selected. No ValueId-to-Tensor AD map
is allowed; the existing bounded cold reconstruction table remains unrelated.

### Manual saved-tensor recipes, attrs/registry bodies or a second executor

Manual recipes burden ordinary users with derivative internals. Tensor/graph attrs or registry
keys hide source containment. Another AD algebra/executor or runtime tape moves responsibilities
away from their existing owners. These are rejected rather than introduced as convenience layers.

### Source Tensor body with canonical residual metadata and ordinary reverse scan

Selected. Original producer construction reserves canonical residual ports. Compiler binds exact
original primals to reverse formals and uses ordinary derivative algebra to build a sibling scan.
Only live downstream/public residual demand allocates histories; CPU finalization emits the loop
after exact resource assignment. This serves the same composition need without arbitrary regions.

## Decision

Broaden the existing recurrent-scan owner in place to own complete fixed/general scan semantics;
retain exactly six incorporated contracts. Root and other scoped contracts establish only their
own boundaries and link to that owner. The new inventory is exactly the approved Model ScanBody,
Scan/ScanResult, ScanKind/ScanAttrs and CompiledScanBody, existing producer/node/query extensions,
and package-private Compiler ScanGradientRules/ScanBodyInference. No extra layer, module, dependency
edge or CPU type/schema is selected here.

The [source/scoping protocol](../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities),
[residual-demand protocol](../../architecture/contracts/recurrent-scan.md#canonical-residual-ports-and-demand)
and [exact primal-binding/reverse protocol](../../architecture/contracts/recurrent-scan.md#reverse-scan-and-exact-primal-bindings)
are enacted together. Reusable templates capture independently without producer escape or
cross-template sharing. All derivative primal operands resolve through explicit bindings; constants
close through declared formals/registered outer splats. Joint output seeds combine once, while
tied positional and cross-occurrence terms each contribute once. Dead residuals, including reverse
residuals, allocate/store no history and impose no residual fusion restriction.

The 2026-10-22 clarification separates valid Model source construction from the first CPU
execution qualification, and selects conservative outer result flags rather than copying local
body flags. The complete rules live only in the
[owning source section](../../architecture/contracts/recurrent-scan.md#source-body-attachments-and-scoped-identities).
Existing known pure semantics do not become custom-operation admission or a backend promise.

Providers qualify the whole body and loop binder with Tensor-free facts; one owner prepares the
body once. Exact checked staging/carry/temporary/output/live-history requirements precede slots;
only finalization emits the time loop plus typed body code. There is no runtime graph interpreter,
per-op lookup, per-step callback/allocation, T-fold code growth or nested RunState. Needed staging
and MATMUL storage remain, with mutable carry/addresses/geometry owned by the run.

## Rationale

Canonical residual ports fit Synaptik's immutable-producer/publication model and permit exact
forward-to-reverse edges without replay. This is our design choice, not an industry requirement.
Separating metadata from demanded history preserves inference fusion freedom and zero dead-history
storage. Explicit primal bindings let existing Tensor formulas retain their semantic algebra
without pretending today's rule entry points already support reverse formals.

Restricting source construction to the first CPU algebra/dtypes would confuse Model meaning with
provider capability: valid FLOAT16 TANH and FLOAT32 RELU source should not need a new Model design
when a later execution slice qualifies them. Conversely, copying a source emission's local flag
can lose a temporal dependency: with non-gradient h0, gradient-bearing x, `nextCarry=h+xStep`
and `emit=h`, later emissions depend on earlier x steps. The selected outer-input OR avoids that
false negative without changing formal flags. Its cost is intentional over-approximation for
unused inputs and T=0; it proves neither a nonzero derivative nor AD/backend support. The
[conceptual temporal example](../../architecture/general-differentiable-scan.md#why-outer-construction-flags-are-conservative)
shows the distinction without claiming implemented scan execution.

The bounded exception supersedes only the blanket prohibition on this specific source/captured
scan body. [ADR 0012](0012-fixed-recurrent-scan-without-regions.md)'s fixed-family design, public
surface, routes and fixed BPTT rejection remain intact. Arbitrary graph regions, if/while,
recursion/nested scans, parallel/associative general scans and mixed-owner bodies remain excluded.

## Consequences

### Positive

Users can eventually express bounded recurrent algebra without time-expanded source or manual
saved-value policies. Compiler remains the single AD owner; existing preparation/runtime lifecycle
and family numerical rules remain the foundation. Independent scopes allow immutable template reuse.

### Costs and risks

Construction reserves metadata/wrappers even when history stays dead. Demanded conservative saved
history costs O(T * retained elements) memory and inhibits fusion; selective saving/recomputation
needs further review. Query/projection/demand and primal-binding protocols are real implementation
work. Existing pointwise/carrier/MATMUL emitter ABIs require explicit step bindings, not blind
composition. No universal no-copy, JIT equivalence, speed, zero-penalty or CPU0029 parity is proven.
Generated compatibility must include every baked body/role/demand fact and reject stale artifacts,
without putting extents/slots/run bindings into CpuKernelIr identity.

### Compatibility, validation and follow-up

First execution is all-valid/static/canonical. A complete bounded F32 pointwise forward/automatic-
reverse milestone precedes remaining declared algebra/generated closure and homogeneous F32/F64
completion, including DIV. Low types follow under existing rounding/saved-state semantics. T=0,
empties, exact saved values, tied roles, both seed classes, scope/escape rejection, demand/resource
overflow, overlap, cleanup/retry/concurrency and unsupported binder/interior/generated cases need
new focused evidence. Residual-root and every scan higher-order/createGraph request fail before
derivative allocation. Current fixed tests prove none of this generic capability.

Padding needs separate row-separability and branch-before-padded-primal/cotangent-read/arithmetic
semantics; masking is not skipping. NN acquisition/training and persistence stay independently
gated, including early rejection before setup effects. Low-level AD grants no Model.train/save/load
admission or new v1/v2/v3 wire field. Existing four-dtype/read-layout fixed CPU and static NN routes
remain unchanged.

Future CPU evidence uses an optimal specialized clean Java algorithm/hot-loop/dataflow oracle,
Class-File inspection and short warmed matched-work generated/direct checks; separate cold
generation/class-load/JIT, entry/preflight and kernel/step costs. Fix or justify diagnosed avoidable
overhead. No measurements or Java/build/test-source changes occur now. Future slices own focused/
module validation; main owns new cross-boundary coverage and one source-matched native-enabled
root test/Javadoc gate at the low-level join. Independent Class C review/main acceptance precede
a separately Ready Model source/IR successor; no task becomes Ready automatically.

## Related documentation

- [Focused explanation and conceptual example](../../architecture/general-differentiable-scan.md)
- [Compiler AD boundary](../../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
- [Planning query boundary](../../architecture/contracts/foundational-modules.md#profile-free-capability-queries)
- [Prepare handoff](../../architecture/runtime-prepare-backend-boundary.md#general-differentiable-scan-handoff)
- [CPU emission boundary](../../architecture/contracts/backend-execution.md#cpu-backend-routes)
- [NN admission and persistence boundary](../../architecture/contracts/extensions-training.md#owned-tensor-models)
- [Approved program and serial gates](../../planning/general-differentiable-scan.md)
- [Current fixed capture/early rejection tests](../../../modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/RecurrentScanCompilerTest.java)
  and [fixed CPU conformance](../../../testing/backend-conformance/src/test/java/io/github/pho001/synaptik/testing/conformance/CpuRecurrentScanConformanceTest.java)
  are baseline evidence only, not generic-body qualification.
