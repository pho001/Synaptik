# From an expression to a completed execution

> This atlas explains the current implementation. [The architecture root](../../../ARCHITECTURE.md)
> and its incorporated [foundational](../../architecture/contracts/foundational-modules.md),
> [Compiler/AD](../../architecture/contracts/compiler-autograd.md), and
> [Runtime/Prepare/Engine](../../architecture/contracts/runtime-prepare-engine.md) contracts are
> authoritative; this page does not extend them.

This page follows the packages that turn a Tensor expression into reusable executable work,
then run that work with concrete values. Read it to understand **which stage makes a decision,
which object owns a resource, and why changing input values need not compile another graph**.
Start with the [learning journey](../../tutorials/index.md) for runnable application code;
this is a source-reading companion, not a recipe for constructing internal execution objects.

Public Java visibility is not the same as an ordinary application entry point. Applications
normally use Model and the public `engine.Engine` lifecycle. The compiler, planning, Prepare and
Runtime collaborations below explain implementation boundaries; package-private classes are not
supported application API. Backend implementations remain in their own modules, not in these
shared packages.

## The distinctions to keep while reading

Read the following flow from top to bottom. The left column is a stage, not a dependency graph.
The right column identifies what survives that stage.

```text
Tensor expressions               meaning and caller-owned leaves
  -> Compiler                    immutable CompileArtifacts: graph + logical recipes
     -> Planning                 backend ownership + partitions + logical value requirements
  -> Prepare                     analysis -> shared slot assignment -> backend finalization
                                 immutable executable/schedule/creation recipes
                                 + explicitly owned persistent prepared resources
  -> Runtime run                 one RunState + borrowed inputs + fresh physical representations
                                 -> completed publication occurrences -> closeable RunResult
  -> explicit materialization    detached HostTensorValue, independent of all execution owners
```

| Term | What it means here | What it is not |
|---|---|---|
| Compile recipe | Immutable graph, input/publication associations and backend-neutral plans | An executable or a buffer allocation |
| Prepared recipe | Fixed routes, code, resource selections and schedule order | Mutable invocation data |
| Persistent prepared resource | A backend physical owner acquired during finalization, such as a compiled native executable | Per-run scratch, or every occurrence of an executable in a schedule |
| Buffer/workspace slot | A plan-local logical position with exact byte/alignment declarations | An address or a concrete device buffer |
| Representation | A concrete backend object supplying physical storage for a slot in one run | A graph `ValueId` or a new logical value |
| Prepared run lease | Runtime's permission for one synchronous call to use persistent prepared resources | The returned result lifetime |
| Run result | An owner over completed run-state resources and publication lifetime | An asynchronous job, future or implicit host copy |
| Detached host value | Immutable canonical host bytes copied explicitly from one publication | A Tensor expression or a lease into backend storage |

In particular, two classes share the name `PreparedExecution`:

- `runtime.execution.PreparedExecution` is the backend-neutral prepared owner: one exact memory
  plan, one same-plan schedule, and unique persistent resource identities.
- `engine.PreparedExecution` is the owner-bound ordinary public handle around that exact inward
  owner. It also retains compiled metadata and direct boundary adapters. It does **not** introduce
  a second prepared-resource lease protocol.

The [glossary's compile/prepare/run comparison](../../glossary.md#compile-versus-prepare-versus-run)
provides a shorter reminder of these terms.

## One numerical thread through the packages

This is a **conceptual walkthrough**, not output from a run performed for this atlas. The complete
program, prerequisites and commands belong to the [examples](../../examples/README.md).

Use the canonical `LearningJourney` Model: FLOAT32 input `X` of shape `[2,1]` with values `[1,3]`,
a supplied gradient-eligible weight `W` of shape `[1,1]` initially `0`, no bias, and target `[2,6]`.
Model copies the supplied weight into its owned parameter state. The prediction is
`X @ transpose(W)`; the objective is the **SUM** of squared errors, not their mean.

```text
prediction at W=0:             [0,0]
errors:                        [-2,-6]
objective:                     4 + 36 = 40
weight gradient:               2 * ((-2)*1 + (-6)*3) = -40
Sgd(0.05) update:               W = 0 - 0.05*(-40) = 2
prediction after the update:   [2,6]
fresh same-shape input [2,4]:   [4,8]
```

Compiler constructs the gradient expression; backend code executes it; Training applies the SGD
update. Runtime neither differentiates nor owns the optimizer. Reusing `[2,4]` changes values,
not the static input signature. Changing `[2,1]` to `[3,1]` is not the same operation: a static
prepared template rejects that descriptor rather than silently specializing it. Symbolic-batch
support is a separately qualified path, not a property inferred from these teaching numbers.

## `io.github.pho001.synaptik.trace`

**Purpose and mechanism.** This is the neutral diagnostic envelope, deliberately independent of
Model, Compiler, Runtime and concrete backends.
[`TraceEvent<T>`](../../../modules/trace/src/main/java/io/github/pho001/synaptik/trace/TraceEvent.java)
pairs an event identity, lifecycle phase, severity, monotonic nanoseconds and typed payload.
Construction validates required references; it does not read a clock, emit, store, filter or
serialize the event. Time is a reading in the producer's clock domain, not an epoch timestamp or
a global ordering guarantee. The envelope is shallowly immutable: an implementation of the open
[`TracePayload`](../../../modules/trace/src/main/java/io/github/pho001/synaptik/trace/TracePayload.java)
marker is responsible for providing immutable detached diagnostic data.

**Relations and state.** Producers construct these values and observers consume them. Trace owns
the vocabulary, not the producers' graph, exceptions, storage, native handles or event sinks.
[`TracePhase`](../../../modules/trace/src/main/java/io/github/pho001/synaptik/trace/TracePhase.java)
classifies compile, prepare and run facts without executing those stages.
[`TraceEventId`](../../../modules/trace/src/main/java/io/github/pho001/synaptik/trace/TraceEventId.java)
rejects negative values; its producer defines uniqueness. A severity is classification, not a
request to abort execution.

**In the example.** Trace does not calculate `40` or `-40`, and the existence of an envelope type
does not mean every ordinary run emits events. An actual optional producer is Engine's
[`EngineTuningTraceProducer`](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/EngineTuningTraceProducer.java):
only the observed tuning path constructs it. It emits at most one terminal fallback/rejection
event with a PREPARE phase; the unobserved path does not create it. This distinction prevents a
source tour from being mistaken for evidence of a trace captured from `LearningJourney`.

## `io.github.pho001.synaptik.trace.id`

**Purpose and mechanism.** These are nominal correlation domains: backend, device, prepared unit,
invocation, node, logical value and public Tensor. Each is an immutable non-negative `long`, with
zero valid. The producer translates its own identity to the appropriate trace type and owns that
translation's uniqueness and lifetime; the package neither allocates IDs nor stores a mapping.
The [package contract](../../../modules/trace/src/main/java/io/github/pho001/synaptik/trace/id/package-info.java)
explains why an equal number need not identify the same object across streams.

For example,
[`TracePreparedUnitId`](../../../modules/trace/src/main/java/io/github/pho001/synaptik/trace/id/TracePreparedUnitId.java)
correlates a finalized prepared unit, while
[`TraceInvocationId`](../../../modules/trace/src/main/java/io/github/pho001/synaptik/trace/id/TraceInvocationId.java)
correlates an invocation. Reusing a prepared recipe therefore need not reuse invocation identity.
[`TraceTensorId`](../../../modules/trace/src/main/java/io/github/pho001/synaptik/trace/id/TraceTensorId.java)
and [`TraceValueId`](../../../modules/trace/src/main/java/io/github/pho001/synaptik/trace/id/TraceValueId.java)
separate source-level Tensor correlation from compiled logical values; neither is a buffer slot.

**Relations, invariants and example.** Payloads retain these detached values, not execution owners.
Negative construction fails with `IllegalArgumentException`; the types provide no process-wide
or cross-stream uniqueness. In an instrumented version of the example, a weight Tensor,
its generated gradient value, a prepared backend unit and repeated calls would be different
correlation roles. The atlas assigns no invented numeric IDs to them. These records retain no
resources and add no execution step to the teaching path.

## `io.github.pho001.synaptik.trace.payload`

**Purpose and mechanism.** Shared payloads describe backend outcomes using neutral categories,
not concrete backend classes.
[`BackendPreparationOutcome`](../../../modules/trace/src/main/java/io/github/pho001/synaptik/trace/payload/BackendPreparationOutcome.java)
records a finalized unit's backend/device/unit correlations, outcome, route, cache fact and
optional native status.
[`BackendInvocationOutcome`](../../../modules/trace/src/main/java/io/github/pho001/synaptik/trace/payload/BackendInvocationOutcome.java)
adds the separate invocation correlation. They validate that success includes native SUCCESS,
and failure never includes native SUCCESS; failure before native return may have no native status.

[`TraceNativeStatus`](../../../modules/trace/src/main/java/io/github/pho001/synaptik/trace/payload/TraceNativeStatus.java)
preserves the exact signed native code. Zero means SUCCESS, and only SUCCESS may have zero;
nonzero codes retain a producer interpretation, including UNKNOWN. There is no universal numeric
error mapping across backends. A cache fact describes what that producer queried, not proof that
an enclosing Engine cache hit or missed.
[`LowPrecisionTraceMetadata`](../../../modules/trace/src/main/java/io/github/pho001/synaptik/trace/payload/LowPrecisionTraceMetadata.java)
is a bounded custom-kernel route/dtype diagnostic, not a numerical policy selector.

**Relations, state and example.** Backend producers supply these immutable data-transfer objects;
observers may retain them without retaining physical resources. Null references or inconsistent
status/code combinations reject at construction. These payloads do not choose routes, retry a
failed native operation or perform a transfer. The FLOAT32 learning example needs no low-precision
metadata or explicit trace payload construction, and no outcome event is claimed here as runtime
evidence.

## `io.github.pho001.synaptik.backend.contract`

**Purpose and mechanism.** This package lets compile-time ownership say *which backend domain*
without holding the backend's live implementation.
[`BackendId`](../../../modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/BackendId.java)
retains an exact nonblank string. It does not trim, case-fold, normalize or resolve aliases:
`"cpu"` and `"CPU"` are different IDs.
[`BackendDeviceId`](../../../modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/BackendDeviceId.java)
scopes a device identity to a backend; `DeviceClass` supplies a coarse class rather than a kernel
capability. Requirements can name a backend, one backend/device, or a device class.

[`BackendAvailabilitySnapshot`](../../../modules/backend-contract/src/main/java/io/github/pho001/synaptik/backend/contract/BackendAvailabilitySnapshot.java)
copies a caller-supplied device-to-class map, checks every device belongs to its backend, and
keeps the backend ID meaningful even when the map is empty. It does not discover devices or
refresh itself. Planning combines that snapshot with a separate capability provider; availability
alone cannot establish operation support or choose a physical device.

**Relations and ownership.** Config carries declarative requirements; Planning consumes them and
returns IDs; Engine later resolves selected owners through its private composition. These are
immutable facts with no resource lifecycle, service locator or kernel registry. Blank IDs,
null fields and cross-backend device entries fail construction.

**In the example.** A CPU-owned compiled node retains a CPU identity, not an OpenBLAS service,
CPU buffer or generated kernel. This is why compile artifacts remain logical recipes even though
they already have ownership decisions. The ordinary example uses `Engine.standard()`'s CPU
composition; it does not infer Metal execution from the workstation's hardware.

## `io.github.pho001.synaptik.config.compile`

**Purpose and mechanism.** These values request graph scope and planning policy without performing
the work. [`CompileMode`](../../../modules/config/src/main/java/io/github/pho001/synaptik/config/compile/CompileMode.java)
distinguishes FORWARD_ONLY, FORWARD_AND_BACKWARD and TRAINING_STEP. Compiler interprets backward
modes by constructing gradient expressions before capture; the TRAINING_STEP name does not itself
insert optimizer updates.

[`BackendIntent.unconstrained()` / `requiring(...)`](../../../modules/config/src/main/java/io/github/pho001/synaptik/config/compile/BackendIntent.java)
carry an optional **hard** eligibility requirement.
[`PartitionScoringConfig.neutral()` / `preferring(...)`](../../../modules/config/src/main/java/io/github/pho001/synaptik/config/compile/PartitionScoringConfig.java)
carry a **soft** device-class preference. A soft preference may rank eligible candidates but cannot
restore one rejected by capability or a hard requirement. Current baseline selection uses provider
order and this coarse preference, not a calibrated numerical cost model.

[`GraphOptimizationConfig.standard()` / `disabled()`](../../../modules/config/src/main/java/io/github/pho001/synaptik/config/compile/GraphOptimizationConfig.java)
permits or disables optional exact optimizations. Disabling them does not disable inference,
mandatory canonicalization or validation, and enabling them permits no approximate arithmetic.
The optional/required references reject null; these immutable records own neither live services
nor runtime state. Compiler and Planning consume the values; concrete backend route selection
belongs later to backend preparation.

**In the example.** The prediction-only call and the first-order training graph have different
compile scope, but the SGD update still belongs to Training. Neither a preference nor an
optimization permission explains the numerical `-40`; that follows from the expression's semantics.

## `io.github.pho001.synaptik.config.tuning`

**Purpose and mechanism.**
[`ModelAutotuningConfig`](../../../modules/config/src/main/java/io/github/pho001/synaptik/config/tuning/ModelAutotuningConfig.java)
is an explicit request for later measured selection, not a tuning algorithm. It carries an
objective, local-workload budget, representative-profile identity, fallback policy, workload-cache
path, independent complete-plan budget and model-plan-cache path. Paths are retained without
normalization or filesystem I/O. The representative identity defensively copies its opaque bytes;
it is not a retained Tensor batch.

The objective selects the first encountered minimum-median candidate. Budget records validate
positive candidate limits, non-negative warmups and positive odd timed sample counts. Complete-plan
bounds are independent of local-workload bounds; later admission checks the actual candidate count
`N` against `N * (1 + W + S)` executions, including correctness, warmups and timed samples, plus
an aggregate correctness-byte ceiling. Null components, invalid counts and an empty model-plan
cache path reject construction.

**Relations and state.** Engine joins this declarative policy with model identity and live
representative inputs, then composes the tuning tool and selected backend's opaque handoffs.
Config performs no measurement, cache mutation, preparation or execution. A safe-heuristic fallback
policy is not permission to change compile-time ownership or silently use another backend.

**In the example.** Autotuning is not required to obtain `[2,6]` or reuse the static prepared
signature. The ordinary learning path does not request these caches or budgets. Introducing a
representative tuning trial would be a separate workflow, not a hidden part of every `compute`.

## `io.github.pho001.synaptik.planning.capability`

**Purpose and actual admission.** Compiler calls
[`BackendOwnerPlanning.selectOwner(...)`](../../../modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/BackendOwnerPlanning.java)
once for each final node occurrence. Its
[`OperationCapabilityQuery`](../../../modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/OperationCapabilityQuery.java)
carries operation semantics and ordered input/output descriptors; a general Scan also carries
its complete independently scoped body and applicable activity metadata. Support for individual
body operations does not prove support for the complete Scan binder.

The package-private
[`BackendEligibility.evaluate(...)`](../../../modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/BackendEligibility.java)
first validates the complete provider/snapshot composition: unique backend IDs and exactly matching
identity sets. Only then does it filter empty availability and unmet hard requirements, calling
`BackendCapabilityProvider.supports(query)` once for each remaining provider in encounter order.
An exact-device requirement proves matching availability, not a selected device. Provider failures
propagate rather than being interpreted as a fallback opportunity.

**Scoring is currently a baseline, not a hidden optimizer.**
[`BackendOwnerSelection.select(...)`](../../../modules/planning/src/main/java/io/github/pho001/synaptik/planning/capability/BackendOwnerSelection.java)
chooses the first hard-eligible owner, or the first eligible owner with the preferred device class.
If no preferred-class candidate exists, it keeps the first eligible owner. It calculates no numeric
cost, selects no route or device, and retains no provider. An empty eligible set fails with
`no hard-eligible backend is available for ownership selection`; Compiler adds node context.

**Relations and example.** Input is compile-time query + intent + provider/availability snapshots;
output is a `BackendId`, consumed by Compiler's node-to-owner map. No physical state is owned here.
In the training example, the provider must support the **actual generated gradient topology**, not
just forward MATMUL and SUM. Requesting an unavailable hard target fails; it does not weaken the
request to make the example run.

## `io.github.pho001.synaptik.planning.partition`

**Purpose and mechanism.**
[`MaximalSameOwnerPartitioning.partition(graph, ownershipByNodeId)`](../../../modules/planning/src/main/java/io/github/pho001/synaptik/planning/partition/MaximalSameOwnerPartitioning.java)
groups the stored topological node sequence into maximal **consecutive** equal-owner runs. It
validates every key, complete graph coverage and every owner before constructing a result.
Graph edges do not redefine adjacency: fan-out, a phase boundary or an output publication does
not by itself split a same-owner run. A multi-output producer stays one indivisible node.

Each [`PlannedPartition`](../../../modules/planning/src/main/java/io/github/pho001/synaptik/planning/partition/PlannedPartition.java)
retains an owner and immutable ordered node IDs. The result is not a fused kernel or a backend
execution unit. Backend preparation may realize a partition with its own selected units, but
Planning has not made that implementation decision. Unknown/missing nodes, null owners and
inconsistent ownership maps reject; a valid zero-node graph produces an empty partition list,
which is not a promise that ordinary Engine preparation accepts it.

**Relations, state and example.** Compiler supplies graph-wide ownership and consumes the returned
recipes; logical-memory planning and Prepare use their exact membership/order. There is no live
resource, allocator or scheduler here. Consecutive CPU-owned forward and backward nodes in the
example can share one partition even across phases. That says who owns them, not whether MATMUL
uses native code or the elementwise work is fused.

## `io.github.pho001.synaptik.planning.memory`

**Purpose and mechanism.**
[`LogicalMemoryPlanning.plan(graph, partitions)`](../../../modules/planning/src/main/java/io/github/pho001/synaptik/planning/memory/LogicalMemoryPlanning.java)
answers who produces and consumes each logical value, not where bytes reside. It first checks
that partitions cover every node exactly once in stored graph order and are maximal adjacent
owner runs. It then indexes producer partitions and derives one requirement per graph value in
stored value order. Each consuming partition appears at most once, even if it reads that value
several times.

[`LogicalMemoryRequirement`](../../../modules/planning/src/main/java/io/github/pho001/synaptik/planning/memory/LogicalMemoryRequirement.java)
retains value ID, descriptor, optional producer partition, ordered consumer partitions and a
graph-output flag. An input has no producer; a dead or pass-through source may have no consumer.
[`LogicalMemoryPlan`](../../../modules/planning/src/main/java/io/github/pho001/synaptik/planning/memory/LogicalMemoryPlan.java)
collects those immutable facts. They are not byte allocations, buffer slots, lifetimes from a
pool, transfer recipes or a saved-gradient tape. Not every logical intermediate needs a shared
physical slot; backend analysis declares which shared resources are actually required.

**Relations, failures and example.** Compiler stores this plan in artifacts; Prepare uses it for
coverage and boundary validation; Engine derives cross-owner paths from producer/consumer facts.
Duplicate, unknown, missing or out-of-order partition nodes and adjacent equal-owner partitions
reject instead of being repaired. In the example, a weight leaf and the prediction result have
different source/consumer/publication facts even when all work has a CPU owner. Logical memory
has no responsibility for applying the weight update or keeping optimizer state.

## `io.github.pho001.synaptik.compiler`

**Entry and purpose.**
[`GraphCompilationPort.compile(...)`](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompilationPort.java)
is the narrow lifecycle-integration boundary used by Engine; ordinary applications use Engine.
The package-private
[`GraphCompiler.compile(...)`](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompiler.java)
turns the ordered Tensor boundary into a verified immutable recipe, never a backend executable.
Its ordinary port supplies empty explicit forward-constant ingress: a stored leaf is still a
caller-bindable input, not a constant inferred by inspecting its bytes.

**Automatic differentiation (AD) precedes capture.** Forward-only mode captures the requested
forward expressions. Backward-capable modes run
[`AutogradPreflight`](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java)
over the requested derivative routes and use
[`FirstOrderAutograd.expand(...)`](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java)
to construct ordinary Tensor expressions for gradients. Per-request identity maps collect
contributions; multiple uses accumulate with ordinary addition. The compiler then captures
forward outputs **and** requested gradient roots together once, with phase and derivative-order
metadata. It is not differentiating captured Runtime instructions, and Tensor has acquired no
mutable backward tape. The functional request can express the separately bounded second stage;
that is not unrestricted higher-order support for every operation.

For the learning objective, SUM restores the cotangent to the error shape, multiplication
contributes through both squared-error operands, and MATMUL uses transposed contractions.
The implementation owners are
[`ReductionGradientRules`](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ReductionGradientRules.java),
[`ElementwiseGradientRules`](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ElementwiseGradientRules.java)
and [`LinearAlgebraGradientRules`](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/LinearAlgebraGradientRules.java).
Broadcasted contributions reduce back to the selected operand shape. These formulas explain the
`[1,1]` weight gradient `-40`; there is no backend-global autograd implementation to rediscover it.

**Inference and transformation.** Captured graph inference verifies family semantics and descriptors.
[`ForwardGraphOptimization.optimize(...)`](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimization.java)
performs mandatory canonicalization/validation, then, when enabled, bounded exact arithmetic
rewriting, logical-splat folding, dead-code elimination, phase/order-local common-subexpression
elimination and cleanup dead-code elimination. Changed candidates are revalidated. These are
not floating arithmetic evaluation of caller storage, approximate rewrites or backend fusion.
Final descriptor-closure passes resolve only their qualified materialized results and constant
sources, not arbitrary unresolved caller/view layouts. Publication and capability queries see
that final graph.

**The output boundary.**
[`CompileArtifacts`](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)
contains eight parts: mode, final graph, maximal partitions, logical memory, publication plan,
input-source classification, deferred diagnostics and exact-graph derivative metadata. Its
constructor cross-checks graph identity, source coverage, mode, partition/memory agreement and
diagnostic membership. Compiler builds ordered forward/gradient publication associations, selects
owners per final node through Planning, partitions them, and derives logical memory. No retained
Tensor, source producer, provider, physical buffer, route or mutable run state crosses this boundary.

**Other callers and failures.** The same port's `captureSource`, `validateSource` and
`reconstructSource` support a separately bounded cold source workflow used by acquisition and
persistence integrations. The raw source graph is not the optimized artifact graph; its borrowed
ordered leaf sidecar is temporary and does not become artifact state. Cold admission does not
prove backend execution. Invalid mode/request combinations, unsupported derivative routes,
inference/closure failures or incompatible artifact parts reject; deferred successful diagnostics
are not permission for Prepare to guess missing facts.

**In the example.** Forward computation and configured training use this same compiler ownership.
Compiler constructs the gradient but never changes `W` to `2`: Training does that after successful
execution. Fresh `[2,4]` values go through binding on an existing compatible preparation, not
through expression recapture or another derivative pass.

## `io.github.pho001.synaptik.prepare.analysis`

**Purpose and input projection.** This package is the first half of preparation: give a backend
all facts it is allowed to analyze without giving it the Compiler aggregate.
[`PrepareContext<I>`](../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PrepareContext.java)
retains an immutable
[`PartitionDag`](../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PartitionDag.java),
projected values and logical requirements, exact-typed logical splats, backend-owned immutable
inputs, and scoped Scan activity associations. It validates partition order, reference closure,
descriptor/requirement agreement, constant roles and the admitted static or bounded named-batch
shape vocabulary before backend analysis. Scan body IDs are local to their body, never aliases
of equal-numbered outer values.

**Actual backend collaboration.**
[`BackendPartitionPreparer<I,P>.analyze(context)`](../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/BackendPartitionPreparer.java)
is implemented by the concrete backend. It deterministically lowers, specializes and selects a
supported route/configuration using explicit inputs. It returns
[`BackendPartitionAnalysis<P>`](../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/BackendPartitionAnalysis.java):
the exact partition reference, an opaque immutable selected plan, and ordered exact
[`PreparationResourceRequirement`](../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/PreparationResourceRequirement.java)
buffer/workspace declarations. Generics preserve the backend-input/plan association; shared
Prepare transports it without casts to concrete CPU/Metal types or a string parameter map.

Requirements name outer logical values or analysis-local workspace IDs and declare byte size
and alignment. Fixed bytes and complete immutable run-size recipes are distinct declarations;
analyzer/finalizer cannot defer discovering another shared slot until a run. Duplicate buffer
IDs, duplicate workspace IDs, unsupported inputs or unresolved route-critical facts reject.
Analysis constructs no executable, physical allocation, persistent resource or per-run state.

**Tuning and example.**
[`BackendPartitionTuningHandoff`](../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/analysis/BackendPartitionTuningHandoff.java)
transports an exact partition, opaque candidate batch and optional decision. It neither measures
candidates nor interprets private settings. Compatible explicit tuning decisions can be inputs
to later analysis; ordinary `analyze` is not a benchmark loop. For the teaching graph, the CPU
analyzer chooses realizations and declares output/scratch needs for the actual forward or
combined graph. Shared analysis does not compute `40`, allocate an SGD state or select the
weight's learning rate.

## `io.github.pho001.synaptik.prepare`

**Purpose and orchestration.**
[`GraphPreparation.prepare(...)`](../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/GraphPreparation.java)
bridges one complete compile recipe to one Runtime prepared owner. Engine supplies one typed
[`PartitionPreparation`](../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PartitionPreparation.java)
per compile partition and one explicit schedule assembler. Shared Prepare projects and validates
**all** contexts before invoking the first analyzer. It checks source-only published caller/constant
resources too; a published graph input must not disappear merely because no partition produces it.
`GraphPreparation.project(...)` exposes the focused single-partition projection, not another
complete execution lifecycle.

**Assignment is between analysis and finalization.** After every analysis has returned and
cross-partition declarations agree,
[`BackendPartitionFinalizationHandoff`](../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizationHandoff.java)
assigns stable `BufferSlot`/`WorkspaceSlot` identities and dense plan positions. Caller buffers
lead in Compiler bindable-input occurrence order; ordinary remaining buffers retain declaration
order, with source-only constants last in graph-value order. Repeated fixed-byte declarations
aggregate the maximum required size and alignment; run-bound declarations must agree structurally.
Distinct backend owners receive stable representation positions:
producer first, or first consumer for a producerless value, then distinct consumers in order.
Workspaces remain partition-owned requirements, not shared logical Tensor values.

[`PreparationResourceAssignment`](../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PreparationResourceAssignment.java)
associates each exact declaration with the slot, dense plan index and, for buffers, the owner's
representation index. The matching
[`BackendPartitionFinalizer`](../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/BackendPartitionFinalizer.java)
then receives its typed analysis and complete assignments. Only now does it construct executable
code/objects and optionally acquire immutable persistent resources. It cannot change route or
add undeclared shared requirements. The returned
[`PreparedPartition`](../../../modules/prepare/src/main/java/io/github/pho001/synaptik/prepare/PreparedPartition.java)
associates the exact planned partition with its finalized executable.

**Schedule and ownership transaction.** Backend contributions describe physical creation recipes
for assigned buffer/workspace coordinates, not alternative schedulers. The explicit assembler
builds the complete schedule; shared Prepare checks exact memory-plan association, contribution
coverage, execution order, representation coordinates and publication order against the compile
recipe. Successful construction transfers persistent-resource ownership to Runtime's
`PreparedExecution`.

A finalizer owns rollback until it returns. Thereafter shared Prepare tracks the unique resource
identities in acquisition order across partitions. A later finalizer, contribution, schedule,
validation or aggregate failure closes them in reverse acquisition order, preserves the primary
failure and suppresses distinct cleanup failures. Duplicate returned resource identities reject;
repeating an executable in a schedule is not another ownership acquisition. Shared Prepare does
not implement physical storage or per-run allocation, and it transports run-size recipes without
evaluating backend geometry.

**In the example.** Preparation freezes the executable realization of the prediction/gradient
graph, not the numeric contents of mutable Model parameters. It neither executes the objective
nor applies SGD. Repeated same-signature calls reuse these recipes and stable slot positions while
Runtime creates distinct mutable run state. Feeding a different shape is not permission to
reanalyze or change the selected route inside a run.

## `io.github.pho001.synaptik.runtime.memory`

**Purpose and mechanism.**
[`PreparedMemoryPlan`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/memory/PreparedMemoryPlan.java)
is the immutable plan-ordered geometry of buffers and workspaces. A
[`BufferSlot`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/memory/BufferSlot.java)
and a [`WorkspaceSlot`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/memory/WorkspaceSlot.java)
are separate nominal domains, even when their numeric values happen to match. Runtime indexes
plan list positions; it does not treat arbitrary slot numeric components as array offsets.

Each entry declares non-negative usable bytes and positive power-of-two byte alignment. The
`ByteSize` boundary distinguishes `FixedByteSize` from an immutable checked numeric recipe
resolved for an admitted batch size. `requiresBatchSize` is explicit: a symbolic plan can require
B even when a particular resource always has zero or constant bytes. A static-mode plan rejects
run-bound declarations rather than sampling a recipe to pretend it is fixed.

**Relations and state.** Shared Prepare constructs the plan; executable, transfer, creation and
publication recipes retain that exact reference; run geometry is authenticated to it. The plan
copies entry lists, checks duplicate slots in each domain and allocates no physical storage.
Invalid size/alignment, null entries and duplicate slots fail construction. It contains no
Compiler graph, backend lookup or source-to-slot mapping.

**In the example.** FLOAT32 `[2,1]` has eight logical dense bytes and `[1,1]` has four, but those
numbers alone are not a complete prepared memory plan: internal materialization and scratch
requirements come from backend analysis. The atlas does not invent a slot count or workspace
size for a graph it has not prepared. Static fresh values reuse the same plan geometry.

## `io.github.pho001.synaptik.runtime.resource`

**Purpose and mechanism.** This package separates a reusable description of physical origins
from the concrete physical objects that exist during a run.
[`PreparedRepresentationPlan`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/PreparedRepresentationPlan.java)
contains, for each dense buffer/representation position, a `CallerInput`, `CreatedBuffer` or
`InitializedBuffer` origin, plus one creator per workspace. Construction snapshots the recipe;
it invokes no callback. A caller occurrence consumes a borrowed input. A created buffer begins
invalid until work writes its logical value; an initialized buffer begins valid. Both are
run-owned. Workspace scratch has no logical-value validity bit.

[`BufferRepresentation`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/BufferRepresentation.java)
and [`WorkspaceRepresentation`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/WorkspaceRepresentation.java)
are nominal close contracts implemented by backends. Runtime can retain and orchestrate cleanup
without knowing a CPU native segment or Metal storage type. Creator callbacks receive exact usable
bytes/alignment and return fresh per-run objects. They do not cause shared Runtime to become a
physical allocator or a backend dispatcher.

[`PreparedResource`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/resource/PreparedResource.java)
is a different lifetime: an immutable persistent physical owner acquired during finalization.
Runtime's prepared owner closes its unique identities; a backend may supply fresh run-owned
read-only child bindings that keep immutable prepared storage resident. Such a child is not
another prepared-resource registration and not shared mutable scratch.

**Invariants and example.** Recipe counts must match the memory plan and every buffer has a
nonempty origin list. Callback results must be fresh and non-null; Runtime creation rejects
reused representation identities and rolls back newly acquired owners. In the example, caller
input storage remains borrowed while internal/output buffers and scratch are run-owned. These
contracts do not expose the parameter as a mutable shared prepared field or authorize a hidden
copy-back from an output into Model state.

## `io.github.pho001.synaptik.runtime.execution`

**Purpose and binding mechanism.**
[`PreparedExecutable`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecutable.java)
is an immutable backend-owned recipe for a region, with dense buffer/representation selections,
workspace selections and explicit READ_ONLY/WRITE_ONLY/READ_WRITE access declarations.
`bind(RunState)` is the cold boundary: shared checks validate plan/ranges; concrete backend checks
validate physical representation compatibility once, then create a typed
[`BoundInvocation`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/BoundInvocation.java)
with direct references. Repeated operand selections express roles, not duplicate ownership.
Execution is not a graph traversal or repeated concrete-type lookup.

[`PreparedBufferTransfer`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedBufferTransfer.java)
and [`BoundBufferTransfer`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/BoundBufferTransfer.java)
apply the same prepared/cold-bound distinction to explicit copying between two representations
of one logical buffer. An already-valid destination is a no-op; otherwise a valid source is
required, physical work occurs, and success marks the destination valid while keeping the source
valid. Failure leaves Runtime validity unchanged. Transfer is prepared work, not on-demand
discovery or an implicit coherence mechanism.

**Prepared ownership.**
[`PreparedExecution`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)
retains exactly one memory plan and same-reference schedule, and snapshots unique persistent
resource identities in acquisition order. `acquireRunLease()` serializes admission with `close()`;
lease allocation precedes incrementing the active-run count. Close rejects new runs immediately
but does not wait for admitted synchronous runs. With active leases, the last release performs
deferred reverse-order attempt-all physical cleanup. Without them, close cleans immediately.
Runtime repeated close is idempotent and does not replay a previous cleanup failure; the outward
Engine wrapper has its own close-failure replay behavior.

**Relations, errors and example.** Shared Prepare constructs this inward owner, the stateless
runner uses it, and the Engine wrapper delegates resource lifetime to it. Foreign plan references,
out-of-range selections and incompatible concrete representations fail before hot execution;
a closed prepared owner reports `prepared execution is closed`. No executable owns selected
run buffers. In the learning path, generated gradient operations are already executable recipes:
Runtime executes them but never dispatches derivative rules or performs SGD itself.

## `io.github.pho001.synaptik.runtime.schedule`

**Purpose and mechanism.**
[`PreparedSchedule`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/schedule/PreparedSchedule.java)
is an immutable ordered list of **occurrences**, all tied to one exact memory-plan reference.
An optional `RepresentationCreationStep` is permitted only at index zero. Execution and explicit
buffer-transfer steps follow; publication steps form a dense final suffix with result indices
`0,1,...`. Once publication starts, no later execution/transfer is permitted. Constructor checks
enforce these structural invariants without allocating physical resources or running work.

The distinction between recipe and occurrence matters: placing the same executable twice means
execute it twice, not own its persistent resources twice. A repeated transfer occurrence invokes
the transfer action again, but its already-valid-destination check may avoid another physical copy.
The schedule does not infer dependencies at run time, select kernels or provide another resource
owner. Shared Prepare validates its relationship to the graph; Runtime
later cold-binds and traverses it. Immutable schedules can be shared by calls, but their mutable
run states cannot.

**In the example.** A CPU-only preparation needs no cross-owner transfer. Its final publications
name the requested forward/gradient occurrences after their producing work; they are not host
copies inserted after each arithmetic operation. In a mixed-owner graph, Engine's assembler
places one explicit transfer before each destination owner's first consumer, without changing
Planning's owners. An empty schedule is structurally representable, but the runner requires a
creation prefix for a nonempty memory plan; representability is not ordinary preparation admission.

## `io.github.pho001.synaptik.runtime.run`

**Purpose and actual call sequence.**
[`PreparedExecutionRunner.run(...)`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/PreparedExecutionRunner.java)
first acquires the inward prepared lease, then validates exact-plan numeric geometry and supplied
inputs. [`RunStateCreation.create(...)`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunStateCreation.java)
validates the complete dense borrowed-input list before any creator call, creates buffers in
buffer/representation order and workspaces afterward, and transactionally constructs exactly one
[`RunState`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunState.java)
for the whole logical run. Failed creation closes only fresh acquired owners in reverse order;
borrowed inputs are never acquired as owned storage.

Before the first action, the runner cold-binds **every** executable, transfer and publication
occurrence into a direct-reference action array. Traversal then follows schedule order. For an
executable, declared reads must be valid; before writes, other copies of each written logical
buffer are invalidated; after successful invocation, the written representations become valid.
A representation is *resident* when the physical object is bound, and *valid* when it contains
the logical value. Allocating storage establishes residency, not a computed result.

**Publication is not copying.**
[`PreparedPublication.bind(...)`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/PreparedPublication.java)
resolves a direct representation once.
[`BoundPublication.publish()`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/BoundPublication.java)
checks the open state, one-shot occurrence and valid copy, then marks only that occurrence complete.
It transfers no bytes and changes no ownership. Multiple publication occurrences can name one
physical representation while retaining different result roles. Runtime
[`RunResult`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunResult.java)
owns the completed state/publication lifetime, not an in-flight command.

**Lifetime and errors.** Successful synchronous return is the completion barrier. The prepared
run lease is released before return; the result's run-state lifetime continues until result close.
Closing preparation therefore need not close an already returned result. A backend child binding
can keep immutable prepared physical storage alive independently. Failure after state creation
closes the state and returns no result; if releasing the prepared lease fails after execution,
the runner closes the unpublished result too. Cleanup preserves the triggering failure and adds
distinct suppressed failures. Closing a result clears direct invocation-payload references even
when cleanup fails, without closing borrowed caller storage. Plan/count/geometry inspection
remains, not physical access. `RunState` and inward result objects are not thread-safe; separate
calls create isolated mutable state. There is no asynchronous/cancellation/polling contract.

[`RunMemoryGeometry`](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/run/RunMemoryGeometry.java)
is authenticated checked numeric sizing for an exact plan, not Model shape inference. Runtime
receives no Compiler graph for symbolic binding and performs no tuning-cache lookup or route search.

**In the example.** The first call's predictions and gradient belong to that call's completed
state; a later call with `[2,4]` gets another state and fresh writable representations. The
Training owner consumes the gradient before its internal result closes and applies SGD outside
Runtime. Keeping one result open is not permission to mutate its borrowed caller storage or reuse
another run's mutable workspace.

## `io.github.pho001.synaptik.engine`

**Ordinary application entry points.**
[`Engine`](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)
is the composition root: `compile`, `prepare`, `run`, `runValues`, explicit `prepareTuned`, and
closure coordinate the owners above. `Engine.standard()` transfers a CPU integration into the
builder; it is not reflective hardware discovery. `Engine.builder().takeOwnership(...)` composes
supported backend-owned integrations explicitly. The builder and Engine own accepted integrations;
concrete backends do not depend back on Engine. Public `AdvancedEngine` and advanced handles expose
lower-level composition, not a second compiler, scheduler or lease system.

**Compile and prepare, concretely.** The package-private orchestration in
[`AdvancedEngine`](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/AdvancedEngine.java)
passes frozen provider/availability lists to Compiler. The outward
[`CompiledGraph`](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/CompiledGraph.java)
retains owner-bound final input and publication metadata. It is not caller storage or a prepared
executable. Ordinary compilation exposes forward outputs, or explicit cotangent seeds and targets
for first-order gradients; scalar-objective convenience constructs the appropriate seed through
the same compiler path.

The private
[`EngineBackendRegistry.prepare(...)`](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/EngineBackendRegistry.java)
rejects empty plans and missing exact owners, then preflights every cross-owner path **before**
backend analysis. It routes each partition by exact `BackendId` to a typed preparation and calls
shared `GraphPreparation`. There is no owner substitution, retry on CPU or normalized-ID alias.
[`CompositePreparedScheduleAssembler.assemble(...)`](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/CompositePreparedScheduleAssembler.java)
combines exact owner-local creation contributions, emits one creation prefix, executions in
compile-partition order, transfers before first destination consumption, and the publication
suffix. It uses fixed producer/consumer facts rather than rescoring ownership. Backend transfer
adapters implement supported physical movement; Engine does not implement arithmetic kernels.

The resulting public
[`PreparedExecution`](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/PreparedExecution.java)
wraps the exact inward Runtime owner, borrows the Engine composition lifetime, and retains direct
input/publication adapters aligned with final occurrence order. Wrapper construction and Engine
registration are part of transactional publication. Failed outward publication rolls back inward
ownership; it is not acceptable to leak a valid inward preparation because wrapping failed.

**Two binding modes on the same preparation.** `run` accepts every exact compiled TensorId once,
in arbitrary order, with the complete matching descriptor. `runValues` accepts fresh values in
`CompiledGraph.inputs()` order, without matching TensorIds; the same fresh Tensor may fill multiple
matching positions. For static templates, type, shape, layout and gradient eligibility match
exactly. Engine checks its own admission, exact prepared owner and closed inward delegate before
inspecting inputs. Under an open Engine, a closed preparation wins over malformed input errors.

Engine snapshots storage associations and validates all selected values for presence, dtype,
liveness, current-thread accessibility and capacity before borrowing any. Non-owning wrappers
borrow only required referenced storage spans, never transfer caller ownership. Qualified
symbolic-batch paths additionally use Compiler's
[`BatchBindingPlan`](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/BatchBindingPlan.java)
for fresh complete semantic binding and checked numeric sizing; that is not implicit static
respecialization. Runtime receives numeric geometry, not the Compiler binding metadata. The
registry/provider inventory has no execution or materialization lookup role after preparation.

**Completed results and detached values.** Ordinary
[`RunResult`](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/RunResult.java)
exposes distinct forward-then-gradient publication occurrences with descriptors, Tensor IDs,
roles and derivative target/order metadata. `materialize(publication, maximumBytes)` authenticates
an exact occurrence object from that result, enforces the byte limit and invokes the already
selected publication adapter synchronously. Equal metadata or an occurrence from another run
is not a valid selector. Each success returns a fresh
[`HostTensorValue`](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/HostTensorValue.java):
immutable canonical dense row-major bytes, with multi-byte elements big-endian. It retains no
Engine, result, representation, arena or storage. It is usable after all those owners close;
`bytes()` provides an independent read-only view. Materialization is explicit, not automatic
publication, dtype conversion or construction of another Tensor expression.

**Closure and failure boundaries.** Caller storage remains live and free of conflicting mutation
until its result closes. Prepared close rejects new runs without waiting for already leased runs
and does not close their completed results. Result materialization remains valid after prepared
close while result and Engine are open. Engine close rejects new operations, waits for admitted
work, then closes results in reverse publication order, prepared handles in reverse publication
order, and integrations in reverse registration order. Cleanup is attempt-all with failure
suppression. A thread cannot synchronously close an Engine from its own admitted operation;
cleanup-owner recursive close is idempotent. Outward close paths serialize concurrent closure
and replay their recorded failure; this must not be confused with inward Runtime repeated-close
semantics. Result-wrapper construction and registration also belong to the rollback boundary:
a completed run is not published to the application unless the entire outward transaction succeeds.

**Tuning stays cold and owner-preserving.** `prepareTuned` joins Config, caller model/profile
identity and live representative values through
[`ModelAutotuningRequest`](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/ModelAutotuningRequest.java).
Engine owns representative execution and correctness collaboration; tooling owns bounded search
and cache work; the backend owns candidate vocabulary. Current admission is one eligible CPU
partition or the bounded singleton-NEG Metal plan with its complete local route batch, not an
arbitrary heterogeneous model. A permitted recoverable safe-heuristic fallback prepares through
the exact selected adapter. It does not replan ownership or introduce run-time tuning. Optional
[`EngineTuningObserver`](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/EngineTuningObserver.java)
receives detached diagnostic envelopes; observation does not become another execution owner.

**In the example.** Model and Training compose this public lifecycle: initial execution explains
`[0,0]` and `40`, Compiler-generated backward execution supplies `-40`, and Training updates the
owned parameter to `2`. Repeated Model computation with same-shape fresh `[2,4]` uses cached
compatible preparation to obtain `[4,8]`. Engine has not become the parameter owner or optimizer.
The concrete static mismatch `[3,1]` is a binding error, not a request to compile another model.

## Following the boundary into backend code

None of these 18 packages owns a C/C++/Objective-C++ kernel or native source tree. Engine contains
Java CPU/Metal composition and transfer adapters, but concrete lowering, Foreign Function & Memory
bindings, allocation/release mechanics and native execution live in backend/provider modules.
Follow the [CPU package atlas](cpu.md) and [Metal package atlas](metal.md) for those physical paths;
this page's `PreparedResource` and representation interfaces are ownership boundaries, not native
implementations. In particular, neither trace's native-status DTO nor `BackendId("metal")` loads
a library or proves native availability.

## Reading failures by the stage that owns them

This table is a source-based diagnostic guide, not a transcript of exercised failures.

| When the failure appears | Question to ask | Do not work around it by |
|---|---|---|
| Compile/AD | Is the exact operation/attribute/derivative route admitted, and is its generated closure supported? | Asking Runtime to differentiate the forward graph |
| Planning | Do provider/snapshot identities agree, and does an available backend satisfy hard requirements and the whole query? | Treating a soft preference as fallback authorization |
| Prepare analysis | Are all facts sufficient to choose one supported route and declare exact resources? | Discovering a new slot or kernel during binding |
| Assignment/finalization | Do exact partition, declaration, plan and resource identities agree? | Reusing a foreign memory plan or registering the same persistent owner twice |
| Engine binding | Are order/IDs, full descriptors, storage lifetime and capacity valid for this prepared handle? | Mutating prototypes or silently specializing a static signature |
| Runtime action/publication | Is the selected physical representation valid at this point in the fixed schedule? | Marking fresh uncomputed storage valid or publishing after a failed producer |
| Materialization/close | Is the exact result occurrence open, within the byte limit and still backed by live owners? | Treating a publication lease as detached host bytes |

For the supported-facing sequence, return to the [tutorials](../../tutorials/index.md) or
[API reference](../../reference/index.md). For relationships beyond execution, continue through
the [framework explanation](../framework.md) and [package index](index.md).
