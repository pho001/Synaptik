# [Training Graph](../glossary.md#training-graph)

This document explains the training graph [model](../glossary.md#term-model) established by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The root contract remains authoritative.

The [compiler](../glossary.md#term-compiler) currently consumes all three [compile](../glossary.md#compile) modes. Backward-capable [compilation](../glossary.md#compiler-engine-and-execution-stage-names) accepts a
bounded functional-[gradient](../glossary.md#term-gradient) request, constructs one or two [reverse-mode](../glossary.md#primal-adjoint-and-differentiation-seed) stages, [captures](../glossary.md#term-capture-meanings) the
forward and [derivative](../glossary.md#term-derivative) expressions together, and derives gradient [publication](../glossary.md#term-publication) [bindings](../glossary.md#term-binding). The
public Training extension now consumes the ordinary [Engine](../glossary.md#term-engine)'s one-stage first-gradient path for
its reusable session; it does not own or alter the combined compile-time [graph](../glossary.md#graph) described here.
This page is subordinate to the Compiler-owned contract.

The **forward** computation calculates outputs; the **backward** computation carries
sensitivities toward requested derivative [targets](../glossary.md#term-target). A **[cotangent](../glossary.md#cotangent)** is such a [sensitivity](../glossary.md#term-sensitivity) supplied
to an output; a **[seed](../glossary.md#term-seed)** is its starting value. A **[primal](../glossary.md#term-primal)** is an original forward value used by
a derivative formula. Combining the graphs means the compiler can see both computations;
it does not move [optimizer](../glossary.md#term-optimizer) updates or derivative-rule [ownership](../glossary.md#term-ownership-meanings) into a [backend](../glossary.md#backend).

### Who requests, constructs and applies a gradient?

The application configures an objective through NN's owned Model, or opens a `TrainingSession`
directly. Training calls public Engine compilation with the scalar objective expression and
ordered parameter targets. Engine delegates to Compiler, which constructs derivative Tensor
expressions **before capture**. Those expressions are descriptions, not gradient arrays.
Backend preparation realizes the combined graph; a later run produces actual objective and
gradient publications. Training copies them while the result is open, closes that result, then
stages and validates the parameter update.

Training borrows Engine and admitted parameter storage; it owns optimizer slots and its
preparation. An owned NN Model additionally owns private parameter copies and any native state
it acquired. Neither Compiler nor a backend takes over the optimizer. The
[update transaction](../explanation/packages/extensions-tools.md#where-an-update-becomes-a-transaction)
explains why a failed late gradient check cannot partially commit an ordinary training step.
Earlier fixed-recurrence rejection notes below retain their staged implementation context;
use [current support evidence](../backend-guide/production-support.md#status-and-interpretation)
for later qualified derivative routes rather than inferring them from a forward cell name.

### A scalar chain-rule example

For finite real, dimensionless input $x$, weight $w$ and target $t$, let $y=xw$ and
$L=(y-t)^2$. Then

$$
\frac{dL}{dw}=\frac{dL}{dy}\frac{dy}{dw}=2(y-t)x.
$$

With $x=3$, $t=6$, $w=0$, the contribution is $2(0-6)3=-36$.
In a two-sample squared-error **sum**, contributions add; they are not averaged.
The [full worked update](../explanation/tensors-and-gradients.md#the-first-training-step)
adds the other sample to obtain $-40$. This real-number derivation teaches the dependency;
actual typed operations preserve their rounding sites, and unsupported derivative/value domains
still reject. It does not extend the bounded derivative order.


## [Owned Model](../glossary.md#model-module-buffer-and-scan-distinctions) consumer

An owned [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [`Model`](../glossary.md#term-model) configures one pure [scalar objective](../glossary.md#loss) on its fixed output expressions and
explicit [target](../glossary.md#term-target) [prototypes](../glossary.md#term-prototype). It delegates to the same public [TrainingSession](../glossary.md#training-session--trainingsession) described here, with
no new backward pipeline. `train(batch, targets)` uses `RESET_AND_STEP` through fresh positional
values; it neither rebuilds forward nor toggles [Module](../glossary.md#term-module) mode. [Parameters](../glossary.md#term-parameter) keep their [identities](../glossary.md#term-identity-meanings)
through one explicit native [admission](../glossary.md#term-admission), and [inference](../glossary.md#term-inference) and save observe subsequent same-store updates.
[Objective](../glossary.md#term-objective) callbacks, [optimizer history](../glossary.md#term-optimizer-history) and random/input progress are not model-file contents.
The implemented all-valid general-[scan](../glossary.md#term-cumulative-scan) owned-Model workflow reuses this same path for qualified
static [canonical](../glossary.md#term-canonical-form) [homogeneous](../glossary.md#term-homogeneous) [FLOAT32](../glossary.md#term-float32)/[FLOAT64](../glossary.md#term-float64) occurrences. Its [graph](../glossary.md#graph)/definition acquisition uses
bounded [Compiler](../glossary.md#term-compiler) [cold reconstruction](../glossary.md#term-cold-reconstruction) through [Checkpoint](../glossary.md#term-checkpoint) attachment-preserving copy/[rebind](../glossary.md#capture-graph-ports-and-reconstruction), not a
[CPU](../glossary.md#term-cpu) support predicate. All-valid means every static time step participates, not arbitrary body
support. Fixed scan-training rejection stays intact. All-valid body [persistence](../glossary.md#term-persistence) uses Checkpoint
v4; static homogeneous length-mode cold/persistence uses Compiler0018/Checkpoint v5. NN0033
separately implements and qualifies the recurrent-composition numerical/[lifecycle](../glossary.md#lifecycle) journey.
See the [NN ownership contract](contracts/extensions-training.md#owned-tensor-models)<span data-glossary-navigation="true"> ([ownership glossary](../glossary.md#term-ownership-meanings))</span> and
[training workflow](../user-guide/training.md#train-an-owned-static-model).

## [Compile](../glossary.md#compile) modes

Synaptik distinguishes three compile modes:

- **Forward-only mode** (`FORWARD_ONLY`) compiles only the forward computation and its requested publications.
- **Forward-and-backward mode** (`FORWARD_AND_BACKWARD`) constructs gradient Tensor expressions
  from the original forward expression and captures both phases together.<span data-glossary-navigation="true"> ([captures glossary](../glossary.md#term-capture-meanings))</span><span data-glossary-navigation="true"> ([gradient glossary](../glossary.md#term-gradient); [Tensor glossary](../glossary.md#tensor))</span>
- **Training-step mode** (`TRAINING_STEP`) initially uses that same combined forward/backward
  construction. Optimizer-update graph work remains future.<span data-glossary-navigation="true"> ([Optimizer glossary](../glossary.md#term-optimizer))</span>

Both backward-capable modes require the same current `FunctionalGradientRequest`. The [compile
mode](../glossary.md#compile-mode--compilemode) changes the compile-time [graph](../glossary.md#graph) and [publication](../glossary.md#term-publication) needs. It does not allow a [backend](../glossary.md#backend) to own
global [autograd](../glossary.md#autograd).

## Combined forward and [backward graph](../glossary.md#backward-graph)

When a [compile mode](../glossary.md#compile-mode--compilemode) requires [gradients](../glossary.md#term-gradient), the [compiler](../glossary.md#term-compiler) builds backward expressions before the only
[capture](../glossary.md#term-capture-meanings):

```text
original forward Tensor expression DAG
  -> fail-closed compiler autograd preflight
  -> gradient formulas through ordinary public Tensor operations
  -> combined forward + gradient Tensor expression DAG
  -> one phase-aware capture
  -> immutable combined forward + backward graph
  -> inference, validation, combined optimization, and final validation
  -> publication binding
  -> planning and partitioning
```

The compiler uses exact [Tensor](../glossary.md#tensor) [identity](../glossary.md#term-reference-identity) only for temporary reverse-accumulation bookkeeping and
ordinary `Tensor.add` for contribution accumulation. This is not a second [graph](../glossary.md#graph) representation.
Named compiler rules own [dispatch](../glossary.md#term-dispatch); [model](../glossary.md#term-model) operations remain [derivative](../glossary.md#term-derivative)-agnostic. [Seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings) and other
derivative [constants](../glossary.md#term-constant) are storage-free [leaves](../glossary.md#term-leaf) registered explicitly as [compile](../glossary.md#compile)-time splats.

Phase-aware capture receives the original forward-[producer](../glossary.md#term-producer) [identity](../glossary.md#term-identity-meanings) set, forward outputs,
all requested first- and optional second-stage gradient roots and [target](../glossary.md#term-target) roles, and exact constant
facts. It assigns graph-local IDs once and retains `FORWARD` or `BACKWARD` on every [node](../glossary.md#node). The
separate derivative sidecar records order zero for original forward producers, order one for
first-stage producers, and order two for second-stage producers. Orders one and two are both
`BACKWARD`; derivative order does not replace [graph phase](../glossary.md#graph-phase).

Gradient [publication](../glossary.md#term-publication) [bindings](../glossary.md#term-binding) use orders one and two and are ordered by derivative order, then by
target position within that stage. Each binding retains its target identity and final gradient
value. Two targets may share one captured gradient value; their bindings remain distinct while
the graph output boundary starts with the forward-output prefix and appends that value only if it
was not already listed.

The combined graph is [immutable](../glossary.md#term-immutable) compile-time graph state. [Backends](../glossary.md#backend) receive only their planned
regions during [prepare](../glossary.md#prepare) and execute only those prepared regions at [runtime](../glossary.md#term-runtime).

Compile-time combination does not require a single runtime schedule. Prepare may expose separate forward and backward schedules or one training-step schedule, depending on prepare-time decisions.

## Why combine the [graph](../glossary.md#graph)

Seeing forward and backward work together allows [compiler](../glossary.md#term-compiler) and [planning](../glossary.md#term-planning) passes to reason across the [autograd](../glossary.md#autograd) boundary. This can enable:

- [dead-code elimination](../glossary.md#term-dead-code-elimination-dce) and common subexpression elimination across both phases;
- cleanup of redundant transposes and no-op unbroadcast operations;
- simpler [gradient accumulation](../glossary.md#term-gradient-accumulation);
- reuse of forward values required by backward computation;
- more accurate [lifetime](../glossary.md#term-lifetime) and logical memory planning; and
- [ownership](../glossary.md#term-ownership-meanings) and [partition](../glossary.md#partition) decisions informed by the full training computation.

These are global graph concerns, so they belong to compiler and [backend](../glossary.md#backend)-neutral planning rather than concrete backends.

The first implementation reuses only already-proved exact rewrites and folding rules. Common
subexpression elimination stays phase-local, and every changed [immutable](../glossary.md#term-immutable) candidate is revalidated.
No new algebraic [identity](../glossary.md#term-identity-meanings) follows merely from combining the phases.

## Failures and bounded [derivative](../glossary.md#term-derivative) order

[Preflight](../glossary.md#term-preflight) rejects any unsupported backward-reachable [operation](../glossary.md#operation), exact attributes, or required
derivative policy before it constructs backward expressions. Full [inference](../glossary.md#term-inference) follows [capture](../glossary.md#term-capture-meanings), so a
later failure can consume temporary [Tensor](../glossary.md#tensor) IDs; callers must already treat those IDs as [opaque](../glossary.md#term-opaque).

The current functional request has exactly one or two stages. Each stage has non-empty ordered
outputs, output-aligned optional [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings), and non-empty [identity](../glossary.md#term-identity-meanings)-unique ordered [targets](../glossary.md#term-target) from the
complete original forward inventory. Stage one selects exact requested forward-output Tensors
through `ForwardTensorReference`. Stage two may select only first-stage [gradients](../glossary.md#term-gradient) through
`FirstStageGradientReference(targetIndex)`.

One stage requires `createGraph == false`. Two stages require stage-one
`createGraph == true` and stage-two `createGraph == false`; retention serves only the immediate
second stage in the same [compile](../glossary.md#compile). An absent [seed](../glossary.md#term-seed) supplies an exact typed one only for an eligible
[scalar](../glossary.md#term-scalar) output. A present seed must match the selected output's exact [Shape](../glossary.md#shape) and floating type and
must not request gradients. `ERROR` rejects a disconnected target, while `ZERO` produces an
ordinary exact typed zero expression that several target roles may share.

This boundary supports bounded second-stage construction, including a suitably seeded
[Hessian-vector product](../glossary.md#term-hessian-vector-product-hvp) when the selected formulas are covered. It does not support a third stage,
derivative orders above two, arbitrary nesting, a persistent or [runtime](../glossary.md#term-runtime) tape, [mutable](../glossary.md#term-mutable) Tensor
gradient state, or `Tensor.backward()`. Operation-specific derivative coverage remains
[fail-closed](../glossary.md#term-fail-closed); successful [graph](../glossary.md#graph) construction alone does not promise preparation or numerical
execution on a particular [backend](../glossary.md#backend).

## Fixed recurrent [scan](../glossary.md#term-cumulative-scan) and the initial [BPTT](../glossary.md#term-backpropagation-through-time-bptt) boundary

The current [Model](../glossary.md#term-model) fixed recurrent scan is an ordinary flat multi-output forward [occurrence](../glossary.md#term-occurrence), not
a nested [training graph](../glossary.md#training-graph). Model task 0025E owns only its fixed forward meanings, [descriptor](../glossary.md#term-descriptor)
construction, and [canonical](../glossary.md#term-canonical-form) outputs. Generic [capture](../glossary.md#term-capture-meanings) preserves one [producer](../glossary.md#term-producer) as one [node](../glossary.md#node), and
current [Compiler](../glossary.md#term-compiler) forward-only [inference](../glossary.md#term-inference) independently revalidates its fully static descriptors.
Both backward-capable modes reject any complete forward inventory containing fixed recurrence before
any [derivative](../glossary.md#term-derivative) [Tensor](../glossary.md#tensor) is allocated.

[Backpropagation through time](../glossary.md#backpropagation-through-time-and-recurrent-state) (BPTT) therefore remains unsupported even though output [gradient](../glossary.md#term-gradient)
eligibility is derived from [differentiable](../glossary.md#derivative-and-chain-rule) floating inputs, states, and [parameters](../glossary.md#term-parameter). The [runtime](../glossary.md#term-runtime)
`INT64[batch]` valid-length input is always non-differentiable and never contributes a gradient.
The [fail-closed](../glossary.md#term-fail-closed) rule avoids a partially constructed backward expression and prevents a [backend](../glossary.md#backend)
from becoming the [owner](../glossary.md#term-owner) of global [autograd](../glossary.md#autograd).

This decision selects no saved-gate outputs, tape, checkpointing policy, recomputation policy,
backward [operation](../glossary.md#operation), derivative formula, or physical saved-state representation. A later Compiler
architecture and implementation task must choose those contracts after the forward executable
path is stable. Current `FORWARD_ONLY` [compilation](../glossary.md#compiler-engine-and-execution-stage-names) is the only [compile mode](../glossary.md#compile-mode--compilemode) that adopts the family
until that family's BPTT is separately designed. The accepted generic-body decision below does
not add its fixed [adjoint](../glossary.md#term-adjoint). See
[ADR 0012](../design/decisions/0012-fixed-recurrent-scan-without-regions.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>.

<a id="accepted-general-body-ad-not-current-capability"></a>

## General-body first-order [AD](../glossary.md#autograd), not execution support

The [sole scan semantic owner](contracts/recurrent-scan.md#general-differentiable-scan) now accepts
fixed-count [scan](../glossary.md#term-cumulative-scan) over ordinary [Tensor](../glossary.md#tensor) body algebra, not arbitrary regions or while loops. Current
Java implements the [Model](../glossary.md#term-model) source/[IR](../glossary.md#term-intermediate-representation-ir) foundation, [Planning](../glossary.md#term-planning) body transport and [Compiler](../glossary.md#term-compiler) scoped
[capture](../glossary.md#term-capture-meanings)/[inference](../glossary.md#term-inference), reverse-scan first-order AD and final body-bearing demand queries. Compiler0013
extends this [compile](../glossary.md#compile)-time path to the bounded length mode. Prepare0011 projects its static
[bindings](../glossary.md#term-binding) and demand through the existing resource [handoff](../glossary.md#term-handoff); CPU0042 separately qualifies fixed-geometry
[FLOAT32](../glossary.md#term-float32)/[FLOAT64](../glossary.md#term-float64) length execution. The implementation follows the existing pre-capture boundary:

```text
original scan wrappers + source Tensor body
  -> preflight, joint output cotangents and exact primal bindings
  -> ordinary reverse sibling Scan, traversing in the opposite direction
  -> one combined capture, with independent body scopes
  -> saved-value edges and demand -> planning -> separately qualified one-owner preparation
```

Read downward: [derivative](../glossary.md#term-derivative) construction precedes capture; [Runtime](../glossary.md#term-runtime) never sees derivative rules.
[Residual](../glossary.md#scan-residual-port-and-history-demand) ports are [canonical](../glossary.md#term-canonical-form) [metadata](../glossary.md#term-metadata) reserved by original [immutable](../glossary.md#term-immutable) [producer](../glossary.md#term-producer) construction, not
allocated memory. Only [publication](../glossary.md#term-publication)/downstream consumption creates live history. The reverse scan
consumes exact saved histories and explicit step/[invariant](../glossary.md#term-invariant) inputs; its own unused residual ports
allocate/store no history and impose no residual [fusion](../glossary.md#term-fusion) restriction. Internal body uses need only
step temporaries. This is Synaptik's producer design choice, not a requirement on other frameworks.

Selected existing Compiler rule entry points accept an explicit original-[primal](../glossary.md#term-primal)-Tensor binding function.
For a [TANH](../glossary.md#term-tanh) step, its derivative uses the exact saved TANH output, not a fresh tanh expression.
Contribution [identities](../glossary.md#term-identity-meanings) remain original Tensors. Joint final-[carry](../glossary.md#term-carry)/emission [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings) combine once
per [occurrence](../glossary.md#term-occurrence); repeated roles and shared [parameters](../glossary.md#term-parameter) contribute once per position, then ordinary
Tensor.add sums tied actual identities across roles/occurrences. TANH/[SIGMOID](../glossary.md#term-sigmoid) oneLike [constants](../glossary.md#term-constant)
become explicit invariant formals bound to registered outer splats, not free [leaves](../glossary.md#term-leaf). Registered
outer zeros initialize [seed](../glossary.md#term-seed)/[accumulator](../glossary.md#accumulator-type) roles through the appropriate carry/invariant formals.

Length-mode original-source AD requires every carry and every original sequence to be ROW(0),
including unused/non-[gradient](../glossary.md#term-gradient) positions. Its opposite-traversal sibling retains the same discrete
lengths actual, with no formal or [cotangent](../glossary.md#cotangent) for it. Generated seeds and exact retained primals
have explicit ROW(0), ROW(1) or SHARED roles pruned with their sequence bindings. Row-shaped
literal/zero seeds close through declared non-gradient sequences; [scalar](../glossary.md#term-scalar) constants remain
invariants. Each incoming cotangent for a SHARED value is normalized before accumulation,
including shared intermediates at B=1. Invariant-shaped suffix carries use one distinct
`old.add(contribution)` each, without a per-row parameter bank. This is [graph](../glossary.md#graph) construction only;
CPU0042 separately qualifies fixed-geometry FLOAT32/FLOAT64 forward and actual reverse execution.
Static [homogeneous](../glossary.md#term-homogeneous) FLOAT32/FLOAT64 length-mode [cold reconstruction](../glossary.md#term-cold-reconstruction) and [Checkpoint](../glossary.md#term-checkpoint) copy/[rebind](../glossary.md#capture-graph-ports-and-reconstruction)
are implemented and accepted under Compiler0018. Checkpoint0014 adds v5 [persistence](../glossary.md#term-persistence) and loaded
generic Model numerical proof; NN0033 completes bounded recurrent composition. [Prepare](../glossary.md#prepare) preserves
the actual reverse's mixed roles and exact saved ports without applying the restricted
original-source AD rule to that generated sibling or executing its body.

[Normalization](../glossary.md#term-normalization) here makes each [sensitivity](../glossary.md#term-sensitivity) match the receiving value's [shape](../glossary.md#shape) and type before
addition, including reversing broadcasts where needed. It does not standardize data values as
LayerNorm does. ROW roles track a [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters)-row [coordinate](../glossary.md#axis-rank-scalar-vector-and-matrix); SHARED values have no such row coordinate,
even when one of their extents happens to equal the batch size.

The user-approved [Model0044 source foundation](../planning/modules/model/tasks/0044-symbolic-length-scan-source-foundation.md)
combines the existing length overload with one named B, retaining static T/[features](../glossary.md#term-feature) and true row
[provenance](../glossary.md#provenance). Source support is Complete under the authority accepted on 2026-10-26; independent
source/docs review and MAIN's root test/[Javadoc](../glossary.md#term-javadoc) checkpoint passed, with MAIN acceptance on 2026-10-26.
For example, h `[Rows,2]` and X `[4,Rows,3]` can be described with [INT64](../glossary.md#data-type--datatype) L `[Rows]` and fixed
[weights](../glossary.md#term-weight), without choosing a numeric Rows or reading L.
[Compiler0015](contracts/recurrent-scan.md#runtime-preflight-and-implementation-gates) implements
independent FLOAT32/FLOAT64 inference, actual first-order sibling AD and complete fresh scoped
all-port binding together for that fixed-T/features one-B source; the Compiler-only task is Complete.
For example, binding Rows=7 inspects [descriptors](../glossary.md#term-descriptor) for h `[7,2]`, X `[4,7,3]` and
INT64 L `[7]`, not L's seven values. Changed storage/length payloads at the same Rows are [numerical
oracle](../glossary.md#adr-ci-and-qualification) cases, not metadata-binding inputs. Compiler [admission](../glossary.md#term-admission) is therefore not runtime support.
Original-source ROW(0) restrictions and generated mixed roles remain unchanged; Prepare still
rejects before numeric [extent](../glossary.md#term-extent) access/analyzers, and combined [CPU](../glossary.md#term-cpu)/[Engine](../glossary.md#term-engine) execution, cold/copy/wire
and [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context)/Training remain closed. No resource, arithmetic or activity algorithm changes. The
[source admission contract](contracts/recurrent-scan.md#header-bindings-and-source-admission)
owns exact [dimension](../glossary.md#dimension) and [layout](../glossary.md#layout) rules; no new body representation or execution [owner](../glossary.md#term-owner) is added.

For example, the [conceptual forward/residual/reverse walkthrough](general-differentiable-scan.md#conceptual-forward-and-reverse-example)
uses two steps of `h + x*w`: final `6` and emissions `[2,6]` jointly seeded with ones produce
dInitialCarry `3`, dSequence `[6,4]` and dWeight `7`. Seeding each port through separate repeated
scan AD would duplicate work/contributions. Saved-output replay would additionally break exact
[rounding](../glossary.md#term-rounding) sites for [nonlinear](../glossary.md#term-nonlinear) bodies.

Residual-root differentiation and every scan higher-order/createGraph request reject before
derivative allocation. Unsupported body occurrences in unrelated original scans also reject before
allocation. The ordinary two-stage request boundary elsewhere does not grant scan higher order. General-body
first-order Compiler inventory includes DIV and the whole required reduction/expand/[view](../glossary.md#term-view-meanings)/[contiguous](../glossary.md#term-contiguous)/
constant/cast closure, not only handwritten forward operators. Internal carry cotangents traverse
false-flag source values without changing public flags or seed guards. Bounded test-only numerical
[evaluation](../glossary.md#term-evaluation) for all-valid scans compares the reverse [topology](../glossary.md#term-topology) with [unrolled](../glossary.md#backpropagation-through-time-and-recurrent-state) existing gradients in both directions;
it is not a production runner or execution [qualification](../glossary.md#term-qualification). A complete bounded [F32](../glossary.md#floating-point-precision-and-rounding) vertical
capability precedes F32/[F64](../glossary.md#floating-point-precision-and-rounding) declared-domain completion; [padding](../glossary.md#padding) and
persistence remain separate. Existing fixed scans and NN static [unrolling](../glossary.md#term-unrolling) retain their contracts.
See [ADR 0045](../design/decisions/0045-general-differentiable-scan.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span> for the decision and costs.

## [Optimizer](../glossary.md#term-optimizer) as a [backend](../glossary.md#backend)-agnostic step

`extensions/nn` owns the [model](../glossary.md#term-model)-[module](../glossary.md#term-module) side of a training step: modules declare their trainable
[`Parameter`](../glossary.md#term-parameter) values, persistent [`Buffer`](../glossary.md#term-buffer) values, and train/eval forward behavior.
`modules/training` owns generic live [admission](../glossary.md#term-admission), the implemented `Sgd`, [`Adam`](../glossary.md#adaptive-optimizer--adam-adamw), and [`AdamW`](../glossary.md#adaptive-optimizer--adam-adamw)
mathematical updates and [`TrainingSession`](../glossary.md#training-session--trainingsession) orchestration without importing [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context). NN's narrow
`Module.trainingParameters()` adapter exposes exact ordered live membership, not [optimizer state](../glossary.md#term-optimizer-state).
This lets non-NN [Tensor](../glossary.md#tensor) [owners](../glossary.md#term-owner) train through the same boundary.

An initialized layer can begin with [heap](../glossary.md#term-heap)-backed parameters, but a session can update only
shareable native parameter storage. The caller may explicitly call
`TrainingParameterStorage.copyToSharedNative(parameters, arena)` before opening a session, with
`parameters = module.trainingParameters()` for NN. Training validates the complete source set,
revalidates live membership before installation, copies exact
represented bytes into fresh aligned native segments, and rebinds only Model's [borrowed](../glossary.md#ownership-borrowing-and-lease)
[host](../glossary.md#term-host)-storage associations. The same Parameter and Tensor objects remain in previously constructed
expressions, so the normal [Engine](../glossary.md#term-engine) [compile](../glossary.md#compile) and [prepare](../glossary.md#prepare) path can use them without a second [graph](../glossary.md#graph)
construction route. The caller owns the destination [Arena](../glossary.md#term-arena) through session close; Training owns
neither arena nor a backend transfer. The [owning Training contract](contracts/extensions-training.md#optimizertraining-lifecycle)
defines the [validation](../glossary.md#term-validation) and [lifetime](../glossary.md#term-lifetime) boundary, and [ADR 0028](../design/decisions/0028-explicit-native-parameter-copy.md)<span data-glossary-navigation="true"> ([ADR glossary](../glossary.md#term-architecture-decision-record-adr))</span>
records why the transition is explicit.

```text
live source (NN adapter or generic owner) -> explicit native copy -> TrainingSession open (Engine.prepare) -> Engine.run -> prepared close -> caller Arena close
```

Read this left to right: storage association changes only before open, while the parameter
[identity](../glossary.md#term-identity-meanings) used by the expression and [compiled graph](../glossary.md#compiled-graph--compiledgraphmodel) does not. For example, a [`Linear`](../glossary.md#term-linear-meanings) constructed
with heap-initialized [weight](../glossary.md#term-weight) and [bias](../glossary.md#term-bias) can build its [loss](../glossary.md#loss) expression, be copied, and then open a
[CPU](../glossary.md#term-cpu) `TrainingSession` on that same expression. By contrast, opening without the copy still fails
at native-only parameter [capture](../glossary.md#term-capture-meanings); attempting the copy while a session is open would invalidate
the session's fixed storage [snapshot](../glossary.md#term-snapshot). Focused unit tests cover byte preservation and failure
before rebinding, and the CPU [integration test](../glossary.md#term-integration-test) covers the end-to-end path. This transition does
not authorize Training to select a backend or change Model or NN [ownership](../glossary.md#term-ownership-meanings).

The public initial [lifecycle](../glossary.md#lifecycle) compiles and prepares the forward/[backward graph](../glossary.md#backward-graph) once through Engine,
then applies the optimizer only after every [objective](../glossary.md#term-objective) and [gradient](../glossary.md#term-gradient) [publication](../glossary.md#term-publication) is [detached](../glossary.md#detached-model-results) and
validated:

```text
one reusable prepared forward/backward execution
  -> target-indexed detached gradients
  -> stage assigned optimizer slots/accumulation candidates
  -> validate all stable parameter bindings again
  -> commit parameter bytes and optimizer/session state
```

[Compiler](../glossary.md#term-compiler) [target](../glossary.md#term-target) order and publication `targetIndex` are the mapping authorities; Training does not
infer a parameter association from [Shape](../glossary.md#shape), storage, or publication order alone. At open, ordered
groups resolve one [immutable](../glossary.md#term-immutable) algorithm per parameter; for example an Adam default, [SGD](../glossary.md#term-stochastic-gradient-descent-sgd) weight
group, and AdamW bias group still use one graph and one global step. `Sgd` supports [momentum](../glossary.md#term-momentum),
[dampening](../glossary.md#term-dampening), coupled [weight decay](../glossary.md#term-weight-decay), and compatible [Nesterov](../glossary.md#term-nesterov) behavior. `Adam` and `AdamW` use
bias-corrected first and second moments; the former couples decay into gradients, while the latter
decays the [master parameter](../glossary.md#term-master-parameter) separately. Their moments and low-[precision](../glossary.md#term-precision) master candidates are
staged before any parameter write. The session supports reset-and-step, accumulate-only,
accumulate-and-step, zeroing, immutable state snapshots,
strict restore, and fail-before-[commit](../glossary.md#term-commit) rejection of non-[finite](../glossary.md#term-finite) optimizer state or arithmetic.
Exact-ID [`run`](../glossary.md#term-run) and positional `runValues` share that [transaction](../glossary.md#term-transaction) and the same prepared [owner](../glossary.md#term-owner).
The latter accepts fresh stored [batches](../glossary.md#term-batch-machine-learning-and-hyperparameters) and targets in `session.inputs()` order with exact full
static [descriptors](../glossary.md#term-descriptor); it inserts the fixed parameters and private [seed](../glossary.md#term-seed) before public Engine
positional execution. No [prototype](../glossary.md#term-prototype) storage is copied or rebound. Caller external storage is
borrowed only through the [synchronous](../glossary.md#term-synchronous) run and result closure. Repeating one nonempty value in two
slots, overlapping external slices, or [aliasing](../glossary.md#term-aliasing) a parameter still rejects before execution.
Training owns none of `Parameter`, `Buffer`, layer behavior, train/[eval mode](../glossary.md#inference-and-evaluation), [backend ownership](../glossary.md#backend-ownership),
or a [fused](../glossary.md#specialization-and-fusion) update route.

## Continuing at the next training input

The [Training lifecycle contract](contracts/extensions-training.md#optimizertraining-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../glossary.md#lifecycle))</span>
defines which [owner](../glossary.md#term-owner) must supply each part of a recovery point. Read the table from left to
right: [capture](../glossary.md#term-capture-meanings) is when the value is observed, while [preflight](../glossary.md#term-preflight) names who can reject a bad restore
before changing its own state. A **[detached](../glossary.md#detached-model-results)** [snapshot](../glossary.md#term-snapshot) owns its values independently of live
[Tensors](../glossary.md#tensor) and storage.

| Participant and owner | Capture point and detachedness | Restore or preflight authority | Unsupported boundary |
|---|---|---|---|
| [Trainable parameter](../glossary.md#trainable-parameter) values and [optimizer](../glossary.md#term-optimizer) slots, Training | `TrainingSession.state()` after an admitted [run](../glossary.md#term-run); `TrainingState` owns [canonical](../glossary.md#term-canonical-form) logical/master bytes, first/second slots, pending [gradients](../glossary.md#term-gradient), optimizer configuration, paths, and counters. | `TrainingSession.restore()` checks its complete fixed [parameter](../glossary.md#term-parameter) [view](../glossary.md#term-view-meanings) and [payload](../glossary.md#artifact-envelope-and-atomic-publication) before its own [commit](../glossary.md#term-commit). | It cannot install a different parameter Tensor or storage [binding](../glossary.md#term-binding). |
| Persistent [buffers](../glossary.md#term-buffer) and bindings, [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) | `Module.stateDictionary()` observes paths and exact Tensor references; it is shallow. A recovery owner needs a proven step-aligned [detached value](../glossary.md#detached-model-results) for every relevant buffer. `TrainingState` already detaches supported parameter values. | NN's `loadStateDictionary()` checks path, kind, type, [Shape](../glossary.md#shape), and parameter eligibility before replacing bindings. | A shallow dictionary is not a detached value [checkpoint](../glossary.md#term-checkpoint); loading during an open [Training session](../glossary.md#training-session--trainingsession) violates its fixed bindings. Training-mode `BatchNorm.forward` binds symbolic next-buffer expressions at [graph](../glossary.md#graph) construction; no current test or [API](../glossary.md#term-application-programming-interface-api) proves a detached value corresponding to a completed session step, so this workflow is ineligible for exact continuation today. |
| Per-[module](../glossary.md#term-module) [forward mode](../glossary.md#term-forward-mode), NN | Caller reads `mode()` on every relevant module when saving; no mode is in `StateDictionary`. | Caller restores the intended tree state before constructing the [objective](../glossary.md#term-objective); recursive `train()`/`eval()` set a uniform tree or reconstruct mixed modes with parent-first child overrides under exclusive access. | A single root mode cannot reconstruct mixed local modes; local mode also does not prove which context a forward used. |
| Effective forward contexts, NN and caller | Caller records the [immutable](../glossary.md#term-immutable) `ForwardContext` choice actually supplied to each mode-sensitive forward when constructing the fixed objective, with enough construction-site association to reproduce it. This is a separate caller-owned value, not a `StateDictionary` field. | Caller validates and rebuilds the same context choices and objective before `TrainingSession.open`. | `Dropout.forward` and `BatchNorm.forward` accept an explicit context that may differ from `Module.mode()`; if the effective choices cannot be reconstructed, refuse exact continuation. |
| Fixed objective and graph random state, [Model](../glossary.md#term-model)/[Engine](../glossary.md#term-engine) and caller | Caller records graph construction and the key/counter supplied to `GraphRngState.initial`; the expression is storage-free and the session compiles it once. | Rebuild the same intended objective before `TrainingSession.open`; [compile](../glossary.md#compile)/[prepare](../glossary.md#prepare) and Training capture check their respective domains. | No current per-run [RNG](../glossary.md#graph-rng-state--graphrngstate) input/output or automatic [dropout](../glossary.md#dropout--inverted-dropout) stream advancement. |
| Next input and program progress, caller | At a [quiescent](../glossary.md#term-quiescent) successful-run boundary, record next [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) [identity](../glossary.md#term-identity-meanings)/order, sampler cursor, [epoch](../glossary.md#term-epoch-meanings), and any external random generator state in the caller's own detached representation. | Caller validates and restores its own source before giving exact-ID inputs to `run` or fresh stored values in `session.inputs()` order to `runValues`. | Training counters do not identify a batch or restore a sampler; positional [admission](../glossary.md#term-admission) does not advance graph RNG. Durable exact encoding requires a supported versioned [schema](../glossary.md#term-schema), bounded [codec](../glossary.md#term-codec), and owner validator/restorer. |
| Scheduler, scaler, or other program state, caller if present | Capture each concrete external participant at the same boundary. | Its owner validates and restores it. | No built-in scheduler/scaler contract or generic state [registry](../glossary.md#term-registry) exists; arbitrary caller objects cannot be encoded as exact state by Checkpoint. |

For the fixed-session route, no new Training public value is needed: the existing
`TrainingState` and a caller-owned, explicit progress/random [handoff](../glossary.md#term-handoff) are sufficient for
the supported state-complete claim only when every affecting participant is demonstrably
capturable and restorable on the same admitted computation route. The current claim excludes
training-mode BatchNorm and any unreconstructible objective context. It is not a cross-[device](../glossary.md#term-device)
bitwise replay promise. The [decision record](../design/decisions/0030-exact-training-resume-state-ownership.md)
explains the tradeoff and the future joint transition.

**Concrete save and restore scenario.** Suppose batch identifiers are `[A, B, C, D]`, the
caller has successfully run `A` and `B`, and the next batch is `C` at cursor `2` in epoch `0`.
The fixed objective was built with `GraphRngState.initial(17, 4)` and uses that same explicit
graph state at each session run. This example uses a tree with no persistent buffers or
training-mode BatchNorm. At this quiescent boundary the caller records cursor `2`, epoch `0`,
its external augmentation generator's next state `81`, graph construction inputs `(17, 4)`,
the effective `ForwardContext(TRAINING)` supplied to dropout when constructing the objective,
and every relevant module's local mode; it also obtains `TrainingState`. The explicit context
must be recorded even if dropout's local mode was [`EVALUATION`](../glossary.md#term-evaluation). For recovery it builds a
distinct private NN tree, loads its saved bindings, restores local modes, copies parameter storage
to the supported native domain, supplies the recorded forward context while reconstructing
the same objective from the final NN bindings, then opens a new private session. It compares
canonical logical parameter bytes, restores `TrainingState`, stages caller progress and external
generator state, publishes the validated holder under exclusion, and supplies `C` as the next
input. The key/counter is rebuilt as a graph expression;
`81` belongs to the separate caller-owned generator; its in-memory handoff is not a Checkpoint
codec. The current APIs do not make that multi-owner sequence [atomic](../glossary.md#resource-transaction-and-rollback). The caller instead keeps
the replacement tree and session private, compares the newly opened private session's canonical
logical parameter bytes from `TrainingSession.state()` with the saved `TrainingState` bytes before
restore, and publishes the replacement and staged caller state only after restore and all other
checks succeed under caller-owned exclusion. The example does not claim a new dropout draw for
each batch: the fixed graph reuses `(17, 4)`.

**Refusal scenario.** If the augmentation generator is seeded again but its state after `B` is
unknown, or the saved batch cursor cannot identify `C`, restoring `TrainingState` still restores
[optimizer state](../glossary.md#term-optimizer-state) but cannot identify the next computation. Label this a partial restore or
restart. The same refusal applies if the forward context actually used for dropout cannot be
reconstructed, or if training-mode BatchNorm's symbolic running buffers have no proved
step-aligned detached [publication](../glossary.md#term-publication). A future durable [artifact](../glossary.md#term-artifact) also refuses an arbitrary caller
RNG/sampler/scheduler/scaler object without a supported versioned codec and owner validator.
Likewise, a program expecting dropout to consume graph state `(17, 4)` at `A` and
automatically use its returned next state at `B` cannot call the current fixed session an exact
advancing-stream continuation: `TrainingSession.run` exposes no per-run graph-RNG handoff.

The [Training 0004 integration fixture](../../testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/PrivateTrainingRestoreIntegrationTest.java)
exercises this bounded caller-coordinated joint restore; it does not introduce a general
[transaction](../glossary.md#term-transaction) API. It uses a model-only artifact for NN values and keeps Training and typed caller
state in memory. It saves the actual quiescent trained tree through ordinary `ModelCheckpoint.save`.
Checkpoint preflights the original state and internally selects raw-preserving [`CONTIGUOUS`](../glossary.md#term-contiguous)
outputs for unique [provenance](../glossary.md#provenance)-free [leaves](../glossary.md#term-leaf) through the existing public Engine path; it does not
replace NN bindings or add generic zero-[node](../glossary.md#node) Engine/Prepare support. The [fixture](../glossary.md#term-fixture) strict-loads a
private [heap](../glossary.md#term-heap) tree, directly re-saves it, checks complete artifact equality, and loads it again
before native parameter copying. This proves a bounded model [round trip](../glossary.md#term-round-trip), not durable Training
continuation or cross-owner [atomicity](../glossary.md#term-atomicity). See the [model checkpoint lifecycle](../reference/checkpoint/values.md#lifecycle-and-invariants).
Read this sequence as moving from detached evidence to private mutation to publication:

```text
quiesce published state; validate available detached facts and participant completeness
  -> load NN values/modes and prepare native parameter storage on a distinct private tree
  -> rebuild objective from its final Tensors; open a private session
  -> compare canonical NN/Training logical bytes by stable path; restore TrainingState
  -> validate staged caller state and candidate; publish under caller exclusion
```

Every check available from detached state precedes mutation of a published object. The NN load,
native copy, Engine preparation, and session restore may mutate or allocate within the private
candidate; their failure closes and discards that candidate without publication. In particular,
`TrainingSession.restore` needs an open session to validate its fixed parameter view and payload.
No separate pure pre-open Training validator is required. The newly opened candidate's
`TrainingSession.state()` provides a detached canonical-byte view for the required comparison
when the NN source does not expose comparable detached bytes; it copies the complete parameter
and optimizer snapshot, so an earlier detached comparison is preferable when those bytes are
already available. `ModelCheckpoint.inspect` exposes [descriptors](../glossary.md#term-descriptor), not value bytes, and the
loaded NN Tensor's native-order storage must not be compared directly with Training's canonical
[big-endian](../glossary.md#text-encoding-and-file-boundaries) bytes. The old session may remain quiescent while a distinct candidate is built; the
caller owns its retirement at handoff. Publication is a caller-controlled
transition under exclusion, not [rollback](../glossary.md#term-rollback) of an already closed old session or a linearizable view
for readers that ignore exclusion. A participant that cannot be staged for that transition
precludes an exact claim. The [restore decision](../design/decisions/0035-private-candidate-training-restore.md)
explains why the [validation](../glossary.md#term-validation) stages are split.

A future Checkpoint adapter may only encode and recover participants with supported versioned,
bounded schemas/codecs and owner validators/restorers; it cannot create a missing sampler,
random-state handoff, or scheduler. The BatchNorm training path needs separate step-aligned
buffer publication evidence before inclusion.

## Future compiled [optimizer](../glossary.md#term-optimizer) [graph](../glossary.md#graph)

A future architecture version may generate optimizer update operations as part of a larger graph:

```text
forward + backward + optimizer update graph
  -> optimize and partition
  -> backend prepare
  -> training-step schedule
```

In that [model](../glossary.md#term-model), [backend](../glossary.md#backend) [prepare](../glossary.md#prepare) may fuse or specialize optimizer updates while the training extension remains backend-agnostic. This direction requires the explicit architecture updates required by the contract; it is not permission to move backend implementation into training now.

## Why `MetalOptimizerBridge` does not belong in training

A `MetalOptimizerBridge` would make the training extension depend on a [concrete backend](../glossary.md#concrete-backend) and give training responsibility for [backend](../glossary.md#backend) execution. That reverses the intended [dependency direction](../glossary.md#java-platform-and-build-tools).

The correct [ownership](../glossary.md#term-ownership-meanings) is:

```text
modules/training     -> optimizer algorithm and mathematical intent
compiler/model       -> graph operations or backend-neutral step representation
modules/backend/metal       -> MPSGraph or custom fused Metal implementation
runtime              -> execution of the prepared executable
```

The same rule excludes [CPU](../glossary.md#term-cpu)- and [CUDA](../glossary.md#term-cuda)-specific [optimizer](../glossary.md#term-optimizer) bridges from training. See [Module Boundaries](module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../glossary.md#term-module))</span> and [Dependency Rules](dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../glossary.md#term-dependency))</span>.
