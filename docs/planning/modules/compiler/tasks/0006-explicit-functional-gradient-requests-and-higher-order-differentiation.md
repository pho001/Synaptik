# Task 0006: Explicit Functional [Gradient](../../../../glossary.md#term-gradient) Requests and Higher-Order Differentiation

## Status

Complete

## Goal

Replace the package-private [scalar](../../../../glossary.md#term-scalar)-[objective](../../../../glossary.md#term-objective)-only first-order request with one [compiler](../../../../glossary.md#term-compiler)-owned
public [immutable](../../../../glossary.md#term-immutable) [functional gradient request](../../../../glossary.md#functional-gradient-request) that can describe either:

- one ordered [reverse-mode](../../../../glossary.md#primal-adjoint-and-differentiation-seed) stage; or
- exactly two ordered reverse-mode stages, where the second stage differentiates exact gradient
  results produced by the first stage.

[Compile](../../../../glossary.md#compile) the requested forward outputs and every requested first- and second-[derivative](../../../../glossary.md#term-derivative) root into
one immutable combined [graph](../../../../glossary.md#graph) in one [capture](../../../../glossary.md#term-capture-meanings). Publish immutable result roles in stage and [target](../../../../glossary.md#term-target)
order, preserve the exact requested target [identity](../../../../glossary.md#term-identity-meanings) through pre-capture [validation](../../../../glossary.md#term-validation) and its
[`TensorId`](../../../../glossary.md#tensorid)-to-[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) projection after capture, and add compiler-owned derivative-order
[metadata](../../../../glossary.md#term-metadata) without changing [Model](../../../../glossary.md#term-model) `GraphPhase`.

The request is functional: it returns graph results and [artifacts](../../../../glossary.md#term-artifact). It does not mutate a [Tensor](../../../../glossary.md#tensor),
store a [gradient](../../../../glossary.md#term-gradient) on a Tensor, [run](../../../../glossary.md#term-run) an [eager](../../../../glossary.md#term-eager-evaluation) tape, or add another public compile [facade](../../../../glossary.md#term-facade).

## [Scope](../../../../glossary.md#term-scope)

### Public functional request

Add this exact public root-package record:

```java
public record FunctionalGradientRequest(
        List<FunctionalGradientRequest.Stage> stages) {

    public enum DisconnectedPolicy {
        ERROR,
        ZERO
    }

    public sealed interface OutputReference
            permits ForwardTensorReference, FirstStageGradientReference {}

    public record ForwardTensorReference(Tensor tensor)
            implements OutputReference {}

    public record FirstStageGradientReference(int targetIndex)
            implements OutputReference {}

    public record Stage(
            List<OutputReference> outputs,
            List<Optional<Tensor>> cotangentSeeds,
            List<Tensor> targets,
            boolean createGraph,
            DisconnectedPolicy disconnectedPolicy) {}
}
```

The request and every nested record [snapshot](../../../../glossary.md#term-snapshot) list membership and retain exact [immutable](../../../../glossary.md#term-immutable) [element](../../../../glossary.md#term-element)
references. They use ordinary record value equality and hashing for their public value [shape](../../../../glossary.md#shape), but
all [Tensor](../../../../glossary.md#tensor) membership, uniqueness, ancestry, and [target](../../../../glossary.md#term-target)-route decisions during [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) use
exact object [identity](../../../../glossary.md#term-reference-identity).

Constructor [validation](../../../../glossary.md#term-validation) is exact:

- `stages` is non-null and contains exactly one or two non-null stages;
- each stage has a non-null, non-empty `outputs` list, a non-null `cotangentSeeds` list of the
  same size, and a non-null, non-empty `targets` list;
- no output reference, [seed](../../../../glossary.md#term-seed) `Optional`, present seed Tensor, target, or policy is null;
- stage targets are exact-object-[identity](../../../../glossary.md#term-identity-meanings)-unique within that stage;
- stage-one output references are all `ForwardTensorReference`;
- a one-stage request has `createGraph == false`;
- a two-stage request has stage one `createGraph == true`, stage two
  `createGraph == false`, and every stage-two output is a
  `FirstStageGradientReference`;
- every first-stage-gradient reference uses a non-negative target index smaller than the exact
  stage-one target-list size; and<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient))</span>
- no other stage count, reference direction, [derivative](../../../../glossary.md#term-derivative) chain, or `createGraph` combination is
  representable.

The constructor does not traverse Tensor [provenance](../../../../glossary.md#provenance), inspect [descriptors](../../../../glossary.md#term-descriptor) beyond non-null record
components, allocate a Tensor, [capture](../../../../glossary.md#term-capture-meanings) a [graph](../../../../glossary.md#graph), or infer a [compile mode](../../../../glossary.md#compile-mode--compilemode). Descriptor, membership,
route, and seed checks belong to [compile](../../../../glossary.md#compile)-time [preflight](../../../../glossary.md#term-preflight) because they depend on the supplied
forward boundary.

`createGraph` is a compile-local permission to retain the first-stage formula as [differentiable](../../../../glossary.md#derivative-and-chain-rule)
ordinary Tensor expressions for the immediately following stage. It is true exactly when a
second stage exists. It is not stored on a Tensor, is not a global recording mode, and does not
permit a third stage or a later call to continue a hidden derivative chain.

### Output, [target](../../../../glossary.md#term-target), and [cotangent](../../../../glossary.md#cotangent)-[seed](../../../../glossary.md#term-seed) contract

For stage one:

- every `ForwardTensorReference.tensor()` must be the exact object reference of one requested
  `forwardOutputs` [element](../../../../glossary.md#term-element);
- output references must be exact-object-identity-unique; callers who want to combine two
  cotangents for one output must add them before the request;
- the referenced output must have a floating [`DataType`](../../../../glossary.md#data-type--datatype) and `requiresGrad == true`; and
- output order determines cotangent seeding and reverse-root encounter order.

For stage two:

- each `FirstStageGradientReference.targetIndex()` resolves to the exact generated gradient for
  that stage-one target role, not to an equal Tensor, a captured [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot), or a reconstructed
  wrapper;
- reference indices must be unique within stage two;
- the referenced [gradient](../../../../glossary.md#term-gradient) must have the exact stage-one target [Shape](../../../../glossary.md#shape) and floating [data type](../../../../glossary.md#data-type--datatype); and
- output order determines second-stage cotangent seeding and reverse-root encounter order.

For both stages:

- `targets` are ordered, exact-object-[identity](../../../../glossary.md#term-identity-meanings)-unique floating [Tensors](../../../../glossary.md#tensor) with
  `requiresGrad == true`;
- each target must occur in the complete original forward Tensor inventory rooted at all
  requested forward outputs, but need not be connected to every selected stage output;
- a target may be a leaf, an intermediate, a selected forward output, or a target used by both
  stages;
- target order determines result-role order within the stage; and
- targets are not traversal stops.

`cotangentSeeds` aligns one-for-one with `outputs`. An empty optional requests the default seed.
The default is legal only when the exact selected output [descriptor](../../../../glossary.md#term-descriptor) has `Shape.scalar()`, a
floating data type, and gradient eligibility. It is the existing exact positive-one [logical
splat](../../../../glossary.md#logical-splat) of that output type.

A present explicit seed:

- is the exact supplied Tensor reference;
- has a floating data type equal to the selected output data type;
- has a Shape equal to the selected output Shape;
- has `requiresGrad == false`;
- may be a [provenance](../../../../glossary.md#provenance)-free bindable [leaf](../../../../glossary.md#term-leaf) or an ordinary Tensor expression; and
- is used directly as the output cotangent without broadcasting, implicit casting, storage
  reads, or numerical evaluation.<span data-glossary-navigation="true"> ([broadcasting glossary](../../../../glossary.md#broadcasting); [evaluation glossary](../../../../glossary.md#term-evaluation))</span>

Seed [producers](../../../../glossary.md#term-producer) already present in the original forward inventory retain [derivative](../../../../glossary.md#term-derivative) order zero.
An independent seed [producer](../../../../glossary.md#term-producer) is inventoried and validated with its stage and receives that
stage's derivative order; a provenance-free seed remains a [graph](../../../../glossary.md#graph) input with no phase or order.
The seed is not a requested forward [publication](../../../../glossary.md#term-publication) unless the caller also lists it in
`forwardOutputs`. A non-[scalar](../../../../glossary.md#term-scalar) output with an absent seed, mismatched Shape or type, non-floating
seed, gradient-eligible seed, or invalid reference fails before derivative construction.

Multiple output roots define the sum of their [vector](../../../../glossary.md#term-vector)-[Jacobian](../../../../glossary.md#term-jacobian) products. Reverse accumulation
starts one contribution per output in output order, then preserves the established producer
postorder, selected output-slot order, input-position order, and left-associated
`Tensor.add` contribution order.

### Disconnected [targets](../../../../glossary.md#term-target)

`DisconnectedPolicy.ERROR` requires at least one [differentiable](../../../../glossary.md#derivative-and-chain-rule) [identity](../../../../glossary.md#term-identity-meanings) route from a selected
stage output to each target. The first disconnected target fails with:

```text
functionalGradientRequest.stages[<stage>].targets[<target>]
is disconnected from the selected stage outputs
```

Stage indices in failures are zero-based public list indices. The failure occurs before any
[Tensor](../../../../glossary.md#tensor) for that stage is allocated. For stage one it therefore consumes no [derivative](../../../../glossary.md#term-derivative)
[`TensorId`](../../../../glossary.md#tensorid); for stage two, valid stage-one formula Tensors may already exist, but no stage-two
Tensor is allocated.

`DisconnectedPolicy.ZERO` returns one ordinary exact positive-zero Tensor expression with the
target's exact [Shape](../../../../glossary.md#shape) and floating [data type](../../../../glossary.md#data-type--datatype). It uses the existing request-local explicit [logical
splat](../../../../glossary.md#logical-splat) and ordinary `expand` when the target is non-[scalar](../../../../glossary.md#term-scalar). It does not add a zero [operation](../../../../glossary.md#operation) kind,
an identity [node](../../../../glossary.md#node), a [mutable](../../../../glossary.md#term-mutable) zero [gradient](../../../../glossary.md#term-gradient), storage, or a [runtime](../../../../glossary.md#term-runtime) special case. Two disconnected
targets may share the exact generated zero Tensor only when their exact Shape and data type are
equal; their ordered result roles remain distinct.

A target absent from the complete original forward inventory is invalid, not disconnected, and
is rejected under both policies.

### [Preflight](../../../../glossary.md#term-preflight) and failure order

Generalize `AutogradPreflight` from `FirstOrderRequest` to `FunctionalGradientRequest` while
retaining `FirstOrderGradientCoverage` as the single `D`/[`ND`](../../../../glossary.md#term-nd)/`FC` role and formula-[owner](../../../../glossary.md#term-owner) source.
Do not add a public [registry](../../../../glossary.md#term-registry), [derivative](../../../../glossary.md#term-derivative) [domain-specific language](../../../../glossary.md#domain-specific-language), captured-value-to-[Tensor](../../../../glossary.md#tensor)
conversion, or second [operation](../../../../glossary.md#operation) algebra.

The two `GraphCompiler.compile(...)` overloads remain package-private and keep their existing
[parameter](../../../../glossary.md#term-parameter) count and order. Replace only:

```java
Optional<AutogradPreflight.FirstOrderRequest>
```

with:

```java
Optional<FunctionalGradientRequest>
```

There is no additional overload, builder, [facade](../../../../glossary.md#term-facade), static public [compile](../../../../glossary.md#compile) method, or public
`GraphCompiler`.

[Validation](../../../../glossary.md#term-validation) and preflight occur in this order:

1. validate the existing top-level compile [arguments](../../../../glossary.md#term-argument) in declaration order;
2. validate the mode/request-presence [matrix](../../../../glossary.md#term-matrix-meanings);
3. validate [immutable](../../../../glossary.md#term-immutable) request structure and `createGraph`/reference order;
4. inventory the complete original forward request and explicit [constant](../../../../glossary.md#term-constant) ingress;
5. validate stage-one forward references, seed references and descriptors, then target
   membership, type, eligibility, and identity uniqueness in list order;<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [identity glossary](../../../../glossary.md#term-identity-meanings))</span>
6. select and validate every stage-one `D` route through the existing coverage checker and
   occurrence-local preflight matrix;
7. resolve every stage-one disconnected [target](../../../../glossary.md#term-target) and its policy;
8. when stage two exists, validate reference indices and use the checker plus the task-0005E
   transitive formula-operation closure to prove that every possible selected second-pass
   occurrence is classified and has an existing formula owner;
9. only after the complete stage-one plan succeeds, allocate stage-one seeds, constants, and
   formulas;
10. resolve exact stage-one gradient references, validate stage-two seed descriptors and target
    facts, run occurrence-local second-stage preflight on the generated expressions, and resolve
    stage-two disconnected targets; and<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [run glossary](../../../../glossary.md#term-run))</span>
11. only after the complete second-stage plan succeeds, allocate stage-two seeds, constants, and
    formulas.

Null checks retain `NullPointerException` with the exact component path. Structural, membership,
[descriptor](../../../../glossary.md#term-descriptor), route, policy, or unsupported differentiation facts use
`IllegalArgumentException` with the first deterministic component/[occurrence](../../../../glossary.md#term-occurrence) path and existing
family reason. Unknown kinds, `ND` routes selected as targets, `FC` rows, malformed
cardinalities, missing [canonical](../../../../glossary.md#term-canonical-form) auxiliaries, and unsupported generated second-pass rows [fail
closed](../../../../glossary.md#admission-preflight-and-fail-closed-validation).

The allocation guarantee is stage-local. Invalid request structure, forward references,
stage-one [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings)/targets/routes, and stage-one disconnected `ERROR` consume no derivative
[`TensorId`](../../../../glossary.md#tensorid). A second-stage failure may follow successful stage-one construction, but it consumes
no stage-two Tensor ID and returns no partial [graph](../../../../glossary.md#graph), role, [publication plan](../../../../glossary.md#publication-plan), or [artifact](../../../../glossary.md#term-artifact). Tensor
IDs already consumed by valid stage-one construction are [opaque](../../../../glossary.md#term-opaque) and are not rolled back.

### [Reverse-mode](../../../../glossary.md#primal-adjoint-and-differentiation-seed) stages and ordinary [Tensor](../../../../glossary.md#tensor) formulas

Retain `FirstOrderAutograd` as the package-private [owner](../../../../glossary.md#term-owner) of one reverse-mode stage. Generalize its
entry to accept the successful stage plan, ordered output/[cotangent](../../../../glossary.md#cotangent) pairs, [target](../../../../glossary.md#term-target) list, and
disconnected policy. `GraphCompiler` invokes it exactly once or twice. Do not add
`SecondOrderAutograd`; the [second derivative](../../../../glossary.md#term-second-derivative) is one more use of the same checked first-order
primitive over generated ordinary Tensor expressions.

Reuse every [Compiler](../../../../glossary.md#term-compiler) 0005E `D`/[`ND`](../../../../glossary.md#term-nd)/`FC` decision, exact formula owner, [canonical](../../../../glossary.md#term-canonical-form) auxiliary,
cotangent-[normalization](../../../../glossary.md#term-normalization), [derivative](../../../../glossary.md#term-derivative) policy, and failure reason. First- and second-stage formulas
use only current public Tensor operations and the same request-local exact splat owner.

The combined expansion retains:

- one [identity](../../../../glossary.md#term-identity-meanings) set of original forward [producers](../../../../glossary.md#term-producer);
- one identity set of stage-one-owned producers, including independent seed expressions and
  generated formulas;<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed))</span>
- one identity set of stage-two-owned producers, including independent seed expressions and
  generated formulas;
- ordered exact target-to-[gradient](../../../../glossary.md#term-gradient) Tensor roles for each stage; and
- caller [constant](../../../../glossary.md#term-constant) [bindings](../../../../glossary.md#term-binding) followed by generated bindings in deterministic first-use order.

The sets and Tensor references are ephemeral [compile](../../../../glossary.md#compile)-local bookkeeping and are discarded after
combined [capture](../../../../glossary.md#term-capture-meanings).

### Public [gradient](../../../../glossary.md#term-gradient) [publication](../../../../glossary.md#term-publication) [bindings](../../../../glossary.md#term-binding)

Add this exact public root-package record:

```java
public record GradientPublicationBinding(
        int derivativeOrder,
        int targetIndex,
        TensorId target,
        ValueId valueId) {}
```

`derivativeOrder` is exactly `1` or `2`. `targetIndex` is the zero-based position in that
[derivative](../../../../glossary.md#term-derivative) stage's [target](../../../../glossary.md#term-target) list. `target` is the [`TensorId`](../../../../glossary.md#tensorid) read from the exact target [Tensor](../../../../glossary.md#tensor)
reference after [identity](../../../../glossary.md#term-identity-meanings) [validation](../../../../glossary.md#term-validation). `valueId` is the final [graph](../../../../glossary.md#graph)-local gradient [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot).

The constructor rejects another order, a negative target index, or a null identity. Result-role
lists are ordered first by derivative order and then by target index. Equal target IDs may occur
once in each derivative order. Within one derivative order target IDs are unique. Several
bindings may share one gradient value, including ZERO-policy results, without manufacturing
identity [nodes](../../../../glossary.md#node).

Exact Tensor object [identity](../../../../glossary.md#term-reference-identity) is authoritative while the expression is available before [capture](../../../../glossary.md#term-capture-meanings).
[Artifacts](../../../../glossary.md#term-artifact) deliberately retain only its stable `TensorId` projection and final `ValueId`; they do
not retain public Tensor, [producer](../../../../glossary.md#term-producer), [provenance](../../../../glossary.md#provenance), or request objects.

Replace package-private `GraphCompilation.GradientResultRole` and publication gradient
`PublicationBinding` values with the public `GradientPublicationBinding`. Rename the public [Model](../../../../glossary.md#term-model)
record used for requested forward publications from `PublicationBinding` to
`ForwardPublicationBinding` so its role is not confused with the richer gradient binding.
`PublicationPlan` remains a public final type with package-private construction and retains
`List<ForwardPublicationBinding>` plus `List<GradientPublicationBinding>` in private members. Its
only public list accessors are `forwardBindings()` and `gradientBindings()`. Boundary validation
flattens all first-order bindings followed by all second-order bindings and appends each
previously unseen gradient `valueId()` after the forward prefix.

### [Derivative](../../../../glossary.md#term-derivative)-order [graph](../../../../glossary.md#graph) [metadata](../../../../glossary.md#term-metadata)

Add this exact public final root-package [artifact](../../../../glossary.md#term-artifact):

```java
public final class DerivativeGraphMetadata {
    public DerivativeGraphMetadata(
            CompiledGraphModel graph,
            Map<NodeId, Integer> derivativeOrderByNode);

    public CompiledGraphModel graph();

    public Map<NodeId, Integer> derivativeOrderByNode();
}
```

The artifact retains the exact final graph reference and an [immutable](../../../../glossary.md#term-immutable) encounter-order [snapshot](../../../../glossary.md#term-snapshot)
with one exact graph [`NodeId`](../../../../glossary.md#nodeid) key per [node](../../../../glossary.md#node):

- order `0` for every original forward [producer](../../../../glossary.md#term-producer);
- order `1` for every producer first owned by stage one; and
- order `2` for every producer first owned by stage two.

A producer reused by a later stage retains the first order that owns it. The constructor
rejects nulls, unknown/missing/duplicate node IDs, another order, or map iteration order different
from `graph.nodes()` order. It contains no [Tensor](../../../../glossary.md#tensor), [seed](../../../../glossary.md#term-seed), request, formula-[owner](../../../../glossary.md#term-owner), storage, [runtime](../../../../glossary.md#term-runtime),
[backend](../../../../glossary.md#backend), or execution state.

`GraphPhase` remains unchanged in [Model](../../../../glossary.md#term-model). Order zero nodes have `GraphPhase.FORWARD`; orders one
and two have `GraphPhase.BACKWARD`. The sidecar augments rather than replaces phase.

`GraphCapture.captureCombined(...)` receives the three producer [identity](../../../../glossary.md#term-identity-meanings) sets and assigns IDs
once. [Canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), exact rewriting, [constant folding](../../../../glossary.md#term-constant-folding), [dead-code elimination](../../../../glossary.md#term-dead-code-elimination-dce), and graph
rebuilds remap the sidecar with the graph. [Common-subexpression elimination](../../../../glossary.md#term-common-subexpression-elimination-cse) remains phase-local
and becomes derivative-order-local: equal expressions may merge only when their phase and
derivative order are equal. This preserves result meaning and metadata without adding or
broadening an algebraic rewrite.

### One combined [capture](../../../../glossary.md#term-capture-meanings) and [artifacts](../../../../glossary.md#term-artifact)

One [compile](../../../../glossary.md#compile) invocation performs this sequence:

```text
validate the complete functional request
  -> construct stage-one Tensor gradients
  -> if requested, construct stage-two Tensor gradients
  -> capture forward outputs plus every stage result root together once
  -> infer and validate once
  -> canonicalize and run the established exact combined-graph pipeline once
  -> revalidate changed candidates
  -> publication and planning once
  -> CompileArtifacts
```

The [graph](../../../../glossary.md#graph)-output boundary is:

1. ordered distinct forward output values;
2. each previously unseen first-order [gradient](../../../../glossary.md#term-gradient) value in first-order [target](../../../../glossary.md#term-target) order; and
3. each previously unseen second-order gradient value in second-order target order.

Capture preserves shared multi-output [producer](../../../../glossary.md#term-producer) occurrences, every output slot, exact
[`TensorProducer`](../../../../glossary.md#producer--tensorproducer) [identity](../../../../glossary.md#term-identity-meanings), repeated input positions, [canonical](../../../../glossary.md#term-canonical-form) saved edges, explicit [constant](../../../../glossary.md#term-constant)
facts, and one allocation of each [`NodeId`](../../../../glossary.md#nodeid)/[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot). It does not compile the [forward graph](../../../../glossary.md#forward-graph),
stage one, and stage two separately.

Add `DerivativeGraphMetadata derivatives` to `GraphCompilation` and as the eighth, final
component of public `CompileArtifacts`:

```java
public record CompileArtifacts(
        CompileMode mode,
        CompiledGraphModel graph,
        List<PlannedPartition> partitions,
        LogicalMemoryPlan memory,
        PublicationPlan publication,
        CompileConstantPlan constants,
        CompileDiagnostics diagnostics,
        DerivativeGraphMetadata derivatives) {}
```

`CompileArtifacts` validates components in declaration order, requires
`derivatives.graph() == graph`, validates exact phase/order consistency, and retains all existing
[partition](../../../../glossary.md#partition), memory, [publication](../../../../glossary.md#term-publication), constant, and diagnostic checks. `FORWARD_ONLY` requires no
gradient roles and order zero for every [node](../../../../glossary.md#node). Backward-capable modes require at least one
gradient role and allow order one and optional order two.

The package-private complete compile overload still performs graph-stage [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) exactly
once and [planning](../../../../glossary.md#term-planning) exactly once. It does not expose the request through `CompileArtifacts` and
does not retain providers, availability snapshots, selected devices, [kernels](../../../../glossary.md#kernel), [buffers](../../../../glossary.md#term-buffer),
preparation, [runtime](../../../../glossary.md#term-runtime), or execution state.

### Planned [API](../../../../glossary.md#term-application-programming-interface-api) examples

These examples are [compiler](../../../../glossary.md#term-compiler)-package integration sketches for the API implemented by this task.
`GraphCompiler` remains package-private, so they do not claim a new public [compile](../../../../glossary.md#compile) [facade](../../../../glossary.md#term-facade).

First-order [vector-Jacobian product](../../../../glossary.md#term-vector-jacobian-product-vjp) in one compile call:

```java
Tensor x = tensor(Shape.of(3), true);
Tensor output = x.mul(x);
Tensor seed = tensor(Shape.of(3), false);

FunctionalGradientRequest gradients = new FunctionalGradientRequest(List.of(
        new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.ForwardTensorReference(output)),
                List.of(Optional.of(seed)),
                List.of(x),
                false,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR)));

CompileArtifacts artifacts = GraphCompiler.compile(
        CompileMode.FORWARD_AND_BACKWARD,
        List.of(output),
        Optional.of(gradients),
        CompileTimeConstantGraph.Ingress.empty(),
        GraphOptimizationConfig.standard(),
        backendIntent,
        PartitionScoringConfig.neutral(),
        providers,
        snapshots);
```

The one first-order role maps `x.id()` to the final value for `seed * 2 * x`. No [Tensor](../../../../glossary.md#tensor) field is
written.

[Second derivative](../../../../glossary.md#term-second-derivative) of a [scalar](../../../../glossary.md#term-scalar) expression in the same one compile call:

```java
Tensor x = tensor(Shape.scalar(), true);
Tensor loss = x.mul(x).sum();

FunctionalGradientRequest gradients = new FunctionalGradientRequest(List.of(
        new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.ForwardTensorReference(loss)),
                List.of(Optional.empty()),
                List.of(x),
                true,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR),
        new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.FirstStageGradientReference(0)),
                List.of(Optional.empty()),
                List.of(x),
                false,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR)));

CompileArtifacts artifacts = GraphCompiler.compile(
        CompileMode.FORWARD_AND_BACKWARD,
        List.of(loss),
        Optional.of(gradients),
        CompileTimeConstantGraph.Ingress.empty(),
        GraphOptimizationConfig.standard(),
        backendIntent,
        PartitionScoringConfig.neutral(),
        providers,
        snapshots);
```

The ordered roles contain [derivative](../../../../glossary.md#term-derivative) order one for `2*x` and derivative order two for `2`.
Both are ordinary [graph](../../../../glossary.md#graph) values in the same combined [immutable](../../../../glossary.md#term-immutable) graph. The example creates no
[mutable](../../../../glossary.md#term-mutable) `x.gradient`, calls no `backward()`, and executes nothing.

## Out of [scope](../../../../glossary.md#term-scope)

- `Tensor.gradient`, `Tensor.backward`, [mutable](../../../../glossary.md#term-mutable) [Tensor](../../../../glossary.md#tensor) [gradient](../../../../glossary.md#term-gradient) state, or a Tensor recording flag
- an eager tape, thread-local gradient scope, public gradient registry, service loader, or
  extension SPI<span data-glossary-navigation="true"> ([eager glossary](../../../../glossary.md#term-eager-evaluation); [registry glossary](../../../../glossary.md#term-registry); [SPI glossary](../../../../glossary.md#term-service-provider-interface-spi))</span>
- Model-owned derivative rules, new Model operation kinds, backward-only kinds, a second
  derivative algebra, or Model/<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [derivative glossary](../../../../glossary.md#term-derivative); [operation glossary](../../../../glossary.md#operation); [second derivative glossary](../../../../glossary.md#term-second-derivative))</span>`GraphPhase` changes
- more than two reverse-mode stages, unlimited higher order, forward-mode differentiation,
  Jacobian materialization, or a persistent derivative chain<span data-glossary-navigation="true"> ([reverse-mode glossary](../../../../glossary.md#primal-adjoint-and-differentiation-seed); [forward-mode glossary](../../../../glossary.md#primal-adjoint-and-differentiation-seed); [Jacobian glossary](../../../../glossary.md#term-jacobian); [materialization glossary](../../../../glossary.md#materialization))</span>
- another public [compile](../../../../glossary.md#compile) [facade](../../../../glossary.md#term-facade), public `GraphCompiler`, [engine](../../../../glossary.md#term-engine) `CompiledGraph`, or
  `CompileConfig` aggregate
- public runtime gradient delivery, optimizer updates, optimizer/training-session work, parameter
  mutation, or training-loop behavior<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [optimizer glossary](../../../../glossary.md#term-optimizer); [parameter glossary](../../../../glossary.md#term-parameter))</span>
- concrete dimension binding, prepare, runtime, backend, schedule, execution, storage, physical
  saved values, or buffer work<span data-glossary-navigation="true"> ([dimension glossary](../../../../glossary.md#dimension); [binding glossary](../../../../glossary.md#term-binding); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [buffer glossary](../../../../glossary.md#term-buffer))</span>
- new algebraic rewrite, backward-only optimizer, broader CSE across derivative orders, numerical
  evaluation, or host-storage constant discovery<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [evaluation glossary](../../../../glossary.md#term-evaluation); [host glossary](../../../../glossary.md#term-host); [constant glossary](../../../../glossary.md#term-constant))</span>
- dependency, Gradle, architecture, ADR, architecture-test, backend-conformance, integration, or
  other-module changes<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [conformance glossary](../../../../glossary.md#term-conformance); [module glossary](../../../../glossary.md#term-module))</span>
- a later [Compiler](../../../../glossary.md#term-compiler) task specification or out-of-order [planning](../../../../glossary.md#term-planning)

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- [Training graph](../../../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../../../glossary.md#training-graph))</span>
- [Module boundaries](../../../../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../../../../glossary.md#term-module))</span>
- [Dependency rules](../../../../architecture/dependency-rules.md)<span data-glossary-navigation="true"> ([Dependency glossary](../../../../glossary.md#term-dependency))</span>
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr))</span>
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)<span data-glossary-navigation="true"> ([Planning glossary](../../../../glossary.md#term-planning))</span>
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler))</span>
- [Compiler 0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Compiler 0005](0005-publication-planning-orchestration-and-compile-artifacts.md)
- [Compiler 0005A](0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
- [Compiler 0005B](0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [Compiler 0005C](0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
- [Compiler 0005D](0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
- [Compiler 0005E](0005e-first-order-gradient-coverage-closure-checkpoint.md)
- [Compile API](../../../../api/compile-api.md)<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile); [API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- [Tensor API](../../../../api/tensor-api.md)<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor))</span>
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative.
- Compiler owns functional gradient requests, preflight, reverse accumulation, formula dispatch,
  derivative-order metadata, combined capture, result roles, and compile artifacts.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [preflight glossary](../../../../glossary.md#term-preflight); [dispatch glossary](../../../../glossary.md#term-dispatch); [metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings); [compile artifacts glossary](../../../../glossary.md#compile-artifacts))</span>
- Tensor remains immutable in identity, descriptor, and provenance and gains no gradient or
  backward lifecycle state.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable); [descriptor glossary](../../../../glossary.md#term-descriptor); [provenance glossary](../../../../glossary.md#provenance); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Generated derivatives remain ordinary public Tensor expressions governed by the same
  inference, validation, numerical, and exact optimization contracts as forward expressions.<span data-glossary-navigation="true"> ([inference glossary](../../../../glossary.md#term-inference); [validation glossary](../../../../glossary.md#term-validation))</span>
- One [compile](../../../../glossary.md#compile) request captures forward and all requested [derivative](../../../../glossary.md#term-derivative) roots together once.
- `GraphPhase` remains Model-owned and unchanged; Compiler derivative-order metadata is a
  sidecar adjacent to the final graph.<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [graph glossary](../../../../glossary.md#graph))</span>
- Exact [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-reference-identity) is compile-local. [Artifacts](../../../../glossary.md#term-artifact) retain only [`TensorId`](../../../../glossary.md#tensorid) and [`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot).
- Compile artifacts remain immutable logical recipes and contain no physical, prepared, runtime,
  backend-executable, or mutable state.<span data-glossary-navigation="true"> ([runtime glossary](../../../../glossary.md#term-runtime); [backend glossary](../../../../glossary.md#backend); [mutable glossary](../../../../glossary.md#term-mutable))</span>
- [Compiler](../../../../glossary.md#term-compiler) keeps its existing allowed dependencies and no [dependency direction](../../../../glossary.md#java-platform-and-build-tools) changes.
- If implementation requires a Model change, another public compile facade, a formula outside
  the closed 0005E matrix, a third derivative stage, runtime state, or an architecture change,
  stop and request clarification.<span data-glossary-navigation="true"> ([facade glossary](../../../../glossary.md#term-facade); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>

## Package impact

Existing package used and changed:

- `io.github.pho001.synaptik.compiler` — continues to own the cohesive compiler front end and its
  narrow public request/artifact value contracts.<span data-glossary-navigation="true"> ([artifact glossary](../../../../glossary.md#term-artifact))</span>

No package is added.

Type placement:

- `io.github.pho001.synaptik.compiler.FunctionalGradientRequest` — public immutable functional
  request and its bounded nested stage/reference/policy vocabulary.<span data-glossary-navigation="true"> ([immutable glossary](../../../../glossary.md#term-immutable))</span>
- `io.github.pho001.synaptik.compiler.GradientPublicationBinding` — public immutable
  derivative-stage target-to-gradient publication binding.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [target glossary](../../../../glossary.md#term-target); [gradient publication binding glossary](../../../../glossary.md#gradient-publication-binding))</span>
- `io.github.pho001.synaptik.compiler.DerivativeGraphMetadata` — public immutable graph-adjacent
  derivative-order sidecar.<span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>
- `io.github.pho001.synaptik.model.graph.ForwardPublicationBinding` — the renamed public immutable
  requested-forward identity binding; its two-component semantics are unchanged.<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings); [binding glossary](../../../../glossary.md#term-binding))</span>
- Existing package-private `GraphCompiler`, `GraphCompilation`, `AutogradPreflight`,
  `FirstOrderAutograd`, [capture](../../../../glossary.md#term-capture-meanings), [validation](../../../../glossary.md#term-validation), and optimization types remain implementation [owners](../../../../glossary.md#term-owner).

The three public types are [compiler](../../../../glossary.md#term-compiler) contracts justified by this concrete current consumer. They
do not form a [facade](../../../../glossary.md#term-facade), [registry](../../../../glossary.md#term-registry), or new package layer.

## Affected files

Production:

- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FunctionalGradientRequest.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GradientPublicationBinding.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/DerivativeGraphMetadata.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/AutogradPreflight.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderAutograd.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverage.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCapture.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CapturedGraphInference.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ValidatedGraph.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCanonicalization.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardExactArithmeticRewriting.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardConstantFolding.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimization.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardCommonSubexpressionElimination.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/ForwardDeadCodeElimination.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompilation.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/GraphCompiler.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/PublicationPlan.java`
- `modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java`

Tests:

- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FunctionalGradientRequestTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AutogradPreflightTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderAutogradTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCaptureTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/CapturedGraphInferenceTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCanonicalizationTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardExactArithmeticRewritingTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardConstantFoldingTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ForwardGraphOptimizationTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilationTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GraphCompilerTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/PublicationPlanTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/CompileArtifactsTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/FirstOrderGradientCoverageTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/OrderingStochasticGradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/ConvolutionAndPoolingGradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/GradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/LayoutWindowGradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/LossGradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/AttentionGradientRulesTest.java`
- `modules/compiler/src/test/java/io/github/pho001/synaptik/compiler/IndexingScatterGradientRulesTest.java`

Documentation/[planning](../../../../glossary.md#term-planning):

- `ARCHITECTURE.md`
- `docs/api/compile-api.md`
- `docs/api/tensor-api.md`
- `docs/api/training-api.md`
- `docs/api/public-api.md`
- `docs/glossary.md`
- `docs/planning/modules/compiler/tasks/0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md`
- `docs/planning/modules/compiler/master-plan.md`
- `docs/planning/roadmap.md`

Approved terminology-correction review:

- `modules/model/src/main/java/io/github/pho001/synaptik/model/graph/PublicationBinding.java`
  renamed to `ForwardPublicationBinding.java`
- `modules/model/src/test/java/io/github/pho001/synaptik/model/graph/PublicationBindingTest.java`
  renamed to `ForwardPublicationBindingTest.java`
- `docs/planning/modules/model/model-capability-contract-closure-audit.md` updates its live
  current-contract source link while preserving the historical audit conclusion
- `docs/architecture/partition-scoring.md`,
  `docs/design/notes/memory-planning-strategy.md`, and
  `docs/user-guide/backend-selection.md` correct only current negative-boundary type names

Review-only unless a contradiction requires stopping: `ARCHITECTURE.md`, focused architecture and
[ADR](../../../../glossary.md#term-architecture-decision-record-adr) 0009; Public/[Runtime](../../../../glossary.md#term-runtime) APIs; [Model](../../../../glossary.md#term-model) capabilities/master/tasks/source/tests; completed [Compiler](../../../../glossary.md#term-compiler)
tasks and family-rule source/tests; compiler package [Javadoc](../../../../glossary.md#term-javadoc); Config, Planning, [Trace](../../../../glossary.md#trace), Runtime,
[Prepare](../../../../glossary.md#prepare), [backends](../../../../glossary.md#backend), [Engine](../../../../glossary.md#term-engine), [NN](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context), training, tools; architecture/backend-[conformance](../../../../glossary.md#term-conformance)/integration
tests; and [Gradle](../../../../glossary.md#term-gradle)/build files.

## Maximum [scope](../../../../glossary.md#term-scope)

The original ceiling was 48 paths. The implementation used 45 paths before the approved final
terminology correction. The final exact 54-path scope contains 20 production paths, 21 test paths,
and 13 documentation/[planning](../../../../glossary.md#term-planning) paths. Renaming the tracked [Model](../../../../glossary.md#term-model) production/test pair accounts for
four delete/create paths; the required architecture identifier and current-contract
documentation account for the remaining approved expansion. This correction changes names and
documentation only; it does not authorize another semantic, [validation](../../../../glossary.md#term-validation), visibility, [ownership](../../../../glossary.md#term-ownership-meanings),
[capture](../../../../glossary.md#term-capture-meanings), optimization, [publication](../../../../glossary.md#term-publication), or ID behavior change.

The limit is intentionally larger than a normal task because [derivative](../../../../glossary.md#term-derivative)-order [metadata](../../../../glossary.md#term-metadata) must
remain synchronized through every existing [immutable](../../../../glossary.md#term-immutable) [graph](../../../../glossary.md#graph) rebuild and because the public
[artifact](../../../../glossary.md#term-artifact) validation changes atomically with request/results. Do not split those consistency
changes into a later cleanup.

If another formula-family source/test, Model behavior, public [facade](../../../../glossary.md#term-facade), [dependency](../../../../glossary.md#term-dependency)/build change, or
third-stage mechanism is required, stop and report the exact gap. Do not use the approved
terminology expansion to hide a formula or architecture problem.

A separate documentation-focused clean context finalizes authorized Javadocs, [API](../../../../glossary.md#term-application-programming-interface-api) references,
glossary impact, examples, planning evidence, and status. It reuses successful Java evidence
unless it changes executable Java behavior or records a concrete reason.

## Acceptance criteria

### Request and [preflight](../../../../glossary.md#term-preflight)

- The exact public request and nested types above [compile](../../../../glossary.md#compile) with complete meaningful [Javadoc](../../../../glossary.md#term-javadoc).
- Records snapshot every list and retain exact elements; all null, empty, size, uniqueness,
  reference-index, stage-count, and <span data-glossary-navigation="true"> ([snapshot glossary](../../../../glossary.md#term-snapshot); [elements glossary](../../../../glossary.md#term-element))</span>`createGraph` failures are deterministic.
- The one-stage and two-stage [matrices](../../../../glossary.md#term-matrix-meanings) are exhaustive and no third stage is representable.
- Stage-one forward references, stage-two exact gradient references, seeds, targets, and
  disconnected policies satisfy the exact contracts and failure order above.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [targets glossary](../../../../glossary.md#term-target))</span>
- Default seeds are accepted only for exact scalar floating outputs. Explicit seeds require exact
  floating Shape/type equality and <span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [Shape glossary](../../../../glossary.md#shape))</span>`requiresGrad == false`.
- Invalid first-stage facts consume no derivative Tensor ID. Invalid second-stage facts consume
  no stage-two Tensor ID and return no partial artifact.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [Tensor glossary](../../../../glossary.md#tensor); [artifact glossary](../../../../glossary.md#term-artifact))</span>
- `FirstOrderGradientCoverage` remains the sole D/[ND](../../../../glossary.md#term-nd)/FC and formula-[owner](../../../../glossary.md#term-owner) source for both stages.

### [Gradient](../../../../glossary.md#term-gradient) construction and results

- One-stage scalar, explicit vector-Jacobian, multi-output, mixed-floating target, repeated
  consumer, shared producer, target-as-output, and target-in-both-stages cases are covered.<span data-glossary-navigation="true"> ([scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [Jacobian glossary](../../../../glossary.md#term-jacobian); [producer glossary](../../../../glossary.md#term-producer))</span>
- Two-stage scalar second derivative, Hessian-vector product, constant/disconnected first
  derivative, shared stage-one result, and unsupported second-pass cases are covered.<span data-glossary-navigation="true"> ([second derivative glossary](../../../../glossary.md#term-second-derivative); [Hessian-vector product glossary](../../../../glossary.md#term-hessian-vector-product-hvp); [constant glossary](../../../../glossary.md#term-constant); [derivative glossary](../../../../glossary.md#term-derivative))</span>
- `ERROR` rejects the first disconnected [target](../../../../glossary.md#term-target); `ZERO` returns exact target-Shape/type zero
  through the existing explicit splat sidecar.<span data-glossary-navigation="true"> ([Shape glossary](../../../../glossary.md#shape))</span>
- Contribution and result ordering is deterministic and exactly output order, then existing
  reverse traversal, then target order.
- `GradientPublicationBinding` lists are ordered by derivative order then stage-local target
  index and preserve target-ID/result-value bindings even when values repeat.<span data-glossary-navigation="true"> ([bindings glossary](../../../../glossary.md#term-binding))</span>
- No [Tensor](../../../../glossary.md#tensor)/request/[provenance](../../../../glossary.md#provenance) reference survives in [graph](../../../../glossary.md#graph)-stage or public [artifacts](../../../../glossary.md#term-artifact).

### [Graph](../../../../glossary.md#graph), [metadata](../../../../glossary.md#term-metadata), optimization, and [artifacts](../../../../glossary.md#term-artifact)

- Forward outputs and every first-/second-stage result are captured together exactly once.
- Shared multi-output producers, canonical auxiliaries, saved edges, repeated positions,
  constants, graph IDs, and publication facts remain correct.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([canonical glossary](../../../../glossary.md#term-canonical-form); [constants glossary](../../../../glossary.md#term-constant); [publication glossary](../../../../glossary.md#term-publication))</span>
- `DerivativeGraphMetadata` covers every final node exactly once with order zero, one, or two and
  exact phase consistency.<span data-glossary-navigation="true"> ([node glossary](../../../../glossary.md#node))</span>
- Every [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), rewrite/fold, [DCE](../../../../glossary.md#term-dead-code-elimination-dce), and [CSE](../../../../glossary.md#term-common-subexpression-elimination-cse) rebuild remaps [derivative](../../../../glossary.md#term-derivative) metadata.
- CSE remains exact, phase-local, and derivative-order-local; no rewrite rule is added or
  broadened.
- The final combined graph receives the established optimization pipeline once and each changed
  candidate is revalidated.
- `GraphCompilation`, `PublicationPlan`, and eight-component `CompileArtifacts` cross-validate
  result order, boundary de-duplication, graph identity, phase/order, planning, memory,
  constants, and diagnostics. <span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-identity-meanings))</span>`PublicationPlan` exposes only `forwardBindings()` and
  `gradientBindings()`.
- The complete package-private [compile](../../../../glossary.md#compile) entry performs graph [compilation](../../../../glossary.md#compiler-engine-and-execution-stage-names) once and [Planning](../../../../glossary.md#term-planning) once.

### Boundaries, documentation, and completion

- No Tensor gradient/backward state, public compile facade, registry, second algebra, new Model
  kind, Model/GraphPhase change, third derivative stage, runtime tape, optimizer/training loop,
  prepare/runtime/backend behavior, dependency, Gradle, or architecture change is present.<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [gradient glossary](../../../../glossary.md#term-gradient); [compile glossary](../../../../glossary.md#compile); [facade glossary](../../../../glossary.md#term-facade); [registry glossary](../../../../glossary.md#term-registry); [Model glossary](../../../../glossary.md#term-model); [derivative glossary](../../../../glossary.md#term-derivative); [runtime glossary](../../../../glossary.md#term-runtime); [optimizer glossary](../../../../glossary.md#term-optimizer); [prepare glossary](../../../../glossary.md#prepare); [backend glossary](../../../../glossary.md#backend); [dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle))</span>
- The separate documentation-focused clean context finalizes public and implementation Javadocs,
  Compile/Tensor/Training API status, glossary terms, both conceptual examples, task evidence,
  master plan, and roadmap.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Compile API explains functional request/reference/result shapes, seed and disconnected
  behavior, order metadata, one capture, and current package-private integration.<span data-glossary-navigation="true"> ([shapes glossary](../../../../glossary.md#shape); [seed glossary](../../../../glossary.md#term-seed); [metadata glossary](../../../../glossary.md#term-metadata); [capture glossary](../../../../glossary.md#term-capture-meanings))</span>
- Tensor API states that derivatives are functional compiler results and Tensor remains free of
  gradient lifecycle state. Training API distinguishes this compiler capability from planned
  optimizer/session/runtime work.<span data-glossary-navigation="true"> ([compiler glossary](../../../../glossary.md#term-compiler); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>
- Reasoned no-change conclusions are recorded for architecture/ADR/tests, Model, Config,
  Planning, Trace, Runtime, Prepare, backends, Engine, NN/training implementation, conformance,
  integration, Gradle, and unrelated APIs.<span data-glossary-navigation="true"> ([ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [Planning glossary](../../../../glossary.md#term-planning); [Trace glossary](../../../../glossary.md#trace); [Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [conformance glossary](../../../../glossary.md#term-conformance))</span>
- Exactly the authorized paths change, no later Compiler task specification exists, links and
  anchors resolve, headings are unique, fences are balanced, final newlines and whitespace are
  valid, statuses synchronize, and `git diff --check` passes.

## Tests / [validation](../../../../glossary.md#term-validation)

Focused executable validation:

```text
./gradlew :modules:compiler:test \
  --tests io.github.pho001.synaptik.compiler.FunctionalGradientRequestTest \
  --tests io.github.pho001.synaptik.compiler.AutogradPreflightTest \
  --tests io.github.pho001.synaptik.compiler.FirstOrderAutogradTest \
  --tests io.github.pho001.synaptik.compiler.GraphCaptureTest \
  --tests io.github.pho001.synaptik.compiler.CapturedGraphInferenceTest \
  --tests io.github.pho001.synaptik.compiler.GraphCanonicalizationTest \
  --tests io.github.pho001.synaptik.compiler.ForwardExactArithmeticRewritingTest \
  --tests io.github.pho001.synaptik.compiler.ForwardConstantFoldingTest \
  --tests io.github.pho001.synaptik.compiler.ForwardGraphOptimizationTest \
  --tests io.github.pho001.synaptik.compiler.GraphCompilationTest \
  --tests io.github.pho001.synaptik.compiler.GraphCompilerTest \
  --tests io.github.pho001.synaptik.compiler.PublicationPlanTest \
  --tests io.github.pho001.synaptik.compiler.CompileArtifactsTest
```

After executable Java stabilizes, [run](../../../../glossary.md#term-run) the affected [module](../../../../glossary.md#term-module) once:

```text
./gradlew :modules:compiler:test
```

This task closes the explicit functional/higher-order [compiler](../../../../glossary.md#term-compiler) capability and changes a public
[compile](../../../../glossary.md#compile)-[artifact](../../../../glossary.md#term-artifact) [shape](../../../../glossary.md#shape). Run the capability [checkpoint](../../../../glossary.md#term-checkpoint):

```text
./gradlew test :testing:architecture-tests:test
```

The documentation-focused pass reuses successful Java evidence unless executable Java changes.
After final Javadocs and documentation:

```text
./gradlew :modules:model:javadoc :modules:compiler:javadoc
python3 /tmp/validate_synaptik_markdown.py
git diff --check
```

If `/tmp/validate_synaptik_markdown.py` is absent, create an equivalent temporary validator
outside the repository. It must validate every changed Markdown link [target](../../../../glossary.md#term-target) and heading anchor,
unique headings, balanced fences, final newlines, and trailing whitespace.

Required manual/source evidence:

- exact public `javap`/reflection surface and constructor validation for the final public
  request, forward/gradient binding, derivative metadata, and eight-component<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [binding glossary](../../../../glossary.md#term-binding); [derivative glossary](../../../../glossary.md#term-derivative); [metadata glossary](../../../../glossary.md#term-metadata))</span>
  `CompileArtifacts` contracts;
- no public `GraphCompiler`, extra compile overload, Tensor gradient/backward member, registry,
  service loader, second algebra, or third-stage path;<span data-glossary-navigation="true"> ([Tensor glossary](../../../../glossary.md#tensor); [registry glossary](../../../../glossary.md#term-registry))</span>
- exact-object identity membership, list snapshots, no-allocation failure deltas, seed
  Shape/DataType/eligibility matrix, and ERROR/ZERO behavior;<span data-glossary-navigation="true"> ([identity glossary](../../../../glossary.md#term-reference-identity))</span><span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [DataType glossary](../../../../glossary.md#data-type--datatype); [matrix glossary](../../../../glossary.md#term-matrix-meanings))</span>
- D/ND/FC coverage and formula-owner equality for both passes, including the exact unsupported
  second-pass reason;<span data-glossary-navigation="true"> ([ND glossary](../../../../glossary.md#term-nd); [owner glossary](../../../../glossary.md#term-owner))</span>
- one combined capture invocation, graph boundary order/de-duplication, shared producer and
  auxiliary identity, constant sidecar, and publication inspection;<span data-glossary-navigation="true"> ([capture glossary](../../../../glossary.md#term-capture-meanings); [graph glossary](../../../../glossary.md#graph); [producer glossary](../../../../glossary.md#term-producer); [constant glossary](../../../../glossary.md#term-constant); [publication glossary](../../../../glossary.md#term-publication))</span>
- derivative-order coverage/remapping through canonicalization, rewrite/fold, DCE, CSE, and final
  artifact validation;<span data-glossary-navigation="true"> ([canonicalization glossary](../../../../glossary.md#occurrence-identity-and-canonical-form); [DCE glossary](../../../../glossary.md#term-dead-code-elimination-dce); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse))</span>
- original 48-path ceiling, recorded 45-path pre-correction scope, approved final
  terminology-only Model rename and documentation expansion, no executable Java outside
  Compiler, no Gradle/dependency/other-module behavior change, and no later task specification;<span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope); [Model glossary](../../../../glossary.md#term-model); [Gradle glossary](../../../../glossary.md#term-gradle); [dependency glossary](../../../../glossary.md#term-dependency))</span>
- task/master/roadmap Ready/In-progress synchronization before implementation and Complete
  synchronization only after all implementation evidence passes; and<span data-glossary-navigation="true"> ([Ready glossary](../../../../glossary.md#term-ready); [synchronization glossary](../../../../glossary.md#term-synchronization))</span>
- General/API-Javadoc/Planning/Example profile review, terminology, examples, links, anchors,
  headings, fences, final newlines, whitespace, and <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning))</span>`git diff --check`.

Do not run [backend](../../../../glossary.md#backend)-[conformance](../../../../glossary.md#term-conformance) or integration suites separately. There is no backend or
end-to-end execution change, and the root plus architecture checkpoint is the proportional
cross-repository gate. Do not repeat successful Java commands in the documentation context
without changed executable behavior or a recorded concrete risk.

## Dependencies

- [Compiler](../../../../glossary.md#term-compiler) 0001–0003B [capture](../../../../glossary.md#term-capture-meanings), [validation](../../../../glossary.md#term-validation), [canonicalization](../../../../glossary.md#occurrence-identity-and-canonical-form), and exact optimization — Complete.
- Compiler 0004–0004B pre-capture first-order [autograd](../../../../glossary.md#autograd) and shared [Tensor](../../../../glossary.md#tensor) algebra — Complete.
- Compiler 0005 [compile artifacts](../../../../glossary.md#compile-artifacts) and [publication](../../../../glossary.md#term-publication)/[planning](../../../../glossary.md#term-planning) orchestration — Complete.
- Compiler 0005A–0005D complete current first-order formulas and policies — Complete.
- Compiler 0005E source-backed 37-family/107-kind/128-signature D/ND/FC and transitive formula
  closure checkpoint — Complete.<span data-glossary-navigation="true"> ([signature glossary](../../../../glossary.md#term-signature); [ND glossary](../../../../glossary.md#term-nd); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Stable current public Tensor, Tensor [producer](../../../../glossary.md#term-producer)/[canonical](../../../../glossary.md#term-canonical-form) output, [Shape](../../../../glossary.md#shape), [DataType](../../../../glossary.md#data-type--datatype), [graph](../../../../glossary.md#graph) ID,
  `GraphPhase`, [compile](../../../../glossary.md#compile)-mode, optimization, publication, Planning, and compile-[artifact](../../../../glossary.md#term-artifact) contracts.

## Follow-up tasks

- No later [Compiler](../../../../glossary.md#term-compiler) task is specified. Reassess the roadmap after this task is Complete.
- A future Engine task may wrap the public functional request in its public compile lifecycle
  without making <span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [compile glossary](../../../../glossary.md#compile); [lifecycle glossary](../../../../glossary.md#lifecycle))</span>`GraphCompiler` public.
- A future focused task may add another derivative order only after an explicit architecture and
  capability decision; it must not infer permission from this bounded two-stage contract.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative))</span>

Do not create a follow-up specification in this task.

## Architecture impact

Expected impact: None.

The architecture already requires [Compiler](../../../../glossary.md#term-compiler)-owned [pre-capture Tensor-expression autograd](../../../../glossary.md#pre-capture-tensor-expression-autograd), one
combined [graph](../../../../glossary.md#graph), [immutable](../../../../glossary.md#term-immutable) [compile artifacts](../../../../glossary.md#compile-artifacts), no [Tensor](../../../../glossary.md#tensor) [gradient](../../../../glossary.md#term-gradient) [lifecycle](../../../../glossary.md#lifecycle) state, and an explicit
future create-graph/[derivative](../../../../glossary.md#term-derivative)-order contract. This task supplies that bounded compiler contract
without moving [ownership](../../../../glossary.md#term-ownership-meanings), changing dependencies, or altering [Model](../../../../glossary.md#term-model) `GraphPhase`.

If implementation requires changing `ARCHITECTURE.md`, Model, a [dependency](../../../../glossary.md#term-dependency) rule, the public
[compile](../../../../glossary.md#compile) [facade](../../../../glossary.md#term-facade) [owner](../../../../glossary.md#term-owner), [runtime](../../../../glossary.md#term-runtime) state, [backend](../../../../glossary.md#backend) behavior, or the two-stage limit, stop and report the
exact conflict.

## Implementation prompt

Use this prompt in a separate agentic task/thread:

```text
You are working in the Synaptik repository. Do not commit or push.

Read AGENTS.md, ARCHITECTURE.md, docs/planning/planning-guide.md, focused architecture and ADR 0009,
documentation rules/profiles, the Compiler master plan, completed Compiler 0004–0005E contracts,
and docs/planning/modules/compiler/tasks/0006-explicit-functional-gradient-requests-and-higher-order-differentiation.md.
Read every affected/review-only source, test, API, and glossary section named by the task.

Implement Compiler 0006 exactly within its 48-path authorized ceiling. Add the exact public functional
request, result role, and derivative-order sidecar; generalize the package-private GraphCompiler
integration to one or two checked reverse-mode stages; preserve one combined capture and the
established exact optimization/publication/planning pipeline. Do not add Tensor gradient/backward
state, another public compile facade, a registry or second algebra, a Model/GraphPhase change,
another derivative stage, runtime tape, optimizer/training-loop, prepare/runtime/backend,
dependency, Gradle, or architecture work. Stop on any formula, policy, package, scope, or
architecture conflict.

Run the focused tests, one final Compiler module test, the root/architecture capability
checkpoint, and required source/surface checks. Then hand the actual diff and successful Java
evidence to a separate documentation-focused clean context. That pass must follow
documentation-rules.md; finalize authorized Javadocs, Compile/Tensor/Training APIs, glossary,
examples, task evidence/summary, master plan, roadmap, links, status, scope, and reasoned
no-change conclusions; and not repeat Java tests unless executable behavior changes or a concrete
risk is recorded.

Return both context IDs, exact paths, commands/results/counts, identity/seed/disconnected and
two-stage evidence, derivative-order/one-capture/artifact evidence, no-change conclusions,
unresolved issues, required follow-up, and the repository completion status format. Mark Complete
only after every acceptance criterion and documentation gate passes.
```

## Local decisions

- The public request owns a nested bounded vocabulary so the functional capability is explicit
  without creating a broad compiler API package or many top-level policy/reference types.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>
- Stage-two references use stage-one target indices because request target order is stable before
  generated Tensor and graph identities exist. They do not expose temporary generated Tensors or
  captured <span data-glossary-navigation="true"> ([identities glossary](../../../../glossary.md#term-identity-meanings))</span><span data-glossary-navigation="true"> ([graph glossary](../../../../glossary.md#graph))</span>[`ValueId`](../../../../glossary.md#logical-valueid-versus-physical-memory-slot) values.
- Targets must belong to the complete forward request, while disconnected policy is evaluated
  against selected stage outputs. This distinguishes an invalid target from a valid disconnected
  target and makes ZERO useful for multi-output requests and constant first derivatives.<span data-glossary-navigation="true"> ([constant glossary](../../../../glossary.md#term-constant))</span>
- Explicit seeds must not request gradients. Independent seed producers belong to their
  derivative stage, while provenance-free seeds remain graph inputs; neither creates a third
  publication category or becomes a differentiation target in this task.<span data-glossary-navigation="true"> ([producers glossary](../../../../glossary.md#term-producer))</span><span data-glossary-navigation="true"> ([seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [gradients glossary](../../../../glossary.md#term-gradient); [seed glossary](../../../../glossary.md#term-seed); [provenance glossary](../../../../glossary.md#provenance); [publication glossary](../../../../glossary.md#term-publication); [differentiation target glossary](../../../../glossary.md#functional-gradient-request))</span>
- The existing `FirstOrderAutograd` primitive is reused once per stage. A separate
  second-order engine would duplicate dispatch and imply a second algebra.<span data-glossary-navigation="true"> ([engine glossary](../../../../glossary.md#term-engine); [dispatch glossary](../../../../glossary.md#term-dispatch))</span>
- Exact [Tensor](../../../../glossary.md#tensor) [identity](../../../../glossary.md#term-reference-identity) is not retained after [capture](../../../../glossary.md#term-capture-meanings). The public
  `GradientPublicationBinding` records the exact [target](../../../../glossary.md#term-target)'s stable [`TensorId`](../../../../glossary.md#tensorid), final `ValueId`,
  derivative order, and target position.
- [Derivative](../../../../glossary.md#term-derivative) order is [compiler](../../../../glossary.md#term-compiler)-owned sidecar state. [Model](../../../../glossary.md#term-model) `GraphPhase.BACKWARD` continues to
  classify both orders, and exact CSE adds order equality to its existing phase equality guard.<span data-glossary-navigation="true"> ([CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [guard glossary](../../../../glossary.md#term-guard))</span>
- `CompileArtifacts` gains the derivative sidecar atomically because downstream consumers need
  the final graph and role/order interpretation to agree.
- No architecture, dependency, Model, Gradle, backend, runtime, preparation, execution, or
  training implementation change is planned.<span data-glossary-navigation="true"> ([dependency glossary](../../../../glossary.md#term-dependency); [Gradle glossary](../../../../glossary.md#term-gradle); [backend glossary](../../../../glossary.md#backend); [runtime glossary](../../../../glossary.md#term-runtime))</span>

## Known limitations

- Exactly two [reverse-mode](../../../../glossary.md#primal-adjoint-and-differentiation-seed) stages are supported. There is no unlimited higher-order chain.
- Stage two can differentiate only exact stage-one result roles, not arbitrary generated
  intermediates.
- [Cotangent](../../../../glossary.md#cotangent) [seeds](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings) are not differentiation [targets](../../../../glossary.md#term-target) in this task.
- One-output attention and positive-static-depth index-loss restrictions remain as recorded by
  Compiler 0005D/0005E.<span data-glossary-navigation="true"> ([loss glossary](../../../../glossary.md#loss); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
- Any [occurrence](../../../../glossary.md#term-occurrence)-dependent row that remains `FC` under the existing preflight contract remains
  unsupported in either stage.<span data-glossary-navigation="true"> ([preflight glossary](../../../../glossary.md#term-preflight))</span>
- ZERO returns symbolic exact zeros; this task does not execute or materialize them.
- Public Engine compilation, preparation, runtime publication, optimizer/session behavior, and
  execution remain planned.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [compilation glossary](../../../../glossary.md#compiler-engine-and-execution-stage-names); [runtime glossary](../../../../glossary.md#term-runtime); [publication glossary](../../../../glossary.md#term-publication); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>

## [Validation](../../../../glossary.md#term-validation) evidence

- The implementation context's final focused terminology command
  `./gradlew :modules:model:test --tests
  io.github.pho001.synaptik.model.graph.ForwardPublicationBindingTest
  :modules:compiler:test --tests io.github.pho001.synaptik.compiler.PublicationPlanTest --tests
  io.github.pho001.synaptik.compiler.GraphCompilationTest --tests
  io.github.pho001.synaptik.compiler.GraphCompilerTest --tests
  io.github.pho001.synaptik.compiler.CompileArtifactsTest` passed 35 tests total
  (`4 + 2 + 3 + 23 + 3`) with no skips, failures, or errors.
- The implementation context's single final
  `./gradlew :modules:model:test :modules:compiler:test` passed Model 127 suites/1,031 tests and
  Compiler 31 suites/208 tests with no skips, failures, or errors. No executable Java changed
  afterward, so documentation context<span data-glossary-navigation="true"> ([Model glossary](../../../../glossary.md#term-model); [Compiler glossary](../../../../glossary.md#term-compiler))</span>
  `/root/rename_publication_bindings/docs_finalize_publication_rename` reused this evidence and
  did not repeat Java tests.
- Final Java scans found no `PublicationBinding` or `GradientResultRole` in
  `modules/**/*.java`, found no retired `PublicationPlan` accessor, and `git diff --check`
  passed before the documentation handoff.<span data-glossary-navigation="true"> ([handoff glossary](../../../../glossary.md#term-handoff))</span>
- Documentation context
  `/root/rename_publication_bindings/docs_finalize_publication_rename` applied the
  General, API/Javadoc, Planning, and Example profiles. It finalized the affected Javadocs,
  current API/glossary/architecture terminology, task/master/roadmap evidence, and the current
  Model capability-audit link while preserving historical task text where it records the
  contract name that existed when that task completed.<span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api); [Javadoc glossary](../../../../glossary.md#term-javadoc); [Planning glossary](../../../../glossary.md#term-planning))</span>
- Documentation context
  `/root/rename_publication_bindings/docs_finalize_publication_rename` ran
  `./gradlew :modules:model:javadoc :modules:compiler:javadoc` exactly once after final Javadoc
  edits; it passed (`BUILD SUCCESSFUL`, 8 actionable tasks: 3 executed and 5 up-to-date).
- `python3 /tmp/validate_synaptik_markdown.py <all 13 changed Markdown paths>` passed after
  validating local links/anchors, effective unique heading anchors, balanced backtick and tilde
  fences, final newlines, and trailing whitespace. The six headings introduced by the overall
  change and the renamed glossary/task headings each occur once.
- Live Java/source scans proved no exact `PublicationBinding` or `GradientResultRole`, no retired
  public `PublicationPlan.forwardOutputs()` or `gradientResults()`, exact
  `ForwardPublicationBinding(TensorId, ValueId)`, exact
  `GradientPublicationBinding(derivativeOrder, targetIndex, target, valueId)`, and retained
  package-private `GraphCompilation.forwardOutputs`/`gradientResults` with `valueId()` use.
- Current-contract documentation scans found no stale exact public identifier in
  `ARCHITECTURE.md`, current API/glossary, current partition/memory/backend-selection
  explanations, compiler master plan, or roadmap. Historical Model task 0009 still contains its
  original <span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [backend glossary](../../../../glossary.md#backend))</span>`PublicationBinding` name, while the current capability audit resolves to the live
  `ForwardPublicationBinding.java` path.
- Final scope/status checks proved exactly 54 paths (20 production, 21 test, 13
  documentation/planning), task/master/roadmap <span data-glossary-navigation="true"> ([scope glossary](../../../../glossary.md#term-scope))</span>`Complete`, and task 0006 as the last detailed
  Compiler specification. Final changed-Markdown validation and `git diff --check` passed after
  this evidence update.

## Implementation notes

- Added the bounded public request, result-role, and derivative-order values while retaining
  package-private compiler entry ownership and immutable Tensor semantics.<span data-glossary-navigation="true"> ([derivative glossary](../../../../glossary.md#term-derivative); [ownership glossary](../../../../glossary.md#term-ownership-meanings); [immutable glossary](../../../../glossary.md#term-immutable); [Tensor glossary](../../../../glossary.md#tensor))</span>
- Reused the checked reverse-mode primitive once per stage, captured forward and all derivative
  roots once, and propagated derivative-order metadata through every graph rebuild. CSE is local
  to both graph phase and derivative order.<span data-glossary-navigation="true"> ([reverse-mode glossary](../../../../glossary.md#primal-adjoint-and-differentiation-seed); [metadata glossary](../../../../glossary.md#term-metadata); [graph glossary](../../../../glossary.md#graph); [CSE glossary](../../../../glossary.md#term-common-subexpression-elimination-cse); [graph phase glossary](../../../../glossary.md#graph-phase))</span>
- Exact explicit/default seed rules, output-order VJP accumulation, ERROR/ZERO disconnected
  behavior, stage-local target identity, and the eight-component artifact boundary are covered by
  the final focused and module evidence.<span data-glossary-navigation="true"> ([seed glossary](../../../../glossary.md#term-seed); [VJP glossary](../../../../glossary.md#term-vector-jacobian-product-vjp); [identity glossary](../../../../glossary.md#term-identity-meanings); [artifact glossary](../../../../glossary.md#term-artifact); [module glossary](../../../../glossary.md#term-module))</span>
- Applied the approved minimal terminology correction: [Model](../../../../glossary.md#term-model)
  `PublicationBinding`/`PublicationBindingTest` became
  `ForwardPublicationBinding`/`ForwardPublicationBindingTest`; [Compiler](../../../../glossary.md#term-compiler)
  `GradientResultRole` became `GradientPublicationBinding` with exact components
  `derivativeOrder`, `targetIndex`, [`target`](../../../../glossary.md#term-target), and `valueId`; and `PublicationPlan` now exposes
  only `forwardBindings()` and `gradientBindings()`.
- No semantic, validation-order, visibility, ownership, capture, optimization, publication, or
  ID behavior changed. <span data-glossary-navigation="true"> ([validation glossary](../../../../glossary.md#term-validation); [capture glossary](../../../../glossary.md#term-capture-meanings); [publication glossary](../../../../glossary.md#term-publication))</span>`GraphCompilation` deliberately retains its package-private
  `forwardOutputs`/`gradientResults` component names while carrying
  `GradientPublicationBinding`.
- Historical Model task 0009 and earlier completed task specifications retain the names and
  paths that were true at their completion; they are evidence, not current API references. The
  current Model capability audit instead uses the live <span data-glossary-navigation="true"> ([API glossary](../../../../glossary.md#term-application-programming-interface-api))</span>`ForwardPublicationBinding` source link.
  The Model master plan needed no edit because it contains no affected current identifier.
- Compile, Public, and Tensor API references plus the glossary required exact name, component, and
  accessor corrections. Training API already described the functional request and publication
  boundary without either retired identifier, so review found no additional rename edit.<span data-glossary-navigation="true"> ([Compile glossary](../../../../glossary.md#compile))</span>
- `ARCHITECTURE.md` required only its Model allowed-list identifier. The focused partition-scoring
  explanation, memory-planning design note, and backend-selection guide required only negative-
  boundary identifier corrections. Other focused architecture pages and ADR 0009 required no
  change because ownership, lifecycle, dependency, and autograd architecture did not move.<span data-glossary-navigation="true"> ([partition glossary](../../../../glossary.md#partition); [planning glossary](../../../../glossary.md#term-planning); [backend glossary](../../../../glossary.md#backend); [ADR glossary](../../../../glossary.md#term-architecture-decision-record-adr); [lifecycle glossary](../../../../glossary.md#lifecycle); [dependency glossary](../../../../glossary.md#term-dependency); [autograd glossary](../../../../glossary.md#autograd))</span>
- Architecture tests required no change because no dependency rule changed. Model capabilities
  and master planning required no semantic update; only the capability audit's live source link
  changed, while historical task records remain historical.
- Config, Planning implementation, Trace, Runtime, Prepare, concrete backends, Engine, NN,
  training implementation, backend conformance, integration, Gradle/build configuration, and
  other modules required no change because this pass adds no behavior, dependency, execution,
  runtime publication, or backend claim.<span data-glossary-navigation="true"> ([Trace glossary](../../../../glossary.md#trace); [Runtime glossary](../../../../glossary.md#term-runtime); [Prepare glossary](../../../../glossary.md#prepare); [Engine glossary](../../../../glossary.md#term-engine); [NN glossary](../../../../glossary.md#neural-network-module-parameter-buffer-and-forward-context); [conformance glossary](../../../../glossary.md#term-conformance); [Gradle glossary](../../../../glossary.md#term-gradle))</span>

## Completion summary

- Completed changes: implemented explicit one/two-stage functional gradient requests, ordered
  first-/second-derivative results, exact seeds and disconnected policies, one combined graph,
  derivative-order metadata, synchronized artifact/publication validation, and the approved
  unambiguous forward/gradient publication-binding terminology correction.<span data-glossary-navigation="true"> ([gradient glossary](../../../../glossary.md#term-gradient); [derivative glossary](../../../../glossary.md#term-derivative); [seeds glossary](../../../../glossary.md#commit-release-distribution-log-and-seed-meanings); [graph glossary](../../../../glossary.md#graph); [metadata glossary](../../../../glossary.md#term-metadata); [artifact glossary](../../../../glossary.md#term-artifact); [publication glossary](../../../../glossary.md#term-publication); [validation glossary](../../../../glossary.md#term-validation); [binding glossary](../../../../glossary.md#term-binding))</span>
- Files changed or created: 54 exact paths—20 production paths, 21 test paths, and 13
  documentation/planning paths. The original implementation used 45 paths before the approved
  Model rename and current-contract documentation expansion.<span data-glossary-navigation="true"> ([planning glossary](../../../../glossary.md#term-planning); [Model glossary](../../../../glossary.md#term-model))</span>
- Validation: focused Compiler 0006 tests passed; full Compiler passed 31 suites/208 tests; the
  repository/architecture checkpoint passed; the separate documentation gates passed.<span data-glossary-navigation="true"> ([Compiler glossary](../../../../glossary.md#term-compiler); [checkpoint glossary](../../../../glossary.md#term-checkpoint))</span>
- Documentation: finalized affected Javadoc, Compile/Tensor/Training/Public API references,
  glossary terminology, scalar-second-derivative and vector/Hessian-product examples,
  architecture identifier, current Model capability-audit link, and task/master/roadmap status.<span data-glossary-navigation="true"> ([Javadoc glossary](../../../../glossary.md#term-javadoc); [Compile glossary](../../../../glossary.md#compile); [Tensor glossary](../../../../glossary.md#tensor); [API glossary](../../../../glossary.md#term-application-programming-interface-api); [scalar glossary](../../../../glossary.md#term-scalar); [vector glossary](../../../../glossary.md#term-vector); [Hessian glossary](../../../../glossary.md#term-hessian))</span>
- Unresolved issues: none.
- Required follow-up: none for this task; later Engine, prepare/runtime publication, optimizer,
  and any derivative order beyond two remain separately planned.<span data-glossary-navigation="true"> ([Engine glossary](../../../../glossary.md#term-engine); [prepare glossary](../../../../glossary.md#prepare); [runtime glossary](../../../../glossary.md#term-runtime); [optimizer glossary](../../../../glossary.md#term-optimizer))</span>

Status: Complete
