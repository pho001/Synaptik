# Task 0006: Explicit Functional Gradient Requests and Higher-Order Differentiation

## Status

Complete

## Goal

Replace the package-private scalar-objective-only first-order request with one compiler-owned
public immutable functional gradient request that can describe either:

- one ordered reverse-mode stage; or
- exactly two ordered reverse-mode stages, where the second stage differentiates exact gradient
  results produced by the first stage.

Compile the requested forward outputs and every requested first- and second-derivative root into
one immutable combined graph in one capture. Publish immutable result roles in stage and target
order, preserve the exact requested target identity through pre-capture validation and its
`TensorId`-to-`ValueId` projection after capture, and add compiler-owned derivative-order
metadata without changing Model `GraphPhase`.

The request is functional: it returns graph results and artifacts. It does not mutate a Tensor,
store a gradient on a Tensor, run an eager tape, or add another public compile facade.

## Scope

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

The request and every nested record snapshot list membership and retain exact immutable element
references. They use ordinary record value equality and hashing for their public value shape, but
all Tensor membership, uniqueness, ancestry, and target-route decisions during compilation use
exact object identity.

Constructor validation is exact:

- `stages` is non-null and contains exactly one or two non-null stages;
- each stage has a non-null, non-empty `outputs` list, a non-null `cotangentSeeds` list of the
  same size, and a non-null, non-empty `targets` list;
- no output reference, seed `Optional`, present seed Tensor, target, or policy is null;
- stage targets are exact-object-identity-unique within that stage;
- stage-one output references are all `ForwardTensorReference`;
- a one-stage request has `createGraph == false`;
- a two-stage request has stage one `createGraph == true`, stage two
  `createGraph == false`, and every stage-two output is a
  `FirstStageGradientReference`;
- every first-stage-gradient reference uses a non-negative target index smaller than the exact
  stage-one target-list size; and
- no other stage count, reference direction, derivative chain, or `createGraph` combination is
  representable.

The constructor does not traverse Tensor provenance, inspect descriptors beyond non-null record
components, allocate a Tensor, capture a graph, or infer a compile mode. Descriptor, membership,
route, and seed checks belong to compile-time preflight because they depend on the supplied
forward boundary.

`createGraph` is a compile-local permission to retain the first-stage formula as differentiable
ordinary Tensor expressions for the immediately following stage. It is true exactly when a
second stage exists. It is not stored on a Tensor, is not a global recording mode, and does not
permit a third stage or a later call to continue a hidden derivative chain.

### Output, target, and cotangent-seed contract

For stage one:

- every `ForwardTensorReference.tensor()` must be the exact object reference of one requested
  `forwardOutputs` element;
- output references must be exact-object-identity-unique; callers who want to combine two
  cotangents for one output must add them before the request;
- the referenced output must have a floating `DataType` and `requiresGrad == true`; and
- output order determines cotangent seeding and reverse-root encounter order.

For stage two:

- each `FirstStageGradientReference.targetIndex()` resolves to the exact generated gradient for
  that stage-one target role, not to an equal Tensor, a captured `ValueId`, or a reconstructed
  wrapper;
- reference indices must be unique within stage two;
- the referenced gradient must have the exact stage-one target Shape and floating data type; and
- output order determines second-stage cotangent seeding and reverse-root encounter order.

For both stages:

- `targets` are ordered, exact-object-identity-unique floating Tensors with
  `requiresGrad == true`;
- each target must occur in the complete original forward Tensor inventory rooted at all
  requested forward outputs, but need not be connected to every selected stage output;
- a target may be a leaf, an intermediate, a selected forward output, or a target used by both
  stages;
- target order determines result-role order within the stage; and
- targets are not traversal stops.

`cotangentSeeds` aligns one-for-one with `outputs`. An empty optional requests the default seed.
The default is legal only when the exact selected output descriptor has `Shape.scalar()`, a
floating data type, and gradient eligibility. It is the existing exact positive-one logical
splat of that output type.

A present explicit seed:

- is the exact supplied Tensor reference;
- has a floating data type equal to the selected output data type;
- has a Shape equal to the selected output Shape;
- has `requiresGrad == false`;
- may be a provenance-free bindable leaf or an ordinary Tensor expression; and
- is used directly as the output cotangent without broadcasting, implicit casting, storage
  reads, or numerical evaluation.

Seed producers already present in the original forward inventory retain derivative order zero.
An independent seed producer is inventoried and validated with its stage and receives that
stage's derivative order; a provenance-free seed remains a graph input with no phase or order.
The seed is not a requested forward publication unless the caller also lists it in
`forwardOutputs`. A non-scalar output with an absent seed, mismatched Shape or type, non-floating
seed, gradient-eligible seed, or invalid reference fails before derivative construction.

Multiple output roots define the sum of their vector-Jacobian products. Reverse accumulation
starts one contribution per output in output order, then preserves the established producer
postorder, selected output-slot order, input-position order, and left-associated
`Tensor.add` contribution order.

### Disconnected targets

`DisconnectedPolicy.ERROR` requires at least one differentiable identity route from a selected
stage output to each target. The first disconnected target fails with:

```text
functionalGradientRequest.stages[<stage>].targets[<target>]
is disconnected from the selected stage outputs
```

Stage indices in failures are zero-based public list indices. The failure occurs before any
Tensor for that stage is allocated. For stage one it therefore consumes no derivative
`TensorId`; for stage two, valid stage-one formula Tensors may already exist, but no stage-two
Tensor is allocated.

`DisconnectedPolicy.ZERO` returns one ordinary exact positive-zero Tensor expression with the
target's exact Shape and floating data type. It uses the existing request-local explicit logical
splat and ordinary `expand` when the target is non-scalar. It does not add a zero operation kind,
an identity node, a mutable zero gradient, storage, or a runtime special case. Two disconnected
targets may share the exact generated zero Tensor only when their exact Shape and data type are
equal; their ordered result roles remain distinct.

A target absent from the complete original forward inventory is invalid, not disconnected, and
is rejected under both policies.

### Preflight and failure order

Generalize `AutogradPreflight` from `FirstOrderRequest` to `FunctionalGradientRequest` while
retaining `FirstOrderGradientCoverage` as the single `D`/`ND`/`FC` role and formula-owner source.
Do not add a public registry, derivative domain-specific language, captured-value-to-Tensor
conversion, or second operation algebra.

The two `GraphCompiler.compile(...)` overloads remain package-private and keep their existing
parameter count and order. Replace only:

```java
Optional<AutogradPreflight.FirstOrderRequest>
```

with:

```java
Optional<FunctionalGradientRequest>
```

There is no additional overload, builder, facade, static public compile method, or public
`GraphCompiler`.

Validation and preflight occur in this order:

1. validate the existing top-level compile arguments in declaration order;
2. validate the mode/request-presence matrix;
3. validate immutable request structure and `createGraph`/reference order;
4. inventory the complete original forward request and explicit constant ingress;
5. validate stage-one forward references, seed references and descriptors, then target
   membership, type, eligibility, and identity uniqueness in list order;
6. select and validate every stage-one `D` route through the existing coverage checker and
   occurrence-local preflight matrix;
7. resolve every stage-one disconnected target and its policy;
8. when stage two exists, validate reference indices and use the checker plus the task-0005E
   transitive formula-operation closure to prove that every possible selected second-pass
   occurrence is classified and has an existing formula owner;
9. only after the complete stage-one plan succeeds, allocate stage-one seeds, constants, and
   formulas;
10. resolve exact stage-one gradient references, validate stage-two seed descriptors and target
    facts, run occurrence-local second-stage preflight on the generated expressions, and resolve
    stage-two disconnected targets; and
11. only after the complete second-stage plan succeeds, allocate stage-two seeds, constants, and
    formulas.

Null checks retain `NullPointerException` with the exact component path. Structural, membership,
descriptor, route, policy, or unsupported differentiation facts use
`IllegalArgumentException` with the first deterministic component/occurrence path and existing
family reason. Unknown kinds, `ND` routes selected as targets, `FC` rows, malformed
cardinalities, missing canonical auxiliaries, and unsupported generated second-pass rows fail
closed.

The allocation guarantee is stage-local. Invalid request structure, forward references,
stage-one seeds/targets/routes, and stage-one disconnected `ERROR` consume no derivative
`TensorId`. A second-stage failure may follow successful stage-one construction, but it consumes
no stage-two Tensor ID and returns no partial graph, role, publication plan, or artifact. Tensor
IDs already consumed by valid stage-one construction are opaque and are not rolled back.

### Reverse-mode stages and ordinary Tensor formulas

Retain `FirstOrderAutograd` as the package-private owner of one reverse-mode stage. Generalize its
entry to accept the successful stage plan, ordered output/cotangent pairs, target list, and
disconnected policy. `GraphCompiler` invokes it exactly once or twice. Do not add
`SecondOrderAutograd`; the second derivative is one more use of the same checked first-order
primitive over generated ordinary Tensor expressions.

Reuse every Compiler 0005E `D`/`ND`/`FC` decision, exact formula owner, canonical auxiliary,
cotangent-normalization, derivative policy, and failure reason. First- and second-stage formulas
use only current public Tensor operations and the same request-local exact splat owner.

The combined expansion retains:

- one identity set of original forward producers;
- one identity set of stage-one-owned producers, including independent seed expressions and
  generated formulas;
- one identity set of stage-two-owned producers, including independent seed expressions and
  generated formulas;
- ordered exact target-to-gradient Tensor roles for each stage; and
- caller constant bindings followed by generated bindings in deterministic first-use order.

The sets and Tensor references are ephemeral compile-local bookkeeping and are discarded after
combined capture.

### Public gradient publication bindings

Add this exact public root-package record:

```java
public record GradientPublicationBinding(
        int derivativeOrder,
        int targetIndex,
        TensorId target,
        ValueId valueId) {}
```

`derivativeOrder` is exactly `1` or `2`. `targetIndex` is the zero-based position in that
derivative stage's target list. `target` is the `TensorId` read from the exact target Tensor
reference after identity validation. `valueId` is the final graph-local gradient `ValueId`.

The constructor rejects another order, a negative target index, or a null identity. Result-role
lists are ordered first by derivative order and then by target index. Equal target IDs may occur
once in each derivative order. Within one derivative order target IDs are unique. Several
bindings may share one gradient value, including ZERO-policy results, without manufacturing
identity nodes.

Exact Tensor object identity is authoritative while the expression is available before capture.
Artifacts deliberately retain only its stable `TensorId` projection and final `ValueId`; they do
not retain public Tensor, producer, provenance, or request objects.

Replace package-private `GraphCompilation.GradientResultRole` and publication gradient
`PublicationBinding` values with the public `GradientPublicationBinding`. Rename the public Model
record used for requested forward publications from `PublicationBinding` to
`ForwardPublicationBinding` so its role is not confused with the richer gradient binding.
`PublicationPlan` remains a public final type with package-private construction and retains
`List<ForwardPublicationBinding>` plus `List<GradientPublicationBinding>` in private members. Its
only public list accessors are `forwardBindings()` and `gradientBindings()`. Boundary validation
flattens all first-order bindings followed by all second-order bindings and appends each
previously unseen gradient `valueId()` after the forward prefix.

### Derivative-order graph metadata

Add this exact public final root-package artifact:

```java
public final class DerivativeGraphMetadata {
    public DerivativeGraphMetadata(
            CompiledGraphModel graph,
            Map<NodeId, Integer> derivativeOrderByNode);

    public CompiledGraphModel graph();

    public Map<NodeId, Integer> derivativeOrderByNode();
}
```

The artifact retains the exact final graph reference and an immutable encounter-order snapshot
with one exact graph `NodeId` key per node:

- order `0` for every original forward producer;
- order `1` for every producer first owned by stage one; and
- order `2` for every producer first owned by stage two.

A producer reused by a later stage retains the first order that owns it. The constructor
rejects nulls, unknown/missing/duplicate node IDs, another order, or map iteration order different
from `graph.nodes()` order. It contains no Tensor, seed, request, formula-owner, storage, runtime,
backend, or execution state.

`GraphPhase` remains unchanged in Model. Order zero nodes have `GraphPhase.FORWARD`; orders one
and two have `GraphPhase.BACKWARD`. The sidecar augments rather than replaces phase.

`GraphCapture.captureCombined(...)` receives the three producer identity sets and assigns IDs
once. Canonicalization, exact rewriting, constant folding, dead-code elimination, and graph
rebuilds remap the sidecar with the graph. Common-subexpression elimination remains phase-local
and becomes derivative-order-local: equal expressions may merge only when their phase and
derivative order are equal. This preserves result meaning and metadata without adding or
broadening an algebraic rewrite.

### One combined capture and artifacts

One compile invocation performs this sequence:

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

The graph-output boundary is:

1. ordered distinct forward output values;
2. each previously unseen first-order gradient value in first-order target order; and
3. each previously unseen second-order gradient value in second-order target order.

Capture preserves shared multi-output producer occurrences, every output slot, exact
`TensorProducer` identity, repeated input positions, canonical saved edges, explicit constant
facts, and one allocation of each `NodeId`/`ValueId`. It does not compile the forward graph,
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
partition, memory, publication, constant, and diagnostic checks. `FORWARD_ONLY` requires no
gradient roles and order zero for every node. Backward-capable modes require at least one
gradient role and allow order one and optional order two.

The package-private complete compile overload still performs graph-stage compilation exactly
once and planning exactly once. It does not expose the request through `CompileArtifacts` and
does not retain providers, availability snapshots, selected devices, kernels, buffers,
preparation, runtime, or execution state.

### Planned API examples

These examples are compiler-package integration sketches for the API implemented by this task.
`GraphCompiler` remains package-private, so they do not claim a new public compile facade.

First-order vector-Jacobian product in one compile call:

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

The one first-order role maps `x.id()` to the final value for `seed * 2 * x`. No Tensor field is
written.

Second derivative of a scalar expression in the same one compile call:

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

The ordered roles contain derivative order one for `2*x` and derivative order two for `2`.
Both are ordinary graph values in the same combined immutable graph. The example creates no
mutable `x.gradient`, calls no `backward()`, and executes nothing.

## Out of scope

- `Tensor.gradient`, `Tensor.backward`, mutable Tensor gradient state, or a Tensor recording flag
- an eager tape, thread-local gradient scope, public gradient registry, service loader, or
  extension SPI
- Model-owned derivative rules, new Model operation kinds, backward-only kinds, a second
  derivative algebra, or Model/`GraphPhase` changes
- more than two reverse-mode stages, unlimited higher order, forward-mode differentiation,
  Jacobian materialization, or a persistent derivative chain
- another public compile facade, public `GraphCompiler`, engine `CompiledGraph`, or
  `CompileConfig` aggregate
- public runtime gradient delivery, optimizer updates, optimizer/training-session work, parameter
  mutation, or training-loop behavior
- concrete dimension binding, prepare, runtime, backend, schedule, execution, storage, physical
  saved values, or buffer work
- new algebraic rewrite, backward-only optimizer, broader CSE across derivative orders, numerical
  evaluation, or host-storage constant discovery
- dependency, Gradle, architecture, ADR, architecture-test, backend-conformance, integration, or
  other-module changes
- a later Compiler task specification or out-of-order planning

## Architecture references

- [`ARCHITECTURE.md`](../../../../../ARCHITECTURE.md)
- [Current architecture plan](../../../../architecture/current-architecture-plan.md)
- [Lifecycle](../../../../architecture/lifecycle.md)
- [Training graph](../../../../architecture/training-graph.md)
- [Module boundaries](../../../../architecture/module-boundaries.md)
- [Dependency rules](../../../../architecture/dependency-rules.md)
- [ADR 0009](../../../../design/decisions/0009-compiler-owned-pre-capture-tensor-expression-autograd.md)
- [Documentation rules](../../../../developer-guide/documentation-rules.md)
- [Planning guide](../../../planning-guide.md)
- [Roadmap](../../../roadmap.md)
- [Compiler master plan](../master-plan.md)
- [Compiler 0004](0004-compiler-owned-pre-capture-autograd-and-combined-graph-compilation.md)
- [Compiler 0005](0005-publication-planning-orchestration-and-compile-artifacts.md)
- [Compiler 0005A](0005a-derivative-policy-and-elementwise-activation-gradient-completion.md)
- [Compiler 0005B](0005b-reduction-scan-softmax-statistics-and-normalization-gradient-completion.md)
- [Compiler 0005C](0005c-layout-window-indexing-scatter-ordering-and-stochastic-gradient-completion.md)
- [Compiler 0005D](0005d-attention-convolution-pooling-and-loss-gradient-completion.md)
- [Compiler 0005E](0005e-first-order-gradient-coverage-closure-checkpoint.md)
- [Compile API](../../../../api/compile-api.md)
- [Tensor API](../../../../api/tensor-api.md)
- [Training API](../../../../api/training-api.md)
- [Glossary](../../../../glossary.md)

## Architecture constraints

- `ARCHITECTURE.md` is authoritative.
- Compiler owns functional gradient requests, preflight, reverse accumulation, formula dispatch,
  derivative-order metadata, combined capture, result roles, and compile artifacts.
- Tensor remains immutable in identity, descriptor, and provenance and gains no gradient or
  backward lifecycle state.
- Generated derivatives remain ordinary public Tensor expressions governed by the same
  inference, validation, numerical, and exact optimization contracts as forward expressions.
- One compile request captures forward and all requested derivative roots together once.
- `GraphPhase` remains Model-owned and unchanged; Compiler derivative-order metadata is a
  sidecar adjacent to the final graph.
- Exact Tensor identity is compile-local. Artifacts retain only `TensorId` and `ValueId`.
- Compile artifacts remain immutable logical recipes and contain no physical, prepared, runtime,
  backend-executable, or mutable state.
- Compiler keeps its existing allowed dependencies and no dependency direction changes.
- If implementation requires a Model change, another public compile facade, a formula outside
  the closed 0005E matrix, a third derivative stage, runtime state, or an architecture change,
  stop and request clarification.

## Package impact

Existing package used and changed:

- `io.github.pho001.synaptik.compiler` — continues to own the cohesive compiler front end and its
  narrow public request/artifact value contracts.

No package is added.

Type placement:

- `io.github.pho001.synaptik.compiler.FunctionalGradientRequest` — public immutable functional
  request and its bounded nested stage/reference/policy vocabulary.
- `io.github.pho001.synaptik.compiler.GradientPublicationBinding` — public immutable
  derivative-stage target-to-gradient publication binding.
- `io.github.pho001.synaptik.compiler.DerivativeGraphMetadata` — public immutable graph-adjacent
  derivative-order sidecar.
- `io.github.pho001.synaptik.model.graph.ForwardPublicationBinding` — the renamed public immutable
  requested-forward identity binding; its two-component semantics are unchanged.
- Existing package-private `GraphCompiler`, `GraphCompilation`, `AutogradPreflight`,
  `FirstOrderAutograd`, capture, validation, and optimization types remain implementation owners.

The three public types are compiler contracts justified by this concrete current consumer. They
do not form a facade, registry, or new package layer.

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

Documentation/planning:

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
ADR 0009; Public/Runtime APIs; Model capabilities/master/tasks/source/tests; completed Compiler
tasks and family-rule source/tests; compiler package Javadoc; Config, Planning, Trace, Runtime,
Prepare, backends, Engine, NN, training, tools; architecture/backend-conformance/integration
tests; and Gradle/build files.

## Maximum scope

The original ceiling was 48 paths. The implementation used 45 paths before the approved final
terminology correction. The final exact 54-path scope contains 20 production paths, 21 test paths,
and 13 documentation/planning paths. Renaming the tracked Model production/test pair accounts for
four delete/create paths; the required architecture identifier and current-contract
documentation account for the remaining approved expansion. This correction changes names and
documentation only; it does not authorize another semantic, validation, visibility, ownership,
capture, optimization, publication, or ID behavior change.

The limit is intentionally larger than a normal task because derivative-order metadata must
remain synchronized through every existing immutable graph rebuild and because the public
artifact validation changes atomically with request/results. Do not split those consistency
changes into a later cleanup.

If another formula-family source/test, Model behavior, public facade, dependency/build change, or
third-stage mechanism is required, stop and report the exact gap. Do not use the approved
terminology expansion to hide a formula or architecture problem.

A separate documentation-focused clean context finalizes authorized Javadocs, API references,
glossary impact, examples, planning evidence, and status. It reuses successful Java evidence
unless it changes executable Java behavior or records a concrete reason.

## Acceptance criteria

### Request and preflight

- The exact public request and nested types above compile with complete meaningful Javadoc.
- Records snapshot every list and retain exact elements; all null, empty, size, uniqueness,
  reference-index, stage-count, and `createGraph` failures are deterministic.
- The one-stage and two-stage matrices are exhaustive and no third stage is representable.
- Stage-one forward references, stage-two exact gradient references, seeds, targets, and
  disconnected policies satisfy the exact contracts and failure order above.
- Default seeds are accepted only for exact scalar floating outputs. Explicit seeds require exact
  floating Shape/type equality and `requiresGrad == false`.
- Invalid first-stage facts consume no derivative Tensor ID. Invalid second-stage facts consume
  no stage-two Tensor ID and return no partial artifact.
- `FirstOrderGradientCoverage` remains the sole D/ND/FC and formula-owner source for both stages.

### Gradient construction and results

- One-stage scalar, explicit vector-Jacobian, multi-output, mixed-floating target, repeated
  consumer, shared producer, target-as-output, and target-in-both-stages cases are covered.
- Two-stage scalar second derivative, Hessian-vector product, constant/disconnected first
  derivative, shared stage-one result, and unsupported second-pass cases are covered.
- `ERROR` rejects the first disconnected target; `ZERO` returns exact target-Shape/type zero
  through the existing explicit splat sidecar.
- Contribution and result ordering is deterministic and exactly output order, then existing
  reverse traversal, then target order.
- `GradientPublicationBinding` lists are ordered by derivative order then stage-local target
  index and preserve target-ID/result-value bindings even when values repeat.
- No Tensor/request/provenance reference survives in graph-stage or public artifacts.

### Graph, metadata, optimization, and artifacts

- Forward outputs and every first-/second-stage result are captured together exactly once.
- Shared multi-output producers, canonical auxiliaries, saved edges, repeated positions,
  constants, graph IDs, and publication facts remain correct.
- `DerivativeGraphMetadata` covers every final node exactly once with order zero, one, or two and
  exact phase consistency.
- Every canonicalization, rewrite/fold, DCE, and CSE rebuild remaps derivative metadata.
- CSE remains exact, phase-local, and derivative-order-local; no rewrite rule is added or
  broadened.
- The final combined graph receives the established optimization pipeline once and each changed
  candidate is revalidated.
- `GraphCompilation`, `PublicationPlan`, and eight-component `CompileArtifacts` cross-validate
  result order, boundary de-duplication, graph identity, phase/order, planning, memory,
  constants, and diagnostics. `PublicationPlan` exposes only `forwardBindings()` and
  `gradientBindings()`.
- The complete package-private compile entry performs graph compilation once and Planning once.

### Boundaries, documentation, and completion

- No Tensor gradient/backward state, public compile facade, registry, second algebra, new Model
  kind, Model/GraphPhase change, third derivative stage, runtime tape, optimizer/training loop,
  prepare/runtime/backend behavior, dependency, Gradle, or architecture change is present.
- The separate documentation-focused clean context finalizes public and implementation Javadocs,
  Compile/Tensor/Training API status, glossary terms, both conceptual examples, task evidence,
  master plan, and roadmap.
- Compile API explains functional request/reference/result shapes, seed and disconnected
  behavior, order metadata, one capture, and current package-private integration.
- Tensor API states that derivatives are functional compiler results and Tensor remains free of
  gradient lifecycle state. Training API distinguishes this compiler capability from planned
  optimizer/session/runtime work.
- Reasoned no-change conclusions are recorded for architecture/ADR/tests, Model, Config,
  Planning, Trace, Runtime, Prepare, backends, Engine, NN/training implementation, conformance,
  integration, Gradle, and unrelated APIs.
- Exactly the authorized paths change, no later Compiler task specification exists, links and
  anchors resolve, headings are unique, fences are balanced, final newlines and whitespace are
  valid, statuses synchronize, and `git diff --check` passes.

## Tests / validation

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

After executable Java stabilizes, run the affected module once:

```text
./gradlew :modules:compiler:test
```

This task closes the explicit functional/higher-order compiler capability and changes a public
compile-artifact shape. Run the capability checkpoint:

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
outside the repository. It must validate every changed Markdown link target and heading anchor,
unique headings, balanced fences, final newlines, and trailing whitespace.

Required manual/source evidence:

- exact public `javap`/reflection surface and constructor validation for the final public
  request, forward/gradient binding, derivative metadata, and eight-component
  `CompileArtifacts` contracts;
- no public `GraphCompiler`, extra compile overload, Tensor gradient/backward member, registry,
  service loader, second algebra, or third-stage path;
- exact-object identity membership, list snapshots, no-allocation failure deltas, seed
  Shape/DataType/eligibility matrix, and ERROR/ZERO behavior;
- D/ND/FC coverage and formula-owner equality for both passes, including the exact unsupported
  second-pass reason;
- one combined capture invocation, graph boundary order/de-duplication, shared producer and
  auxiliary identity, constant sidecar, and publication inspection;
- derivative-order coverage/remapping through canonicalization, rewrite/fold, DCE, CSE, and final
  artifact validation;
- original 48-path ceiling, recorded 45-path pre-correction scope, approved final
  terminology-only Model rename and documentation expansion, no executable Java outside
  Compiler, no Gradle/dependency/other-module behavior change, and no later task specification;
- task/master/roadmap Ready/In-progress synchronization before implementation and Complete
  synchronization only after all implementation evidence passes; and
- General/API-Javadoc/Planning/Example profile review, terminology, examples, links, anchors,
  headings, fences, final newlines, whitespace, and `git diff --check`.

Do not run backend-conformance or integration suites separately. There is no backend or
end-to-end execution change, and the root plus architecture checkpoint is the proportional
cross-repository gate. Do not repeat successful Java commands in the documentation context
without changed executable behavior or a recorded concrete risk.

## Dependencies

- Compiler 0001–0003B capture, validation, canonicalization, and exact optimization — Complete.
- Compiler 0004–0004B pre-capture first-order autograd and shared Tensor algebra — Complete.
- Compiler 0005 compile artifacts and publication/planning orchestration — Complete.
- Compiler 0005A–0005D complete current first-order formulas and policies — Complete.
- Compiler 0005E source-backed 37-family/107-kind/128-signature D/ND/FC and transitive formula
  closure checkpoint — Complete.
- Stable current public Tensor, Tensor producer/canonical output, Shape, DataType, graph ID,
  `GraphPhase`, compile-mode, optimization, publication, Planning, and compile-artifact contracts.

## Follow-up tasks

- No later Compiler task is specified. Reassess the roadmap after this task is Complete.
- A future Engine task may wrap the public functional request in its public compile lifecycle
  without making `GraphCompiler` public.
- A future focused task may add another derivative order only after an explicit architecture and
  capability decision; it must not infer permission from this bounded two-stage contract.

Do not create a follow-up specification in this task.

## Architecture impact

Expected impact: None.

The architecture already requires Compiler-owned pre-capture Tensor-expression autograd, one
combined graph, immutable compile artifacts, no Tensor gradient lifecycle state, and an explicit
future create-graph/derivative-order contract. This task supplies that bounded compiler contract
without moving ownership, changing dependencies, or altering Model `GraphPhase`.

If implementation requires changing `ARCHITECTURE.md`, Model, a dependency rule, the public
compile facade owner, runtime state, backend behavior, or the two-stage limit, stop and report the
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
  without creating a broad compiler API package or many top-level policy/reference types.
- Stage-two references use stage-one target indices because request target order is stable before
  generated Tensor and graph identities exist. They do not expose temporary generated Tensors or
  captured `ValueId` values.
- Targets must belong to the complete forward request, while disconnected policy is evaluated
  against selected stage outputs. This distinguishes an invalid target from a valid disconnected
  target and makes ZERO useful for multi-output requests and constant first derivatives.
- Explicit seeds must not request gradients. Independent seed producers belong to their
  derivative stage, while provenance-free seeds remain graph inputs; neither creates a third
  publication category or becomes a differentiation target in this task.
- The existing `FirstOrderAutograd` primitive is reused once per stage. A separate
  second-order engine would duplicate dispatch and imply a second algebra.
- Exact Tensor identity is not retained after capture. The public
  `GradientPublicationBinding` records the exact target's stable `TensorId`, final `ValueId`,
  derivative order, and target position.
- Derivative order is compiler-owned sidecar state. Model `GraphPhase.BACKWARD` continues to
  classify both orders, and exact CSE adds order equality to its existing phase equality guard.
- `CompileArtifacts` gains the derivative sidecar atomically because downstream consumers need
  the final graph and role/order interpretation to agree.
- No architecture, dependency, Model, Gradle, backend, runtime, preparation, execution, or
  training implementation change is planned.

## Known limitations

- Exactly two reverse-mode stages are supported. There is no unlimited higher-order chain.
- Stage two can differentiate only exact stage-one result roles, not arbitrary generated
  intermediates.
- Cotangent seeds are not differentiation targets in this task.
- One-output attention and positive-static-depth index-loss restrictions remain as recorded by
  Compiler 0005D/0005E.
- Any occurrence-dependent row that remains `FC` under the existing preflight contract remains
  unsupported in either stage.
- ZERO returns symbolic exact zeros; this task does not execute or materialize them.
- Public Engine compilation, preparation, runtime publication, optimizer/session behavior, and
  execution remain planned.

## Validation evidence

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
  afterward, so documentation context
  `/root/rename_publication_bindings/docs_finalize_publication_rename` reused this evidence and
  did not repeat Java tests.
- Final Java scans found no `PublicationBinding` or `GradientResultRole` in
  `modules/**/*.java`, found no retired `PublicationPlan` accessor, and `git diff --check`
  passed before the documentation handoff.
- Documentation context
  `/root/rename_publication_bindings/docs_finalize_publication_rename` applied the
  General, API/Javadoc, Planning, and Example profiles. It finalized the affected Javadocs,
  current API/glossary/architecture terminology, task/master/roadmap evidence, and the current
  Model capability-audit link while preserving historical task text where it records the
  contract name that existed when that task completed.
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
  original `PublicationBinding` name, while the current capability audit resolves to the live
  `ForwardPublicationBinding.java` path.
- Final scope/status checks proved exactly 54 paths (20 production, 21 test, 13
  documentation/planning), task/master/roadmap `Complete`, and task 0006 as the last detailed
  Compiler specification. Final changed-Markdown validation and `git diff --check` passed after
  this evidence update.

## Implementation notes

- Added the bounded public request, result-role, and derivative-order values while retaining
  package-private compiler entry ownership and immutable Tensor semantics.
- Reused the checked reverse-mode primitive once per stage, captured forward and all derivative
  roots once, and propagated derivative-order metadata through every graph rebuild. CSE is local
  to both graph phase and derivative order.
- Exact explicit/default seed rules, output-order VJP accumulation, ERROR/ZERO disconnected
  behavior, stage-local target identity, and the eight-component artifact boundary are covered by
  the final focused and module evidence.
- Applied the approved minimal terminology correction: Model
  `PublicationBinding`/`PublicationBindingTest` became
  `ForwardPublicationBinding`/`ForwardPublicationBindingTest`; Compiler
  `GradientResultRole` became `GradientPublicationBinding` with exact components
  `derivativeOrder`, `targetIndex`, `target`, and `valueId`; and `PublicationPlan` now exposes
  only `forwardBindings()` and `gradientBindings()`.
- No semantic, validation-order, visibility, ownership, capture, optimization, publication, or
  ID behavior changed. `GraphCompilation` deliberately retains its package-private
  `forwardOutputs`/`gradientResults` component names while carrying
  `GradientPublicationBinding`.
- Historical Model task 0009 and earlier completed task specifications retain the names and
  paths that were true at their completion; they are evidence, not current API references. The
  current Model capability audit instead uses the live `ForwardPublicationBinding` source link.
  The Model master plan needed no edit because it contains no affected current identifier.
- Compile, Public, and Tensor API references plus the glossary required exact name, component, and
  accessor corrections. Training API already described the functional request and publication
  boundary without either retired identifier, so review found no additional rename edit.
- `ARCHITECTURE.md` required only its Model allowed-list identifier. The focused partition-scoring
  explanation, memory-planning design note, and backend-selection guide required only negative-
  boundary identifier corrections. Other focused architecture pages and ADR 0009 required no
  change because ownership, lifecycle, dependency, and autograd architecture did not move.
- Architecture tests required no change because no dependency rule changed. Model capabilities
  and master planning required no semantic update; only the capability audit's live source link
  changed, while historical task records remain historical.
- Config, Planning implementation, Trace, Runtime, Prepare, concrete backends, Engine, NN,
  training implementation, backend conformance, integration, Gradle/build configuration, and
  other modules required no change because this pass adds no behavior, dependency, execution,
  runtime publication, or backend claim.

## Completion summary

- Completed changes: implemented explicit one/two-stage functional gradient requests, ordered
  first-/second-derivative results, exact seeds and disconnected policies, one combined graph,
  derivative-order metadata, synchronized artifact/publication validation, and the approved
  unambiguous forward/gradient publication-binding terminology correction.
- Files changed or created: 54 exact paths—20 production paths, 21 test paths, and 13
  documentation/planning paths. The original implementation used 45 paths before the approved
  Model rename and current-contract documentation expansion.
- Validation: focused Compiler 0006 tests passed; full Compiler passed 31 suites/208 tests; the
  repository/architecture checkpoint passed; the separate documentation gates passed.
- Documentation: finalized affected Javadoc, Compile/Tensor/Training/Public API references,
  glossary terminology, scalar-second-derivative and vector/Hessian-product examples,
  architecture identifier, current Model capability-audit link, and task/master/roadmap status.
- Unresolved issues: none.
- Required follow-up: none for this task; later Engine, prepare/runtime publication, optimizer,
  and any derivative order beyond two remain separately planned.

Status: Complete
