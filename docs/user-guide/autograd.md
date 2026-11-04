# Use the current [autograd](../glossary.md#autograd) boundaries

## Outcome

This guide helps you choose the current [automatic differentiation](../glossary.md#autograd) ([autograd](../glossary.md#autograd)) entry point and
understand the result. Autograd constructs [reverse-mode](../glossary.md#primal-adjoint-and-differentiation-seed) [gradient](../glossary.md#term-gradient) expressions from a forward
[Tensor](../glossary.md#tensor) expression. Synaptik currently offers a narrow one-shot [scalar](../glossary.md#term-scalar)-[objective](../glossary.md#term-objective) convenience, an
ordinary explicitly seeded first-order [compile](../glossary.md#compile), and an advanced bounded one- or two-stage
functional request.

A **gradient** measures how a chosen result changes when selected input values change.
**Forward** computes the original expression; **[reverse mode](../glossary.md#term-reverse-mode)** works backward through its
dependencies to build those sensitivities. **First order** means a [derivative](../glossary.md#term-derivative) of the original
expression; **second order** differentiates a derivative. Computing either does not update a
[weight](../glossary.md#term-weight): that is the [optimizer](../glossary.md#term-optimizer)'s separate job.
For the concrete [FLOAT32](../glossary.md#term-float32) `[2,1]`, no-[bias](../glossary.md#term-bias) learning case, start with the
[loss `40` and weight gradient `-40`](../tutorials/loss-and-gradients.md)<span data-glossary-navigation="true"> ([loss glossary](../glossary.md#loss))</span> and follow the existing
[reverse-gradient diagram](../explanation/tensors-and-gradients.md).

## Choose an entry point

| Goal | Current entry point | Fixed boundary |
|---|---|---|
| Compute one [scalar objective](../glossary.md#loss) and [detached](../glossary.md#detached-model-results) first [gradients](../glossary.md#term-gradient) now | `Engine.backward(objective, targets, maximumTotalBytes)` | One absent-[seed](../glossary.md#term-seed) first-order stage, `createGraph == false`, `DisconnectedPolicy.ERROR` |
| [Compile](../glossary.md#compile) a reusable first-order [graph](../glossary.md#graph) with explicit [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings) | `Engine.compile(forwardOutputs, cotangentSeeds, targets)` | One stage, one non-null seed per output, `createGraph == false`, `DisconnectedPolicy.ERROR` |
| Select multiple outputs, optional seeds, disconnected-zero results, or a second reverse stage | `AdvancedEngine.compile(...)` with `FunctionalGradientRequest` | Exactly one or two stages under the complete bounded policy |

All three paths leave public [Tensors](../glossary.md#tensor) unchanged. They do not add a gradient field, a recording
tape, or a `Tensor.backward()` method.

### Who asks for the derivative?

In these examples the Java application supplies Tensor expressions and calls Engine.
Engine calls Compiler to construct the derivative descriptions; Compiler does not read
batch bytes to perform this construction. The one-shot `backward` path then executes and
copies results before returning. The `compile` paths return handles for later preparation
and execution, not current derivative values.

During an actual run, input stores remain the supplying application's responsibility.
An explicit `RunResult` borrows them until it closes; `backward` closes its temporary result
before returning detached copies. See
[run and materialize](running-models.md#run-and-materialize) for that lifetime difference.

## Prerequisites and inputs

Start with an open [Engine](../glossary.md#term-engine) composition and [Tensor](../glossary.md#tensor) expressions whose selected [derivative](../glossary.md#term-derivative) routes are
supported. [Targets](../glossary.md#term-target) are always explicit: Synaptik does not infer which Tensors should receive
[gradients](../glossary.md#term-gradient). Every target list is non-empty, ordered, unique by exact Tensor object [identity](../glossary.md#term-reference-identity), and
drawn from the complete original forward expression inventory.

Here a **target** is a Tensor with respect to which a derivative is requested, such as the
[weight](../glossary.md#term-weight). It is not necessarily a [training target](../glossary.md#loss) containing desired answers. **[Identity](../glossary.md#term-identity-meanings)** means
the exact Java Tensor object, not another object with equal values. A **disconnected** target
has no supported [differentiable](../glossary.md#derivative-and-chain-rule) [dependency](../glossary.md#term-dependency) path from the selected outputs.

The snippets below are conceptual [API](../glossary.md#term-application-programming-interface-api)-[shape](../glossary.md#shape) examples. They assume that the shown Tensors have
compatible [descriptors](../glossary.md#term-descriptor) and live input storage and that `engine` or `advanced` is an open Engine.
See the [ordinary and advanced compile reference](../reference/compiler/entry-points.md#current-ordinary-and-advanced-engine-compile-boundaries)<span data-glossary-navigation="true"> ([compile glossary](../glossary.md#compile))</span>
for complete [lifecycle](../glossary.md#lifecycle) [ownership](../glossary.md#term-ownership-meanings) and current execution restrictions.

## Compute a [scalar objective](../glossary.md#loss) and first [gradients](../glossary.md#term-gradient)

For one [scalar](../glossary.md#term-scalar) floating [objective](../glossary.md#term-objective), use the ordinary one-shot convenience:

```java
// Ask for derivatives of this loss with respect to these exact Tensors; do not update them.
ScalarObjectiveBackwardResult result =
        engine.backward(loss, List.of(weight, bias), maximumTotalBytes);
```

The call freshly compiles, prepares, runs, copies the objective first and the gradients in [target](../glossary.md#term-target)
order, then cleans up temporary execution state. The returned values are [detached](../glossary.md#detached-model-results) and remain
readable after [Engine](../glossary.md#term-engine) and caller storage closure. The byte limit applies to the combined returned
payloads.

This method always constructs one [reverse-mode](../glossary.md#primal-adjoint-and-differentiation-seed) stage. Its only output is [`loss`](../glossary.md#loss), its [seed](../glossary.md#term-seed) is
absent, `createGraph` is false, and its disconnected policy is `ERROR`. The [compiler](../glossary.md#term-compiler) turns the
absent seed into an exact typed one only if `loss` is scalar, floating, and gradient-eligible.
Use another entry point when you need an explicit seed, several outputs, a zero for disconnected
targets, or a second stage.

## [Compile](../glossary.md#compile) an explicitly seeded first-order [graph](../glossary.md#graph)

The ordinary compile overload aligns one explicit [seed](../glossary.md#term-seed) with each forward output:

```java
// The seed selects the output sensitivities; compile creates formulas, not computed bytes.
CompiledGraph compiled = engine.compile(
        List.of(vectorOutput),
        List.of(cotangentSeed),
        List.of(input));
```

This constructs the [vector-Jacobian product](../glossary.md#term-vector-jacobian-product-vjp) selected by `cotangentSeed`. The seed must match the
output's exact [Shape](../glossary.md#shape) and floating [data type](../glossary.md#data-type--datatype) and must not request [gradients](../glossary.md#term-gradient). This overload still
uses one stage, `createGraph == false`, and `DisconnectedPolicy.ERROR`; it does not expose the
second-stage or disconnected-zero policy.

A **[cotangent seed](../glossary.md#primal-adjoint-and-differentiation-seed)** supplies a [sensitivity](../glossary.md#term-sensitivity) for each output [element](../glossary.md#term-element). [Reverse mode](../glossary.md#term-reverse-mode) combines
those sensitivities to ask how the selected weighted output depends on the [targets](../glossary.md#term-target).
The **[Jacobian](../glossary.md#term-jacobian)** is the array of all output-to-input [derivatives](../glossary.md#term-derivative); a **vector-Jacobian product**
combines its rows using the seed, without requiring a separately published full Jacobian.
For a [scalar](../glossary.md#term-scalar) [loss](../glossary.md#loss), seed one asks for the loss's own derivative.

The result is a reusable compile handle bound to this Engine, not computed gradient bytes.
Call `engine.prepare(compiled)` once, then `run` or `runValues` for each batch and inspect
the forward and gradient publications. Close each result before releasing borrowed input
storage. The [prepared-run guide](running-models.md#compile-and-prepare-once) shows those calls.
Successful compile does not promise that every planned backend and transfer can be prepared.

### [Pool3d](../glossary.md#pool3d--ncdhw-pooling) first-order training

Maximum and average Pool3d use ordinary [Compiler](../glossary.md#term-compiler) [gradient](../glossary.md#term-gradient) expressions and shared unfold/fold
[owners](../glossary.md#term-owner). Qualified static [CPU](../glossary.md#term-cpu) floating types and [Metal](../glossary.md#term-metal) [FLOAT32](../glossary.md#term-float32) execute forward and explicitly
seeded input gradients, including overlapping windows. [Conv3d](../glossary.md#conv3d--ncdhw) → [pooling](../glossary.md#term-pooling) → flatten → [Linear](../glossary.md#term-linear-meanings)
training uses the same owners for [parameter](../glossary.md#term-parameter) gradients; there is no separate [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [derivative](../glossary.md#term-derivative) path.
Low CPU arithmetic uses FLOAT32 working values and final low-result [narrowing](../glossary.md#term-narrowing). Metal [FLOAT64](../glossary.md#term-float64)/low overlap, Metal Scan and
ordinary public CPU named-B [convolution](../glossary.md#term-channels-first-convolution-layers)/pooling expansion are not granted by this capability.

Cold source copying and save/load preserve exact Pool3d geometry but are separate from execution
[qualification](../glossary.md#term-qualification): Pool3d needs at least `graph-model.v6`, while [FP8](../glossary.md#term-fp8) sources need v7. Loaded Models require a fresh training configuration: [optimizer](../glossary.md#term-optimizer),
executable and session state are not restored. The
[support matrix](../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary)<span data-glossary-navigation="true"> ([support matrix glossary](../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([matrix glossary](../glossary.md#term-review-matrix))</span>
defines the [backend](../glossary.md#backend) [occurrence](../glossary.md#term-occurrence) limits.

## Build a bounded functional request

`FunctionalGradientRequest` contains exactly one or two ordered [reverse-mode](../glossary.md#primal-adjoint-and-differentiation-seed) stages:

- Each stage has non-empty ordered output references, one optional seed position per output, and
  a non-empty ordered target list.<span data-glossary-navigation="true"> ([seed glossary](../glossary.md#term-seed))</span>
- Stage one selects exact [Tensors](../glossary.md#tensor) from the requested forward-output boundary with
  `ForwardTensorReference`.
- Stage two may select only generated first-stage [gradients](../glossary.md#term-gradient) with
  `FirstStageGradientReference(targetIndex)`.
- One stage requires `createGraph == false`. Two stages require `true` for stage one and `false`
  for stage two.
- An absent seed means an exact typed one only for an eligible scalar output. A present seed must
  match the output's exact Shape and floating type and must not request gradients.<span data-glossary-navigation="true"> ([scalar glossary](../glossary.md#term-scalar); [Shape glossary](../glossary.md#shape))</span>
- `DisconnectedPolicy.ERROR` rejects a [target](../glossary.md#term-target) without a [differentiable](../glossary.md#derivative-and-chain-rule) route.
  `DisconnectedPolicy.ZERO` returns an ordinary exact typed zero expression; several target
  roles may share that same value.

`createGraph` retains first-stage formulas only for the immediate second stage in the same
[compile](../glossary.md#compile). It does not open a persistent [derivative](../glossary.md#term-derivative) recording [scope](../glossary.md#term-scope) or authorize arbitrary nesting.

## Conceptual example: [compile](../glossary.md#compile) a [Hessian-vector product](../glossary.md#term-hessian-vector-product-hvp)

### Goal and inputs

Assume `x` is a floating [Tensor](../glossary.md#tensor) of [Shape](../glossary.md#shape) `[2]`, `loss = sum(x * x)` is [scalar](../glossary.md#term-scalar), and [`vector`](../glossary.md#term-vector) is a
non-[gradient](../glossary.md#term-gradient) Tensor of exact Shape `[2]` and the same floating type as the first-stage gradient.
The goal is to retain the first [derivative](../glossary.md#term-derivative) and construct one second-stage [vector-Jacobian product](../glossary.md#term-vector-jacobian-product-vjp).

The **[Hessian](../glossary.md#term-hessian)** collects second derivatives of a [scalar objective](../glossary.md#loss). A **[Hessian-vector product](../glossary.md#term-hessian-vector-product-hvp)**
combines them along a supplied direction, rather than producing the entire second-derivative
array. This advanced example teaches that separate concept; it is not an additional prerequisite
or a replacement [model](../glossary.md#term-model) for the one-[weight](../glossary.md#term-weight) learning journey.

For a finite dimensionless real vector $x=(x_1,x_2)$ and
$L(x)=x_1^2+x_2^2$, the first derivative is
$\nabla L(x)=(2x_1,2x_2)$. The Hessian has diagonal entries $2$ and other entries $0$.
A seed vector $v=(v_1,v_2)$ therefore selects
$H_L(x)v=(2v_1,2v_2)$, where $H_L$ denotes the second-derivative matrix.
For $v=(1,-1)$ the mathematical result is $(2,-2)$.
This explains which derivative the request describes; it does not establish execution support
for a backend, NaN/infinite input behavior or a particular floating-point evaluation order.

### Meaningful steps

The following is conceptual Java because [Tensor](../glossary.md#tensor) storage and [Engine](../glossary.md#term-engine) composition setup are omitted:

```java
// Retain first-stage formulas only so the immediately following stage can differentiate them.
var first = new FunctionalGradientRequest.Stage(
        List.of(new FunctionalGradientRequest.ForwardTensorReference(loss)),
        List.of(Optional.empty()),
        List.of(x),
        true,
        FunctionalGradientRequest.DisconnectedPolicy.ERROR);

// Select generated gradient number zero, not a forward Tensor with similar values.
var second = new FunctionalGradientRequest.Stage(
        List.of(new FunctionalGradientRequest.FirstStageGradientReference(0)),
        List.of(Optional.of(vector)),
        List.of(x),
        false,
        FunctionalGradientRequest.DisconnectedPolicy.ERROR);

var request = new FunctionalGradientRequest(List.of(first, second));

AdvancedCompiledGraph compiled = advanced.compile(
        CompileMode.FORWARD_AND_BACKWARD,
        List.of(loss),
        Optional.of(request),
        GraphOptimizationConfig.standard(),
        BackendIntent.unconstrained(),
        PartitionScoringConfig.neutral());
```

Stage one uses the eligible [scalar](../glossary.md#term-scalar) default [seed](../glossary.md#term-seed) and produces the [gradient](../glossary.md#term-gradient) selected by [target](../glossary.md#term-target) index
zero. Stage two uses [`vector`](../glossary.md#term-vector) as the [cotangent](../glossary.md#cotangent) for that exact first-stage result. The [compiler](../glossary.md#term-compiler)
preflights both selected routes, constructs their formulas with ordinary Tensor operations, and
[captures](../glossary.md#term-capture-meanings) the forward output and both gradient roots together once.

### Result and interpretation

Successful [compilation](../glossary.md#compiler-engine-and-execution-stage-names) returns an [opaque](../glossary.md#term-opaque) `AdvancedCompiledGraph`. The underlying [compiler](../glossary.md#term-compiler)
[artifacts](../glossary.md#term-artifact) retain one order-one [gradient](../glossary.md#term-gradient) [binding](../glossary.md#term-binding) followed by one order-two binding, both for `x`.
Per-[node](../glossary.md#node) [derivative](../glossary.md#term-derivative) [metadata](../glossary.md#term-metadata) uses order zero for original forward [producers](../glossary.md#term-producer), one for producers
first owned by stage one, and two for producers first owned by stage two. Orders one and two both
remain [graph phase](../glossary.md#graph-phase) `BACKWARD`.

For this expression, the second binding represents the [Hessian-vector product](../glossary.md#term-hessian-vector-product-hvp) selected by
[`vector`](../glossary.md#term-vector). The example proves bounded [graph](../glossary.md#graph) construction and ordering only. It does not prove that
a particular [backend](../glossary.md#backend) can [prepare](../glossary.md#prepare) or numerically execute the graph, and it does not create a
third-stage derivative chain.

If `vector` has the wrong [Shape](../glossary.md#shape) or floating type, or requests gradients, compilation rejects the
request before derivative formulas are allocated.

## [Publication](../glossary.md#term-publication) and shared values

[Gradient](../glossary.md#term-gradient) publication [bindings](../glossary.md#term-binding) are ordered first by [derivative](../glossary.md#term-derivative) order and then by [target](../glossary.md#term-target) index
within that stage. Each binding retains the target [identity](../glossary.md#term-identity-meanings) and final gradient [graph value](../glossary.md#graph-value).
Bindings remain target-distinct even when disconnected-zero handling or equivalent formulas make
several targets share one value. The [graph](../glossary.md#graph) output boundary starts with the ordered forward values
and then appends each previously unseen gradient value once in binding order.

## Common errors

| Symptom | Likely cause | Correction |
|---|---|---|
| An absent [seed](../glossary.md#term-seed) is rejected | The selected output is not [scalar](../glossary.md#term-scalar), floating, and [gradient](../glossary.md#term-gradient)-eligible. | Supply a matching explicit seed, or select an eligible scalar output. |
| An explicit seed is rejected | Its [Shape](../glossary.md#shape) or floating type differs from the output, or it requests gradients. | Use the exact output Shape and type with gradient eligibility disabled. |
| A [target](../glossary.md#term-target) is rejected as disconnected | The selected outputs have no [differentiable](../glossary.md#derivative-and-chain-rule) route to that target under `ERROR`. | Select a connected target or use the advanced request with `ZERO` when a typed zero is intended. |
| A public [Tensor](../glossary.md#tensor) method exists but [autograd](../glossary.md#autograd) rejects the route | Expression construction does not guarantee a [derivative](../glossary.md#term-derivative) rule for every [operation](../glossary.md#operation), attribute, or role. | Check the maintained [Compiler derivative matrix](../reference/compiler/autograd.md#current-package-private-pre-capture-autograd)<span data-glossary-navigation="true"> ([matrix glossary](../glossary.md#term-matrix-meanings))</span>. |
| A two-stage request is rejected structurally | Stage one does not use `createGraph == true`, stage two uses it, or stage two references a forward Tensor. | Use `true` only on stage one and select stage-two outputs by first-stage target index. |
| [Compilation](../glossary.md#compiler-engine-and-execution-stage-names) succeeds but preparation fails | Current [backend](../glossary.md#backend) preparation supports a narrower executable subset than [compiler](../glossary.md#term-compiler) [graph](../glossary.md#graph) construction. | Treat [compile](../glossary.md#compile) and [prepare](../glossary.md#prepare) as separate boundaries and stay within the documented backend subset. |

## Limitations

The current boundary has no third reverse stage, [derivative](../glossary.md#term-derivative) order above two, arbitrary nested
differentiation, persistent or [runtime](../glossary.md#term-runtime) tape, [mutable](../glossary.md#term-mutable) [Tensor](../glossary.md#tensor) [gradient](../glossary.md#term-gradient) state, no-[argument](../glossary.md#term-argument) backward,
`Tensor.backward()`, [optimizer](../glossary.md#term-optimizer) update, or [training session](../glossary.md#training-session--trainingsession). `TRAINING_STEP` currently constructs
the same bounded derivative [graph](../glossary.md#graph) as `FORWARD_AND_BACKWARD`; it does not add an optimizer update.

Derivative coverage remains [fail-closed](../glossary.md#term-fail-closed) and [operation](../glossary.md#operation)-specific. The maintained [Compile](../glossary.md#compile) [API](../glossary.md#term-application-programming-interface-api) lists
the current formula [matrix](../glossary.md#term-matrix-meanings) and exclusions; this guide does not duplicate that inventory or imply
numerical or [backend](../glossary.md#backend) coverage from formula construction alone.

## Related documentation

- [Compile API and functional-gradient examples](../reference/compiler/autograd.md#functional-gradient-examples)<span data-glossary-navigation="true"> ([Compile glossary](../glossary.md#compile); [API glossary](../glossary.md#term-application-programming-interface-api); [gradient glossary](../glossary.md#term-gradient))</span>
- [Compiler-owned automatic differentiation contract](../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../glossary.md#term-compiler); [automatic differentiation glossary](../glossary.md#autograd))</span>
- [Training graph explanation](../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../glossary.md#training-graph))</span>
- [Public API lifecycle](../reference/engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle)<span data-glossary-navigation="true"> ([lifecycle glossary](../glossary.md#lifecycle))</span>
- [Tensor API](../reference/tensor/index.md)<span data-glossary-navigation="true"> ([Tensor glossary](../glossary.md#tensor))</span>
- [Training API status](../reference/training/index.md)
