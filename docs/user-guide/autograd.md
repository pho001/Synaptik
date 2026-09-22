# Use the current autograd boundaries

## Outcome

This guide helps you choose the current automatic differentiation (autograd) entry point and
understand the result. Autograd constructs reverse-mode gradient expressions from a forward
Tensor expression. Synaptik currently offers a narrow one-shot scalar-objective convenience, an
ordinary explicitly seeded first-order compile, and an advanced bounded one- or two-stage
functional request.

## Choose an entry point

| Goal | Current entry point | Fixed boundary |
|---|---|---|
| Compute one scalar objective and detached first gradients now | `Engine.backward(objective, targets, maximumTotalBytes)` | One absent-seed first-order stage, `createGraph == false`, `DisconnectedPolicy.ERROR` |
| Compile a reusable first-order graph with explicit seeds | `Engine.compile(forwardOutputs, cotangentSeeds, targets)` | One stage, one non-null seed per output, `createGraph == false`, `DisconnectedPolicy.ERROR` |
| Select multiple outputs, optional seeds, disconnected-zero results, or a second reverse stage | `AdvancedEngine.compile(...)` with `FunctionalGradientRequest` | Exactly one or two stages under the complete bounded policy |

All three paths leave public Tensors unchanged. They do not add a gradient field, a recording
tape, or a `Tensor.backward()` method.

## Prerequisites and inputs

Start with an open Engine composition and Tensor expressions whose selected derivative routes are
supported. Targets are always explicit: Synaptik does not infer which Tensors should receive
gradients. Every target list is non-empty, ordered, unique by exact Tensor object identity, and
drawn from the complete original forward expression inventory.

The snippets below are conceptual API-shape examples. They assume that the shown Tensors have
compatible descriptors and live input storage and that `engine` or `advanced` is an open Engine.
See the [ordinary and advanced compile reference](../api/compile-api.md#current-ordinary-and-advanced-engine-compile-boundaries)
for complete lifecycle ownership and current execution restrictions.

## Compute a scalar objective and first gradients

For one scalar floating objective, use the ordinary one-shot convenience:

```java
ScalarObjectiveBackwardResult result =
        engine.backward(loss, List.of(weight, bias), maximumTotalBytes);
```

The call freshly compiles, prepares, runs, copies the objective first and the gradients in target
order, then cleans up temporary execution state. The returned values are detached and remain
readable after Engine and caller storage closure. The byte limit applies to the combined returned
payloads.

This method always constructs one reverse-mode stage. Its only output is `loss`, its seed is
absent, `createGraph` is false, and its disconnected policy is `ERROR`. The compiler turns the
absent seed into an exact typed one only if `loss` is scalar, floating, and gradient-eligible.
Use another entry point when you need an explicit seed, several outputs, a zero for disconnected
targets, or a second stage.

## Compile an explicitly seeded first-order graph

The ordinary compile overload aligns one explicit seed with each forward output:

```java
CompiledGraph compiled = engine.compile(
        List.of(vectorOutput),
        List.of(cotangentSeed),
        List.of(input));
```

This constructs the vector-Jacobian product selected by `cotangentSeed`. The seed must match the
output's exact Shape and floating data type and must not request gradients. This overload still
uses one stage, `createGraph == false`, and `DisconnectedPolicy.ERROR`; it does not expose the
second-stage or disconnected-zero policy.

The result is an owner-bound reusable compile handle, not computed gradient bytes. Prepare and
run it through the ordinary Engine lifecycle, then inspect its forward and gradient publication
occurrences. A successful compile does not by itself promise that the current CPU-only
composition can prepare or execute every accepted graph.

## Build a bounded functional request

`FunctionalGradientRequest` contains exactly one or two ordered reverse-mode stages:

- Each stage has non-empty ordered output references, one optional seed position per output, and
  a non-empty ordered target list.
- Stage one selects exact Tensors from the requested forward-output boundary with
  `ForwardTensorReference`.
- Stage two may select only generated first-stage gradients with
  `FirstStageGradientReference(targetIndex)`.
- One stage requires `createGraph == false`. Two stages require `true` for stage one and `false`
  for stage two.
- An absent seed means an exact typed one only for an eligible scalar output. A present seed must
  match the output's exact Shape and floating type and must not request gradients.
- `DisconnectedPolicy.ERROR` rejects a target without a differentiable route.
  `DisconnectedPolicy.ZERO` returns an ordinary exact typed zero expression; several target
  roles may share that same value.

`createGraph` retains first-stage formulas only for the immediate second stage in the same
compile. It does not open a persistent derivative recording scope or authorize arbitrary nesting.

## Conceptual example: compile a Hessian-vector product

### Goal and inputs

Assume `x` is a floating Tensor of Shape `[2]`, `loss = sum(x * x)` is scalar, and `vector` is a
non-gradient Tensor of exact Shape `[2]` and the same floating type as the first-stage gradient.
The goal is to retain the first derivative and construct one second-stage vector-Jacobian product.

### Meaningful steps

The following is conceptual Java because Tensor storage and Engine composition setup are omitted:

```java
var first = new FunctionalGradientRequest.Stage(
        List.of(new FunctionalGradientRequest.ForwardTensorReference(loss)),
        List.of(Optional.empty()),
        List.of(x),
        true,
        FunctionalGradientRequest.DisconnectedPolicy.ERROR);

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

Stage one uses the eligible scalar default seed and produces the gradient selected by target index
zero. Stage two uses `vector` as the cotangent for that exact first-stage result. The compiler
preflights both selected routes, constructs their formulas with ordinary Tensor operations, and
captures the forward output and both gradient roots together once.

### Result and interpretation

Successful compilation returns an opaque `AdvancedCompiledGraph`. The underlying compiler
artifacts retain one order-one gradient binding followed by one order-two binding, both for `x`.
Per-node derivative metadata uses order zero for original forward producers, one for producers
first owned by stage one, and two for producers first owned by stage two. Orders one and two both
remain graph phase `BACKWARD`.

For this expression, the second binding represents the Hessian-vector product selected by
`vector`. The example proves bounded graph construction and ordering only. It does not prove that
a particular backend can prepare or numerically execute the graph, and it does not create a
third-stage derivative chain.

If `vector` has the wrong Shape or floating type, or requests gradients, compilation rejects the
request before derivative formulas are allocated.

## Publication and shared values

Gradient publication bindings are ordered first by derivative order and then by target index
within that stage. Each binding retains the target identity and final gradient graph value.
Bindings remain target-distinct even when disconnected-zero handling or equivalent formulas make
several targets share one value. The graph output boundary starts with the ordered forward values
and then appends each previously unseen gradient value once in binding order.

## Common errors

| Symptom | Likely cause | Correction |
|---|---|---|
| An absent seed is rejected | The selected output is not scalar, floating, and gradient-eligible. | Supply a matching explicit seed, or select an eligible scalar output. |
| An explicit seed is rejected | Its Shape or floating type differs from the output, or it requests gradients. | Use the exact output Shape and type with gradient eligibility disabled. |
| A target is rejected as disconnected | The selected outputs have no differentiable route to that target under `ERROR`. | Select a connected target or use the advanced request with `ZERO` when a typed zero is intended. |
| A public Tensor method exists but autograd rejects the route | Expression construction does not guarantee a derivative rule for every operation, attribute, or role. | Check the maintained [Compiler derivative matrix](../api/compile-api.md#current-package-private-pre-capture-autograd). |
| A two-stage request is rejected structurally | Stage one does not use `createGraph == true`, stage two uses it, or stage two references a forward Tensor. | Use `true` only on stage one and select stage-two outputs by first-stage target index. |
| Compilation succeeds but preparation fails | Current backend preparation supports a narrower executable subset than compiler graph construction. | Treat compile and prepare as separate boundaries and stay within the documented backend subset. |

## Limitations

The current boundary has no third reverse stage, derivative order above two, arbitrary nested
differentiation, persistent or runtime tape, mutable Tensor gradient state, no-argument backward,
`Tensor.backward()`, optimizer update, or training session. `TRAINING_STEP` currently constructs
the same bounded derivative graph as `FORWARD_AND_BACKWARD`; it does not add an optimizer update.

Derivative coverage remains fail-closed and operation-specific. The maintained Compile API lists
the current formula matrix and exclusions; this guide does not duplicate that inventory or imply
numerical or backend coverage from formula construction alone.

## Related documentation

- [Compile API and functional-gradient examples](../api/compile-api.md#functional-gradient-examples)
- [Compiler-owned automatic differentiation contract](../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)
- [Training graph explanation](../architecture/training-graph.md)
- [Public API lifecycle](../api/public-api.md#current-ordinary-and-advanced-cpu-lifecycle)
- [Tensor API](../api/tensor-api.md)
- [Training API status](../api/training-api.md)
