# Reference: find the contract for your task

Use this reference when you know what you want to do and need exact semantics, ownership, validation or supported boundaries. For a first working program, start with the [learning path](../tutorials/index.md) and [complete examples](../examples/README.md). For how the parts fit together, read [the framework explanation](../explanation/framework.md).

A Tensor expression describes a computation; compilation produces a logical recipe; preparation acquires reusable executable resources; a run binds values and returns a result lease. A detached host value is a copied result that can outlive that lease. These are different objects and lifetimes, not interchangeable names for an array.

## Choose by goal

| I want to… | Start here | Then consult |
|---|---|---|
| Create values or understand an expression | [Tensor families](tensor/index.md) | [Shapes](tensor/shapes.md), [descriptors](tensor/descriptors.md), [borrowed storage](tensor/storage-association.md) |
| Build a small owned neural network | [Fluent Model builder](nn/model-builder.md) | [Owned Model execution](nn/owned-model.md), [NN contracts](nn/index.md) |
| Compile once and run fresh inputs | [Engine lifecycle](engine/lifecycle.md) | [Exact bindings and results](runtime/engine-boundary.md), [explicit CPU/Metal composition](engine/composition.md) |
| Train parameters and recover optimizer state | [Training sessions](training/sessions.md) | [Optimizers and transactions](training/optimizers.md), [caller-coordinated recovery](training/recovery.md) |
| Save or load a graph or Model | [Checkpoint choices](checkpoint/index.md) | [Graph sources](checkpoint/graph.md), [format versions](checkpoint/graph-formats.md), [whole-model recovery](checkpoint/models.md) |
| Pad sequences or encode text | [Sequence batches](data/sequences.md) | [Frozen tokenization](text/tokenization.md), [runtime-length NN composition](nn/recurrent.md) |
| Understand logical compilation outputs | [Compiler boundaries](compiler/index.md) | [Artifacts](compiler/artifacts.md), [functional gradients](compiler/autograd.md), [batch binding](compiler/batch-binding.md) |
| Integrate below the application facade | [Runtime mental model](runtime/index.md) | [Prepare](runtime/prepare.md), [representation ownership](runtime/representations.md), [run orchestration](runtime/orchestration.md) |
| Look up exact Java signatures | [Generated Javadoc access](javadoc.md) | Build-required local links and extraction instructions for all 18 supplied Javadoc JARs |
| Check whether a backend accepts my occurrence | [Current production matrix](../backend-guide/production-support.md) | [Public API compatibility](compatibility.md); Model-valid does not mean backend-supported |

## Follow one small computation

The shared learning example uses FLOAT32 input `[2,1]` with values `[1,3]`, a supplied gradient-eligible weight `[1,1]` initialized to `0`, no bias, and target `[2,6]`. [Linear projection](tensor/matmul-linear.md) describes `input @ transpose(weight)`. [Arithmetic](tensor/arithmetic.md) and [reductions](tensor/reductions.md) describe `sum((prediction-target)^2)`. Initially the prediction is `[0,0]`, the objective is `40`, and the weight gradient is `-40`. [SGD](training/optimizers.md) at learning rate `0.05` updates the privately owned weight to `2`, yielding `[2,6]`; fresh input `[2,4]` yields `[4,8]`.

These are exact teaching values, not a convergence or performance claim. The [complete learning program](../examples/README.md) supplies setup, execution and interpretation. General Scan, tokenization, convolution, checkpoint formats, backend integration and runtime service-provider interfaces are not needed to understand that first step.

## How to read precise details

- **Application API:** normally use owned `Model` or ordinary `Engine`. Public visibility of a lower-level collaboration does not make it a recommended application facade.
- **Implementation detail:** package-private Compiler and backend sketches explain mechanisms, not a supported extension API.
- **Independent qualification gates:** source construction, differentiation, execution, cold reconstruction and persistence have separate admission rules. A historical numerical slice is not today's complete backend matrix.
- **Authority:** [the architecture root](../../ARCHITECTURE.md#authority-incorporation-and-precedence) and its six incorporated contracts are normative. These pages explain existing behavior; no API, numerical, ownership or release policy changes are introduced by their reorganization.
- **Evidence:** 55 retained complete Java classes compiled against installed public artifacts;
  all 53 with main methods ran, and all 47 literal stdout blocks matched. The learning project
  independently matched its complete transcript. Conceptual fragments and package-local sketches
  are not executable application promises; these checks do not qualify all backend geometries.

## Existing links

The former `docs/api/*.md` pages are migration portals. Every original heading anchor and explicit HTML ID has a direct link to its canonical topic. Superseded qualification inventories are archived rather than used as current task guidance; detailed current contracts remain in the topic families above.
