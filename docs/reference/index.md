# Reference: find the contract for your task

Use this reference when you know what you want to do and need exact semantics, [ownership](../glossary.md#term-ownership-meanings), [validation](../glossary.md#term-validation) or supported boundaries. For a first working program, start with the [learning path](../tutorials/index.md) and [complete examples](../examples/README.md). For how the parts fit together, read [the framework explanation](../explanation/framework.md).

A [Tensor](../glossary.md#tensor) expression describes a computation; [compilation](../glossary.md#compiler-engine-and-execution-stage-names) produces a logical recipe; preparation acquires reusable executable resources; a [run](../glossary.md#term-run) binds values and returns a result [lease](../glossary.md#term-lease). A [detached](../glossary.md#detached-model-results) [host](../glossary.md#term-host) value is a copied result that can outlive that lease. These are different objects and lifetimes, not interchangeable names for an array.

In this reference, a **contract** is the behavior and constraints a caller relies on, including
who must keep resources alive and who closes them. **[Metadata](../glossary.md#term-metadata)** describes a value, such as its
[shape](../glossary.md#shape) and [element](../glossary.md#term-element) type; it is not the value's numeric contents. The topic pages separate those
descriptions from the storage and work needed to produce results.

## Choose by goal

| I want to… | Start here | Then consult |
|---|---|---|
| Create values or understand an expression | [Tensor families](tensor/index.md) | [Shapes](tensor/shapes.md), [descriptors](tensor/descriptors.md), [borrowed storage](tensor/storage-association.md) |
| Build a small owned [neural network](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) | [Fluent Model builder](nn/model-builder.md) | [Owned Model execution](nn/owned-model.md), [NN contracts](nn/index.md) |
| [Compile](../glossary.md#compile) once and [run](../glossary.md#term-run) fresh inputs | [Engine lifecycle](engine/lifecycle.md) | [Exact bindings and results](runtime/engine-boundary.md), [explicit CPU/Metal composition](engine/composition.md) |
| Train [parameters](../glossary.md#term-parameter) and recover [optimizer state](../glossary.md#term-optimizer-state) | [Training sessions](training/sessions.md) | [Optimizers and transactions](training/optimizers.md), [caller-coordinated recovery](training/recovery.md) |
| Save or load a [graph](../glossary.md#graph) or [Model](../glossary.md#term-model) | [Checkpoint choices](checkpoint/index.md) | [Graph sources](checkpoint/graph.md), [format versions](checkpoint/graph-formats.md), [whole-model recovery](checkpoint/models.md) |
| Pad sequences or encode text | [Sequence batches](data/sequences.md) | [Frozen tokenization](text/tokenization.md), [runtime-length NN composition](nn/recurrent.md) |
| Understand logical [compilation](../glossary.md#compiler-engine-and-execution-stage-names) outputs | [Compiler boundaries](compiler/index.md) | [Artifacts](compiler/artifacts.md), [functional gradients](compiler/autograd.md), [batch binding](compiler/batch-binding.md) |
| Integrate below the application [facade](../glossary.md#term-facade) | [Runtime mental model](runtime/index.md) | [Prepare](runtime/prepare.md), [representation ownership](runtime/representations.md), [run orchestration](runtime/orchestration.md) |
| Look up exact Java [signatures](../glossary.md#term-signature) | [Generated Javadoc access](javadoc.md) | Build-required local links and extraction instructions for all 18 supplied [Javadoc](../glossary.md#term-javadoc) JARs |
| Check whether a [backend](../glossary.md#backend) accepts my [occurrence](../glossary.md#term-occurrence) | [Current production matrix](../backend-guide/production-support.md) | [Public API compatibility](compatibility.md); Model-valid does not mean backend-supported |

## Follow one small computation

The shared learning example has two examples, each with one feature: FLOAT32 input `[2,1]`
containing $x_1=1$, $x_2=3$, and target values $y_1=2$, $y_2=6$. A supplied trainable scalar
weight starts at $w=0$ and there is no bias. [Linear projection](tensor/matmul-linear.md) describes
predictions $\widehat y_i=x_iw$ for $i\in\{1,2\}$; [arithmetic](tensor/arithmetic.md) and
[reductions](tensor/reductions.md) describe the squared-error sum

$$
L(w)=\sum_{i=1}^{2}(x_iw-y_i)^2,\qquad
\frac{dL}{dw}=2\sum_{i=1}^{2}x_i(x_iw-y_i).
$$

At $w=0$, $L=4+36=40$ and $dL/dw=-40$. [SGD](training/optimizers.md) with learning rate
$\eta=0.05$ computes $w_{\mathrm{new}}=0-0.05(-40)=2$, yielding `[2,6]`; fresh input `[2,4]`
then yields `[4,8]`. These real-number calculations explain the example; execution follows
FLOAT32 rounding rather than arbitrary algebraic reassociation. The Model acquires a private
copy of the application's supplied weight, and the step reports the **pre-update** loss $40$.

These are exact teaching values, not a convergence or performance claim. The [complete learning program](../examples/README.md) supplies setup, execution and interpretation. [General Scan](../glossary.md#general-differentiable-scan), [tokenization](../glossary.md#frozen-whitespace-tokenizer), [convolution](../glossary.md#term-channels-first-convolution-layers), [checkpoint](../glossary.md#term-checkpoint) formats, [backend](../glossary.md#backend) integration and [runtime](../glossary.md#term-runtime) service-provider interfaces are not needed to understand that first step.

## How to read precise details

- **Application [API](../glossary.md#term-application-programming-interface-api):** normally use owned [`Model`](../glossary.md#term-model) or ordinary [`Engine`](../glossary.md#term-engine). Public visibility of a lower-level collaboration does not make it a recommended application [facade](../glossary.md#term-facade).
- **Implementation detail:** package-private [Compiler](../glossary.md#term-compiler) and [backend](../glossary.md#backend) sketches explain mechanisms, not a supported extension API.
- **Independent [qualification](../glossary.md#term-qualification) gates:** source construction, differentiation, execution, [cold reconstruction](../glossary.md#term-cold-reconstruction) and [persistence](../glossary.md#term-persistence) have separate [admission](../glossary.md#term-admission) rules. A historical numerical slice is not today's complete backend [matrix](../glossary.md#term-matrix-meanings).
- **Authority:** [the architecture root](../../ARCHITECTURE.md#authority-incorporation-and-precedence) and its six incorporated contracts are [normative](../glossary.md#term-normative). These pages explain existing behavior; no API, numerical, [ownership](../glossary.md#term-ownership-meanings) or [release](../glossary.md#term-release) policy changes are introduced by their reorganization.
- **Evidence:** 55 retained complete Java classes compiled against installed public artifacts;
  all 53 with main methods ran, and all 47 literal stdout blocks matched. The learning project
  independently matched its complete transcript. Conceptual fragments and package-local sketches
  are not executable application promises; these checks do not qualify all backend geometries.

## Existing links

The former `docs/api/*.md` pages are migration portals. Every original heading anchor and explicit HTML ID has a direct link to its [canonical](../glossary.md#term-canonical-form) topic. Superseded [qualification](../glossary.md#term-qualification) inventories are archived rather than used as current task guidance; detailed current contracts remain in the topic families above.
