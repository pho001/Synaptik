# [Tensor](../glossary.md#tensor) geometry and [gradients](../glossary.md#term-gradient)

> This explanation follows the [Model semantic contract](../architecture/contracts/foundational-modules.md#modulesmodel)<span data-glossary-navigation="true"> ([Model glossary](../glossary.md#term-model))</span>
> and [Compiler AD contract](../architecture/contracts/compiler-autograd.md#compiler-owned-automatic-differentiation)<span data-glossary-navigation="true"> ([Compiler glossary](../glossary.md#term-compiler); [AD glossary](../glossary.md#autograd))</span>.
> It adds a worked mental model, not different arithmetic rules.<span data-glossary-navigation="true"> ([mental model glossary](../glossary.md#term-mental-model))</span>

## [Shape](../glossary.md#shape) says what an index means

A [tensor](../glossary.md#tensor) is typed, indexed data. Its **shape** lists [axis](../glossary.md#term-axis) extents; its **[rank](../glossary.md#term-rank)** is the number of
axes, not its number of [elements](../glossary.md#term-element). The learning input `[2,1]` has rank two and two [FLOAT32](../glossary.md#term-float32) elements:
two [samples](../glossary.md#term-sample-machine-learning-and-hyperparameters), one [feature](../glossary.md#term-feature) per sample. The [weight](../glossary.md#term-weight) `[1,1]` maps that feature to one output.

[Matrix multiplication](../glossary.md#matrix-multiplication--matmul) contracts the input-feature axis. The final output keeps the sample axis:

```text
X: [samples=2, input_features=1]
W: [output_features=1, input_features=1]
X @ transpose(W): [samples=2, output_features=1]
```

This is [dimension](../glossary.md#dimension) reasoning, not executable Java. The complete program imports and constructs
these values in [LearningJourney](../examples/README.md). The low-level [Model](../glossary.md#term-model) supports some named
symbolic dimensions, but the tutorial's owned builder [signature](../glossary.md#term-signature) is deliberately fully static.
Changing its [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) from two to three is not supported by that same Model.

## [Layout](../glossary.md#layout) says which stored [element](../glossary.md#term-element) an index references

A **layout** maps logical [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix) to element offsets. For a [canonical](../glossary.md#term-canonical-form) [contiguous](../glossary.md#term-contiguous) `[2,3]`
[FLOAT32](../glossary.md#term-float32) [matrix](../glossary.md#term-matrix-meanings), row [stride](../glossary.md#term-stride) is three elements and column stride is one. Element `[1,2]` is at
`1*3 + 2*1 = 5`, hence byte offset `5*4 = 20`. An offset is not a [Tensor](../glossary.md#tensor) [identity](../glossary.md#term-identity-meanings).

A [transpose](../glossary.md#term-transpose) can describe different index geometry without changing the meaning of the underlying
values. Whether execution needs a physical copy is a separate preparation/[backend](../glossary.md#backend) question.
A resolved layout does not allocate storage, and a [view](../glossary.md#term-view-meanings) does not own or prolong an [Arena](../glossary.md#term-arena).
See [layout reference](../reference/tensor/index.md) and [ownership](ownership.md)<span data-glossary-navigation="true"> ([ownership glossary](../glossary.md#term-ownership-meanings))</span>.

Read this table as two ways to name cells in the **same conceptual six-element storage**, not as
an executed transpose or an automatically shared Tensor storage association. The transpose has a
different [shape](../glossary.md#shape) and strides, while a hypothetical contiguous copy would store the logical result
in a new order.

| Description | Shape | Strides in elements | Logical `[0,1]` reads storage position |
|---|---|---|---|
| Original matrix | `[2,3]` | `[3,1]` | `1` |
| Transposed mapping | `[3,2]` | `[1,3]` | `3` |
| Separate contiguous realization of the transpose | `[3,2]` | `[2,1]` | `1` in the new storage |

“Canonical” here means the conventional dense [row-major](../glossary.md#term-row-major) ordering, with the last [axis](../glossary.md#term-axis) changing
fastest. It is a storage-order convention, not a promise that every intermediate is copied.

## [Broadcasting](../glossary.md#broadcasting) expands an indexing rule, not the [dataset](../glossary.md#term-dataset)

Broadcasting aligns [shapes](../glossary.md#shape) from the right. Equal extents agree; an [extent](../glossary.md#term-extent) of one can be reused
along the other operand's matching [axis](../glossary.md#term-axis). For example, `[2,1] + [1,3]` produces `[2,3]`.
The diagram shows a separate [FLOAT32](../glossary.md#term-float32) addition example, not an extra [bias](../glossary.md#term-bias) in the no-bias learning [Model](../glossary.md#term-model).

```mermaid
flowchart LR
    accTitle: Broadcasting two rows against three column offsets
    accDescr: Shape two by one with values one and three adds shape one by three containing ten, twenty and thirty; output rows are eleven twenty-one thirty-one and thirteen twenty-three thirty-three.
    X["X [2,1]<br/>1<br/>3"] --> ADD["Elementwise addition"]
    B["Offsets [1,3]<br/>10, 20, 30"] --> ADD
    ADD --> Y["Output [2,3]<br/>11, 21, 31<br/>13, 23, 33"]
```

The same source [element](../glossary.md#term-element) may be read at several output [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix). Broadcasting by itself does
not promise a materialized repeated input array. `[2,2]` and `[2,3]` cannot broadcast because the
rightmost extents are unequal and neither is one. [Matrix multiplication](../glossary.md#matrix-multiplication--matmul)'s contracted axes are a
different rule: adding a broadcast axis does not repair an invalid [matrix](../glossary.md#term-matrix-meanings) product.

## The first training step

For inputs `X=[1,3]`, [targets](../glossary.md#term-target) `T=[2,6]` and [scalar](../glossary.md#term-scalar) [weight](../glossary.md#term-weight) `w=0`, each [prediction](../glossary.md#term-prediction) is `y_i=x_i*w`.
The [objective](../glossary.md#term-objective) is the **sum**, not mean, of squared errors. The table follows one forward and
reverse pass, before the [optimizer](../glossary.md#term-optimizer) mutates the weight.

| [Sample](../glossary.md#term-sample-machine-learning-and-hyperparameters) | Input x | Target t | Prediction y | Error y−t | Squared error | Contribution 2x(y−t) to dL/dw |
|---|---:|---:|---:|---:|---:|---:|
| First | 1 | 2 | 0 | -2 | 4 | -4 |
| Second | 3 | 6 | 0 | -6 | 36 | -36 |
| Aggregate | | | | | **40** | **-40** |

A [gradient](../glossary.md#term-gradient) is a local [sensitivity](../glossary.md#term-sensitivity). [Reverse-mode](../glossary.md#primal-adjoint-and-differentiation-seed) [automatic differentiation](../glossary.md#autograd) applies the [chain rule](../glossary.md#term-chain-rule)
from the scalar [loss](../glossary.md#loss) back to requested [parameters](../glossary.md#term-parameter); it is not [finite-difference](../glossary.md#term-finite-difference-gradient-check) probing or a
[backend](../glossary.md#backend)-specific training loop.

```mermaid
flowchart TD
    accTitle: Reverse chain rule for the learning objective
    accDescr: Sensitivity flows from summed squared errors through predictions and then to the weight by multiplying each prediction sensitivity by its input and summing.
    LOSS[Scalar loss: sum of squared errors] --> ERROR[Sensitivity to errors: twice each error]
    ERROR --> PRED[Prediction sensitivity: negative four and negative twelve]
    PRED --> WEIGHT[Weight sensitivity: multiply by each input, then sum]
```

[SGD](../glossary.md#term-stochastic-gradient-descent-sgd) applies `new_weight = old_weight - learning_rate * gradient`. The project's `Sgd(0.05)` gives
`0 - 0.05*(-40) = 2`, so a fresh forward pass produces `[2,6]` and zero squared error on this [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters).
These deliberately chosen numbers teach one update. They do not show that this [learning rate](../glossary.md#term-learning-rate)
converges for arbitrary data, that lower [precision](../glossary.md#term-precision) has the same result, or that fitting one batch
generalizes to unseen data.

With a [mean](../glossary.md#term-mean)-squared objective the initial loss and gradient would both be divided by two. Reusing
the same learning rate would give a different update. If you change the reduction, reason about
that scale instead of assuming that optimizer settings are independent of the objective.

## [Parameters](../glossary.md#term-parameter), data and [evaluation](../glossary.md#term-evaluation) have different roles

The [weight](../glossary.md#term-weight) is [gradient](../glossary.md#term-gradient)-eligible state; input and [target](../glossary.md#term-target) are fresh data, not parameters to update.
The returned `TrainingStep` contains the [objective](../glossary.md#term-objective) evaluated before the update. Use a new compute
(or explicit objective evaluation) to assess the updated [model](../glossary.md#term-model), preferably on separate examples.
The tutorial uses fresh inputs `[2,4]`, whose learned [predictions](../glossary.md#term-prediction) are `[4,8]`.

A Model's train/[eval mode](../glossary.md#inference-and-evaluation) controls mode-sensitive layers; it is not the [optimizer](../glossary.md#term-optimizer) itself.
`configureTraining` and `train(input,target)` do not toggle it. This [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions) [captures](../glossary.md#term-capture-meanings)
evaluation mode: **do not call inherited no-[argument](../glossary.md#term-argument) `model.train()`** to start its optimizer.
Changing a captured mode invalidates its source rather than rebuilding it. Our [Linear](../glossary.md#term-linear-meanings) has no
mode-dependent arithmetic, but its captured-state [guard](../glossary.md#term-guard) still applies; the same distinction is
essential for [Dropout](../glossary.md#dropout--inverted-dropout) and BatchNorm.

Next: [follow these expressions through compilation and execution](framework.md)<span data-glossary-navigation="true"> ([compilation glossary](../glossary.md#compiler-engine-and-execution-stage-names))</span>, or
[run the training chapter](../tutorials/index.md)<span data-glossary-navigation="true"> ([run glossary](../glossary.md#term-run))</span>.
