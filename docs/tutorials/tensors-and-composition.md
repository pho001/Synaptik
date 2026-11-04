# 2. Give data [shape](../glossary.md#shape) and compose expressions

## Outcome and prerequisites

Import two Java floats as a [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters), distinguish stored data from an unevaluated expression,
compose a [Model](../glossary.md#term-model)'s cold forward output, and reject a [shape](../glossary.md#shape) that has the right number of values
but the wrong [axes](../glossary.md#term-axis). Complete [setup](../examples/README.md#build-the-complete-project) and the
[first-model chapter](first-model.md); commands run from the example project with [JDK](../glossary.md#term-java-development-kit-jdk) 26 and
`LEARNING_CP` configured. Each mode still starts independently.

## Data is more than an array

A **[tensor](../glossary.md#tensor)** describes an ordered multidimensional numerical value. Its **[shape](../glossary.md#shape)** lists [axis](../glossary.md#term-axis)
sizes; **[rank](../glossary.md#term-rank)** is the number of axes. `[2,1]` has rank two and two [elements](../glossary.md#term-element). A [scalar](../glossary.md#term-scalar) has rank
zero, no axes, and one element. Neither a shape nor a [descriptor](../glossary.md#term-descriptor) allocates data by itself.

For this [model](../glossary.md#term-model) axis zero means [samples](../glossary.md#term-sample-machine-learning-and-hyperparameters) and the last axis means [features](../glossary.md#term-feature). The Java flat array
`{1,3}` becomes logical rows `[1]`, `[3]`, not the single row `[1,3]`. [FLOAT32](../glossary.md#term-float32) means each number
uses a 32-bit floating representation. A **[layout](../glossary.md#layout)** describes how logical [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix) map to
storage; [canonical](../glossary.md#term-canonical-form) [contiguous](../glossary.md#term-contiguous) layout here has zero offset, no [view](../glossary.md#term-view-meanings) and ordinary [row-major](../glossary.md#term-row-major)
ordering. A full **descriptor** combines [dtype](../glossary.md#data-type--datatype), shape, optional layout and [gradient](../glossary.md#term-gradient) eligibility.

An **axis** is one coordinate direction; in `[2,1]`, the first coordinate selects a sample and the
second selects its sole feature. **Row-major** means all values of one row come before the next
row. **Contiguous** means the values have no storage gaps; an **offset** would skip a prefix of
storage, and a **view** would describe an existing store with another geometry. A **dtype** is
the element's [data type](../glossary.md#data-type--datatype), here FLOAT32. These facts explain why `[1,2]` cannot replace `[2,1]`.
For combining compatible but differently shaped operands, use the existing
[broadcast illustration](../explanation/tensors-and-gradients.md); [broadcasting](../glossary.md#broadcasting) is not a general
permission to reshape a Model input.

The [`matrix`](../glossary.md#term-matrix-meanings)/`descriptor` helpers in
[LearningJourney.java](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java)
only spell out those public types; they do not add a data framework. The application calls
`TensorFactory.fromFlatArray` with a descriptor and Java array. The factory copies that array
into independent heap storage and returns a Tensor associated with the copy. It requires
exactly the logical number of elements and a matching dtype/resolved dense layout.
The application may subsequently change its original array without changing the imported Tensor.

There is no loader in this experiment: input and [target](../glossary.md#term-target) are standalone stored data. Larger
applications may load/[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) data through separate data facilities before supplying exact Tensor
values. The `data`, `text` and [ONNX](../glossary.md#term-onnx) extensions are not required by this path.

ONNX expands to [Open Neural Network Exchange](../glossary.md#cpu-gpu-host-and-device), a model-interchange format. Its mention here
identifies a separate extension, not another [dependency](../glossary.md#term-dependency) or import step for these two samples.

## Construction is not execution

Read the source's `tensors` method. The application calls `x.linear(w)` to construct an
expression with [provenance](../glossary.md#provenance): it records the operation and exact input
Tensors, rather than multiplying their values or filling result storage now. The application
then calls `Engine.compute(expression)`. Engine arranges compilation, preparation and execution,
borrowing the live stored inputs during the call, and returns independent copied output bytes.
Here the same zero weight produces $[0,0]$. See
[fresh-value binding](../user-guide/running-models.md#run-fresh-values-in-compiled-order)
when the same description should run with different stored inputs.

The later `model.forward(prototype).relu()` demonstrates **composition**: use the [Model](../glossary.md#term-model) as one
part of a larger expression. The [prototype](../glossary.md#term-prototype) is storage-free [FLOAT32](../glossary.md#term-float32) `[2,1]` with the exact Model
input [descriptor](../glossary.md#term-descriptor). Cold `forward` applies the owned chain to that expression input; `relu`
records an [elementwise operation](../glossary.md#elementwise-unary-binary-and-interval-vocabulary) that keeps positive numbers and maps nonpositive numbers to
zero. The mode checks that this combined result has no computed storage. It does not execute
this extra composed [graph](../glossary.md#graph) or claim it trained another model. The [canonical](../glossary.md#term-canonical-form) learning rule stays
one [bias](../glossary.md#term-bias)-free [Linear](../glossary.md#term-linear-meanings) layer.

**Cold** here means describing computation before execution resources are prepared.
A **prototype** states the required type and geometry without supplying data. **Provenance**
records which [operation](../glossary.md#operation) and inputs produced an expression. Thus the prototype, the expression
and its later numerical result serve three different purposes even though all describe tensors.

Conceptual [pseudocode](../glossary.md#term-pseudocode), not [API](../glossary.md#term-application-programming-interface-api) calls:

```text
stored x [2,1] + stored w [1,1]
    -> describe linear(x,w) -> expression p [2,1], no data yet
    -> Engine execution    -> detached numerical prediction [0,0]
```

Use `Model.compute` for the [owned model](../glossary.md#model-module-buffer-and-scan-distinctions)'s cached execution; use cold `forward` to describe
further computation. Calling `forward` on every [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) is not how to reuse an executable.

## Run the data/composition experiment

```bash
# Run construction and execution side by side: only the explicit compute produces values.
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney tensors
```

Expected application stdout, checked against the installed private candidate:

```text
== tensors ==
tensors.input=FLOAT32 [2,1], values=[1.0, 3.0]
tensors.array-copy=ok
tensors.expression-has-storage=false
tensors.prediction=[0.0, 0.0]
tensors.forward-composition-has-storage=false
```

The two `false` values distinguish constructed expressions from evaluated data. They do not
mean the operations failed or that result values are known without execution.

## Run the actual error path

```bash
# A matching value count cannot compensate for swapped sample and feature axes.
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney invalid-shape
```

Expected application stdout:

```text
== invalid-shape ==
invalid-shape.compute=IllegalArgumentException
invalid-shape.valid-after-rejection=[0.0, 0.0]
```

This calls public `Model.compute` with [FLOAT32](../glossary.md#term-float32) `[1,2]`, catches its `IllegalArgumentException`,
and fails with `AssertionError` if the call is accepted. It then supplies a valid `[2,1]` input
to the same [Model](../glossary.md#term-model). The second line demonstrates recovery from invalid caller input; it does
not promise recovery from arbitrary execution failures. Exception text is deliberately not
part of expected stdout.

Model inputs must match the **entire** declared [descriptor](../glossary.md#term-descriptor), not only [element](../glossary.md#term-element) count: [dtype](../glossary.md#data-type--datatype),
[shape](../glossary.md#shape), [layout](../glossary.md#layout) presence/geometry and `requiresGrad=false`. An unresolved/absent layout is not
silently the [canonical](../glossary.md#term-canonical-form) layout. There is no automatic reshape or changed-[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) [specialization](../glossary.md#term-specialization)
in this [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions) path.

## Useful variation and relevant errors

Change only the imported valid values to `[2,4]` while preserving `[2,1]`. At initial `w=0` the
[prediction](../glossary.md#term-prediction) is still `[0,0]`; after training, [reuse](reusable-execution.md) will show `[4,8]`.
The fixed printed input label and assertions must reflect any experiment you keep. Rebuild with
the [setup command](../examples/README.md#build-the-complete-project).

A three-[element](../glossary.md#term-element) array for [shape](../glossary.md#shape) `[2,1]` fails import with `IllegalArgumentException` before
execution. Fix the data count or declare the intended shape; do not pad or silently drop values.
A [storage-free prototype](../glossary.md#leaf-prototype-and-constant) is suitable for composition, but not for a compute input that must
contain live stored values.

## Source and next step

See [`TensorDescriptor`](../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java),
[`TensorFactory`](../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java),
[`Tensor`](../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java) and
[`Model.forward/compute`](../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java).
For the separation of [tensor](../glossary.md#tensor) meaning from execution, use the
[foundational contract](../architecture/contracts/foundational-modules.md#modulesmodel).

Next: [turn prediction errors into one loss and its derivative](loss-and-gradients.md)<span data-glossary-navigation="true"> ([prediction glossary](../glossary.md#term-prediction); [loss glossary](../glossary.md#loss); [derivative glossary](../glossary.md#term-derivative))</span>.
