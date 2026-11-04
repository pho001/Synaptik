# 2. Give data shape and compose expressions

## Outcome and prerequisites

Import two Java floats as a batch, distinguish stored data from an unevaluated expression,
compose a Model's cold forward output, and reject a shape that has the right number of values
but the wrong axes. Complete [setup](../examples/README.md#build-the-complete-project) and the
[first-model chapter](first-model.md); commands run from the example project with JDK 26 and
`LEARNING_CP` configured. Each mode still starts independently.

## Data is more than an array

A **tensor** describes an ordered multidimensional numerical value. Its **shape** lists axis
sizes; **rank** is the number of axes. `[2,1]` has rank two and two elements. A scalar has rank
zero, no axes, and one element. Neither a shape nor a descriptor allocates data by itself.

For this model axis zero means samples and the last axis means features. The Java flat array
`{1,3}` becomes logical rows `[1]`, `[3]`, not the single row `[1,3]`. FLOAT32 means each number
uses a 32-bit floating representation. A **layout** describes how logical coordinates map to
storage; canonical contiguous layout here has zero offset, no view and ordinary row-major
ordering. A full **descriptor** combines dtype, shape, optional layout and gradient eligibility.

The `matrix`/`descriptor` helpers in
[LearningJourney.java](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java)
only spell out those public types; they do not add a data framework. `fromFlatArray` copies the
source array, requires exactly the logical number of elements and a matching dtype/resolved
dense layout. Modifying the original Java array afterward does not modify the stored Tensor.

There is no loader in this experiment: input and target are standalone stored data. Larger
applications may load/batch data through separate data facilities before supplying exact Tensor
values. The `data`, `text` and ONNX extensions are not required by this path.

## Construction is not execution

Read the source's `tensors` method. `x.linear(w)` constructs an expression with provenance:
it records how a result depends on the inputs. It does not multiply their values now and does
not fill result storage. `Engine.compute` is the explicit execution boundary for this standalone
expression and returns a detached output. Here the same zero weight gives `[0,0]`.

The later `model.forward(prototype).relu()` demonstrates **composition**: use the Model as one
part of a larger expression. The prototype is storage-free FLOAT32 `[2,1]` with the exact Model
input descriptor. Cold `forward` applies the owned chain to that expression input; `relu`
records an elementwise operation that keeps positive numbers and maps nonpositive numbers to
zero. The mode checks that this combined result has no computed storage. It does not execute
this extra composed graph or claim it trained another model. The canonical learning rule stays
one bias-free Linear layer.

Conceptual pseudocode, not API calls:

```text
stored x [2,1] + stored w [1,1]
    -> describe linear(x,w) -> expression p [2,1], no data yet
    -> Engine execution    -> detached numerical prediction [0,0]
```

Use `Model.compute` for the owned model's cached execution; use cold `forward` to describe
further computation. Calling `forward` on every batch is not how to reuse an executable.

## Run the data/composition experiment

```bash
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney tensors
```

Expected application stdout (source-derived, not runtime evidence):

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
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney invalid-shape
```

Expected application stdout:

```text
== invalid-shape ==
invalid-shape.compute=IllegalArgumentException
invalid-shape.valid-after-rejection=[0.0, 0.0]
```

This calls public `Model.compute` with FLOAT32 `[1,2]`, catches its `IllegalArgumentException`,
and fails with `AssertionError` if the call is accepted. It then supplies a valid `[2,1]` input
to the same Model. The second line demonstrates recovery from invalid caller input; it does
not promise recovery from arbitrary execution failures. Exception text is deliberately not
part of expected stdout.

Model inputs must match the **entire** declared descriptor, not only element count: dtype,
shape, layout presence/geometry and `requiresGrad=false`. An unresolved/absent layout is not
silently the canonical layout. There is no automatic reshape or changed-batch specialization
in this owned Model path.

## Useful variation and relevant errors

Change only the imported valid values to `[2,4]` while preserving `[2,1]`. At initial `w=0` the
prediction is still `[0,0]`; after training, [reuse](reusable-execution.md) will show `[4,8]`.
The fixed printed input label and assertions must reflect any experiment you keep. Rebuild with
the [setup command](../examples/README.md#build-the-complete-project).

A three-element array for shape `[2,1]` fails import with `IllegalArgumentException` before
execution. Fix the data count or declare the intended shape; do not pad or silently drop values.
A storage-free prototype is suitable for composition, but not for a compute input that must
contain live stored values.

## Source and next step

See [`TensorDescriptor`](../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorDescriptor.java),
[`TensorFactory`](../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java),
[`Tensor`](../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java) and
[`Model.forward/compute`](../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java).
For the separation of tensor meaning from execution, use the
[foundational contract](../architecture/contracts/foundational-modules.md#modulesmodel).

Next: [turn prediction errors into one loss and its derivative](loss-and-gradients.md).
