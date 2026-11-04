# 1. Compute with your first [Model](../glossary.md#term-model)

## Outcome and prerequisites

Compute `[0,0]` from input `[1,3]` and a single zero [weight](../glossary.md#term-weight), then read the result after the [Model](../glossary.md#term-model)
closes. You will also see executable checks that the Model copies supplied [parameter](../glossary.md#term-parameter) storage
instead of retaining its caller's [lifetime](../glossary.md#term-lifetime).

Complete the [project setup](../examples/README.md#build-the-complete-project) with [JDK](../glossary.md#term-java-development-kit-jdk) 26, [Maven](../glossary.md#term-maven)
and the supplied candidate [file repository](../glossary.md#term-file-repository). Run the commands below from the example project
with `LEARNING_CP` exported. No previous mode needs to have run.

## Inputs and mechanism

The input is [FLOAT32](../glossary.md#term-float32) `[2,1]`: two [samples](../glossary.md#term-sample-machine-learning-and-hyperparameters), one [feature](../glossary.md#term-feature) each. The supplied [weight](../glossary.md#term-weight) is FLOAT32
`[1,1]`, value `0`, with `requiresGrad=true`. That flag makes the floating [leaf](../glossary.md#term-leaf) eligible for
requested differentiation; it is not a [gradient](../glossary.md#term-gradient) [buffer](../glossary.md#term-buffer) and does not initiate training.

A **feature** is one input measurement: `1` for the first sample and `3` for the second.
A **[parameter](../glossary.md#term-parameter)** is [model](../glossary.md#term-model) state that training may change; here it is the single weight, not either
input number. A **leaf** is a [Tensor](../glossary.md#tensor) with no producing expression. This supplied leaf already has
stored values; other leaves can be storage-free input [prototypes](../glossary.md#term-prototype). Leaf status alone does not
establish the presence of storage.

### Who supplies the weight, and who executes the rule?

The Java application calls `Model.builder().input(...).linear(supplied).build()`.
The builder reads the application's stored weight while its storage is live and creates a
Model with a **private copy**. Here the application is the caller and the builder is the
callee; Engine has not been created and is not the caller at this boundary.

The rule is $p_i=x_iw$ for sample index $i\in\{1,2\}$, input measurements
$x_1=1$, $x_2=3$, weight $w=0$, and predictions $p_1,p_2$. These are dimensionless
teaching numbers. In the general matrix form, $P=XW^{\mathsf T}$, rows of $X$ are
samples and rows of $W$ are output-feature weights; $\mathsf T$ swaps matrix axes.
Here `linear(weight)` has **no [bias](../glossary.md#term-bias)**: there is no extra
adjustable number $b$ in a rule $p_i=x_iw+b$.

Build creates an [evaluation](../glossary.md#term-evaluation)-mode Model but no Engine.
It copies a stored leaf, not an arbitrary weight expression awaiting evaluation. Later the
application calls `model.compute(input)`: Model uses its execution path to calculate values
from the supplied batch and its own weight. The input is borrowed for the synchronous call,
not adopted as Model state; keep its storage live and unchanged until that call returns.

**Evaluation mode** is a [Module](../glossary.md#term-module) behavior setting, not a request to update parameters or assess
[prediction](../glossary.md#term-prediction) quality. This chapter performs [inference](../glossary.md#term-inference): it computes predictions without training.

Read `copiedModel` and `first` in the
[complete Java source](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java).
The important decisions are:

1. The application's `copiedModel` helper opens an [`Arena`](../glossary.md#term-arena),
   which controls the lifetime of one native float holding zero. `TensorFactory.create`
   attaches that storage to `supplied` without copying it. The Tensor has no producing
   expression, so it is a leaf; the application still controls its Arena.
2. The helper calls `build()` before closing that Arena. It checks that the Model's weight
   has different Tensor and storage identities, writes `99` to the original storage, and
   checks that the Model's private value is still zero.
3. The helper closes the Arena before returning the Model to `first`. Computation still
   produces $[1\cdot0,3\cdot0]=[0,0]$, because it reads the private copy.
4. `first` calls `compute`, which lazily obtains execution through the Model's Engine path
   and copies the output into an immutable
   [`HostTensorValue`](../glossary.md#host-tensor-value--hosttensorvalue). Closing the Model
   releases its cached resources, not the independent output bytes.

This is why a tensor object, its storage and a [detached](../glossary.md#detached-model-results) result are not interchangeable:

| Object | Owns/borrows | [Lifetime](../glossary.md#term-lifetime) in this mode |
|---|---|---|
| Supplied Tensor | Borrows caller native storage | Storage ends before compute |
| Owned Model | Private state; lazily owned execution resources | Ends at try-with-resources close |
| `HostTensorValue` | Independent [canonical](../glossary.md#term-canonical-form) output bytes | Usable after Model close; no close method |

**Owned** means the Model is responsible for releasing its private resources. **[Borrowed](../glossary.md#ownership-borrowing-and-lease)** means
another [owner](../glossary.md#term-owner) must keep storage alive; merely retaining a Java Tensor reference cannot keep a
closed native Arena usable. **Detached** means output bytes were copied into an independent
value. Follow the [ownership diagram](../explanation/ownership.md#a-result-is-not-always-an-independent-value)<span data-glossary-navigation="true"> ([ownership glossary](../glossary.md#term-ownership-meanings))</span>
for these different lifetimes. Native storage is memory managed outside the ordinary Java [heap](../glossary.md#term-heap);
the Arena determines when this caller-supplied [native memory](../glossary.md#aliasing-and-arena) stops being usable.

The explicit `99` mutation is a copy-isolation experiment, not a recommended application pattern.
Keep supplied storage live and exclude concurrent mutation during build.

## Run and interpret

```bash
# This process checks both parameter-copy isolation and output survival after Model close.
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney first
```

Expected application stdout, checked against the installed private candidate:

```text
== first ==
supplied-weight.copy-and-close=ok
first.prediction=[0.0, 0.0]
first.after-model-close=[0.0, 0.0]
```

The first marker follows checks and a successful compute after supplied-[owner](../glossary.md#term-owner) close. The final
line is read **after [Model](../glossary.md#term-model) close**, so it checks a different boundary: [detached](../glossary.md#detached-model-results) outputs do not
borrow their originating Model. In this example zero predictions follow from multiplying the
nonzero inputs by the chosen zero weight; they do not indicate an uninitialized Model or ignored
input. No [optimizer](../glossary.md#term-optimizer) has run.

## Useful variation and relevant error

As an independent experiment, change the supplied initial zero to `2`, remove the artificial
`99` mutation/check or adjust its private-[weight](../glossary.md#term-weight) expectation, and expect `[2,6]`. Update the
mode's numerical assertions too. Rebuild with the exact [Maven](../glossary.md#term-maven) command in the
[setup](../examples/README.md#build-the-complete-project). This shows [prediction](../glossary.md#term-prediction) with a chosen
[parameter](../glossary.md#term-parameter); it is not learning. Restore the [canonical](../glossary.md#term-canonical-form) source before later chapters.

A supplied weight with `requiresGrad=false`, wrong [rank](../glossary.md#term-rank)/[dtype](../glossary.md#data-type--datatype), no stored values, or incompatible
[feature](../glossary.md#term-feature) width is not a substitute for this parameter. Recipe/[schema](../glossary.md#term-schema) [admission](../glossary.md#term-admission) can throw
`IllegalArgumentException`; unavailable storage at build can throw `IllegalStateException`.
Provide a stored [FLOAT32](../glossary.md#term-float32) `[1,1]` [gradient](../glossary.md#term-gradient)-eligible [leaf](../glossary.md#term-leaf) and keep its [owner](../glossary.md#term-owner) live until build
returns. Do not close a [Model](../glossary.md#term-model) and then call `compute` again: read the already [detached value](../glossary.md#detached-model-results),
or create another Model.

## Source and next step

The actual public entry points and [ownership](../glossary.md#term-ownership-meanings) contracts are documented in
[`Model`](../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)<span data-glossary-navigation="true"> ([Model glossary](../glossary.md#term-model))</span>,
[`ModelBuilder`](../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelBuilder.java),
[`TensorFactory`](../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java)
and [`HostTensorValue`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/HostTensorValue.java)<span data-glossary-navigation="true"> ([HostTensorValue glossary](../glossary.md#host-tensor-value--hosttensorvalue))</span>.
The [owned-Model contract](../architecture/contracts/extensions-training.md#owned-tensor-models)
is authoritative; this chapter teaches one usage, not a replacement contract.

Next: [give tensors meaning through shape, storage and composition](tensors-and-composition.md)<span data-glossary-navigation="true"> ([tensors glossary](../glossary.md#tensor); [shape glossary](../glossary.md#shape))</span>.
