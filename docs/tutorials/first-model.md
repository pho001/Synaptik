# 1. Compute with your first Model

## Outcome and prerequisites

Compute `[0,0]` from input `[1,3]` and a single zero weight, then read the result after the Model
closes. You will also see executable checks that the Model copies supplied parameter storage
instead of retaining its caller's lifetime.

Complete the [project setup](../examples/README.md#build-the-complete-project) with JDK 26, Maven
and the supplied candidate file repository. Run the commands below from the example project
with `LEARNING_CP` exported. No previous mode needs to have run.

## Inputs and mechanism

The input is FLOAT32 `[2,1]`: two samples, one feature each. The supplied weight is FLOAT32
`[1,1]`, value `0`, with `requiresGrad=true`. That flag makes the floating leaf eligible for
requested differentiation; it is not a gradient buffer and does not initiate training.

A **feature** is one input measurement: `1` for the first sample and `3` for the second.
A **parameter** is model state that training may change; here it is the single weight, not either
input number. A **leaf** is a Tensor with no producing expression, so the supplied weight already
has stored values rather than instructions that must first be evaluated.

`Model.builder().input(...).linear(supplied).build()` creates the rule `p = x @ transpose(w)`.
The weight is `[output features,input features]`, so this case maps one feature to one feature.
`linear(weight)` has **no bias**. A bias would add another parameter and change the equation.
Build creates an evaluation-mode owned Model, but no Engine yet. A successful build copies the
supplied stored leaf; it does not evaluate an arbitrary supplied weight expression.

Here `@` denotes matrix multiplication, and transpose swaps the weight's row and column axes.
With one weight this is just `p = x*w` for each row. A bias would instead give `p = x*w+b`;
“no bias” means no such extra stored number, not a bias initialized to zero.
**Evaluation mode** is a Module behavior setting, not a request to update parameters or assess
prediction quality. This chapter performs inference: it computes predictions without training.

Read `copiedModel` and `first` in the
[complete Java source](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java).
The important decisions are:

1. A caller-owned `Arena` allocates one native float holding zero. `TensorFactory.create` borrows
   that storage and associates its exact contiguous descriptor with it.
2. Build runs while the supplied storage is still live. The program checks that Model state has
   different Tensor and storage identities, changes the caller's original value to `99`, and
   checks that the Model's private value is still `0`.
3. The caller Arena closes **before** the returned Model computes. A correct private copy can
   still predict `[1*0,3*0] = [0,0]`.
4. `compute` lazily obtains execution through the Model's Engine path and returns a detached
   immutable `HostTensorValue`. Try-with-resources closes the Model's cached owners; the output
   bytes are still readable afterward.

This is why a tensor object, its storage and a detached result are not interchangeable:

| Object | Owns/borrows | Lifetime in this mode |
|---|---|---|
| Supplied Tensor | Borrows caller native storage | Storage ends before compute |
| Owned Model | Private state; lazily owned execution resources | Ends at try-with-resources close |
| `HostTensorValue` | Independent canonical output bytes | Usable after Model close; no close method |

**Owned** means the Model is responsible for releasing its private resources. **Borrowed** means
another owner must keep storage alive; merely retaining a Java Tensor reference cannot keep a
closed native Arena usable. **Detached** means output bytes were copied into an independent
value. Follow the [ownership diagram](../explanation/ownership.md#a-result-is-not-always-an-independent-value)
for these different lifetimes. Native storage is memory managed outside the ordinary Java heap;
the Arena determines when this caller-supplied native memory stops being usable.

The explicit `99` mutation is a copy-isolation experiment, not a recommended application pattern.
Keep supplied storage live and exclude concurrent mutation during build.

## Run and interpret

```bash
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

The first marker follows checks and a successful compute after supplied-owner close. The final
line is read **after Model close**, so it checks a different boundary: detached outputs do not
borrow their originating Model. Zero predictions mean the chosen weight is zero, not that the
model is uninitialized or that the input was ignored. No optimizer has run.

## Useful variation and relevant error

As an independent experiment, change the supplied initial zero to `2`, remove the artificial
`99` mutation/check or adjust its private-weight expectation, and expect `[2,6]`. Update the
mode's numerical assertions too. Rebuild with the exact Maven command in the
[setup](../examples/README.md#build-the-complete-project). This shows prediction with a chosen
parameter; it is not learning. Restore the canonical source before later chapters.

A supplied weight with `requiresGrad=false`, wrong rank/dtype, no stored values, or incompatible
feature width is not a substitute for this parameter. Recipe/schema admission can throw
`IllegalArgumentException`; unavailable storage at build can throw `IllegalStateException`.
Provide a stored FLOAT32 `[1,1]` gradient-eligible leaf and keep its owner live until build
returns. Do not close a Model and then call `compute` again: read the already detached value,
or create another Model.

## Source and next step

The actual public entry points and ownership contracts are documented in
[`Model`](../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java),
[`ModelBuilder`](../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/ModelBuilder.java),
[`TensorFactory`](../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/TensorFactory.java)
and [`HostTensorValue`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/HostTensorValue.java).
The [owned-Model contract](../architecture/contracts/extensions-training.md#owned-tensor-models)
is authoritative; this chapter teaches one usage, not a replacement contract.

Next: [give tensors meaning through shape, storage and composition](tensors-and-composition.md).
