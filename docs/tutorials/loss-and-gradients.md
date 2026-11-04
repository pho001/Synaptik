# 3. Turn [prediction](../glossary.md#term-prediction) errors into a [loss](../glossary.md#loss) and [gradient](../glossary.md#term-gradient)

## Outcome and prerequisites

Obtain the [scalar objective](../glossary.md#loss) `40` and actual [weight](../glossary.md#term-weight) [derivative](../glossary.md#term-derivative) `-40` without updating the weight.
Complete [setup](../examples/README.md#build-the-complete-project) and
[tensors and composition](tensors-and-composition.md)<span data-glossary-navigation="true"> ([tensors glossary](../glossary.md#tensor))</span>. Commands [run](../glossary.md#term-run) in the example project
with [JDK](../glossary.md#term-java-development-kit-jdk) 26 and `LEARNING_CP` set. The [`gradients`](../glossary.md#term-gradient) mode is independent of previous processes.

## Why one [scalar objective](../glossary.md#loss)?

A **[loss](../glossary.md#loss)** measures how wrong a [prediction](../glossary.md#term-prediction) is. For these two [samples](../glossary.md#term-sample-machine-learning-and-hyperparameters), subtraction gives errors
`[-2,-6]`; squaring gives `[4,36]`. Training needs an explicit [scalar](../glossary.md#term-scalar) **[objective](../glossary.md#term-objective)**, the one
number whose [derivative](../glossary.md#term-derivative) guides updates. The choice of reduction defines what is being optimized:
we use a **sum**, not a [mean](../glossary.md#term-mean), and no implicit division by [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) size occurs.

The **[target](../glossary.md#term-target)** supplies the desired outputs `[2,6]`; it is not a [differentiation target](../glossary.md#functional-gradient-request) in this
calculation. A **reduction** combines several [elements](../glossary.md#term-element) into fewer elements. SUM combines `4` and
`36` into `40`; MEAN would divide that sum by two. A **scalar** is one number with no [axes](../glossary.md#term-axis), not
merely a one-element row. The objective is the scalar we choose to minimize, while the
per-sample squared losses still describe the separate errors.

Read `squaredErrorSum` and [`gradients`](../glossary.md#term-gradient) in the
[complete Java source](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java).
The function constructs `error = prediction.sub(target)` and `error.mul(error).sum()`.
Those are [Tensor](../glossary.md#tensor) expressions, not a Java loop reading prediction bytes. `sum()` reduces every
logical element to a [rank-zero](../glossary.md#axis-rank-scalar-vector-and-matrix) [FLOAT32](../glossary.md#term-float32) scalar, preserving its [dependency](../glossary.md#term-dependency) on the [weight](../glossary.md#term-weight).

The arithmetic before any update is:

| Stage | [Shape](../glossary.md#shape) | Values |
|---|---|---|
| `p = x*w` with `w=0` | `[2,1]` | `[0,0]` |
| `p-t` with `t=[2,6]` | `[2,1]` | `[-2,-6]` |
| `(p-t)*(p-t)` | `[2,1]` | `[4,36]` |
| Sum | scalar | `40` |

**Gradient** means the derivative of the objective with respect to selected values. Here there
is one weight, so its gradient says how the loss changes when that one number changes. With
`L(w) = (1*w-2)^2 + (3*w-6)^2`, ordinary differentiation gives:

```text
L'(w) = 2*(w-2)*1 + 2*(3*w-6)*3
L'(0) = 2*(-2)*1 + 2*(-6)*3 = -4 - 36 = -40
```

The negative sign says increasing the weight is the local downhill direction; it does not mean
the squared loss is negative. The derivative has the [parameter](../glossary.md#term-parameter)'s shape `[1,1]`, not one weight
per sample. Contributions from both rows add into that same parameter derivative.

A **derivative** measures local [sensitivity](../glossary.md#term-sensitivity): approximately how much the objective changes per
small change in the selected number. With many parameters the collection of those derivatives
is the gradient. Follow the existing
[reverse-gradient diagram](../explanation/tensors-and-gradients.md) from the loss back to the
shared weight; both samples contribute to one weight derivative, not to two independent weights.

## What performs differentiation?

The mode supplies a stored [gradient](../glossary.md#term-gradient)-eligible [weight](../glossary.md#term-weight) and calls public
`Engine.backward(objective, List.of(w), 8)`. [Compiler](../glossary.md#term-compiler) owns the actual [derivative](../glossary.md#term-derivative) construction;
[Engine](../glossary.md#term-engine) orchestrates [compile](../glossary.md#compile), [prepare](../glossary.md#prepare), execution and [detached](../glossary.md#detached-model-results) copies. The explicit [target](../glossary.md#term-target) list
asks for the weight derivative, not input/target derivatives. No `Tensor.backward()` or [mutable](../glossary.md#term-mutable)
[Tensor](../glossary.md#tensor) gradient field is involved.

The `8` is an aggregate output-copy ceiling: one [FLOAT32](../glossary.md#term-float32) [objective](../glossary.md#term-objective) plus one FLOAT32 weight
gradient uses `4 + 4` bytes. It does not bound the entire execution workspace. This one-shot
convenience compiles/prepares per call; use [prepared execution](reusable-execution.md)<span data-glossary-navigation="true"> ([prepared execution glossary](../glossary.md#prepared-execution--preparedexecution))</span> when
reuse is the goal. The result's objective and target-aligned gradients are detached values.
Backward alone does not own an [optimizer](../glossary.md#term-optimizer) and does not mutate `w`.

In this [API](../glossary.md#term-application-programming-interface-api), **[differentiation target](../glossary.md#functional-gradient-request)** means “the Tensor with respect to which a derivative is
requested,” here `w`. In the [loss](../glossary.md#loss) builder, **[training target](../glossary.md#loss)** means the desired answer, here
`[2,6]`. The same word does not give those Tensors the same role. A **workspace** is temporary
memory needed to execute intermediate work; the output-copy ceiling limits returned bytes,
not that temporary memory.

## [Run](../glossary.md#term-run) and interpret

```bash
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney gradients
```

Expected application stdout, checked against the installed private candidate:

```text
== gradients ==
gradients.objective=[40.0]
gradients.weight=[-40.0]
gradients.weight-without-optimizer=0.0
gradients.after-engine-close=[-40.0]
```

Brackets are the example's flattened byte-reader presentation: `[40.0]` is still a [rank-zero](../glossary.md#axis-rank-scalar-vector-and-matrix)
[scalar](../glossary.md#term-scalar), while `[-40.0]` is the one-[element](../glossary.md#term-element) `[1,1]` [gradient](../glossary.md#term-gradient). Both numbers are read from actual
[Engine](../glossary.md#term-engine) results and checked. The unchanged [weight](../glossary.md#term-weight) separates differentiation from optimization.
The final line checks that gradient bytes remain readable after Engine close.

## Useful variation and relevant errors

As a separate experiment change `sum()` to `mean()`. There are two errors, so the initial
[objective](../glossary.md#term-objective) becomes `20` and the [derivative](../glossary.md#term-derivative) becomes `-20`. At [learning rate](../glossary.md#term-learning-rate) `0.05`, one update
would yield `w=1`, not `2`. Update the [gradient](../glossary.md#term-gradient) expectations and any downstream training checks
before rebuilding with the [setup command](../examples/README.md#build-the-complete-project).
Restore SUM for the [canonical](../glossary.md#term-canonical-form) journey; reduction is a mathematical choice, not just a display
preference.

Returning the unreduced `[2,1]` squared-error [tensor](../glossary.md#tensor) to this [scalar](../glossary.md#term-scalar)-backward convenience or to
`configureTraining` is invalid: choose an explicit reduction. A disconnected [target](../glossary.md#term-target), or a
[weight](../glossary.md#term-weight) not eligible for gradients, also fails semantic [admission](../glossary.md#term-admission). Keep the stored gradient-
eligible weight connected through [prediction](../glossary.md#term-prediction) and [loss](../glossary.md#loss), and request that exact Tensor [identity](../glossary.md#term-reference-identity).
A copy of a weight is not automatically the [differentiation target](../glossary.md#functional-gradient-request) in an existing expression.

## Source and next step

See [`Engine.backward`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java),
[`ScalarObjectiveBackwardResult`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/ScalarObjectiveBackwardResult.java)
and [`Tensor` expressions](../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)<span data-glossary-navigation="true"> ([Tensor glossary](../glossary.md#tensor))</span>.
[Operation](../glossary.md#operation) meaning belongs to the
[profile-free numerical-semantics contract](../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics);
the [framework explanation](../explanation/framework.md) connects its [owners](../glossary.md#term-owner) without turning
internal packages into application [API](../glossary.md#term-application-programming-interface-api).

Next: [use that derivative for one training step, then evaluate](train-and-evaluate.md)<span data-glossary-navigation="true"> ([derivative glossary](../glossary.md#term-derivative))</span>.
