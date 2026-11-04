# 3. Turn prediction errors into a loss and gradient

## Outcome and prerequisites

Obtain the scalar objective `40` and actual weight derivative `-40` without updating the weight.
Complete [setup](../examples/README.md#build-the-complete-project) and
[tensors and composition](tensors-and-composition.md). Commands run in the example project
with JDK 26 and `LEARNING_CP` set. The `gradients` mode is independent of previous processes.

## Why one scalar objective?

A **loss** measures how wrong a prediction is. For these two samples, subtraction gives errors
`[-2,-6]`; squaring gives `[4,36]`. Training needs an explicit scalar **objective**, the one
number whose derivative guides updates. The choice of reduction defines what is being optimized:
we use a **sum**, not a mean, and no implicit division by batch size occurs.

The **target** supplies the desired outputs `[2,6]`; it is not a differentiation target in this
calculation. A **reduction** combines several elements into fewer elements. SUM combines `4` and
`36` into `40`; MEAN would divide that sum by two. A **scalar** is one number with no axes, not
merely a one-element row. The objective is the scalar we choose to minimize, while the
per-sample squared losses still describe the separate errors.

Read `squaredErrorSum` and `gradients` in the
[complete Java source](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java).
The function constructs `error = prediction.sub(target)` and `error.mul(error).sum()`.
Those are Tensor expressions, not a Java loop reading prediction bytes. `sum()` reduces every
logical element to a rank-zero FLOAT32 scalar, preserving its dependency on the weight.

The arithmetic before any update is:

| Stage | Shape | Values |
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
the squared loss is negative. The derivative has the parameter's shape `[1,1]`, not one weight
per sample. Contributions from both rows add into that same parameter derivative.

A **derivative** measures local sensitivity: approximately how much the objective changes per
small change in the selected number. With many parameters the collection of those derivatives
is the gradient. Follow the existing
[reverse-gradient diagram](../explanation/tensors-and-gradients.md) from the loss back to the
shared weight; both samples contribute to one weight derivative, not to two independent weights.

## What performs differentiation?

The mode supplies a stored gradient-eligible weight and calls public
`Engine.backward(objective, List.of(w), 8)`. Compiler owns the actual derivative construction;
Engine orchestrates compile, prepare, execution and detached copies. The explicit target list
asks for the weight derivative, not input/target derivatives. No `Tensor.backward()` or mutable
Tensor gradient field is involved.

The `8` is an aggregate output-copy ceiling: one FLOAT32 objective plus one FLOAT32 weight
gradient uses `4 + 4` bytes. It does not bound the entire execution workspace. This one-shot
convenience compiles/prepares per call; use [prepared execution](reusable-execution.md) when
reuse is the goal. The result's objective and target-aligned gradients are detached values.
Backward alone does not own an optimizer and does not mutate `w`.

In this API, **differentiation target** means “the Tensor with respect to which a derivative is
requested,” here `w`. In the loss builder, **training target** means the desired answer, here
`[2,6]`. The same word does not give those Tensors the same role. A **workspace** is temporary
memory needed to execute intermediate work; the output-copy ceiling limits returned bytes,
not that temporary memory.

## Run and interpret

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

Brackets are the example's flattened byte-reader presentation: `[40.0]` is still a rank-zero
scalar, while `[-40.0]` is the one-element `[1,1]` gradient. Both numbers are read from actual
Engine results and checked. The unchanged weight separates differentiation from optimization.
The final line checks that gradient bytes remain readable after Engine close.

## Useful variation and relevant errors

As a separate experiment change `sum()` to `mean()`. There are two errors, so the initial
objective becomes `20` and the derivative becomes `-20`. At learning rate `0.05`, one update
would yield `w=1`, not `2`. Update the gradient expectations and any downstream training checks
before rebuilding with the [setup command](../examples/README.md#build-the-complete-project).
Restore SUM for the canonical journey; reduction is a mathematical choice, not just a display
preference.

Returning the unreduced `[2,1]` squared-error tensor to this scalar-backward convenience or to
`configureTraining` is invalid: choose an explicit reduction. A disconnected target, or a
weight not eligible for gradients, also fails semantic admission. Keep the stored gradient-
eligible weight connected through prediction and loss, and request that exact Tensor identity.
A copy of a weight is not automatically the differentiation target in an existing expression.

## Source and next step

See [`Engine.backward`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java),
[`ScalarObjectiveBackwardResult`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/ScalarObjectiveBackwardResult.java)
and [`Tensor` expressions](../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java).
Operation meaning belongs to the
[profile-free numerical-semantics contract](../architecture/contracts/foundational-modules.md#profile-free-numerical-semantics);
the [framework explanation](../explanation/framework.md) connects its owners without turning
internal packages into application API.

Next: [use that derivative for one training step, then evaluate](train-and-evaluate.md).
