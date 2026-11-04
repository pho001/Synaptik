# 3. Turn [prediction](../glossary.md#term-prediction) errors into a [loss](../glossary.md#loss) and [gradient](../glossary.md#term-gradient)

## Outcome and prerequisites

Obtain the [scalar objective](../glossary.md#loss) `40` and actual [weight](../glossary.md#term-weight) [derivative](../glossary.md#term-derivative) `-40` without updating the weight.
Complete [setup](../examples/README.md#build-the-complete-project) and
[tensors and composition](tensors-and-composition.md)<span data-glossary-navigation="true"> ([tensors glossary](../glossary.md#tensor))</span>. Commands [run](../glossary.md#term-run) in the example project
with [JDK](../glossary.md#term-java-development-kit-jdk) 26 and `LEARNING_CP` set. The [`gradients`](../glossary.md#term-gradient) mode is independent of previous processes.

## Why one [scalar objective](../glossary.md#loss)?

A **[loss](../glossary.md#loss)** measures prediction error. The application supplies two input
measurements $x_1=1$, $x_2=3$ and desired answers $t_1=2$, $t_2=6$.
Sample index $i\in\{1,2\}$ selects a row; $w\in\mathbb{R}$ is the one shared weight,
$p_i(w)=x_iw$ is that row's prediction, and $e_i(w)=p_i(w)-t_i$ is its error.
All values are dimensionless in this synthetic experiment.

The scalar **objective** is the one number chosen for minimization. We reduce the two squared
errors by **summing**, not averaging:

$$
L(w)=\sum_{i=1}^{2}e_i(w)^2=(w-2)^2+(3w-6)^2.
$$

At the initial weight $w=0$, errors are $(-2,-6)$, squared errors are $(4,36)$, and
$L(0)=4+36=40$. A mean would be $L(0)/2=20$; SUM performs no division by sample count.
The training target $t_i$ is the desired answer, not the Tensor with respect to which a derivative
is requested. A scalar has no axes, unlike the two-row tensor of per-sample losses.

Read `squaredErrorSum` and [`gradients`](../glossary.md#term-gradient) in the
[complete Java source](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java).
The function constructs `error = prediction.sub(target)` and `error.mul(error).sum()`.
Those are [Tensor](../glossary.md#tensor) expressions, not a Java loop reading prediction bytes. `sum()` reduces every
logical element to a [rank-zero](../glossary.md#axis-rank-scalar-vector-and-matrix) [FLOAT32](../glossary.md#term-float32) scalar, preserving its [dependency](../glossary.md#term-dependency) on the [weight](../glossary.md#term-weight).

### How do both samples affect the same weight?

For finite real inputs, the chain rule differentiates each squared error and adds the
contributions. The prime in $L'(w)=dL/dw$ denotes the derivative with respect to the shared
weight, not to the sample index:

$$
\begin{aligned}
L'(w)&=\sum_{i=1}^{2}2(x_iw-t_i)x_i\\
     &=2(w-2)\cdot1+2(3w-6)\cdot3,\\
L'(0)&=2(-2)\cdot1+2(-6)\cdot3=-4-36=-40.
\end{aligned}
$$

The negative derivative says that a sufficiently small increase in weight reduces the objective
locally. It does not say the squared loss is negative. Both sample contributions reach the same
`[1,1]` weight derivative, not a separate weight for each row.

These equations teach real-number calculus. The program evaluates FLOAT32 operations with
rounding at their defined boundaries; the displayed initial predictions, objective and derivative
are exactly representable here. This derivation is not a rule for NaN, infinity or an empty
reduction, nor permission to reorder arbitrary floating-point sums. The configured training
path separately rejects nonfinite update state.

A **derivative** measures local [sensitivity](../glossary.md#term-sensitivity): approximately how much the objective changes per
small change in the selected number. With many parameters the collection of those derivatives
is the gradient. Follow the existing
[reverse-gradient diagram](../explanation/tensors-and-gradients.md) from the loss back to the
shared weight; both samples contribute to one weight derivative, not to two independent weights.

## What performs differentiation?

The Java application's `gradients` method creates the stored gradient-eligible weight `w`,
builds the objective expression, and calls
`Engine.backward(objective, List.of(w), 8)`. Engine is the application's callee; it in turn
asks Compiler to construct derivative expressions and arranges preparation and execution.
The input, desired-answer and weight storage remain supplied by the application and must be
usable during this call. The explicit differentiation-target list selects the exact weight
Tensor, not input or desired-answer derivatives. No `Tensor.backward()` or mutable Tensor
gradient field is involved.

The `8` is an aggregate output-copy ceiling: one FLOAT32 objective and one FLOAT32 weight
gradient require $1\cdot4+1\cdot4=8$ bytes (four bytes per element). It does not bound
the entire execution workspace. This one-shot
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
# backward computes derivatives but does not perform an optimizer update.
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

As a separate experiment change `sum()` to `mean()`. Dividing by the two samples gives
$L_{\mathrm{mean}}(0)=40/2=20$ and $L'_{\mathrm{mean}}(0)=-40/2=-20$.
At learning rate $\eta=0.05$, plain SGD would produce $w_{\mathrm{new}}=0-\eta(-20)=1$,
not $2$. Update gradient expectations and downstream training checks before rebuilding with
the [setup command](../examples/README.md#build-the-complete-project).
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
