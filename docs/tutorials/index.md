# Learn Synaptik by fitting one weight

Build a Java program that learns `y = 2x` from two examples. Follow the same numbers and complete
source from a first prediction to training, persistence and reusable execution. You need Java
knowledge, not prior machine-learning experience.

## Outcome and prerequisites

By the end you can explain what a tensor describes, why a loss must become one scalar objective,
what a gradient changes (and does not change), and which objects must be closed. You will run
`LearningJourney`, a public-API-only consumer with one direct `nn` dependency.

Use JDK 26, Maven 3.9+, and the supplied private candidate artifacts. Complete the
[project setup and build commands](../examples/README.md#build-the-complete-project) first.
Those instructions require an explicit caller-supplied file repository URL and pin
`0.1.0-public-audit.20261104`; this tutorial does not authorize publishing or assume a public
artifact download. All chapter commands run from `docs/examples/learning-journey`, with
`LEARNING_CP` exported by that setup.

## The experiment in concrete terms

A **sample** is one input paired with the desired answer. A **feature** is one measured input
number. A **batch** groups samples for one execution. Here the batch has two samples and each
sample has one feature:

| Value | FLOAT32 shape | Contents | Role |
|---|---|---|---|
| Input `x` | `[2,1]` | rows `[1]`, `[3]` | Supplied data, not a parameter |
| Target `t` | `[2,1]` | rows `[2]`, `[6]` | Desired predictions |
| Weight `w` | `[1,1]` | initially `[0]` | One trainable parameter, no bias |
| Prediction `p` | `[2,1]` | initially `[0,0]` | `x @ transpose(w)` |
| Objective `L` | scalar (rank zero) | initially `40` | Sum of squared prediction errors |

A **model** applies a parameterized rule. A **parameter** is stored state the optimizer may
update; input and target data are not learned state. The learning rate `0.05` controls the size
of the stochastic gradient descent (SGD) update. These numbers were chosen so one step has an
exact, inspectable answer, not to demonstrate general convergence.

The **target** is the answer supplied for comparison, not a value the model already knows.
The **weight** multiplies each input feature; the same weight is shared by both sample rows.
A **bias** would add an adjustable constant after multiplication, but this model has none.
FLOAT32 is Java-`float`-sized numerical storage; `[2,1]` means two rows with one value each,
not one row with two features.

Training changes the parameter; **inference** uses its current value to predict without an
optimizer update. **Evaluation** assesses those predictions, for example by comparing them with
targets. A complete pass through a chosen training dataset is an **epoch**. This journey performs
one batch update; it does not introduce a dataset iterator or an automatic epoch loop.

Read this flow left to right: data flows through the model into a comparison, and training uses
the derivative of that comparison to update the private parameter.

```text
x [1,3] -- model: p = x*w --> p [0,0] -- compare with t [2,6] --> L = 40
                    ^                                             |
                    +------ SGD: w = 0 - 0.05*(-40) = 2 <----------+
```

Once `w=2`, the original batch predicts `[2,6]` and a fresh batch `[2,4]` predicts `[4,8]`.
This demonstrates the learned multiplication, not statistical generalization: two synthetic
points and a hand-selected model are not a validation dataset.

For the existing [broadcast and reverse-gradient diagrams](../explanation/tensors-and-gradients.md),
read the shape picture as a rule for matching coordinates and the gradient picture as the return
path from one error measure to the shared weight. The
[ownership diagram](../explanation/ownership.md#a-result-is-not-always-an-independent-value)
separately shows which results need a live execution owner.

## Follow the chapters

1. [First model](first-model.md): acquire a private supplied weight, compute, and close safely.
2. [Tensors, shapes, data and composition](tensors-and-composition.md): distinguish descriptions,
   stored numbers and expressions; deliberately reject the wrong shape.
3. [Loss and gradients](loss-and-gradients.md): reduce two errors to one scalar and obtain the
   actual derivative without changing the weight.
4. [Train and evaluate](train-and-evaluate.md): commit one optimizer step, then inspect predictions.
5. [Save, close and load](save-close-load.md): restore graph/current values and configure fresh training.
6. [Reusable execution](reusable-execution.md): feed fresh values through cached Model execution
   and an explicit prepared handle.
7. [Backend and performance choices](backend-and-performance.md): keep semantics fixed while
   choosing the next measurement and supported execution route.

The canonical implementation is
[LearningJourney.java](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java),
with its [Maven POM](../examples/learning-journey/pom.xml). Chapters explain its named methods
instead of repeating partial Java snippets. Any mathematical procedure labelled pseudocode is
explanatory, not another API.

## Run the entire journey

After setup:

```bash
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney all target/learning-journey.graph
```

Expected application stdout is [the complete expected transcript](../examples/learning-journey/expected-stdout.txt).
It includes the initial prediction, objective `40`, derivative `-40`, fitted weight `2`, and
fresh prediction `[4,8]`. The complete transcript was checked against an external Maven consumer
of the installed private candidate. This does not qualify different backends or geometry.
The executable checks its assumptions and fails rather than printing an unearned success marker.
The graph destination is replaced if it exists.

**Useful variation:** run `first` instead of `all` to inspect only ownership and prediction;
all modes are independent, so no earlier process needs to remain alive.

**Relevant error:** a `[1,2]` tensor contains two numbers but is not a `[2,1]` batch. The
`invalid-shape` mode demonstrates the real rejection and subsequent successful correct call.
An unresolved Maven artifact is a setup problem, not a model or learning failure; consult the
[consumer troubleshooting table](../examples/README.md#useful-variation-and-failures).

## Understand the larger framework afterward

The [framework explanation](../explanation/framework.md) relates these calls to compile, prepare
and run. The [package atlas](../explanation/packages/index.md) shows which package owns each
mechanism, including packages this tiny example never uses. The [reference entry](../reference/index.md)
is for exact public contracts after the guided path; the existing
[prepared-graph guide](../user-guide/running-models.md) covers larger and advanced workflows.
