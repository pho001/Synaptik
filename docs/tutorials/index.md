# Learn Synaptik by fitting one [weight](../glossary.md#term-weight)

Build a Java program that learns `y = 2x` from two examples. Follow the same numbers and complete
source from a first [prediction](../glossary.md#term-prediction) to training, [persistence](../glossary.md#term-persistence) and reusable execution. You need Java
knowledge, not prior machine-learning experience.

## Outcome and prerequisites

By the end you can explain what a [tensor](../glossary.md#tensor) describes, why a [loss](../glossary.md#loss) must become one [scalar objective](../glossary.md#loss),
what a [gradient](../glossary.md#term-gradient) changes (and does not change), and which objects must be closed. You will [run](../glossary.md#term-run)
`LearningJourney`, a public-[API](../glossary.md#term-application-programming-interface-api)-only consumer with one direct `nn` [dependency](../glossary.md#term-dependency).

Use [JDK](../glossary.md#term-java-development-kit-jdk) 26, [Maven](../glossary.md#term-maven) 3.9+, and the supplied private candidate [artifacts](../glossary.md#term-artifact). Complete the
[project setup and build commands](../examples/README.md#build-the-complete-project) first.
Those instructions require an explicit caller-supplied [file repository](../glossary.md#term-file-repository) URL and pin
`0.1.0-public-audit.20261104`; this tutorial does not authorize publishing or assume a public
artifact download. All chapter commands run from `docs/examples/learning-journey`, with
`LEARNING_CP` exported by that setup.

## The experiment in concrete terms

A **[sample](../glossary.md#term-sample-machine-learning-and-hyperparameters)** is one input paired with the desired answer. A **[feature](../glossary.md#term-feature)** is one measured input
number. A **[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters)** groups samples for one execution. Here the batch has two samples and each
sample has one feature:

| Value | [FLOAT32](../glossary.md#term-float32) [shape](../glossary.md#shape) | Contents | Role |
|---|---|---|---|
| Input `x` | `[2,1]` | rows `[1]`, `[3]` | Supplied data, not a [parameter](../glossary.md#term-parameter) |
| [Target](../glossary.md#term-target) `t` | `[2,1]` | rows `[2]`, `[6]` | Desired [predictions](../glossary.md#term-prediction) |
| [Weight](../glossary.md#term-weight) `w` | `[1,1]` | initially `[0]` | One [trainable parameter](../glossary.md#trainable-parameter), no [bias](../glossary.md#term-bias) |
| Prediction `p` | `[2,1]` | initially `[0,0]` | `x @ transpose(w)` |
| [Objective](../glossary.md#term-objective) `L` | [scalar](../glossary.md#term-scalar) ([rank](../glossary.md#term-rank) zero) | initially `40` | Sum of squared prediction errors |

A **[model](../glossary.md#term-model)** applies a parameterized rule. A **parameter** is stored state the [optimizer](../glossary.md#term-optimizer) may
update; input and target data are not learned state. The [learning rate](../glossary.md#term-learning-rate) `0.05` controls the size
of the [stochastic gradient descent](../glossary.md#term-stochastic-gradient-descent-sgd) ([SGD](../glossary.md#optimizer-and-stochastic-gradient-descent)) update. These numbers were chosen so one step has an
exact, inspectable answer, not to demonstrate general convergence.

The **target** is the answer supplied for comparison, not a value the model already knows.
The **weight** multiplies each input feature; the same weight is shared by both sample rows.
A **bias** would add an adjustable [constant](../glossary.md#term-constant) after multiplication, but this model has none.
FLOAT32 is Java-`float`-sized numerical storage; `[2,1]` means two rows with one value each,
not one row with two features.

Training changes the parameter; **[inference](../glossary.md#term-inference)** uses its current value to predict without an
optimizer update. **[Evaluation](../glossary.md#term-evaluation)** assesses those predictions, for example by comparing them with
targets. A complete pass through a chosen training [dataset](../glossary.md#term-dataset) is an **[epoch](../glossary.md#term-epoch-meanings)**. This journey performs
one batch update; it does not introduce a dataset iterator or an automatic epoch loop.

Read this flow left to right: data flows through the model into a comparison, and training uses
the [derivative](../glossary.md#term-derivative) of that comparison to update the private parameter.

```text
x [1,3] -- model: p = x*w --> p [0,0] -- compare with t [2,6] --> L = 40
                    ^                                             |
                    +------ SGD: w = 0 - 0.05*(-40) = 2 <----------+
```

Once `w=2`, the original batch predicts `[2,6]` and a fresh batch `[2,4]` predicts `[4,8]`.
This demonstrates the learned multiplication, not statistical [generalization](../glossary.md#term-generalization): two synthetic
points and a hand-selected model are not a [validation](../glossary.md#term-validation) dataset.

For the existing [broadcast and reverse-gradient diagrams](../explanation/tensors-and-gradients.md),
read the shape picture as a rule for matching [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix) and the [gradient](../glossary.md#term-gradient) picture as the return
path from one error measure to the shared weight. The
[ownership diagram](../explanation/ownership.md#a-result-is-not-always-an-independent-value)<span data-glossary-navigation="true"> ([ownership glossary](../glossary.md#term-ownership-meanings))</span>
separately shows which results need a live execution [owner](../glossary.md#term-owner).

## Follow the chapters

1. [First model](first-model.md)<span data-glossary-navigation="true"> ([model glossary](../glossary.md#term-model))</span>: acquire a private supplied [weight](../glossary.md#term-weight), compute, and close safely.
2. [Tensors, shapes, data and composition](tensors-and-composition.md)<span data-glossary-navigation="true"> ([Tensors glossary](../glossary.md#tensor); [shapes glossary](../glossary.md#shape))</span>: distinguish descriptions,
   stored numbers and expressions; deliberately reject the wrong shape.
3. [Loss and gradients](loss-and-gradients.md)<span data-glossary-navigation="true"> ([Loss glossary](../glossary.md#loss); [gradients glossary](../glossary.md#term-gradient))</span>: reduce two errors to one scalar and obtain the
   actual derivative without changing the weight.<span data-glossary-navigation="true"> ([scalar glossary](../glossary.md#term-scalar); [derivative glossary](../glossary.md#term-derivative))</span>
4. [Train and evaluate](train-and-evaluate.md): [commit](../glossary.md#term-commit) one [optimizer step](../glossary.md#optimizer-and-stochastic-gradient-descent), then inspect [predictions](../glossary.md#term-prediction).
5. [Save, close and load](save-close-load.md): restore [graph](../glossary.md#graph)/current values and configure fresh training.
6. [Reusable execution](reusable-execution.md): feed fresh values through cached Model execution
   and an explicit prepared handle.
7. [Backend and performance choices](backend-and-performance.md)<span data-glossary-navigation="true"> ([Backend glossary](../glossary.md#backend))</span>: keep semantics fixed while
   choosing the next measurement and supported execution route.

The [canonical](../glossary.md#term-canonical-form) implementation is
[LearningJourney.java](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java),
with its [Maven POM](../examples/learning-journey/pom.xml)<span data-glossary-navigation="true"> ([Maven glossary](../glossary.md#term-maven); [POM glossary](../glossary.md#term-pom))</span>. Chapters explain its named methods
instead of repeating partial Java snippets. Any mathematical procedure labelled [pseudocode](../glossary.md#term-pseudocode) is
explanatory, not another [API](../glossary.md#term-application-programming-interface-api).

## [Run](../glossary.md#term-run) the entire journey

After setup:

```bash
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney all target/learning-journey.graph
```

Expected application stdout is [the complete expected transcript](../examples/learning-journey/expected-stdout.txt).
It includes the initial [prediction](../glossary.md#term-prediction), [objective](../glossary.md#term-objective) `40`, [derivative](../glossary.md#term-derivative) `-40`, fitted [weight](../glossary.md#term-weight) `2`, and
fresh prediction `[4,8]`. The complete transcript was checked against an external [Maven](../glossary.md#term-maven) consumer
of the installed private candidate. This does not qualify different [backends](../glossary.md#backend) or geometry.
The executable checks its assumptions and fails rather than printing an unearned success marker.
The [graph](../glossary.md#graph) destination is replaced if it exists.

**Useful variation:** run `first` instead of `all` to inspect only [ownership](../glossary.md#term-ownership-meanings) and prediction;
all modes are independent, so no earlier process needs to remain alive.

**Relevant error:** a `[1,2]` [tensor](../glossary.md#tensor) contains two numbers but is not a `[2,1]` [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters). The
`invalid-shape` mode demonstrates the real rejection and subsequent successful correct call.
An unresolved Maven [artifact](../glossary.md#term-artifact) is a setup problem, not a [model](../glossary.md#term-model) or learning failure; consult the
[consumer troubleshooting table](../examples/README.md#useful-variation-and-failures).

## Understand the larger framework afterward

The [framework explanation](../explanation/framework.md) relates these calls to [compile](../glossary.md#compile), [prepare](../glossary.md#prepare)
and [run](../glossary.md#term-run). The [package atlas](../explanation/packages/index.md) shows which package owns each
mechanism, including packages this tiny example never uses. The [reference entry](../reference/index.md)
is for exact public contracts after the guided path; the existing
[prepared-graph guide](../user-guide/running-models.md)<span data-glossary-navigation="true"> ([graph glossary](../glossary.md#graph))</span> covers larger and advanced workflows.
