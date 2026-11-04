# 5. Save, close, load, then configure fresh training

## Outcome and prerequisites

Save the fitted rule with [weight](../glossary.md#term-weight) `2`, close its original [owner](../glossary.md#term-owner), and load a new [Model](../glossary.md#term-model) that maps
fresh `[2,4]` to `[4,8]`. Then configure a new [training session](../glossary.md#training-session--trainingsession) rather than mistaking model
[persistence](../glossary.md#term-persistence) for [optimizer](../glossary.md#term-optimizer) continuation.

Complete [setup](../examples/README.md#build-the-complete-project) and
[train/evaluate](train-and-evaluate.md). Use [JDK](../glossary.md#term-java-development-kit-jdk) 26 and `LEARNING_CP` from the example project.
The [Maven](../glossary.md#term-maven)-created [`target`](../glossary.md#term-target) directory must exist and be writable. The destination below is
**replaced if it already exists**; choose another filename if you need to retain an earlier file.
The mode trains its own Model, so it does not depend on an earlier `train` process.

## What crosses the file boundary?

A saved [model](../glossary.md#term-model) contains the supported cold [graph](../glossary.md#graph) semantics and current named stored values.
**Cold** means descriptions and data from which execution may later be prepared, not a live
[Engine](../glossary.md#term-engine), loaded native [kernel](../glossary.md#kernel) or running session. For this body-free [Linear](../glossary.md#term-linear-meanings) graph, the existing
smallest sufficient graph-model format is used; the example does not choose or implement a
wire [codec](../glossary.md#term-codec).

A **graph** records operations and the dependencies connecting their inputs and outputs.
**[Persistence](../glossary.md#term-persistence)** stores an independently readable representation in a file. A **wire codec**
encodes that representation as bytes and decodes it later; it is not the executable that performs
[matrix multiplication](../glossary.md#matrix-multiplication--matmul). “Body-free” means this example has no nested loop computation to encode.
The saved [parameter](../glossary.md#term-parameter) is the current `[1,1]` [weight](../glossary.md#term-weight), while `[2,1]` input [batches](../glossary.md#term-batch-machine-learning-and-hyperparameters) remain caller data.

Read `saveLoad` in the
[canonical Java source](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java)<span data-glossary-navigation="true"> ([canonical glossary](../glossary.md#term-canonical-form))</span>.
Its order matters:

```text
original: build w=0 -> configure -> train w=2 -> save -> CLOSE
                                                            |
file:                                  graph + current values
                                                            |
restored:                          load -> compute -> configure new training
```

Read downward across the file, not as two Models sharing one parameter allocation. The original
try-with-resources block ends before `Model.load` starts. Loading creates a new [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions) in
[evaluation](../glossary.md#term-evaluation) mode, with no Engine prepared yet. Its current weight is `2`, so fresh [FLOAT32](../glossary.md#term-float32)
`[2,1]` values `[2,4]` give `[4,8]`. The [sample](../glossary.md#term-sample-machine-learning-and-hyperparameters) count is still two: fresh data does not imply a
changed [signature](../glossary.md#term-signature).

| Saved by this workflow | Not restored by this workflow |
|---|---|
| Supported graph semantics and input/output [descriptors](../glossary.md#term-descriptor) | Cached compiled/[prepared execution](../glossary.md#prepared-execution--preparedexecution) or [backend](../glossary.md#backend) eligibility |
| Current named parameter/[buffer](../glossary.md#term-buffer)/[constant](../glossary.md#term-constant) values | [Objective](../glossary.md#term-objective) [callback](../glossary.md#term-callback) or configured [TrainingSession](../glossary.md#training-session--trainingsession) |
| The rule needed to recompute [predictions](../glossary.md#term-prediction) | [Optimizer history](../glossary.md#term-optimizer-history), random-stream state or data cursor |

An [optimizer](../glossary.md#term-optimizer)'s **history** can include earlier update counts and running averages of [gradients](../glossary.md#term-gradient);
it affects later updates even when two Models have the same current weights. A **data cursor**
records which sample or batch to read next. A **[random stream](../glossary.md#graph-rng-state--graphrngstate)** supplies successive random
choices. Neither that progress nor an [epoch](../glossary.md#term-epoch-meanings) count (full passes through a [dataset](../glossary.md#term-dataset)) follows from
restoring weight `2`. This is why matching predictions do not establish exact training [resume](../glossary.md#schema-serialization-and-checkpoint).

The program deliberately calls `restored.train(input,target)` **before** reconfiguration and
checks that it raises `IllegalStateException`. It then supplies the [target](../glossary.md#term-target) descriptor, SUM
objective builder and `Sgd(0.05)` again. The fresh session's first committed [optimizer step](../glossary.md#optimizer-and-stochastic-gradient-descent) is
`1`, not a continuation of the original step counter. Its pre-update [loss](../glossary.md#loss) is `0`, because the
loaded weight already fits the original `[1,3] -> [2,6]` batch, and the weight remains `2`.
No mode toggle is needed or valid for this captured owned source.

## [Run](../glossary.md#term-run) and interpret

```bash
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney save-load target/learning-journey.graph
```

Expected application stdout, checked against the installed private candidate:

```text
== save-load ==
save-load.fresh-prediction=[4.0, 8.0]
save-load.unconfigured-train=rejected
save-load.fresh-training-objective=[0.0]
save-load.fresh-optimizer-step=1
save-load.after-model-close=[4.0, 8.0]
```

A regular [graph](../glossary.md#graph) file remains at the specified path. The final [prediction](../glossary.md#term-prediction) is read after the
restored [Model](../glossary.md#term-model) closes, checking [detached](../glossary.md#detached-model-results) output [lifetime](../glossary.md#term-lifetime) a second time. The rejection and
step number show that the file did not secretly preserve a [training session](../glossary.md#training-session--trainingsession). This example
teaches a fresh training configuration after load, not exact training [checkpoint](../glossary.md#term-checkpoint) [resume](../glossary.md#schema-serialization-and-checkpoint).

## Useful variation and relevant errors

Save before the original training step instead. The restored [weight](../glossary.md#term-weight) is then `0`, fresh values
predict `[0,0]`, and fresh configured training has [objective](../glossary.md#term-objective) `40` before its first update to `2`.
Adjust the saved/restored checks accordingly and rebuild with the
[setup command](../examples/README.md#build-the-complete-project). The point is that save [captures](../glossary.md#term-capture-meanings)
**current** values, not a recipe to rerun the original [optimizer step](../glossary.md#optimizer-and-stochastic-gradient-descent).

A nonexistent/unwritable parent directory, invalid [graph](../glossary.md#graph) file, unsupported format or damaged
[payload](../glossary.md#artifact-envelope-and-atomic-publication) can produce `IOException`; correct the file/path rather than falling back to a new
random [model](../glossary.md#term-model). The source's successful-save path does not create parent directories for you.
An unconfigured training rejection is expected and recoverable by explicit configuration in
this mode. Trying to configure twice is not how to reset a live session: create/load another
Model when you need a new independent configuration.

## Source and next step

See [`Model.save/load`](../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java),
[`GraphCheckpoint`](../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java),
and the authoritative [generic graph source/value boundary](../architecture/contracts/extensions-training.md#generic-graph-source-and-values)<span data-glossary-navigation="true"> ([graph glossary](../glossary.md#graph))</span>.
[Checkpoint](../glossary.md#term-checkpoint) handles cold source/value [persistence](../glossary.md#term-persistence), while [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) composes the [owned model](../glossary.md#model-module-buffer-and-scan-distinctions) [lifecycle](../glossary.md#lifecycle);
it does not serialize a second executor.

Next: [reuse execution with fresh values and explicit owners](reusable-execution.md)<span data-glossary-navigation="true"> ([owners glossary](../glossary.md#term-owner))</span>.
