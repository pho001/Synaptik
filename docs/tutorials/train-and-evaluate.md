# 4. Train once, then evaluate

## Outcome and prerequisites

Commit one SGD step that changes only the Model's private weight from `0` to `2`. Compute the
new prediction `[2,6]`, and read both prediction and training objective after Model close.
Complete [setup](../examples/README.md#build-the-complete-project) and the
[loss/gradient walkthrough](loss-and-gradients.md). Commands use JDK 26 and `LEARNING_CP` from
the example project. No earlier process needs to be running.

## Configure what training means

An **optimizer** turns derivatives into parameter updates. Plain stochastic gradient descent
(SGD), as configured here without momentum or decay, uses `newWeight = oldWeight - rate*gradient`.
The method `configureTraining` supplies three separate choices:

- The target descriptor: FLOAT32 `[2,1]`, canonical contiguous, `requiresGrad=false`.
- A pure objective builder that constructs the scalar sum of squared errors from prediction and
  target prototypes. It is a setup callback, not a callback invoked once per sample or batch.
- `new Sgd(0.05)`, the immutable optimizer configuration.

Read `configure` and `train` in
[LearningJourney.java](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java).
The Model owns the working weight and cached session; Training owns optimizer mathematics and
transaction admission. Configuration can copy private heap parameter storage into Model-owned
shared native memory for the supported training path while preserving parameter Tensor identity.
Do not retain a storage handle before configuration and assume it is still the current binding.
The program looks up live Model state only for inspection.

Configuration succeeds once. It neither updates the weight by itself nor changes Module mode.
Keep the Model's acquired evaluation mode: the owned source captures modes and other structural
bindings. Calling inherited **no-argument** `model.train()` to toggle mode would change a frozen
source and invalidate this owned execution capability. The overloaded
`model.train(input,target)` is the configured optimizer transaction; it is a different operation.
Likewise, evaluation in this journey means `compute` without an optimizer update, not toggling
mode around each batch. Larger custom Module forward workflows have their own mode use cases.

## Work through the update

The complete starting state is input `[1,3]`, target `[2,6]`, private weight `0`, no bias and SUM
objective. The preceding chapter gave objective `40` and weight derivative `-40`.

Conceptual optimizer pseudocode, not an API:

```text
w := 0 - 0.05*(-40) = 2
p := [1*2, 3*2] = [2,6]
```

`model.train(input(), target())` performs one forward/backward/update transaction. Its
`TrainingStep.objective()` is the loss from the forward pass **before** the update: `40`.
It is not a second evaluation of the fitted weight. `optimizerStep()` is `1`, recording the
first committed update. A subsequent `model.compute(input())` observes the updated weight and
predicts `[2,6]`, whose squared-error sum would be `0`.

The example keeps the original supplied heap Tensor to check that its value is still `0`.
Only the Model's private copy becomes `2`. Parameter inspection through `stateDictionary` is
borrowed state, not a detached snapshot; read it only while the Model is live and without
concurrent mutation. `TrainingStep` and `HostTensorValue`, in contrast, remain readable after
all Model-owned execution/native resources close.

## Run and interpret

```bash
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney train
```

Expected application stdout (not yet an observed run):

```text
== train ==
train.objective-before-step=[40.0]
train.private-weight=2.0
train.optimizer-step=1
train.prediction-after-step=[2.0, 6.0]
train.caller-weight=0.0
train.objective-after-model-close=[40.0]
train.prediction-after-model-close=[2.0, 6.0]
```

The old objective next to a new prediction is intentional. The final lines check lifetime,
not another execution after close. Evaluation on the training batch checks the arithmetic;
it does not measure accuracy on independent real-world data. The later fresh `[2,4]` batch
checks the learned rule on different values while keeping the exact input descriptor.

## Useful variation and relevant errors

In an independent experiment use `Sgd(0.025)` with the same SUM loss. The derivative remains
`-40`, so one step gives weight `1` and prediction `[1,3]`; post-step squared error is `10`.
Update the numerical assertions/expected output, then rebuild with the
[setup command](../examples/README.md#build-the-complete-project). Restore `0.05` before running
the canonical full journey. Learning rate and reduction must be considered together.

| Symptom | Meaning and correction |
|---|---|
| `IllegalStateException` before a training transaction | Training was not configured, was configured twice, or the Model/source lifecycle is invalid. Configure a fresh live Model once. |
| Mode/source invalidation after no-argument `train()` | You changed captured Module mode, not performed the optimizer transaction. Rebuild a fresh Model and use `train(input,target)`. |
| `IllegalArgumentException` for batch/target | Use exact declared descriptors and independent live stored inputs; do not alias them with parameter storage. |
| `ArithmeticException` for nonfinite training values | The optimizer transaction rejected invalid numerical state. Inspect data/objective/gradient; do not suppress the error or report an uncommitted update. |

## Source and next step

See [`Model.configureTraining/train`](../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java),
[`TrainingStep`](../../modules/training/src/main/java/io/github/pho001/synaptik/training/TrainingStep.java),
[`Sgd`](../../modules/training/src/main/java/io/github/pho001/synaptik/training/Sgd.java),
and the authoritative [owned-Model lifecycle](../architecture/contracts/extensions-training.md#owned-tensor-models)
and [Training ownership](../architecture/contracts/extensions-training.md#modulestraining).

Next: [save current graph/weights, close, and load a new Model](save-close-load.md).
