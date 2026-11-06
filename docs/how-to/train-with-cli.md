# Train, save and predict from the command line

The [CLI](../glossary.md#term-cli) is an outer caller of the existing public
[Model](../reference/nn/owned-model.md), [TrainingSession](../glossary.md#training-session--trainingsession)
and [Engine](../reference/engine/lifecycle.md) APIs. It reads your files and presents progress;
it does not implement another optimizer, executor or model format.

## Build the separate local application

Use JDK 26 and the repository's Gradle wrapper:

```sh
# Build a local application image; this does not publish the eighteen-product release.
./gradlew :tools:cli:installDist

# The generated launcher supplies the required JDK native-access option.
tools/cli/build/install/synaptik/bin/synaptik --help
```

The image includes the Java dependencies and their [third-party notices](../../THIRD_PARTY_NOTICES.md).
It does not include a JDK, OpenBLAS, oneDNN or the Synaptik Metal native package. JLine terminal
support is a separate dependency, not a numerical backend. The optional Java Vector module is
not a launcher requirement. See [private distribution rules](../developer-guide/release-process.md).

## First real training run

Each [CSV](../glossary.md#term-csv) row below contains one feature followed by one desired output.
The rule is $y=2x+1$; these are actual training examples, not an artificial progress animation.

```sh
# Numeric input, target. No header or quoted fields.
printf '%s\n' '-2,-3' '-1,-1' '0,1' '1,3' '2,5' > train.csv

# Build a seeded FLOAT32 Linear model with bias, then update it from these examples.
# --plain is useful in logs; omit it in a supported terminal for the live display.
tools/cli/build/install/synaptik/bin/synaptik train \
  --data train.csv --output line.synaptik --epochs 200 --plain

# Inspect saved graph/signature metadata, not an optimizer continuation.
tools/cli/build/install/synaptik/bin/synaptik inspect --model line.synaptik

# Prediction rows contain features only; stdout contains actual numeric predictions.
printf '%s\n' '0' '1' '2' > features.csv
tools/cli/build/install/synaptik/bin/synaptik predict \
  --model line.synaptik --data features.csv --output predictions.csv
```

The default new model is a single biased [Linear layer](../glossary.md#term-linear-meanings),
with seed 0, batch size 1, one output, [SGD](../glossary.md#term-stochastic-gradient-descent-sgd)
and learning rate 0.01. `--outputs` changes the output width. `--optimizer adam` selects the
existing [Adam](../glossary.md#adaptive-optimizer--adam-adamw) implementation with default learning
rate 0.001, $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$ and zero weight decay.
`--learning-rate` changes the selected optimizer's learning rate; the CLI never duplicates its update math.

### What loss means here

For batch size $B$ and output width $O$, training uses the existing
[mean-squared-error](../glossary.md#mean-squared-error--mse) operation:

$$
L=\frac{1}{BO}\sum_{r=0}^{B-1}\sum_{c=0}^{O-1}
  (\widehat y_{rc}-y_{rc})^2.
$$

Here $\widehat y$ is the model prediction and $y$ is the supplied target. The displayed step loss
is captured **before that step's parameter update**. An epoch average combines losses observed
while parameters change; it is not a new evaluation of the final epoch model. Use `predict` on
separate evaluation data when assessing the final result. A small training loss alone does not
prove generalization.

## Dataset and static-batch rules
- UTF-8 numeric CSV without a header, quoted fields or missing values. Each nonblank row is
  one example; numbers must decode to finite FLOAT32 values, with normal [floating-point rounding](../glossary.md#floating-point-precision-and-rounding).
- With $F$ features and $O$ outputs, training requires $F+O$ columns. The last $O$ are targets.
  Prediction requires exactly $F$ columns.
- The CLI validates the complete row geometry/count before the first update. Keep the input
  file stable during execution; a later read failure does not undo already committed steps.
- Batch size 1 handles every positive number of rows. For static $B>1$, the row count must be
  divisible by $B$ **before training starts**. No last row is dropped, repeated, padded or masked.
  [Changing the static batch shape](../glossary.md#shape) is not added by this tool.
- Training output must not alias the dataset. Prediction output must not alias the dataset or
  saved model. Training may replace the loaded model only after successful completion.

## Continue from current saved values, not optimizer state

Pass `train --model existing.synaptik --data ... --output updated.synaptik` to acquire a saved
one-input/one-output, static rank-two FLOAT32 graph. The file's signatures determine its geometry.
The checkpoint acquisition boundary admits absent or canonical contiguous, offset-zero,
non-view leaf layouts; arbitrary strided checkpoint inputs are not promised. An absent layout
is preserved, not silently made contiguous: the selected backend can still reject it.
The current CPU MUL path does reject such an unresolved input; inspection can still describe it.

The CLI loads the public graph source once, retains its descriptors, then calls `Model.graph`
with the borrowed Engine. The Model acquires its own private state. This cold acquisition copy
is deliberate; there is no private metadata access, second file read or new ownership API.
See [save/restart boundaries](save-and-restart-training.md) and
[save, close, load](../tutorials/save-close-load.md).

Saving records the admitted graph and **current values**. It does not record optimizer moments,
step number, dataset position, epoch history or exact training continuation. Loading for training
configures a new optimizer. Cancellation observed before save admission prevents saving.
An already admitted synchronous save may finish; late cancellation reports whether that fully
trained checkpoint was saved. A later terminal or cleanup failure cannot retract published bytes.

## Read the live display

Progress goes to [stderr](../glossary.md#term-standard-streams); prediction data stays on stdout
or in the selected output file. In a supported [TTY](../glossary.md#term-tty), the display redraws
its owned lines every **500 ms**, including during a long synchronous training step:

- ASCII SYNAPTIK wordmark; current epoch, completed steps and progress.
- Elapsed/current/recent epoch times and completed steps per second.
- Actual decoded input-plus-target [Mbit/s](../glossary.md#term-mbit-per-second).
- A bounded [text loss plot](../glossary.md#term-text-loss-plot), to the right when space permits
  and below on a narrow terminal.

For one completed FLOAT32 batch, the logical payload is $b=4B(F+O)$ bytes. Over one stated
measurement interval, throughput is

$$
\operatorname{Mbit/s}=\frac{8\,\Delta\mathrm{bytes}}{10^6\,\Delta\mathrm{seconds}}.
$$

Repeated epochs consume the examples again and count again. This is not CSV-file, network,
parameter-copy or GPU-memory bandwidth. Counts advance only after a successful completed
`TrainingStep`; a zero elapsed interval cannot produce an invented infinite speed.

`--refresh-ms` changes the period. `--plain`, redirected/nonterminal streams, dumb terminals
or an unavailable qualified terminal provider use append-only output without
[ANSI cursor controls](../glossary.md#term-ansi). A final state is emitted even for a run shorter
than one refresh period. The UI reads detached counters/loss only, never live Model state.

A nonfinite detached loss is shown honestly, with a plot gap/marker. It does not itself invalidate
an already committed Training step whose required gradients/update were finite. No finite value
or smooth curve is invented to make the display look better.

### Resize, errors and Ctrl-C

The live renderer queries actual dimensions, fits its output and restores the cursor.
Pinned JLine FFM live output is qualified only for matching POSIX input/stderr terminal identities;
its Windows writer does not reliably target stderr, so that combination uses plain output.

Ctrl-C requests **cooperative** cancellation: the current admitted synchronous step finishes,
its real result is counted, and no next step is scheduled. Cancellation before save admission
prevents saving. If an admitted save finishes before the interrupt is observed, status 130
truthfully reports that the fully trained checkpoint was already saved. The final outcome is
admitted after numerical owners close; a subsequent presentation failure does not undo publication.
Normal cleanup closes the Model before its borrowed Engine. No rollback or hard mid-kernel
interruption is promised. Cursor restoration after SIGKILL, VM failure or a disconnected terminal
cannot be guaranteed.

## Select an existing backend

CPU is the default. Native opt-ins require the real absolute library path:

```sh
# Replace the path with your separately built, qualified Metal native library.
synaptik train --data train.csv --output metal.synaptik \
  --backend metal --metal-library /absolute/path/libsynaptik_metal_foundation.dylib

# Use the exact separately supplied oneDNN library; the preference restricts eligible APIs.
synaptik train --data train.csv --output onednn.synaptik \
  --backend onednn --onednn-library /absolute/path/libdnnl.dylib --onednn-api auto
```

Here `synaptik` means the installed launcher above, optionally placed on your PATH. Metal and
oneDNN eligibility still belongs to their backends; choosing an option is not evidence that every
operation used a native kernel. Unsupported requests and native loading errors remain errors,
not a silently substituted model or hidden new backend. See [backend choice](choose-a-backend.md),
the [CPU native contract](../backend-guide/cpu-backend.md),
the [Metal contract](../backend-guide/metal-backend.md), and the
[bounded autotuning audit](../planning/autotuning-audit.md). This CLI composes ordinary public
Model execution; it does not invent a training-autotuning API or broaden that audit's candidate domain.
