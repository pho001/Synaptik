# 6. Reuse execution with fresh data

## Outcome and prerequisites

[Run](../glossary.md#term-run) the fitted multiplication on `[1,3]` and `[2,4]` without rebuilding the [model](../glossary.md#term-model) per [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters).
Then expose the same idea through public `compile -> prepare -> runValues` [owners](../glossary.md#term-owner) and retain
the final result after every execution [owner](../glossary.md#term-owner) closes.

Complete [setup](../examples/README.md#build-the-complete-project),
[train/evaluate](train-and-evaluate.md) and the [save/load distinction](save-close-load.md).
Commands run from the example project with [JDK](../glossary.md#term-java-development-kit-jdk) 26 and `LEARNING_CP`. This mode independently
starts at [weight](../glossary.md#term-weight) `0`, configures SUM [loss](../glossary.md#loss) and `Sgd(0.05)`, and trains to weight `2` before reuse.
It does not read the previous chapter's file.

## Prefer the [Model](../glossary.md#term-model) boundary when it is enough

Read `reuse` in
[LearningJourney.java](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java).
Its first two computations use `model.compute`: input [FLOAT32](../glossary.md#term-float32) `[2,1]` `[1,3]` produces `[2,6]`;
a fresh [Tensor](../glossary.md#tensor) with the **same complete [descriptor](../glossary.md#term-descriptor)** and values `[2,4]` produces `[4,8]`.
The [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions) caches public preparation and a positional [binding](../glossary.md#term-binding) plan. Fresh input [identity](../glossary.md#term-identity-meanings)
and values are allowed; per-[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) builder calls, cold forward composition or [graph](../glossary.md#graph) rebinding
are not required.

### Which caller supplies each batch?

The application's `reuse` method calls `model.compute(batch)`. Model is the callee and hides
its cached compilation/preparation. Internally Model calls Engine with current input values
and its own parameter. The application must keep batch storage usable and unchanged until
`compute` returns; Model closes the internal run result and returns copied output bytes.
Changing batch values does not change the prototype that described the input.

**[Capture](../glossary.md#term-capture-meanings)** records expression operations and their connections as a graph; it does not record
an already-computed [prediction](../glossary.md#term-prediction). **[Compile](../glossary.md#compile)** validates that meaning and fixes execution [ownership](../glossary.md#term-ownership-meanings).
**[Prepare](../glossary.md#prepare)** builds reusable execution resources from it. A **[cache](../glossary.md#term-cache)** retains setup for reuse,
not the answer to every future input. **Binding** supplies this [run](../glossary.md#term-run)'s values to required input
positions. Consequently `[2,4]` reaches the same prepared multiplication as `[1,3]` without
reusing the earlier prediction `[2,6]`.

The application constructs `Engine.standard()` and passes that Engine to `model.withEngine`
before training/compute setup freezes the choice. The **application owns the Engine** and
closes it after Model; Model only borrows it. Without this selection, Model lazily creates and
owns its standard CPU Engine. Neither choice changes the equation or gives Engine responsibility
for optimizer mathematics.

## Expose the reusable [owner](../glossary.md#term-owner) once

The second part deliberately uses the lower-level public [facade](../glossary.md#term-facade) to show what [Model](../glossary.md#term-model) normally
hides. It is instructional, not a recommendation to maintain two [inference](../glossary.md#term-inference) preparations for
such a tiny [graph](../glossary.md#graph).

1. Create a storage-free [canonical](../glossary.md#term-canonical-form) `[2,1]` input [prototype](../glossary.md#term-prototype).
   The application calls `model.forward(prototype)` to compose one expression against Model's
   private [weight](../glossary.md#term-weight), whose current value is `2` after training.
   No per-[batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) recomposition follows.
2. `engine.compile` determines graph meaning and the final input [occurrence](../glossary.md#term-occurrence) order.
3. `engine.prepare(compiled)` creates one closeable [`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution) tied to this Engine.
   Preparation does not bind caller input values or run the graph.
4. For each batch, the application maps source Tensor identity to a current stored value.
   The weight is a borrowed reference from live Model state; the input is this iteration's
   batch. It iterates `compiled.inputs()` to build **Compiler's positional order**, not
   an assumed expression traversal order.
5. The application calls `engine.runValues(prepared, bindings)`. Engine borrows those stores
   until the returned `RunResult` closes; it neither adopts them nor permanently replaces the
   prototype. Positions and full descriptors must match, but fresh Tensor identities are allowed.
   By contrast, `engine.run` requires the exact compiled input identities.
6. The application asks that result to materialize its sole output with an `8`-byte ceiling:
   two FLOAT32 elements at four bytes each. It closes each `RunResult` before the next
   iteration. Only the copied `HostTensorValue` is retained afterward.

A **publication** is an output occurrence the [run](../glossary.md#term-run) exposes, such as this [prediction](../glossary.md#term-prediction).
**[Materialization](../glossary.md#materialization)** copies its numerical bytes into an independent [host](../glossary.md#term-host) value. “Host” means
memory accessible from the Java process, as distinct from [device](../glossary.md#term-device)-only storage. The
[ownership diagram](../explanation/ownership.md#a-result-is-not-always-an-independent-value)
shows why a live result is needed for copying but a detached copy can outlast it.

Read the [ownership](../glossary.md#term-ownership-meanings) stack from the inside out at closure:

```text
Engine (created and closed by the Java application)
  Model (borrows Engine, owns private parameter storage/session)
    PreparedExecution (borrows Engine, runs with live Model state)
      RunResult (one invocation) -> materialize -> detached HostTensorValue
```

The loop's result closes first, then the explicit preparation, then Model, then [Engine](../glossary.md#term-engine).
Do not train, mutate/[rebind](../glossary.md#capture-graph-ports-and-reconstruction) state, change mode or close the Model while this explicit preparation
borrows its [parameter](../glossary.md#term-parameter) [Tensor](../glossary.md#tensor)/storage. This demo runs serially; it does not teach [concurrency](../glossary.md#term-concurrency)
or make a shared [mutable](../glossary.md#term-mutable) Model [thread-safe](../glossary.md#term-thread-safe). Prepared reuse avoids repeated setup, not all
per-run work, allocation or output copying.

## [Run](../glossary.md#term-run) and interpret

```bash
# Reuse setup, not previous answers: both same-descriptor batches execute.
java --enable-native-access=ALL-UNNAMED -cp "$LEARNING_CP" \
  io.github.pho001.synaptik.examples.LearningJourney reuse
```

Expected application stdout, checked against the installed private candidate; this is not a benchmark:

```text
== reuse ==
reuse.model-first=[2.0, 6.0]
reuse.model-fresh=[4.0, 8.0]
reuse.prepared=[2.0, 6.0]
reuse.prepared=[4.0, 8.0]
reuse.after-all-owners-close=[4.0, 8.0]
```

The two routes have the same [predictions](../glossary.md#term-prediction) and the last line reads a [detached](../glossary.md#detached-model-results) copy after
[Model](../glossary.md#term-model)/[Engine](../glossary.md#term-engine)/preparation/result close. This does not mean a closed [`RunResult`](../glossary.md#run-result--runresult) can still
materialize data or a closed Model can compute again. Copy while the appropriate [owners](../glossary.md#term-owner) are
live; retain only the [detached value](../glossary.md#detached-model-results) afterward.

## Useful variation and relevant errors

Add a third same-[descriptor](../glossary.md#term-descriptor) [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) `[5,7]` to the explicit loop. Its [prediction](../glossary.md#term-prediction) should be
`[10,14]`. Update the final [detached](../glossary.md#detached-model-results)-value expectation and expected stdout, then rebuild with
the [setup command](../examples/README.md#build-the-complete-project). No second [compile](../glossary.md#compile) or
[prepare](../glossary.md#prepare) belongs inside the loop.

A `[1,2]` or `[3,1]` batch cannot replace `[2,1]` in this fixed [signature](../glossary.md#term-signature), even though another
public low-level [Engine](../glossary.md#term-engine) domain may separately qualify symbolic batch sizes. The
[`invalid-shape` experiment](tensors-and-composition.md#run-the-actual-error-path) demonstrates
owned-[Model](../glossary.md#term-model) rejection. Engine `runValues` likewise requires exact static descriptors here.
Preparing a [graph](../glossary.md#graph) compiled by a different Engine raises `IllegalArgumentException`. Reusing a
closed prepared handle or calling `withEngine` after setup raises `IllegalStateException`.
Keep one Engine [identity](../glossary.md#term-identity-meanings) and the closure order shown above; changing values is not changing [owners](../glossary.md#term-owner).

## Source and next step

See [`Engine`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)<span data-glossary-navigation="true"> ([Engine glossary](../glossary.md#term-engine))</span>,
[`CompiledGraph`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/CompiledGraph.java),
[`PreparedExecution`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/PreparedExecution.java)<span data-glossary-navigation="true"> ([PreparedExecution glossary](../glossary.md#prepared-execution--preparedexecution))</span>
and [`RunResult`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/RunResult.java)<span data-glossary-navigation="true"> ([RunResult glossary](../glossary.md#run-result--runresult))</span>.
The authoritative [public reusable execution contract](../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)
and [prepared-graph user guide](../user-guide/running-models.md)<span data-glossary-navigation="true"> ([graph glossary](../glossary.md#graph))</span> cover boundaries beyond this example.
Do not import similarly named [Runtime](../glossary.md#term-runtime) internals: they are not supported application [API](../glossary.md#term-application-programming-interface-api).

Next: [choose backend/performance work deliberately](backend-and-performance.md)<span data-glossary-navigation="true"> ([backend glossary](../glossary.md#backend))</span>.
