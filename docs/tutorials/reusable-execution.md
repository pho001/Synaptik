# 6. Reuse execution with fresh data

## Outcome and prerequisites

Run the fitted multiplication on `[1,3]` and `[2,4]` without rebuilding the model per batch.
Then expose the same idea through public `compile -> prepare -> runValues` owners and retain
the final result after every execution owner closes.

Complete [setup](../examples/README.md#build-the-complete-project),
[train/evaluate](train-and-evaluate.md) and the [save/load distinction](save-close-load.md).
Commands run from the example project with JDK 26 and `LEARNING_CP`. This mode independently
starts at weight `0`, configures SUM loss and `Sgd(0.05)`, and trains to weight `2` before reuse.
It does not read the previous chapter's file.

## Prefer the Model boundary when it is enough

Read `reuse` in
[LearningJourney.java](../examples/learning-journey/src/main/java/io/github/pho001/synaptik/examples/LearningJourney.java).
Its first two computations use `model.compute`: input FLOAT32 `[2,1]` `[1,3]` produces `[2,6]`;
a fresh Tensor with the **same complete descriptor** and values `[2,4]` produces `[4,8]`.
The owned Model caches public preparation and a positional binding plan. Fresh input identity
and values are allowed; per-batch builder calls, cold forward composition or graph rebinding
are not required.

**Capture** records expression operations and their connections as a graph; it does not record
an already-computed prediction. **Compile** validates that meaning and fixes execution ownership.
**Prepare** builds reusable execution resources from it. A **cache** retains setup for reuse,
not the answer to every future input. **Binding** supplies this run's values to required input
positions. Consequently `[2,4]` reaches the same prepared multiplication as `[1,3]` without
reusing the earlier prediction `[2,6]`.

The mode constructs `Engine.standard()` explicitly and selects it with `withEngine` before
training/compute setup freezes Engine choice. The Model **borrows** this Engine; Model close
will not close it. Without an explicit selection, the usual Model path lazily owns its standard
CPU Engine. Neither choice changes the equation or transfers optimizer ownership to Engine.

## Expose the reusable owner once

The second part deliberately uses the lower-level public facade to show what Model normally
hides. It is instructional, not a recommendation to maintain two inference preparations for
such a tiny graph.

1. Create a storage-free canonical `[2,1]` input prototype. `model.forward(prototype)` composes
   one expression against the current private weight `2`. No per-batch recomposition follows.
2. `engine.compile` determines graph meaning and the final input occurrence order.
3. `engine.prepare(compiled)` creates one closeable `PreparedExecution` tied to this Engine.
   Preparation does not bind caller input values or run the graph.
4. For each batch, construct a map from source Tensor identity to current value. The weight
   comes from the live Model state; the input comes from this iteration. Iterate
   `compiled.inputs()` to build the **Compiler's** positional order, rather than assuming
   expression traversal matches the caller's list.
5. `engine.runValues(prepared, bindings)` borrows live input/parameter storage for the run.
   It matches positions and exact descriptors, not the fresh Tensor's identity. This differs
   from `engine.run`, whose binding uses the exact compiled TensorIds.
6. Materialize the sole publication with an `8`-byte ceiling: two FLOAT32 values. Close each
   `RunResult` before the next iteration. The detached copy is independent of that result.

A **publication** is an output occurrence the run exposes, such as this prediction.
**Materialization** copies its numerical bytes into an independent host value. “Host” means
memory accessible from the Java process, as distinct from device-only storage. The
[ownership diagram](../explanation/ownership.md#a-result-is-not-always-an-independent-value)
shows why a live result is needed for copying but a detached copy can outlast it.

Read the ownership stack from the inside out at closure:

```text
Engine (caller-owned)
  Model (borrows Engine, owns private parameter storage/session)
    PreparedExecution (borrows Engine, runs with live Model state)
      RunResult (one invocation) -> materialize -> detached HostTensorValue
```

The loop's result closes first, then the explicit preparation, then Model, then Engine.
Do not train, mutate/rebind state, change mode or close the Model while this explicit preparation
borrows its parameter Tensor/storage. This demo runs serially; it does not teach concurrency
or make a shared mutable Model thread-safe. Prepared reuse avoids repeated setup, not all
per-run work, allocation or output copying.

## Run and interpret

```bash
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

The two routes have the same predictions and the last line reads a detached copy after
Model/Engine/preparation/result close. This does not mean a closed `RunResult` can still
materialize data or a closed Model can compute again. Copy while the appropriate owners are
live; retain only the detached value afterward.

## Useful variation and relevant errors

Add a third same-descriptor batch `[5,7]` to the explicit loop. Its prediction should be
`[10,14]`. Update the final detached-value expectation and expected stdout, then rebuild with
the [setup command](../examples/README.md#build-the-complete-project). No second compile or
prepare belongs inside the loop.

A `[1,2]` or `[3,1]` batch cannot replace `[2,1]` in this fixed signature, even though another
public low-level Engine domain may separately qualify symbolic batch sizes. The
[`invalid-shape` experiment](tensors-and-composition.md#run-the-actual-error-path) demonstrates
owned-Model rejection. Engine `runValues` likewise requires exact static descriptors here.
Preparing a graph compiled by a different Engine raises `IllegalArgumentException`. Reusing a
closed prepared handle or calling `withEngine` after setup raises `IllegalStateException`.
Keep one Engine identity and the closure order shown above; changing values is not changing owners.

## Source and next step

See [`Engine`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java),
[`CompiledGraph`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/CompiledGraph.java),
[`PreparedExecution`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/PreparedExecution.java)
and [`RunResult`](../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/RunResult.java).
The authoritative [public reusable execution contract](../architecture/contracts/runtime-prepare-engine.md#public-reusable-prepared-execution)
and [prepared-graph user guide](../user-guide/running-models.md) cover boundaries beyond this example.
Do not import similarly named Runtime internals: they are not supported application API.

Next: [choose backend/performance work deliberately](backend-and-performance.md).
