<a id="training-api"></a>
# Training sessions and owned Model convenience

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java)

<a id="purpose-and-implementation-status"></a>
## Purpose and implementation status

`modules/training` (Gradle `:extensions:training`, unchanged Java packages) provides the public
backend-neutral optimizer/session lifecycle. It accepts generic live `TrainingParameters`, Model
Tensor/storage types, and the public Engine facade. Its only direct project dependencies are
public Model and Engine; it imports no NN, Runtime, Prepare, tuning, or concrete backend.
One `TrainingSession` compiles and prepares a scalar forward/backward graph once, reuses it for
every run, maps gradients by Compiler `targetIndex`, and commits optimizer changes only after
complete validation.

The public surface is:

| Type | Contract |
|---|---|
| `Optimizer` | Sealed marker for immutable built-in optimizer configuration. |
| `Sgd` | Learning rate plus optional momentum, dampening, coupled weight decay, and Nesterov. |
| `Adam` / `AdamW` | Bias-corrected first and second moments with coupled / decoupled weight decay. |
| `ParameterGroup` | Immutable ordered stable source `Entry.path()` values with one built-in optimizer configuration; recursive paths only for NN adapters. |
| `GradientMode` | `RESET_AND_STEP`, `ACCUMULATE`, or `ACCUMULATE_AND_STEP`. |
| `TrainingParameters` / `Entry` | Immutable ordered path/Tensor membership with live-owner validation. |
| `TrainingSession` | Closeable owner of one Engine PreparedExecution, parameter bindings, counters, accumulation, and optimizer slots. |
| `TrainingParameterStorage` | Explicit copy of a complete live source into caller-owned shareable native host memory before session open. |
| `TrainingStep` | Detached objective and successful execution/update numbering for one run. |
| `TrainingState` | Detached immutable parameter/optimizer/accumulation snapshot for strict in-memory restore. |

<a id="owned-model-training-convenience"></a>
## Owned Model training convenience

NN's `Model.defineTensor`/`graph` and `load`/`loadGraph` own one private static source.
`configureTraining(targetSignature, objectiveBuilder, optimizer)` invokes the pure builder once
per explicit attempt on owned outputs and a target prototype. The list form takes explicit ordered
target prototypes and a list builder. The scalar objective must be gradient-eligible FLOAT32/FLOAT64;
arbitrary external objective leaves reject. Configuration preserves eval mode and succeeds once.
Failed setup permits explicit retry but Engine selection freezes before callback/native/preparation
work, even on failure. Successful native admission is not repeated on retry.

Qualified fixed recurrent sources now use that same configured TrainingSession and Compiler
first-order AD, including mixed fixed/general sources, ordinary heads and final-state roots.
Fixed presence is not a separate inference-only admission guard. Keep the source's original
fixed/general recipe, storage carriers and rounding boundaries; low-output objectives explicitly
cast to FLOAT32/FLOAT64. CPU low types retain the existing family and actual derivative guards.
Actual backend/AD rejection follows the existing setup ordering: target/source validation,
Engine-choice freeze, objective callback and validation, inference-preparation closure and the
one native parameter transition, then session compilation/preparation. A failed later attempt
can retain changed storage associations and closed inference preparations; it is not an
effect-free fixed-scan rejection. Retry retains a successful native transition.
No optimizer/session history is restored by graph load; configure fresh training on restored values.

Qualified general Scan sources use the same owners, including CPU low types under actual forward,
first-order derivative and binding guards. The explicit F32/F64 `rnnScan` builder uses general
composition; GRU/LSTM, low RNN and old fixed files retain their fixed recipes and numerical sites.
Configure fresh training after graph load; graph files restore no session history.


`train(batch, target)` and the list form delegate `RESET_AND_STEP` to one TrainingSession.runValues.
NN maps declared batch/target slots and owned buffers/constants to final session input order;
Training retains parameters, seed, overlap checks and complete transaction validation. Inference
preparations close before the one native parameter copy; parameter Tensor identity stays fixed.
Compute and graph save observe updated bytes, but a model file contains no objective or optimizer
continuation. The advanced TrainingSession API below is unchanged. See the
[owned training example](../../user-guide/training.md#train-an-owned-static-model).

<a id="session-construction-and-input-mapping"></a>
## Session construction and input mapping

`TrainingSession.open(engine, parameters, objective, optimizer)` borrows the Engine and live source.
The objective must be one gradient-eligible scalar `FLOAT32` or `FLOAT64` expression. The source
must expose at least one parameter, and every parameter is an identity-unique connected gradient
target. Construction snapshots each path, exact Tensor and host-storage identity in caller
encounter order, never sorted, before calling `Engine.compile(...)` exactly once and
`Engine.prepare(...)` exactly once. It owns that prepared handle. Identity-based `run` executes
through `Engine.run(prepared, inputs)`; positional `runValues` executes through
`Engine.runValues(prepared, values)`. Both share the same optimizer transaction and fixed captures.

`TrainingSession.open(engine, parameters, objective, defaultOptimizer, groups)` additionally assigns
an optimizer configuration to each path named by a `ParameterGroup`. Paths are the stable
`Entry.path()` values in the `parameters.entries()` list (for NN, `module.trainingParameters()`
derives recursive paths such as `weight`, `bias`, and dot-separated child paths). Unlisted
parameters use `defaultOptimizer`; an empty groups list is
equivalent to the four-argument overload. Every parameter remains a gradient target. The
session defensively captures group order and each group's path order, and resolves one algorithm per
parameter in source encounter order before compile. A null or duplicate group/path, empty or
blank path, unknown path, or assigned dtype-incompatible coefficient fails before private seed
allocation, compile, or prepare. Group order and path order are exact state/restore identity,
not separate optimizer-step namespaces.

For example, after the `Linear` parameters in the native-copy example below are ready, this
two-rate variation uses learning rate `0.05` for `weight` and the default `0.1` for `bias`:

```java
List<ParameterGroup> groups = List.of(
        new ParameterGroup(List.of("weight"), new Sgd(0.05)));
try (TrainingSession session = TrainingSession.open(
        engine, layer.trainingParameters(), objective, new Sgd(0.1), groups)) {
    session.run(List.of(input, target), GradientMode.RESET_AND_STEP);
    TrainingState snapshot = session.state();
    assert snapshot.parameterGroups().equals(groups);
}
```

Here `layer`, `objective`, `engine`, `input`, and `target` are the live values constructed in the
complete example under [explicit native parameter copy](parameters.md#explicit-native-parameter-copy). The
single run advances one global optimizer step, not one step per group.

For a three-parameter Module with recursive paths `first`, `second`, and `third`, this assignment
fragment shows all three algorithms in one session. The unlisted `first` uses Adam; the ordered
groups assign SGD to `second` and AdamW to `third`. All three still update at the same global step:

```java
List<ParameterGroup> mixed = List.of(
        new ParameterGroup(List.of("second"), new Sgd(0.2)),
        new ParameterGroup(List.of("third"),
                new AdamW(0.1, 0.5, 0.25, 0.01, 0.1)));
TrainingSession session = TrainingSession.open(engine, module.trainingParameters(), objective,
        new Adam(0.1, 0.5, 0.25, 0.01, 0.1), mixed);
```

Here `module` must own those three live native parameters and `objective` must be a scalar
expression connected to all three; the caller closes `session` after use. The
[CPU integration test](../../../testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/TrainingSessionCpuIntegrationTest.java)
constructs and runs that complete scenario.

`session.inputs()` contains only the compiled inputs not supplied internally as captured
parameters or the private cotangent seed, preserving relative final compiled order and including
targets and every other external bindable leaf. Every `run(inputs, mode)` must provide each listed Tensor
identity exactly once; list order is irrelevant. Missing, duplicate, foreign, descriptor-
incompatible, absent, dead, inaccessible, overlapping, or parameter-aliasing storage is rejected.
Gradient publications must contain the scalar objective followed by one derivative-order-one
occurrence for each parameter, with the exact matching `targetIndex`.

`runValues(values, mode)` instead takes exactly one stored value per `session.inputs()` position,
with fresh TensorIds allowed. Full static descriptors must match exactly: dtype, shape, layout
presence/geometry, and requiresGrad. Values are not expressions to execute or recapture. The
session inserts its captured parameters and private seed into final compiled positions without
copying or rebinding prototype storage. Engine checks referenced spans for resolved layouts and
canonical static geometry for absent layouts; unknown/overflowing geometry rejects. The extra
absent-layout capacity check does not change old `run` diagnostics or checks.

Training rejects nonempty external/external, external/parameter and seed overlap, including one
value repeated across matching positions; zero-byte overlap remains exempt. This is deliberately
stronger than generic Engine positional binding. Both methods borrow caller external storage only
through the synchronous call and internal result closure, never for the next run. Keep it alive,
accessible and free of conflicting mutation until return. See the
[fresh-batch scalar example](../../user-guide/training.md#train-fresh-batches-and-targets) for compiled-order mapping
and independently calculated losses and updates.


---

[Next topic](parameters.md)
