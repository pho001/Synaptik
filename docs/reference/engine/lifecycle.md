# Ordinary Engine compile, prepare and run lifecycle

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="current-ordinary-explicit-composition-and-advanced-lifecycle"></a>
## Current ordinary, explicit-composition, and advanced lifecycle

`Engine.standard()` remains the ordinary CPU-only convenience. Every invocation opens one fresh
CPU integration, transfers ownership through `Engine.Builder`, and returns a distinct `Engine`.
Callers that need explicit composition use `Engine.builder()` and transfer opened CPU or Metal
integrations. There is no discovery, service lookup, process-global Engine, implicit fallback, or
runtime owner substitution. `isClosed()` observes when delegated closure begins, and `close()`
uses the advanced owner's thread-safe, idempotent, failure-retaining cleanup protocol.

The ordinary reusable lifecycle is:

```text
ordered Tensor output expressions -> CompiledGraph -> engine.prepare -> PreparedExecution
prepared + exact-ID Tensors in any order -> engine.run       -> RunResult
prepared + fresh values in compiled order -> engine.runValues -> RunResult
exact publication occurrence -> materialize -> detached HostTensorValue
```

`compile(List<Tensor>)` creates a forward-only handle. The three-list overload accepts ordered
outputs, explicit output-aligned cotangent seeds and identity-unique ordered gradient targets.
It infers neither seeds nor targets. `CompiledGraph.inputs()` is the final input membership and
Compiler occurrence-order authority, retaining descriptors and IDs, not caller storage.

`prepare(graph)` accepts a graph from that exact Engine, performs one preparation, and registers
one closeable outward `PreparedExecution` owner. Its `compiledGraph()` returns the exact graph.
The handle borrows the Engine's composition lifetime; Runtime owns persistent resources and the
unique run-lease protocol. Failed preparation, wrapper construction or publication rolls back
acquired resources while preserving the primary failure and suppressed cleanup failures.

`run(prepared, inputs)` matches every compiled TensorId exactly once in arbitrary order and checks
the complete descriptor. `runValues(prepared, values)` instead binds by final compiled input order,
with no ID requirement. It requires exact count and fully static exact descriptors: dtype, shape,
layout presence/geometry and requiresGrad. One fresh Tensor may fill distinct matching slots.
There is no shape specialization: [64,784] rejects a compiled [32,784] slot.

The qualified CPU-only exception is an explicitly symbolic ordinary or bounded all-valid Scan graph, not a static graph
silently specialized at run time. Compile/prepare once with FLOAT32 X `[B,F]` and fixed W `[F,H]`;
`compile(List.of(X.matmul(W).mean()), List.of(scalarOne), List.of(X,W))` supports actual first-order
gradients at changing B through existing `runValues`. Actual inputs retain type/gradient flags and
fixed axes; symbolic inputs require present zero-offset canonical dense layouts (dense read views
qualify). Compiler binds B and predicates; Engine checks logical counts/byte prefixes/storage
before borrowing. Runtime receives same-plan numeric geometry and actual publication descriptors
follow generated values, including dX `[B,F]` and fixed dW `[F,H]`.
At B=0, empty MEAN is NaN, dX is empty and nonempty dW is positive zero; the whole run is not skipped.
Slots/routes/code do not change, and different-B runs own independent writable state.
See the [complete public example](../../user-guide/running-models.md#ordinary-cpu-execution-with-changing-batch-size).
Current symbolic Scan composes ordinary CPU-supported PURE units with fixed time/features/state
widths and the same ordinary dtype, layout and attribute guards. Each complete forward and
selected ordinary-VJP reverse body qualifies independently; source validity alone grants no
CPU or derivative support. BFLOAT16, FLOAT16 and both FP8 types reuse these same CPU owners
with FLOAT32 working arithmetic and declared logical narrowing. The same public lifecycle supplies actual
canonical final/emission/residual descriptors and fixed parameter-shaped gradients; see the
[Scan changing-B example](../../user-guide/running-models.md#symbolic-scan-with-changing-batch-size).
Final qualification is tracked in [Prepare0014](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md).
This does not enable NN Model/Training reuse, symbolic cold loading or wire formats, native,
mixed-owner, tuning, vector or parallel dynamic execution.

Both calls snapshot each selected host-storage association once and validate all inputs before
any borrow: presence, storage dtype, capacity, liveness and current-thread accessibility.
runValues uses checked referenced element/byte spans for resolved layouts and checked canonical
static geometry for absent layouts, rejecting unknown/overflow geometry. That extra absent-layout
check does not change identity-based run. The snapshot neither copies nor pins bytes and is not
atomic across inputs. The caller must keep storage usable and free from conflicting mutation until
result close; replacing a Tensor association later cannot redirect an existing run.

Both methods first admit the Engine, then validate prepared nullness/exact ownership, then
retrieve/check its exact inward delegate, then inspect inputs. This deliberately changes former
direct-run diagnostic precedence: closed preparation wins null/malformed/dead inputs with
`prepared execution is closed` and zero borrows. Null/foreign prepared handles also precede
input-list errors. Closed Engine still reports `advanced engine is closed` before all arguments.
A later close race is decided only by Runtime's unique nonwaiting lease; no outward monitor is
held through execution.

A successful run is a synchronous completion barrier, not a future or in-flight command.
`RunResult` owns completed run-state resources and the borrowed input wrappers. Its immutable
publication list is dense forward-then-gradient order. Gradient metadata identifies the requested
target, derivative order and target index; aliased values remain distinct occurrence objects.
`materialize` authenticates the exact occurrence from that result and returns a fresh detached
`HostTensorValue`, without caching or deduplication.

Repeated/concurrent runs reuse immutable recipes, precomputed geometry and direct input/publication
adapters, with isolated mutable Runtime state and results. They do not mutate prototypes, construct
synthetic Tensors, recapture, compile, prepare, query providers/registries, or add per-element binding
work. There is no device-overlap, throughput, fairness, asynchronous or cancellation promise.

Prepared close rejects new runs but does not close returned results. A leased run may finish;
Runtime defers prepared-resource cleanup to its last lease release without making close wait.
Result materialization remains valid after prepared close while result and Engine are open.
Engine shutdown rejects new work, waits for admitted operations, then closes results in reverse
run-publication order, ordinary/advanced preparations in reverse preparation-publication order,
and integrations in reverse registration order. Cleanup is attempt-all and failure-retaining.
Metadata stays readable after closure; detached values outlive result, preparation, Engine and
caller storage.

`InferenceSession` and `Engine.session` are removed without a compatibility shim. Migrate
`engine.session(graph)` to `engine.prepare(graph)` and `session.run(inputs)` to
`engine.run(prepared, inputs)`, then recompile. Fresh values explicitly opt into runValues.
`TrainingSession` and the advanced lifecycle remain; this is not a Training API redesign.
See the [fresh-value example](../../user-guide/running-models.md#run-fresh-values-in-compiled-order).

Optional `prepareTuned(...)` is a cache-first preparation path for eligible single-owner CPU plans
and exact singleton-NEG Metal plans. This CPU example uses a `CONTIGUOUS` expression, which has no
eligible local tuning handoff in the current standard composition, so the allowed policy
deterministically demonstrates explicit safe fallback:

```java
import io.github.pho001.synaptik.config.tuning.ModelAutotuningConfig;
import io.github.pho001.synaptik.engine.CompiledGraph;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.ModelAutotuningPreparation;
import io.github.pho001.synaptik.engine.ModelAutotuningRequest;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    Shape shape = Shape.of(2);
    TensorDescriptor descriptor = new TensorDescriptor(
            DataType.FLOAT32, shape,
            Optional.of(LayoutDescriptor.contiguous(shape)), false);
    MemorySegment storage = arena.allocate(2L * Float.BYTES, Float.BYTES);
    storage.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 0, 1.0f);
    storage.setAtIndex(java.lang.foreign.ValueLayout.JAVA_FLOAT, 1, 2.0f);
    Tensor input = TensorFactory.create(descriptor, Optional.empty(), Optional.of(
            new MemorySegmentStorage(DataType.FLOAT32, 2, storage)));
    CompiledGraph graph = engine.compile(List.of(input.contiguous()));

    ModelAutotuningConfig config = new ModelAutotuningConfig(
            ModelAutotuningConfig.Objective.MIN_MEDIAN_ELAPSED_NANOS,
            new ModelAutotuningConfig.Budget(1, 2, 1, 3),
            new ModelAutotuningConfig.RepresentativeProfileIdentity(
                    1, new byte[] {0x01}),
            ModelAutotuningConfig.FallbackPolicy.ALLOW_SAFE_HEURISTIC,
            Path.of("build", "example-workload-cache.bin"),
            new ModelAutotuningConfig.CompletePlanBudget(1, 0, 1, 1L, 0L),
            Path.of("build", "unused-model-plan-cache.bin"));
    ModelAutotuningRequest request = new ModelAutotuningRequest(
            config,
            new ModelAutotuningRequest.ModelIdentity(1, new byte[] {0x42}),
            List.of(input));

    ModelAutotuningPreparation preparation = engine.prepareTuned(graph, request);
    assert preparation.outcome()
            == ModelAutotuningPreparation.Outcome.SAFE_HEURISTIC_FALLBACK;
    assert preparation.evidence().isEmpty();
    try (var prepared = preparation.preparedExecution();
            var result = engine.run(prepared, List.of(input))) {
        assert result.publications().size() == 1;
    }
}
```

The request snapshots the list and identity bytes but retains the live caller-owned Tensor and
storage. Phase 1 reuses or measures the sole eligible local workload. Its authenticated winner is
then fixed while Phase 2 considers the backend-owned complete candidates. For CPU those are the
retained complete topology and representation alternatives; for Metal there is one complete
candidate fixed to the Phase-1-selected route. Phase 2 performs every exact correctness action
before timing; correctness, warmup, and timed executions each use a fresh preparation and fresh
Runtime state. After winner authentication and representative cleanup, the returned production
handle is prepared afresh. Every trial closes its result before its temporary preparation on every
path. The `ModelAutotuningPreparation` metadata carrier is not another closeable owner: its
contained `PreparedExecution` alone owns the production preparation.

A tuned result has outcome `TUNED`. Its immutable evidence contains the Phase-1 workload rows and
required `completePlan` evidence: compatibility and reuse scope, the exact Phase-2 budget, source,
winner identity and summary, plus one correctness-and-sample row per measured candidate. A future
persistent hit has no candidate rows because compact reusable cache records do not contain rich
per-call correctness actions or raw samples. Current CPU and Metal complete-plan evidence is
always `SESSION` and measured, and its explicit model-plan path receives no filesystem access.
Metal workload evidence is also always `SESSION` and measured. Model and representative-profile
identities are caller-defined tuning-evidence labels, not a numerical selector or workload-cache
key. Current scope excludes multiple
workloads or occurrences, Compiler graph alternatives, Planning owner or partition alternatives,
multiple partitions, and mixed backends.

The current one-shot forms are exactly:

```java
HostTensorValue value = engine.compute(output);
HostTensorValue boundedValue = engine.compute(output, maximumTotalBytes);
List<HostTensorValue> values = engine.compute(outputs);
List<HostTensorValue> boundedValues = engine.compute(outputs, maximumTotalBytes);
```

Each call transiently and iteratively inventories provenance-free leaves reachable through Model
Tensor-expression provenance by exact object identity. After compilation,
`CompiledGraph.inputs()` alone selects membership and order by `TensorId`; Engine does not traverse
Compiler intermediate representation (IR), reconstruct liveness, or retain Tensor/provenance
state. Parameters and buffers need no Engine-specific category: their current Tensor bindings are
discovered like any other reachable provenance-free leaf. Each call freshly compiles with the
ordinary fixed forward-only settings, prepares, runs once, verifies the complete forward-publication set,
preflights the sum of canonical returned payload lengths before copying, materializes every
occurrence in requested order, and closes its temporary result before its temporary preparation
before the call returns. The ordered form
returns an immutable list of immutable detached values; distinct occurrences are copied
independently even if they select one inward representation. The limit covers returned canonical
payload bytes only, not inputs, recipes, Runtime buffers or workspaces, object overhead, defensive
copies, peak memory, or other allocation.

One Engine admission covers that complete sequence. Engine closure waits for an admitted call,
including its result cleanup; a call that loses admission observes the closed-Engine failure
before argument validation. A failure returns no partial result. Temporary result-then-preparation
cleanup still runs, with distinct cleanup failures suppressed on the primary failure in cleanup
order. Selected leaf storage remains
caller-owned and must stay live, accessible, and free from conflicting mutation until the
synchronous call completes; unselected discovered storage is not inspected. The `compute(...)`
forms create no compile/prepared/result/value/Tensor-inventory cache or reuse and add no backward,
implicit target/seed behavior, tuning, cross-backend transfer, or Tensor execution method.

The current one-shot backward form is:

```java
ScalarObjectiveBackwardResult result =
        engine.backward(objective, targets, maximumTotalBytes);
```

`objective` must be a scalar floating gradient-eligible Tensor expression. `targets` is an
explicit ordered non-empty list whose elements are unique by exact object identity; its order
defines `result.gradients()`. Targets may be leaves, intermediates, or the objective itself when
Compiler accepts their gradient semantics and differentiable connection. Engine inventories
reachable provenance-free leaves before compilation, then lets final `CompiledGraph.inputs()`
select authoritative input membership and order. It accepts no explicit input list, infers
neither targets nor liveness, mutates no Tensor, and creates no eager tape.

Every call freshly compiles one stage with an absent scalar cotangent seed and
`DisconnectedPolicy.ERROR`, prepares, runs, validates the objective-first then target-ordered
publications, and preflights every static/resolved per-value byte count plus the exact aggregate
before the first copy. Each occurrence is copied independently, including inward aliases. The
limit covers only the objective and gradient canonical payloads. Cleanup completes before return;
Engine close waits through an admitted call, concurrent calls use isolated run state, and cleanup
failures follow the ordinary primary/suppressed failure rules. The returned
`ScalarObjectiveBackwardResult`, its objective, and all gradients are detached and remain readable
after Engine and caller storage close.

The fixed seed and policy keep this convenience narrow. Use ordinary explicit-seed
`compile -> prepare -> engine.run -> materialize` for reusable execution, explicit cotangent
seeds, or multiple forward outputs. Direct `prepare -> run` remains available when the prepared
handle itself must be owned. Use `AdvancedEngine.compile(...)` with a
`FunctionalGradientRequest` for `ZERO`, two stages, higher-order construction, or the full policy
surface.

This complete supported example uses only the current CPU static/resolved `CONTIGUOUS` path:

```java
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;

static Tensor input(Arena arena, float... values) {
    Shape shape = Shape.of(values.length);
    TensorDescriptor descriptor = new TensorDescriptor(
            DataType.FLOAT32,
            shape,
            Optional.of(LayoutDescriptor.contiguous(shape)),
            false);
    MemorySegment source = MemorySegment.ofArray(values);
    MemorySegment storage = arena.allocate(source.byteSize(), Float.BYTES);
    MemorySegment.copy(source, 0, storage, 0, source.byteSize());
    return TensorFactory.create(
            descriptor,
            Optional.empty(),
            Optional.of(new MemorySegmentStorage(
                    DataType.FLOAT32, values.length, storage)));
}

List<HostTensorValue> retained;
try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    Tensor left = input(arena, 1.5f, -2.25f);
    Tensor right = input(arena, 3.25f, 4.5f);
    retained = engine.compute(
            List.of(left.contiguous(), right.contiguous()),
            4L * Float.BYTES);
    assert retained.size() == 2;
    assert retained.get(0).bytes().getFloat(0) == 1.5f;
    assert retained.get(1).bytes().getFloat(0) == 3.25f;
}
ByteBuffer first = retained.get(0).bytes();
assert first.isReadOnly();
assert first.remaining() == 2 * Float.BYTES;
```

The expression leaves are discovered automatically, while the returned positions still follow
the output request. Both values remain readable after Engine and caller arena closure, while the
arena itself stayed caller-owned. Every call compiles and prepares afresh. Use
`compile -> prepare -> engine.run -> materialize` when a prepared recipe should be reused or only
selected publications should be copied. The direct prepared-handle lifecycle remains available
when its lower-level ownership is specifically required.

This complete one-shot backward example uses the currently supported resolved scalar `FLOAT32`
CPU path. Its input storage remains caller-owned, while the two returned values remain readable
after both the Engine and storage arena close:

```java
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.ScalarObjectiveBackwardResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Optional;

ScalarObjectiveBackwardResult retainedBackward;
try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    Shape shape = Shape.scalar();
    TensorDescriptor descriptor = new TensorDescriptor(
            DataType.FLOAT32,
            shape,
            Optional.of(LayoutDescriptor.contiguous(shape)),
            true);
    MemorySegment source = MemorySegment.ofArray(new float[] {2.5f});
    MemorySegment storage = arena.allocate(source.byteSize(), Float.BYTES);
    MemorySegment.copy(source, 0, storage, 0, source.byteSize());
    Tensor input = TensorFactory.create(
            descriptor,
            Optional.empty(),
            Optional.of(new MemorySegmentStorage(DataType.FLOAT32, 1, storage)));
    Tensor objective = input.contiguous();

    retainedBackward = engine.backward(
            objective,
            List.of(input),
            2L * Float.BYTES);
    assert retainedBackward.objective().bytes().getFloat(0) == 2.5f;
    assert retainedBackward.gradients().getFirst().bytes().getFloat(0) == 1.0f;
}
ByteBuffer objectiveBytes = retainedBackward.objective().bytes();
ByteBuffer gradientBytes = retainedBackward.gradients().getFirst().bytes();
assert objectiveBytes.getFloat(0) == 2.5f;
assert gradientBytes.getFloat(0) == 1.0f;
```

`contiguous()` supplies the non-empty supported CPU operation. Compiler supplies the absent
positive-one seed, which the completed Compiler 0006B5 -> Prepare 0005 -> CPU 0010H chain carries
as a source-only publication beside that partition. This proves the scalar objective and its
positive-one gradient on the current path; it does not promise pure zero-node or mixed-backend
composition, broader operation support, optimizer behavior, or a training session.

For a single already-constructed Tensor expression, the complete one-shot execution shape is:

```java
try (Engine engine = Engine.standard()) {
    HostTensorValue result = engine.compute(output);
}
```

Here `output` already retains the Model expression provenance leading to its leaves. The returned
value is detached; closing the Engine does not invalidate it.

An old generic/custom/declarative NN `Model` follows the same boundary. This conceptual example intentionally assumes only the
existing `model`, `input`, and `Tensor` contracts and invents no construction or initialization
API:

```java
Tensor output = model.forward(input);
try (Engine engine = Engine.standard()) {
    HostTensorValue result = engine.compute(output);
}
```

`model.forward(input)` constructs the Tensor expression directed acyclic graph (DAG); it does not
execute it. `engine.compute(output)` performs execution and host materialization. Repeated model
execution should retain the expression and use
`compile -> prepare -> engine.run(explicit inputs)` rather than paying the one-shot compile and
prepare cost on every call.


---

[Previous topic](../runtime/prepare.md) · [Next topic](../nn/owned-model.md)
