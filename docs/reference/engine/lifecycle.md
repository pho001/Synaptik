# Ordinary [Engine](../../glossary.md#term-engine) [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare) and [run](../../glossary.md#term-run) [lifecycle](../../glossary.md#lifecycle)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/engine/src/main/java/io/github/pho001/synaptik/engine/Engine.java)

The three lifecycle stages have different jobs: **compile** creates a logical recipe,
**prepare** arranges [backend](../../glossary.md#backend) executable resources, and **run** executes with supplied values.
The returned result is a resource **[lease](../../glossary.md#term-lease)**, not an [asynchronous](../../glossary.md#term-asynchronous) promise. Explicit
**[materialization](../../glossary.md#materialization)** makes a [detached](../../glossary.md#detached-model-results) [host](../../glossary.md#term-host) copy that can survive closing that result.

<a id="current-ordinary-explicit-composition-and-advanced-lifecycle"></a>
## Current ordinary, explicit-composition, and advanced [lifecycle](../../glossary.md#lifecycle)

`Engine.standard()` remains the ordinary [CPU](../../glossary.md#term-cpu)-only convenience. Every invocation opens one fresh
CPU integration, transfers [ownership](../../glossary.md#term-ownership-meanings) through `Engine.Builder`, and returns a distinct [`Engine`](../../glossary.md#term-engine).
Callers that need explicit composition use `Engine.builder()` and transfer opened CPU or [Metal](../../glossary.md#term-metal)
integrations. There is no discovery, service [lookup](../../glossary.md#term-lookup), process-global Engine, implicit fallback, or
[runtime](../../glossary.md#term-runtime) [owner](../../glossary.md#term-owner) substitution. `isClosed()` observes when delegated closure begins, and `close()`
uses the advanced owner's [thread-safe](../../glossary.md#term-thread-safe), [idempotent](../../glossary.md#term-idempotent), failure-retaining cleanup protocol.

The ordinary reusable lifecycle is:

```text
ordered Tensor output expressions -> CompiledGraph -> engine.prepare -> PreparedExecution
prepared + exact-ID Tensors in any order -> engine.run       -> RunResult
prepared + fresh values in compiled order -> engine.runValues -> RunResult
exact publication occurrence -> materialize -> detached HostTensorValue
```

`compile(List<Tensor>)` creates a forward-only handle. The three-list overload accepts ordered
outputs, explicit output-aligned [cotangent](../../glossary.md#cotangent) [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings) and [identity](../../glossary.md#term-identity-meanings)-unique ordered [gradient](../../glossary.md#term-gradient) [targets](../../glossary.md#term-target).
It infers neither seeds nor targets. `CompiledGraph.inputs()` is the final input membership and
[Compiler](../../glossary.md#term-compiler) [occurrence](../../glossary.md#term-occurrence)-order authority, retaining [descriptors](../../glossary.md#term-descriptor) and IDs, not caller storage.

`prepare(graph)` accepts a [graph](../../glossary.md#graph) from that exact Engine, performs one preparation, and registers
one closeable outward [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) owner. Its `compiledGraph()` returns the exact graph.
The handle borrows the Engine's composition [lifetime](../../glossary.md#term-lifetime); Runtime owns persistent resources and the
unique [run](../../glossary.md#term-run)-[lease](../../glossary.md#term-lease) protocol. Failed preparation, wrapper construction or [publication](../../glossary.md#term-publication) rolls back
acquired resources while preserving the primary failure and suppressed cleanup failures.

`run(prepared, inputs)` matches every compiled [TensorId](../../glossary.md#tensorid) exactly once in arbitrary order and checks
the complete descriptor. `runValues(prepared, values)` instead binds by final compiled input order,
with no ID requirement. It requires exact count and fully static exact descriptors: [dtype](../../glossary.md#data-type--datatype), [shape](../../glossary.md#shape),
[layout](../../glossary.md#layout) presence/geometry and requiresGrad. One fresh [Tensor](../../glossary.md#tensor) may fill distinct matching slots.
There is no shape [specialization](../../glossary.md#term-specialization): [64,784] rejects a compiled [32,784] slot.

The qualified CPU-only exception is an explicitly symbolic ordinary or bounded all-valid [Scan](../../glossary.md#term-cumulative-scan) graph, not a static graph
silently specialized at run time. [Compile](../../glossary.md#compile)/[prepare](../../glossary.md#prepare) once with [FLOAT32](../../glossary.md#term-float32) X `[B,F]` and fixed W `[F,H]`;
`compile(List.of(X.matmul(W).mean()), List.of(scalarOne), List.of(X,W))` supports actual first-order
gradients at changing B through existing `runValues`. Actual inputs retain type/gradient flags and
fixed [axes](../../glossary.md#term-axis); symbolic inputs require present zero-offset [canonical](../../glossary.md#term-canonical-form) dense layouts (dense read [views](../../glossary.md#term-view-meanings)
qualify). Compiler binds B and predicates; Engine checks logical counts/byte prefixes/storage
before [borrowing](../../glossary.md#term-borrowing). Runtime receives same-plan numeric geometry and actual publication descriptors
follow generated values, including dX `[B,F]` and fixed dW `[F,H]`.
At B=0, empty [MEAN](../../glossary.md#term-mean) is [NaN](../../glossary.md#term-nan), dX is empty and nonempty dW is [positive zero](../../glossary.md#nan-infinity-and-signed-zero); the whole run is not skipped.
Slots/routes/code do not change, and different-B runs own independent writable state.
See the [complete public example](../../user-guide/running-models.md#ordinary-cpu-execution-with-changing-batch-size).
Current [symbolic Scan](../../glossary.md#symbolic-scan) composes ordinary CPU-supported [PURE](../../glossary.md#term-pure) units with fixed time/[features](../../glossary.md#term-feature)/state
widths and the same ordinary dtype, layout and attribute guards. Each complete forward and
selected ordinary-[VJP](../../glossary.md#term-vector-jacobian-product-vjp) reverse body qualifies independently; source validity alone grants no
CPU or [derivative](../../glossary.md#term-derivative) support. [BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16) and both [FP8](../../glossary.md#term-fp8) types reuse these same CPU owners
with FLOAT32 [working arithmetic](../../glossary.md#accumulator-type) and declared logical [narrowing](../../glossary.md#term-narrowing). The same public lifecycle supplies actual
canonical final/emission/[residual](../../glossary.md#scan-residual-port-and-history-demand) descriptors and fixed [parameter](../../glossary.md#term-parameter)-shaped gradients; see the
[Scan changing-B example](../../user-guide/running-models.md#symbolic-scan-with-changing-batch-size).
Final [qualification](../../glossary.md#term-qualification) is tracked in [Prepare0014](../../planning/modules/prepare/tasks/0014-symbolic-scan-cpu-vertical.md).
This does not enable [NN Model](../../glossary.md#model-module-buffer-and-scan-distinctions)/Training reuse, symbolic cold loading or wire formats, native,
mixed-owner, tuning, [vector](../../glossary.md#term-vector) or parallel dynamic execution.

Both calls [snapshot](../../glossary.md#term-snapshot) each selected [host](../../glossary.md#term-host)-storage association once and validate all inputs before
any borrow: presence, storage dtype, capacity, [liveness](../../glossary.md#term-liveness) and current-thread accessibility.
runValues uses checked referenced [element](../../glossary.md#term-element)/byte spans for resolved layouts and checked canonical
static geometry for absent layouts, rejecting unknown/[overflow](../../glossary.md#term-overflow) geometry. That extra absent-layout
check does not change identity-based run. The snapshot neither copies nor pins bytes and is not
[atomic](../../glossary.md#resource-transaction-and-rollback) across inputs. The caller must keep storage usable and free from conflicting mutation until
result close; replacing a Tensor association later cannot redirect an existing run.

Both methods first admit the Engine, then validate prepared nullness/exact ownership, then
retrieve/check its exact inward delegate, then inspect inputs. This deliberately changes former
direct-run diagnostic precedence: closed preparation wins null/malformed/dead inputs with
`prepared execution is closed` and zero borrows. Null/foreign prepared handles also precede
input-list errors. Closed Engine still reports `advanced engine is closed` before all [arguments](../../glossary.md#term-argument).
A later close [race](../../glossary.md#term-race) is decided only by Runtime's unique nonwaiting lease; no outward monitor is
held through execution.

A successful run is a [synchronous](../../glossary.md#term-synchronous) completion barrier, not a future or in-flight command.
[`RunResult`](../../glossary.md#run-result--runresult) owns completed run-state resources and the [borrowed](../../glossary.md#ownership-borrowing-and-lease) input wrappers. Its [immutable](../../glossary.md#term-immutable)
publication list is dense forward-then-gradient order. Gradient [metadata](../../glossary.md#term-metadata) identifies the requested
target, derivative order and target index; aliased values remain distinct occurrence objects.
`materialize` authenticates the exact occurrence from that result and returns a fresh [detached](../../glossary.md#detached-model-results)
[`HostTensorValue`](../../glossary.md#host-tensor-value--hosttensorvalue), without caching or deduplication.

Repeated/concurrent runs reuse immutable recipes, precomputed geometry and direct input/publication
adapters, with isolated [mutable](../../glossary.md#term-mutable) Runtime state and results. They do not mutate [prototypes](../../glossary.md#term-prototype), construct
synthetic Tensors, recapture, compile, prepare, query providers/registries, or add per-element [binding](../../glossary.md#term-binding)
work. There is no [device](../../glossary.md#term-device)-overlap, [throughput](../../glossary.md#term-throughput), fairness, [asynchronous](../../glossary.md#term-asynchronous) or [cancellation](../../glossary.md#term-cancellation) promise.

Prepared close rejects new runs but does not close returned results. A leased run may finish;
Runtime defers prepared-resource cleanup to its last lease [release](../../glossary.md#term-release) without making close wait.
Result [materialization](../../glossary.md#materialization) remains valid after prepared close while result and Engine are open.
Engine shutdown rejects new work, waits for admitted operations, then closes results in reverse
run-publication order, ordinary/advanced preparations in reverse preparation-publication order,
and integrations in reverse registration order. Cleanup is attempt-all and failure-retaining.
Metadata stays readable after closure; detached values outlive result, preparation, Engine and
caller storage.

[`InferenceSession`](../../glossary.md#former-inference-session--inferencesession) and `Engine.session` are removed without a compatibility shim. Migrate
`engine.session(graph)` to `engine.prepare(graph)` and `session.run(inputs)` to
`engine.run(prepared, inputs)`, then recompile. Fresh values explicitly opt into runValues.
[`TrainingSession`](../../glossary.md#training-session--trainingsession) and the advanced lifecycle remain; this is not a Training [API](../../glossary.md#term-application-programming-interface-api) redesign.
See the [fresh-value example](../../user-guide/running-models.md#run-fresh-values-in-compiled-order).

Optional `prepareTuned(...)` is a [cache](../../glossary.md#term-cache)-first preparation path for eligible single-owner CPU plans
and exact [singleton](../../glossary.md#term-singleton)-NEG Metal plans. This CPU example uses a [`CONTIGUOUS`](../../glossary.md#term-contiguous) expression, which has no
eligible local tuning [handoff](../../glossary.md#term-handoff) in the current standard composition, so the allowed policy
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
then fixed while Phase 2 considers the [backend](../../glossary.md#backend)-owned complete candidates. For CPU those are the
retained complete [topology](../../glossary.md#term-topology) and representation alternatives; for Metal there is one complete
candidate fixed to the Phase-1-selected route. Phase 2 performs every exact correctness action
before timing; correctness, [warmup](../../glossary.md#term-warmup), and timed executions each use a fresh preparation and fresh
Runtime state. After winner [authentication](../../glossary.md#term-authentication) and representative cleanup, the returned production
handle is prepared afresh. Every trial closes its result before its temporary preparation on every
path. The `ModelAutotuningPreparation` metadata [carrier](../../glossary.md#term-storage-carrier) is not another closeable owner: its
contained `PreparedExecution` alone owns the production preparation.

A tuned result has outcome `TUNED`. Its immutable evidence contains the Phase-1 workload rows and
required `completePlan` evidence: compatibility and reuse [scope](../../glossary.md#term-scope), the exact Phase-2 budget, source,
winner identity and summary, plus one correctness-and-[sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) row per measured candidate. A future
persistent hit has no candidate rows because compact reusable cache records do not contain rich
per-call correctness actions or raw samples. Current CPU and Metal complete-plan evidence is
always `SESSION` and measured, and its explicit [model](../../glossary.md#term-model)-plan path receives no filesystem access.
Metal workload evidence is also always `SESSION` and measured. Model and representative-profile
identities are caller-defined tuning-evidence labels, not a numerical selector or workload-[cache
key](../../glossary.md#term-cache-key). Current scope excludes multiple
workloads or occurrences, Compiler graph alternatives, [Planning](../../glossary.md#term-planning) owner or [partition](../../glossary.md#partition) alternatives,
multiple partitions, and mixed backends.

The current one-shot forms are exactly:

```java
HostTensorValue value = engine.compute(output);
HostTensorValue boundedValue = engine.compute(output, maximumTotalBytes);
List<HostTensorValue> values = engine.compute(outputs);
List<HostTensorValue> boundedValues = engine.compute(outputs, maximumTotalBytes);
```

Each call transiently and iteratively inventories [provenance](../../glossary.md#provenance)-free [leaves](../../glossary.md#term-leaf) reachable through Model
Tensor-expression provenance by exact object [identity](../../glossary.md#term-reference-identity). After [compilation](../../glossary.md#compiler-engine-and-execution-stage-names),
`CompiledGraph.inputs()` alone selects membership and order by `TensorId`; Engine does not traverse
Compiler [intermediate representation](../../glossary.md#term-intermediate-representation-ir) ([IR](../../glossary.md#intermediate-representation-dag-and-ssa)), reconstruct liveness, or retain Tensor/provenance
state. Parameters and [buffers](../../glossary.md#term-buffer) need no Engine-specific category: their current Tensor bindings are
discovered like any other reachable provenance-free leaf. Each call freshly compiles with the
ordinary fixed forward-only settings, prepares, runs once, verifies the complete forward-publication set,
preflights the sum of canonical returned [payload](../../glossary.md#artifact-envelope-and-atomic-publication) lengths before copying, materializes every
occurrence in requested order, and closes its temporary result before its temporary preparation
before the call returns. The ordered form
returns an immutable list of immutable detached values; distinct occurrences are copied
independently even if they select one inward representation. The limit covers returned canonical
payload bytes only, not inputs, recipes, Runtime buffers or workspaces, object overhead, defensive
copies, peak memory, or other allocation.

One Engine [admission](../../glossary.md#term-admission) covers that complete sequence. Engine closure waits for an admitted call,
including its result cleanup; a call that loses admission observes the closed-Engine failure
before argument [validation](../../glossary.md#term-validation). A failure returns no partial result. Temporary result-then-preparation
cleanup still runs, with distinct cleanup failures suppressed on the primary failure in cleanup
order. Selected leaf storage remains
caller-owned and must stay live, accessible, and free from conflicting mutation until the
synchronous call completes; unselected discovered storage is not inspected. The `compute(...)`
forms create no compile/prepared/result/value/Tensor-inventory cache or reuse and add no backward,
implicit target/[seed](../../glossary.md#term-seed) behavior, tuning, cross-backend transfer, or Tensor execution method.

The current one-shot backward form is:

```java
ScalarObjectiveBackwardResult result =
        engine.backward(objective, targets, maximumTotalBytes);
```

[`objective`](../../glossary.md#term-objective) must be a [scalar](../../glossary.md#term-scalar) floating gradient-eligible Tensor expression. `targets` is an
explicit ordered non-empty list whose elements are unique by exact object identity; its order
defines `result.gradients()`. Targets may be leaves, intermediates, or the objective itself when
Compiler accepts their gradient semantics and [differentiable](../../glossary.md#derivative-and-chain-rule) connection. Engine inventories
reachable provenance-free leaves before compilation, then lets final `CompiledGraph.inputs()`
select authoritative input membership and order. It accepts no explicit input list, infers
neither targets nor liveness, mutates no Tensor, and creates no [eager](../../glossary.md#term-eager-evaluation) tape.

Every call freshly compiles one stage with an absent scalar [cotangent seed](../../glossary.md#primal-adjoint-and-differentiation-seed) and
`DisconnectedPolicy.ERROR`, prepares, runs, validates the objective-first then target-ordered
publications, and preflights every static/resolved per-value byte count plus the exact aggregate
before the first copy. Each occurrence is copied independently, including inward [aliases](../../glossary.md#aliasing-and-arena). The
limit covers only the objective and gradient canonical payloads. Cleanup completes before return;
Engine close waits through an admitted call, concurrent calls use isolated [run state](../../glossary.md#run-state--runstate), and cleanup
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
the output request. Both values remain readable after Engine and caller [arena](../../glossary.md#term-arena) closure, while the
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

`contiguous()` supplies the non-empty supported CPU [operation](../../glossary.md#operation). Compiler supplies the absent
positive-one seed, which the completed Compiler 0006B5 -> Prepare 0005 -> CPU 0010H chain carries
as a source-only publication beside that partition. This proves the [scalar objective](../../glossary.md#loss) and its
positive-one gradient on the current path; it does not promise pure zero-[node](../../glossary.md#node) or mixed-[backend
composition](../../glossary.md#backend-composition), broader operation support, [optimizer](../../glossary.md#term-optimizer) behavior, or a [training session](../../glossary.md#training-session--trainingsession).

For a single already-constructed Tensor expression, the complete one-shot execution shape is:

```java
try (Engine engine = Engine.standard()) {
    HostTensorValue result = engine.compute(output);
}
```

Here `output` already retains the Model expression provenance leading to its leaves. The returned
value is detached; closing the Engine does not invalidate it.

An old generic/custom/declarative [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) `Model` follows the same boundary. This conceptual example intentionally assumes only the
existing `model`, `input`, and `Tensor` contracts and invents no construction or initialization
API:

```java
Tensor output = model.forward(input);
try (Engine engine = Engine.standard()) {
    HostTensorValue result = engine.compute(output);
}
```

`model.forward(input)` constructs the Tensor expression [directed acyclic graph](../../glossary.md#term-directed-acyclic-graph-dag) ([DAG](../../glossary.md#intermediate-representation-dag-and-ssa)); it does not
execute it. `engine.compute(output)` performs execution and host materialization. Repeated model
execution should retain the expression and use
`compile -> prepare -> engine.run(explicit inputs)` rather than paying the one-shot compile and
prepare cost on every call.


---

[Previous topic](../runtime/prepare.md) · [Next topic](../nn/owned-model.md)
