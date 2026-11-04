# Public Engine binding and results

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

<a id="current-ordinary-engine-boundary"></a>
## Current ordinary Engine boundary

`Engine.standard()` creates a fresh independent CPU-only composition and privately owns its
advanced lifecycle delegate. `compile(...)` returns an immutable owner-bound `CompiledGraph`,
whose `inputs()` list reports final caller-bindable `TensorId` and descriptor pairs.
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

Advanced and ordinary result publication includes wrapper construction and Engine registration:
failure rolls back the completed run and releases admission even when cleanup fails. Result/state
closure drops these owners' direct references to fresh invocation payloads, including borrowed
input wrappers, without closing caller storage. Retained metadata does not require keeping those
payloads live. Runtime plan/count/geometry inspection remains available; Runtime repeated close
does not replay failure, while the outward Engine result preserves its exact failure replay.

`InferenceSession` and `Engine.session` are removed without a compatibility shim. Migrate
`engine.session(graph)` to `engine.prepare(graph)` and `session.run(inputs)` to
`engine.run(prepared, inputs)`, then recompile. Fresh values explicitly opt into runValues.
`TrainingSession` and the advanced lifecycle remain; this is not a Training API redesign.
See the [fresh-value example](../../user-guide/running-models.md#run-fresh-values-in-compiled-order).

The current fixed CPU path requires fully static Shapes and resolved compatible layouts. Ordinary
builder composition may prepare multiple CPU/Metal partitions and transfers only fully static
canonical contiguous `FLOAT32` values between them. Zero-node pass-through remains rejected. In
particular, public `ADD` currently has an unresolved result layout; applying `contiguous()`
afterward resolves only the new CONTIGUOUS occurrence and does not make the preceding ADD
executable. Compile success is therefore not a prepare or run guarantee.

The following complete setup creates exact-size, resolved `FLOAT32` input leaves backed by a
caller-owned shared arena:

```java
import io.github.pho001.synaptik.engine.CompiledGraph;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.PreparedExecution;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Optional;

static Tensor input(Arena arena, boolean requiresGrad) {
    Shape shape = Shape.of(2);
    TensorDescriptor descriptor = new TensorDescriptor(
            DataType.FLOAT32,
            shape,
            Optional.of(LayoutDescriptor.contiguous(shape)),
            requiresGrad);
    MemorySegmentStorage storage = new MemorySegmentStorage(
            DataType.FLOAT32,
            2,
            arena.allocate(2L * DataType.FLOAT32.byteWidth(), Float.BYTES));
    return TensorFactory.create(descriptor, Optional.empty(), Optional.of(storage));
}
```

A forward-only run uses two supported CONTIGUOUS occurrences, deliberately supplies inputs in the
reverse of final binding order, and materializes one exact publication object:

```java
HostTensorValue retained;
try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    Tensor left = input(arena, false);
    Tensor right = input(arena, false);
    Tensor leftOutput = left.contiguous();
    Tensor rightOutput = right.contiguous();

    CompiledGraph graph = engine.compile(List.of(leftOutput, rightOutput));
    try (PreparedExecution prepared = engine.prepare(graph);
            RunResult result = engine.run(prepared, List.of(right, left))) {
        assert graph.inputs().stream().map(CompiledGraph.Input::tensorId).toList()
                .equals(List.of(left.id(), right.id()));
        assert result.publications().stream()
                .map(RunResult.Publication::tensorId).toList()
                .equals(List.of(leftOutput.id(), rightOutput.id()));
        RunResult.Publication selected = result.publications().getFirst();
        retained = result.materialize(selected, 2L * Float.BYTES);
        ByteBuffer bytes = retained.bytes();
        assert bytes.isReadOnly();
        assert bytes.order() == ByteOrder.BIG_ENDIAN;
        assert bytes.remaining() == 2 * Float.BYTES;
    }
}
assert retained.bytes().getFloat(0) == 0.0f;
```

The inputs are zero-initialized `left` and `right`; the result contains two ordered forward
occurrences. The reversed input list demonstrates identity matching. The exact first occurrence
selects a bounded eight-byte CPU copy. `bytes()` returns a fresh read-only big-endian view, and the
final indexed read demonstrates that the detached value remains usable after result, Engine, and
arena closure. It is not a Tensor, typed array, cache, transfer, persistence format, training
workflow, or cross-backend guarantee.

An explicitly seeded first-order run reuses one produced seed value for two outputs while retaining
two identity-distinct gradient-target occurrences:

```java
try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
    Tensor left = input(arena, true);
    Tensor right = input(arena, true);
    Tensor seedLeaf = input(arena, false);
    Tensor leftOutput = left.contiguous();
    Tensor rightOutput = right.contiguous();
    Tensor seed = seedLeaf.contiguous();

    CompiledGraph graph = engine.compile(
            List.of(leftOutput, rightOutput),
            List.of(seed, seed),
            List.of(left, right));
    try (PreparedExecution prepared = engine.prepare(graph);
            RunResult result = engine.run(prepared, List.of(seedLeaf, right, left))) {
        assert result.publications().stream().map(RunResult.Publication::role).toList()
                .equals(List.of(
                        RunResult.Role.FORWARD,
                        RunResult.Role.FORWARD,
                        RunResult.Role.GRADIENT,
                        RunResult.Role.GRADIENT));
        assert result.publications().get(2).tensorId().equals(left.id());
        assert result.publications().get(2).targetIndex().orElseThrow() == 0;
        assert result.publications().get(3).tensorId().equals(right.id());
        assert result.publications().get(3).targetIndex().orElseThrow() == 1;
    }
}
```

The compiled inputs are `left`, `right`, and `seedLeaf`; the seed expression itself is produced
inside the graph. The four occurrences expose metadata until the caller explicitly selects one
for materialization; this example proves neither optimizer integration nor training. A shared seed leaf without `contiguous()` would be a zero-node pass-through
publication with no current prepared buffer assignment.

Host materialization is current for fully static resolved publications and uses the exact CPU or
Metal adapter captured for each publication during cold preparation. The four ordinary
`Engine.compute(...)` overloads provide Engine-owned one-shot forward execution for
one output or an ordered non-empty output list, with or without an explicit aggregate byte bound.
They transiently inventory reachable provenance-free Tensor leaves by exact object identity and
join them by `TensorId` to final `CompiledGraph.inputs()` membership and order. This is Model
expression-provenance traversal, not Compiler IR traversal or Engine liveness reconstruction.
The bounded forms accept one non-negative `maximumTotalBytes` bound on the sum of canonical
returned payload lengths; the no-limit forms delegate with `Long.MAX_VALUE`, while checked
arithmetic and per-value JVM array ceilings still apply. One
Engine admission covers a fresh forward-only compile, prepare, run, complete forward-publication
and aggregate-byte preflight before any copy, ordered materialization, and temporary-result
cleanup. Values are detached and the ordered form is immutable; no `RunResult`, prepared handle,
or caller storage escapes through the return value.

The convenience creates no cache and reuses no compile or prepared recipe. A failure returns no
partial value list; cleanup still runs, with a distinct cleanup failure suppressed on the primary
failure. Engine close waits for an admitted call through cleanup, while a call that loses
admission fails before argument inspection. Current support requires one non-empty plan whose
owners have exact registered adapters. Single-owner plans remain supported; mixed CPU/Metal plans
must stay inside the all-seven-carrier rank-0..16 fully static canonical or positive-stride
non-overlapping transfer domain with checked physical spans. The aggregate limit is not a bound on
inputs, Runtime allocation, workspaces, object overhead, defensive copies, or peak memory. Use
`compile -> prepare -> run -> materialize` for ordinary repeated runs or selective output copies.
Each reusable run requires explicit inputs, through exact-ID run or positional runValues.
Automatic discovery is only the one-shot convenience. Selected
storage remains caller-owned through synchronous completion, no Tensor or provenance state escapes
the call, and no cache is retained. This adds no `output.execute()` Tensor method, implicit
fallback, general conversion or transfer domain, tuning, or cache.

`Engine.backward(objective, targets, maximumTotalBytes)` is the current one-shot backward
counterpart. It discovers candidate leaves from the objective expression, uses final compiled
inputs as the authoritative selection, and runs one fresh absent-positive-one-seed,
disconnected-target-`ERROR` request. The temporary Engine result must contain one forward
objective followed by one first-order gradient occurrence per target in target-list order. Engine
validates every occurrence, preflights all fully static/resolved per-value byte counts and their
exact aggregate before the first physical copy, then copies objective first and gradients in
order. Aliased occurrences are copied independently.

One admission covers compile, prepare, isolated run state, preflight, copies, and cleanup. Engine
close waits through that sequence; concurrent calls use distinct Runtime states. The returned
`ScalarObjectiveBackwardResult` and its `HostTensorValue` components are detached and remain
readable after temporary result, Engine, and caller storage close. No explicit input list, eager
tape, Tensor mutation, inferred target, no-argument backward, or Runtime liveness inference is
introduced. Explicit seeds, multiple outputs, reusable execution, `ZERO`, and higher-order/full
request policies stay on the existing ordinary or advanced lower-level lifecycle. See the
[complete one-shot examples](../engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle).


---

[Previous topic](index.md) · [Next topic](prepared-execution.md)
