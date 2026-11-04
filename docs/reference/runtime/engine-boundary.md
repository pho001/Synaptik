# Public [Engine](../../glossary.md#term-engine) [binding](../../glossary.md#term-binding) and results

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

Input **binding** connects supplied values to the compiled computation. [Identity](../../glossary.md#term-identity-meanings)-based binding
matches the original [Tensor](../../glossary.md#tensor) identifiers; positional binding follows final compiled input order.
A successful [synchronous](../../glossary.md#term-synchronous) [run](../../glossary.md#term-run) has finished computing, but its **result [lease](../../glossary.md#term-lease)** keeps resources
alive until close. **[Materialization](../../glossary.md#materialization)** explicitly copies a selected result into [detached](../../glossary.md#detached-model-results) [host](../../glossary.md#term-host) bytes.

## Which objects carry descriptions, and which carry values?

The application gives output **expressions** to `Engine.compile`; the resulting graph handle
contains input identities/descriptors, not the original arrays. At `run` or `runValues`, the
application supplies **current stored inputs**. Engine snapshots storage associations and borrows
the selected storage; it does not pin an Arena or copy every input. Keep those bytes usable and
unmodified until the result closes. Engine's result owns the run resources, while `materialize`
creates a separate immutable host value with no close protocol. Compare
[Training's shorter external-batch borrow](../training/sessions.md#session-construction-and-input-mapping),
where Training closes its internal Engine result before the synchronous call returns.

<a id="current-ordinary-engine-boundary"></a>
## Current ordinary [Engine](../../glossary.md#term-engine) boundary

`Engine.standard()` creates a fresh independent [CPU](../../glossary.md#term-cpu)-only composition and privately owns its
advanced [lifecycle](../../glossary.md#lifecycle) delegate. `compile(...)` returns an [immutable](../../glossary.md#term-immutable) [owner](../../glossary.md#term-owner)-bound `CompiledGraph`,
whose `inputs()` list reports final caller-bindable [`TensorId`](../../glossary.md#tensorid) and [descriptor](../../glossary.md#term-descriptor) pairs.
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
[Compiler](../../glossary.md#term-compiler) [occurrence](../../glossary.md#term-occurrence)-order authority, retaining descriptors and IDs, not caller storage.

`prepare(graph)` accepts a [graph](../../glossary.md#graph) from that exact Engine, performs one preparation, and registers
one closeable outward [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) owner. Its `compiledGraph()` returns the exact graph.
The handle borrows the Engine's composition [lifetime](../../glossary.md#term-lifetime); [Runtime](../../glossary.md#term-runtime) owns persistent resources and the
unique [run](../../glossary.md#term-run)-[lease](../../glossary.md#term-lease) protocol. Failed preparation, wrapper construction or [publication](../../glossary.md#term-publication) rolls back
acquired resources while preserving the primary failure and suppressed cleanup failures.

### Bind current stored inputs

`run(prepared, inputs)` matches every compiled TensorId exactly once in arbitrary order and checks
the complete descriptor. `runValues(prepared, values)` instead binds by final compiled input order,
with no ID requirement. It requires exact count and fully static exact descriptors: [dtype](../../glossary.md#data-type--datatype), [shape](../../glossary.md#shape),
[layout](../../glossary.md#layout) presence/geometry and requiresGrad. One fresh [Tensor](../../glossary.md#tensor) may fill distinct matching slots.
There is no shape [specialization](../../glossary.md#term-specialization): [64,784] rejects a compiled [32,784] slot.

Both calls [snapshot](../../glossary.md#term-snapshot) each selected [host](../../glossary.md#term-host)-storage association once and validate all inputs before
any borrow: presence, storage dtype, capacity, [liveness](../../glossary.md#term-liveness) and current-thread accessibility.
runValues uses checked referenced [element](../../glossary.md#term-element)/byte spans for resolved layouts and checked [canonical](../../glossary.md#term-canonical-form)
static geometry for absent layouts, rejecting unknown/[overflow](../../glossary.md#term-overflow) geometry. That extra absent-layout
check does not change identity-based run. The snapshot neither copies nor pins bytes and is not
[atomic](../../glossary.md#resource-transaction-and-rollback) across inputs. The caller must keep storage usable and free from conflicting mutation until
result close; replacing a Tensor association later cannot redirect an existing run.

Both methods first admit the Engine, then validate prepared nullness/exact [ownership](../../glossary.md#term-ownership-meanings), then
retrieve/check its exact inward delegate, then inspect inputs. This deliberately changes former
direct-run diagnostic precedence: closed preparation wins null/malformed/dead inputs with
`prepared execution is closed` and zero borrows. Null/foreign prepared handles also precede
input-list errors. Closed Engine still reports `advanced engine is closed` before all [arguments](../../glossary.md#term-argument).
A later close [race](../../glossary.md#term-race) is decided only by Runtime's unique nonwaiting lease; no outward monitor is
held through execution.

### Keep results alive separately from completion

A successful run is a [synchronous](../../glossary.md#term-synchronous) completion barrier, not a future or in-flight command.
[`RunResult`](../../glossary.md#run-result--runresult) owns completed run-state resources and the [borrowed](../../glossary.md#ownership-borrowing-and-lease) input wrappers. Its immutable
publication list is dense forward-then-gradient order. Gradient [metadata](../../glossary.md#term-metadata) identifies the requested
target, [derivative](../../glossary.md#term-derivative) order and target index; aliased values remain distinct occurrence objects.
`materialize` authenticates the exact occurrence from that result and returns a fresh [detached](../../glossary.md#detached-model-results)
[`HostTensorValue`](../../glossary.md#host-tensor-value--hosttensorvalue), without caching or deduplication.

Repeated/concurrent runs reuse immutable recipes, precomputed geometry and direct input/publication
adapters, with isolated [mutable](../../glossary.md#term-mutable) Runtime state and results. They do not mutate [prototypes](../../glossary.md#term-prototype), construct
synthetic Tensors, recapture, [compile](../../glossary.md#compile), [prepare](../../glossary.md#prepare), query providers/registries, or add per-element [binding](../../glossary.md#term-binding)
work. There is no [device](../../glossary.md#term-device)-overlap, [throughput](../../glossary.md#term-throughput), fairness, [asynchronous](../../glossary.md#term-asynchronous) or [cancellation](../../glossary.md#term-cancellation) promise.

Prepared close rejects new runs but does not close returned results. A leased run may finish;
Runtime defers prepared-resource cleanup to its last lease [release](../../glossary.md#term-release) without making close wait.
Result [materialization](../../glossary.md#materialization) remains valid after prepared close while result and Engine are open.
Engine shutdown rejects new work, waits for admitted operations, then closes results in reverse
run-publication order, ordinary/advanced preparations in reverse preparation-publication order,
and integrations in reverse registration order. Cleanup is attempt-all and failure-retaining.
Metadata stays readable after closure; detached values outlive result, preparation, Engine and
caller storage.

Advanced and ordinary result publication includes wrapper construction and Engine registration:
failure rolls back the completed run and [releases](../../glossary.md#commit-release-distribution-log-and-seed-meanings) [admission](../../glossary.md#term-admission) even when cleanup fails. Result/state
closure drops these owners' direct references to fresh invocation payloads, including borrowed
input wrappers, without closing caller storage. Retained metadata does not require keeping those
payloads live. Runtime plan/count/geometry inspection remains available; Runtime repeated close
does not replay failure, while the outward Engine result preserves its exact failure replay.

[`InferenceSession`](../../glossary.md#former-inference-session--inferencesession) and `Engine.session` are removed without a compatibility shim. Migrate
`engine.session(graph)` to `engine.prepare(graph)` and `session.run(inputs)` to
`engine.run(prepared, inputs)`, then recompile. Fresh values explicitly opt into runValues.
[`TrainingSession`](../../glossary.md#training-session--trainingsession) and the advanced lifecycle remain; this is not a Training [API](../../glossary.md#term-application-programming-interface-api) redesign.
See the [fresh-value example](../../user-guide/running-models.md#run-fresh-values-in-compiled-order).

The fixed CPU path requires fully static Shapes and resolved compatible final layouts; qualified
named-batch execution has its own [metadata-binding boundary](../compiler/batch-binding.md).
Compiler's [final static-result closure](../compiler/optimization.md) can resolve materializing
results such as ADD before capability planning. The public expression's initially unresolved
layout is therefore not by itself a current CPU rejection. Unknown/unproved layouts and
unsupported occurrences still fail; appending `contiguous()` is not a general repair. Mixed-owner
transfers retain their separately qualified type/layout domain, and zero-node pass-through
remains rejected. Compile success is not a prepare or run guarantee.

The following complete setup creates exact-size, resolved `FLOAT32` input [leaves](../../glossary.md#term-leaf) backed by a
caller-owned shared [arena](../../glossary.md#term-arena):

```java
// The Arena owns these bytes; the Tensor describes and borrows their storage association.
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
// Keep the Arena open through result close; materialization copies only the selected publication.
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
selects a bounded eight-byte CPU copy. `bytes()` returns a fresh read-only [big-endian](../../glossary.md#text-encoding-and-file-boundaries) [view](../../glossary.md#term-view-meanings), and the
final indexed read demonstrates that the [detached value](../../glossary.md#detached-model-results) remains usable after result, Engine, and
arena closure. It is not a Tensor, typed array, [cache](../../glossary.md#term-cache), transfer, [persistence](../../glossary.md#term-persistence) format, training
workflow, or cross-[backend](../../glossary.md#backend) guarantee.

An explicitly seeded first-order run reuses one produced [seed](../../glossary.md#term-seed) value for two outputs while retaining
two identity-distinct gradient-target occurrences:

```java
// A repeated seed expression still contributes to two ordered gradient-publication roles.
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
for materialization; this example proves neither [optimizer](../../glossary.md#term-optimizer) integration nor training.
A shared seed leaf without `contiguous()` is a source-only publication, handled separately from
the required nonempty executable partition; it does not grant pure zero-node execution.

Host materialization is current for fully static resolved publications and uses the exact CPU or
Metal adapter captured for each publication during cold preparation. The four ordinary
`Engine.compute(...)` overloads provide Engine-owned [one-shot forward execution](../../glossary.md#one-shot-forward-execution) for
one output or an ordered non-empty output list, with or without an explicit aggregate byte bound.
They transiently inventory reachable [provenance](../../glossary.md#provenance)-free Tensor leaves by exact object [identity](../../glossary.md#term-reference-identity) and
join them by `TensorId` to final `CompiledGraph.inputs()` membership and order. This is [Model](../../glossary.md#term-model)
expression-provenance traversal, not Compiler [IR](../../glossary.md#term-intermediate-representation-ir) traversal or Engine liveness reconstruction.
The bounded forms accept one non-negative `maximumTotalBytes` bound on the sum of canonical
returned payload lengths; the no-limit forms delegate with `Long.MAX_VALUE`, while checked
arithmetic and per-value [JVM](../../glossary.md#term-java-virtual-machine-jvm) array ceilings still apply. One
Engine admission covers a fresh forward-only compile, prepare, run, complete forward-publication
and aggregate-byte [preflight](../../glossary.md#term-preflight) before any copy, ordered materialization, and temporary-result
cleanup. Values are detached and the ordered form is immutable; no `RunResult`, prepared handle,
or caller storage escapes through the return value.

The convenience creates no cache and reuses no compile or prepared recipe. A failure returns no
partial value list; cleanup still runs, with a distinct cleanup failure suppressed on the primary
failure. Engine close waits for an admitted call through cleanup, while a call that loses
admission fails before argument inspection. Current support requires one non-empty plan whose
owners have exact registered adapters. Single-owner plans remain supported; mixed CPU/Metal plans
must stay inside the all-seven-[carrier](../../glossary.md#term-storage-carrier) [rank](../../glossary.md#term-rank)-0..16 fully static canonical or positive-[stride](../../glossary.md#term-stride)
non-overlapping transfer domain with checked physical spans. The aggregate limit is not a bound on
inputs, Runtime allocation, workspaces, object overhead, defensive copies, or peak memory. Use
`compile -> prepare -> run -> materialize` for ordinary repeated runs or selective output copies.
Each reusable run requires explicit inputs, through exact-ID run or positional runValues.
Automatic discovery is only the one-shot convenience. Selected
storage remains caller-owned through synchronous completion, no Tensor or provenance state escapes
the call, and no cache is retained. This adds no `output.execute()` Tensor method, implicit
fallback, general conversion or transfer domain, tuning, or cache.

`Engine.backward(objective, targets, maximumTotalBytes)` is the current one-shot backward
counterpart. It discovers candidate leaves from the [objective](../../glossary.md#term-objective) expression, uses final compiled
inputs as the authoritative selection, and runs one fresh absent-positive-one-seed,
disconnected-target-`ERROR` request. The temporary Engine result must contain one forward
objective followed by one first-order gradient occurrence per target in target-list order. Engine
validates every occurrence, preflights all fully static/resolved per-value byte counts and their
exact aggregate before the first physical copy, then copies objective first and gradients in
order. Aliased occurrences are copied independently.

One admission covers compile, prepare, isolated [run state](../../glossary.md#run-state--runstate), preflight, copies, and cleanup. Engine
close waits through that sequence; concurrent calls use distinct Runtime states. The returned
`ScalarObjectiveBackwardResult` and its `HostTensorValue` components are detached and remain
readable after temporary result, Engine, and caller storage close. No explicit input list, [eager](../../glossary.md#term-eager-evaluation)
tape, Tensor mutation, inferred target, no-argument backward, or Runtime liveness [inference](../../glossary.md#term-inference-meanings) is
introduced. Explicit seeds, multiple outputs, reusable execution, `ZERO`, and higher-order/full
request policies stay on the existing ordinary or advanced lower-level lifecycle. See the
[complete one-shot examples](../engine/lifecycle.md#current-ordinary-explicit-composition-and-advanced-lifecycle).


---

[Previous topic](index.md) · [Next topic](prepared-execution.md)
