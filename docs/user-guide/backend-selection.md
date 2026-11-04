# Select explicit [backend ownership](../glossary.md#backend-ownership)

## Outcome

This guide explains how a Java application chooses which execution integrations Engine may
use. The application calls `Engine.standard()` for a CPU-only composition, or opens CPU/Metal
integrations and passes them to `Engine.builder().takeOwnership(...)`. That transfer makes
the builder, then the built Engine, responsible for closing integrations; the application must
not also close or reuse them. It retains responsibility for closing the Engine itself.

The builder freezes registration order, capability providers and availability snapshots.
Compilation assigns one backend to each operation. Preparation can combine CPU and Metal
through a checked schedule and explicit bounded transfers. These assignments do not adopt the
application's input storage; actual runs borrow it until their result closes.

A **[backend](../glossary.md#backend)** supplies execution for a supported processor or [device](../glossary.md#term-device). An **integration** is the
opened Java-facing connection that [Engine](../glossary.md#term-engine) can own. A **[partition](../glossary.md#partition)** is a group of operations
assigned to one backend; a **schedule** specifies their execution and any required value
transfers. CPU means central processor; Metal is the supported Apple graphics-processor
integration. Capability is support for the exact operation, type and geometry, not merely
whether a device is present.

## [Mental model](../glossary.md#term-mental-model)

```text
ordinary: Engine.standard() -> fixed explicit CPU composition

explicit: caller opens CPU and/or Metal integrations -> transfers them to Engine.Builder
          compile chooses one owner per operation from the frozen inventory
          prepare validates the complete owner set and every required transfer before analysis
          shared Prepare finalizes every partition and assembles one immutable schedule

run:      binds caller inputs through captured per-occurrence adapters
          executes direct partition/transfer references; no selection or discovery
result:   materializes each publication through its captured adapter
```

[CPU](../glossary.md#term-cpu) [scalar](../glossary.md#cpu-portable-execution-strategy), Java [Vector API](../glossary.md#term-vector-api), generated [JVM](../glossary.md#term-java-virtual-machine-jvm) [bytecode](../glossary.md#term-bytecode), and [OpenBLAS](../glossary.md#term-openblas) are CPU-internal routes. They are
not separate [backends](../glossary.md#backend) and are not selected through [Engine](../glossary.md#term-engine) [ownership](../glossary.md#term-ownership-meanings) configuration.

**Ownership transfer** makes Engine responsible for releasing an integration; it is not the
same as [borrowing](../glossary.md#term-borrowing) an Engine while its caller remains responsible for closing it. The
[existing ownership diagram](../explanation/ownership.md#a-result-is-not-always-an-independent-value)
shows that resource relationship. A **captured adapter** is a connection routine fixed during
setup so each [run](../glossary.md#term-run) can bind input storage or copy output without rediscovering that routine.

## Use the ordinary composition

```java
// The application closes Engine; compute returns copied bytes independent of that close.
try (Engine engine = Engine.standard()) {
    HostTensorValue result = engine.compute(output);
}
```

Every call to `standard()` opens one fresh independent [CPU](../glossary.md#term-cpu) integration. [Engine](../glossary.md#term-engine) owns it and closes
it. There is no classpath search, `ServiceLoader`, process-global registry, fallback backend, or
[Runtime](../glossary.md#term-runtime) service [lookup](../glossary.md#term-lookup).

## Compose [CPU](../glossary.md#term-cpu) and [Metal](../glossary.md#term-metal) explicitly

Metal requires the absolute path to the already-built [native bridge](../glossary.md#term-native-bridge). [Backend](../glossary.md#backend) configuration and
native opening happen before [Engine](../glossary.md#term-engine) takes [ownership](../glossary.md#term-ownership-meanings):

```java
MetalBackendConfiguration metalConfiguration =
        new MetalBackendConfiguration(nativeLibrary.toAbsolutePath());
try (Engine.Builder builder = Engine.builder()) {
    // Ownership transfers at each call's entry, including when registration reports an error.
    builder.takeOwnership(MetalBackendIntegration.open(metalConfiguration));
    builder.takeOwnership(CpuBackendIntegration.open());
    try (Engine engine = builder.build()) {
        HostTensorValue result = engine.compute(output);
    }
}
```

Each non-null `takeOwnership(...)` call transfers the integration at method entry. Do not close or
reuse it afterward, even when registration reports an error. Registration order is deterministic
[Planning](../glossary.md#term-planning) input, not fallback priority. A [graph](../glossary.md#graph) supported entirely by one [owner](../glossary.md#term-owner) stays on that
owner. A CPU/Metal graph is prepared only when every directed cross-owner edge is supported;
current transfer is exact native-[host](../glossary.md#term-host)-staged in either direction for all seven carriers over fully
static [rank](../glossary.md#term-rank)-0..16 [canonical](../glossary.md#term-canonical-form) or positive-[stride](../glossary.md#term-stride) non-overlapping [layouts](../glossary.md#layout) with checked physical spans.
It performs no type/layout conversion or fallback.

A **[carrier](../glossary.md#term-storage-carrier)** is the concrete representation holding [elements](../glossary.md#term-element). **Rank** counts [axes](../glossary.md#term-axis); rank zero
is a [scalar](../glossary.md#term-scalar). A **stride** is the storage distance between neighboring [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix) along an axis.
**Non-overlapping** means different logical elements do not reuse the same storage positions;
the **physical span** includes all referenced bytes, including gaps. **Host-staged** transfer
moves through processor-accessible memory instead of implying a direct [device](../glossary.md#term-device)-to-device copy.

## Use explicit advanced [ownership](../glossary.md#term-ownership-meanings)

Advanced integration code may transfer one supported [CPU](../glossary.md#term-cpu) integration and pass the existing
standalone [compile](../glossary.md#compile) settings:

```java
// AdvancedEngine now owns the opened CPU integration; this scope closes it.
try (AdvancedEngine engine =
        AdvancedEngine.takeOwnership(CpuBackendIntegration.open())) {
    AdvancedCompiledGraph graph = engine.compile(
            CompileMode.FORWARD_ONLY,
            List.of(output),
            Optional.empty(),
            GraphOptimizationConfig.disabled(),
            BackendIntent.unconstrained(),
            PartitionScoringConfig.neutral());
}
```

`BackendIntent.unconstrained()` means only that no hard eligibility [target](../glossary.md#term-target) is present.
`PartitionScoringConfig.neutral()` records no coarse [device](../glossary.md#term-device)-class preference. Neither value
discovers a [backend](../glossary.md#backend) or guarantees a valid [owner](../glossary.md#term-owner). With the current composition, CPU must support
every [occurrence](../glossary.md#term-occurrence) and preparation must receive exactly one non-empty maximal CPU [partition](../glossary.md#partition).

Hard requirements and soft preferences remain distinct. A hard requirement removes ineligible
candidates; a soft device-class preference ranks candidates that remain. [Planning](../glossary.md#term-planning) selects a
[`BackendId`](../glossary.md#backend-identity--backendid) owner and does not choose a CPU route, [vector](../glossary.md#term-vector) width, thread count, [kernel](../glossary.md#kernel), or [tuning
candidate](../glossary.md#candidate-batch).

## Optional single-[owner](../glossary.md#term-owner) tuning is not [backend](../glossary.md#backend) selection

`Engine.prepareTuned(...)` can select a compatible local workload candidate before production
execution for one eligible [CPU](../glossary.md#term-cpu) [partition](../glossary.md#partition) or a one-operation [FLOAT32](../glossary.md#term-float32) [Metal](../glossary.md#term-metal) NEG partition with both
custom-[kernel](../glossary.md#kernel) and [MPSGraph](../glossary.md#term-mpsgraph) candidates. The current mapping is one [occurrence](../glossary.md#term-occurrence) at index 0,
partition 0, and [weight](../glossary.md#term-weight) 1. Metal decisions have session-only reuse and no persistent [cache](../glossary.md#term-cache) access.
[Heterogeneous](../glossary.md#term-heterogeneous) or multiple-partition plans fail before representative trial work. Cache [lookup](../glossary.md#term-lookup),
measurements, [ranking](../glossary.md#term-ranking) and explicit safe fallback stay inside the selected owner; tuning never
selects another backend. Registering both backends does not make a single-owner plan heterogeneous.

**[Autotuning](../glossary.md#autotuning--model-autotuning)** compares allowed implementation candidates using a representative workload.
A **cache** retains a compatible earlier decision, not earlier numerical outputs.
**Session-only** reuse ends with its live owner; a **persistent cache** survives as stored data.
A **[heuristic](../glossary.md#term-heuristic)** chooses by a rule rather than a new timing trial. Any allowed fallback here stays
within the already-selected backend; it is not recovery by moving work to another [device](../glossary.md#term-device).

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| A caller expects `Engine.standard()` to discover [Metal](../glossary.md#term-metal) or [CUDA](../glossary.md#term-cuda) | Fixed standard composition was mistaken for explicit registration. | Use `Engine.builder()` for explicit Metal [ownership](../glossary.md#term-ownership-meanings); CUDA has no public [lifecycle](../glossary.md#lifecycle) integration. |
| Registration order is treated as fallback order | Deterministic [Planning](../glossary.md#term-planning) input was mistaken for retry priority. | Expect a missing [owner](../glossary.md#term-owner) or unsupported transfer to fail without owner substitution. |
| A mixed plan fails before [backend](../glossary.md#backend) analysis | One required directed edge is outside the current all-seven-[carrier](../glossary.md#term-storage-carrier) [rank](../glossary.md#term-rank)-0..16 static [canonical](../glossary.md#term-canonical-form) or positive-[stride](../glossary.md#term-stride) non-overlapping [CPU](../glossary.md#term-cpu)/Metal transfer domain. | Use supported [descriptors](../glossary.md#term-descriptor) with checked physical spans and native [host storage](../glossary.md#host-storage) for CPU endpoints, or keep that value on one owner. |
| A [scalar](../glossary.md#term-scalar) cross-owner transfer fails | [Rank-zero](../glossary.md#axis-rank-scalar-vector-and-matrix) itself is supported, but its type, [layout](../glossary.md#layout), physical byte span, or endpoint representation did not match exactly. | Keep the scalar descriptor canonical with an exact one-[element](../glossary.md#term-element) physical span and use one of the seven [model](../glossary.md#term-model) carriers. |
| CPU scalar and [OpenBLAS](../glossary.md#term-openblas) appear as separate owners | [Backend route](../glossary.md#backend-route) and [backend identity](../glossary.md#backend-identity--backendid) were confused. | Let CPU preparation choose its internal route. |
| [Runtime](../glossary.md#term-runtime) changes owner after a failure | Ownership was deferred past [compilation](../glossary.md#compiler-engine-and-execution-stage-names)/preparation. | Runtime must execute the already [prepared schedule](../glossary.md#prepared-schedule--preparedschedule). |
| `unconstrained()` is treated as guaranteed fallback | Absence of a hard requirement was mistaken for a valid candidate. | Expect compilation to fail when no supplied backend is eligible. |
| A tuning result is treated as a generic plan | The bounded single-owner [handoff](../glossary.md#term-handoff) was generalized. | Keep later multi-[occurrence](../glossary.md#term-occurrence) and [graph](../glossary.md#graph)/plan tuning separate. |


## Numerical behavior and [backend](../glossary.md#backend) support

`Engine.builder()` has no numerical-profile selector. [Model](../glossary.md#term-model) defines one family- and [dtype](../glossary.md#data-type--datatype)-specific
semantic contract; [backend capability](../glossary.md#backend-capability) remains an exact question about an [operation](../glossary.md#operation) [occurrence](../glossary.md#term-occurrence).
[Metal](../glossary.md#term-metal) admits its documented exact and arithmetic domains. Every supported [homogeneous](../glossary.md#term-homogeneous) [FLOAT32](../glossary.md#term-float32)
occurrence has independently queried [BFLOAT16](../glossary.md#term-bfloat16) and [FLOAT16](../glossary.md#term-float16) counterparts with
FLOAT32 working values and accumulators and one final [ties-to-even](../glossary.md#term-ties-to-even) [narrowing](../glossary.md#term-narrowing). Direct
BFLOAT16/FLOAT16 mixed-low arithmetic remains unsupported; use explicit FLOAT32 casts. Every
[partition](../glossary.md#partition) containing BFLOAT16 or FLOAT16 values uses `CUSTOM_PROGRAM`, including exact homogeneous
no-[gradient](../glossary.md#term-gradient) raw-preserving `RESHAPE`, simple `PERMUTE`, materializing [`CONTIGUOUS`](../glossary.md#term-contiguous), [`SLICE`](../glossary.md#slice),
`CONCAT`, and `TILE`. It exposes no alternate [MPSGraph](../glossary.md#term-mpsgraph), [classic-MPS](../glossary.md#mps-mpsgraph-and-openblas), [MPP](../glossary.md#term-mpp), [CPU](../glossary.md#term-cpu), retry, or fallback
route. Locally produced [scalar](../glossary.md#term-scalar)
reductions may publish and cross the CPU/Metal boundary through the same exact all-seven-[carrier](../glossary.md#term-storage-carrier)
[rank](../glossary.md#term-rank)-0..16 transfer contract. There is no [owner](../glossary.md#term-owner) substitution after an unsupported occurrence;
preparation fails closed.

## Limitations

There is no current `CompileConfig` aggregate, reflective or service-based registration/discovery,
[CUDA](../glossary.md#term-cuda) [lifecycle](../glossary.md#lifecycle) adapter, [device](../glossary.md#term-device)-level public selector, general transfer/conversion system, or
[runtime](../glossary.md#term-runtime) fallback. Mixed-[owner](../glossary.md#term-owner) execution is specifically bounded to the registered [CPU](../glossary.md#term-cpu)/[Metal](../glossary.md#term-metal)
integrations and exact all-seven-[carrier](../glossary.md#term-storage-carrier) [rank](../glossary.md#term-rank)-0..16 static [canonical](../glossary.md#term-canonical-form) or positive-[stride](../glossary.md#term-stride)
non-overlapping transfer domain.

## Related documentation

- [Public API status](../reference/index.md)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>
- [Module boundaries](../architecture/module-boundaries.md)<span data-glossary-navigation="true"> ([Module glossary](../glossary.md#term-module))</span>
- [Partition scoring](../architecture/partition-scoring.md)<span data-glossary-navigation="true"> ([Partition glossary](../glossary.md#partition))</span>
- [Prepare execution](preparing-execution.md)<span data-glossary-navigation="true"> ([Prepare glossary](../glossary.md#prepare))</span>
