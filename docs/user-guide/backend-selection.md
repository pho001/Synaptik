# Select explicit backend ownership

## Outcome

This guide explains what callers can select today. `Engine.standard()` owns a fixed CPU-only
composition. `Engine.builder()` accepts explicitly opened CPU and Metal integrations and freezes
their registration order, capability providers, and availability snapshots. Compilation selects
one owner per operation; ordinary preparation can combine both registered owners through a
validated shared schedule and explicit bounded transfers.

## Mental model

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

CPU scalar, Java Vector API, generated JVM bytecode, and OpenBLAS are CPU-internal routes. They are
not separate backends and are not selected through Engine ownership configuration.

## Use the ordinary composition

```java
try (Engine engine = Engine.standard()) {
    HostTensorValue result = engine.compute(output);
}
```

Every call to `standard()` opens one fresh independent CPU integration. Engine owns it and closes
it. There is no classpath scan, `ServiceLoader`, process-global registry, fallback backend, or
Runtime service lookup.

## Compose CPU and Metal explicitly

Metal requires the absolute path to the already-built native bridge. Backend configuration and
native opening happen before Engine takes ownership:

```java
MetalBackendConfiguration metalConfiguration =
        new MetalBackendConfiguration(nativeLibrary.toAbsolutePath());
try (Engine.Builder builder = Engine.builder()) {
    builder.takeOwnership(MetalBackendIntegration.open(metalConfiguration));
    builder.takeOwnership(CpuBackendIntegration.open());
    try (Engine engine = builder.build()) {
        HostTensorValue result = engine.compute(output);
    }
}
```

Each non-null `takeOwnership(...)` call transfers the integration at method entry. Do not close or
reuse it afterward, even when registration reports an error. Registration order is deterministic
Planning input, not fallback priority. A graph supported entirely by one owner stays on that
owner. A CPU/Metal graph is prepared only when every directed cross-owner edge is supported;
current transfer is exact native-host-staged in either direction for all seven carriers over fully
static rank-0..16 canonical or positive-stride non-overlapping layouts with checked physical spans.
It performs no type/layout conversion or fallback.

## Use explicit advanced ownership

Advanced integration code may transfer one supported CPU integration and pass the existing
standalone compile settings:

```java
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

`BackendIntent.unconstrained()` means only that no hard eligibility target is present.
`PartitionScoringConfig.neutral()` records no coarse device-class preference. Neither value
discovers a backend or guarantees a valid owner. With the current composition, CPU must support
every occurrence and preparation must receive exactly one non-empty maximal CPU partition.

Hard requirements and soft preferences remain distinct. A hard requirement removes ineligible
candidates; a soft device-class preference ranks candidates that remain. Planning selects a
`BackendId` owner and does not choose a CPU route, vector width, thread count, kernel, or tuning
candidate.

## Optional CPU-local tuning is not backend selection

`Engine.prepareTuned(...)` can select a compatible CPU-local workload candidate before production
execution, including when Metal is also registered but CPU owns the complete plan. Its current
mapping is bounded to one occurrence at index 0, partition 0, and weight 1. A Metal-owned plan
fails before representative input borrowing or trial work. Cache lookup, measurements, ranking,
and explicit safe fallback stay inside the selected CPU owner; tuning never selects another
backend.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| A caller expects `Engine.standard()` to discover Metal or CUDA | Fixed standard composition was mistaken for explicit registration. | Use `Engine.builder()` for explicit Metal ownership; CUDA has no public lifecycle integration. |
| Registration order is treated as fallback order | Deterministic Planning input was mistaken for retry priority. | Expect a missing owner or unsupported transfer to fail without owner substitution. |
| A mixed plan fails before backend analysis | One required directed edge is outside the current all-seven-carrier rank-0..16 static canonical or positive-stride non-overlapping CPU/Metal transfer domain. | Use supported descriptors with checked physical spans and native host storage for CPU endpoints, or keep that value on one owner. |
| A scalar cross-owner transfer fails | Rank-zero itself is supported, but its type, layout, physical byte span, or endpoint representation did not match exactly. | Keep the scalar descriptor canonical with an exact one-element physical span and use one of the seven model carriers. |
| CPU scalar and OpenBLAS appear as separate owners | Backend route and backend identity were confused. | Let CPU preparation choose its internal route. |
| Runtime changes owner after a failure | Ownership was deferred past compilation/preparation. | Runtime must execute the already prepared schedule. |
| `unconstrained()` is treated as guaranteed fallback | Absence of a hard requirement was mistaken for a valid candidate. | Expect compilation to fail when no supplied backend is eligible. |
| A tuning result is treated as a generic plan | The bounded CPU-local handoff was generalized. | Keep later multi-occurrence and graph/plan tuning separate. |


## Select numerical behavior

`Engine.builder()` defaults to `NumericalProfile.STRICT_IEEE`. Call
`numericalProfile(NumericalProfile.ACCELERATOR)` before `build()` to request the bounded Model
profile explicitly. CPU executes either choice through the same exact capability and routes. Metal
admits a broad exact common domain under both profiles and a larger `ACCELERATOR` domain. Every
supported homogeneous accelerator FLOAT32 occurrence has BFLOAT16 and FLOAT16 counterparts with
FLOAT32 working values and accumulators and one final ties-to-even narrowing. Direct
BFLOAT16/FLOAT16 mixed-low arithmetic remains unsupported; use explicit FLOAT32 casts. Every
partition containing BFLOAT16 or FLOAT16 values uses `CUSTOM_PROGRAM`, including exact homogeneous
no-gradient raw-preserving `RESHAPE`, simple `PERMUTE`, materializing `CONTIGUOUS`, `SLICE`,
`CONCAT`, and `TILE`. It exposes no alternate MPSGraph, classic-MPS, MPP, CPU, retry, or fallback
route. Locally produced scalar
reductions may publish and cross the CPU/Metal boundary through the same exact all-seven-carrier
rank-0..16 transfer contract. There is no fallback to strict or owner substitution after an
accelerator request; every unsupported occurrence fails closed.

## Limitations

There is no current `CompileConfig` aggregate, reflective or service-based registration/discovery,
CUDA lifecycle adapter, device-level public selector, general transfer/conversion system, or
runtime fallback. Mixed-owner execution is specifically bounded to the registered CPU/Metal
integrations and exact all-seven-carrier rank-0..16 static canonical or positive-stride
non-overlapping transfer domain.

## Related documentation

- [Public API status](../api/public-api.md)
- [Module boundaries](../architecture/module-boundaries.md)
- [Partition scoring](../architecture/partition-scoring.md)
- [Prepare execution](preparing-execution.md)
