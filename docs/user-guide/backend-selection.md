# Understand current backend ownership

## Outcome

This guide explains what callers can select today. Ordinary `Engine.standard()` owns a fixed
CPU-only composition; it is runnable but offers no backend-selection parameter. The advanced
surface can supply standalone compile intent explicitly, but it still owns exactly one supported
CPU integration and does not provide generic registration or mixed-backend execution.

## Mental model

```text
ordinary: Engine.standard() -> fixed explicit CPU composition

advanced: caller opens CPU integration -> transfers ownership to AdvancedEngine
          standalone compile intent -> Planning chooses among supplied eligible owners

prepare:  selected owner chooses its private route and executable
run:      follows the prepared schedule; no selection or discovery
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
execution. Its current mapping is bounded to one occurrence at index 0, partition 0, and weight 1.
Cache lookup, measurements, ranking, and explicit safe fallback happen outside Runtime. This does
not select between CPU and another backend and is not generic graph/plan tuning.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| A caller expects `Engine.standard()` to discover Metal or CUDA | Fixed standard composition was mistaken for plugin discovery. | Treat the current inventory as CPU only. |
| CPU scalar and OpenBLAS appear as separate owners | Backend route and backend identity were confused. | Let CPU preparation choose its internal route. |
| Runtime changes owner after a failure | Ownership was deferred past compilation/preparation. | Runtime must execute the already prepared schedule. |
| `unconstrained()` is treated as guaranteed fallback | Absence of a hard requirement was mistaken for a valid candidate. | Expect compilation to fail when no supplied backend is eligible. |
| A tuning result is treated as a generic plan | The bounded CPU-local handoff was generalized. | Keep later multi-occurrence and graph/plan tuning separate. |

## Limitations

There is no current `CompileConfig` aggregate, generic backend builder, registration/discovery
surface, Metal or CUDA lifecycle adapter, device-level public selector, mixed-owner schedule
composition, or runtime fallback. Those remain planned and must preserve the architecture's
explicit composition and no-runtime-service-locator rules.

## Related documentation

- [Public API status](../api/public-api.md)
- [Module boundaries](../architecture/module-boundaries.md)
- [Partition scoring](../architecture/partition-scoring.md)
- [Prepare execution](preparing-execution.md)
