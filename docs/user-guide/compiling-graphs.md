# Compile a graph (planned workflow)

## Outcome

This guide explains what graph compilation currently produces internally and how to interpret the
planned public workflow. Public `Tensor` expression construction, four standalone
compile-configuration values, Planning's three package-owned callable operations, and the public
immutable compiler artifact types are current. The complete `GraphCompiler` entry remains
package-private, while `CompileConfig` and the engine facade remain planned, so no user-callable
compile command exists yet.

The current values can record a backend target, graph scope, optional-optimization permission, and
soft coarse device-class preference:

```java
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;

BackendIntent unconstrained = BackendIntent.unconstrained();
BackendIntent requireCpu =
        BackendIntent.requiring(
                new BackendIdRequirement(new BackendId("cpu")));
CompileMode graphScope = CompileMode.FORWARD_AND_BACKWARD;
GraphOptimizationConfig optimization = GraphOptimizationConfig.standard();
PartitionScoringConfig preferCpu = PartitionScoringConfig.preferring(DeviceClass.CPU);
```

These values are runnable metadata construction. The current package-private complete compiler
entry accepts them directly, but no public entry point does.
`unconstrained` promises neither default selection nor fallback, and `requireCpu` does not verify
that CPU is available or capable. `graphScope` requests current internal compiler-owned autograd
and a combined forward/backward graph; constructing the value does not perform either action.
`optimization` permits the current internal standard semantics-preserving pipeline without
selecting or exposing its passes. `preferCpu` records only a soft input for current ranking after
hard eligibility. It does not filter
an eligible accelerator, weaken `requireCpu` or another hard target, calculate a score, or promise
that CPU ownership succeeds.

Use `GraphOptimizationConfig.disabled()` to request skipping only optional compiler optimization.
That setting cannot suppress inference, validation, mandatory canonical representation,
mode-required autograd, publication binding, planning, preparation, or execution. Neither setting
permits approximate mathematics, changed numerical semantics, or backend-specific fusion.

## Current internal steps and planned public call

1. Build a public tensor expression. Provenance on public tensor state will let graph capture discover producers and inputs without turning `Tensor` into an intermediate-representation node.
2. Choose declarative compile configuration. Backend intent, compile mode, graph-optimization
   permission, and the optional coarse class preference are current standalone values; cost
   profiles and their `CompileConfig` aggregate remain planned.
3. The current internal compiler captures, infers, validates, optimizes, optionally expands
   automatic differentiation, creates publication/constant/diagnostic plans, and coordinates
   backend-neutral ownership, partition, and logical-memory planning.
4. The internal result is public immutable `CompileArtifacts`, but no public lifecycle object
   currently returns it.

```java
// Conceptual API; not currently runnable.
CompiledGraph graph = CompiledGraph.compile(output, CompileConfig.auto());
```

## Expected result

Current internal compilation produces an immutable recipe with exactly seven components: compile
mode, the final graph model, partitions assigned to backend identities, logical memory,
publication roles, constant/input roles, and deferred diagnostics. It does not create physical
buffers or choose a CPU, Metal, or CUDA kernel.

Forward publication bindings identify requested Tensor IDs and final graph values. Gradient
bindings identify differentiation-target Tensor IDs and gradient values without adding gradient
state to Tensor. Constant sources are exact logical splats rather than dense payloads. Diagnostics
describe successful deferred constraints; they are not trace events or a public binding language.

For example, a partition may record `owner = CPU`. That means CPU preparation is responsible for it; it does not mean compilation selected scalar, Vector API, or OpenBLAS execution.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| `CompiledGraph` or `CompileConfig` cannot be imported | The public compile lifecycle and aggregate are planned. | Follow the [roadmap](../planning/roadmap.md) and do not create substitute APIs in another module. |
| `CompileArtifacts` can be imported but there is no public compile call | The artifact contract is current while its only producing entry remains package-private. | Treat it as output-only lifecycle data until the engine/compiler facade is implemented. |
| A design puts buffers in compile artifacts | Compile-time and prepared state were mixed. | Keep allocation in prepare/backend/runtime layers. |
| A planner chooses a kernel | Ownership and implementation selection were mixed. | Let planning choose a backend identity and backend prepare choose the route. |

## Related documentation

- [Compile API status](../api/compile-api.md)
- [Lifecycle](../architecture/lifecycle.md)
- [Partition scoring](../architecture/partition-scoring.md)
