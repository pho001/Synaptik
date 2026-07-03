# Metal backend strategy

## Purpose and status

This pre-implementation note explains how MPSGraph, custom Metal kernels, storage, and native integration remain inside one concrete backend. It does not promise platform versions, operation coverage, or performance.

## Strategy

Planning selects `owner = Metal` from declarative capability and scoring facts. Metal prepare lowers the assigned region and chooses MPSGraph, a custom kernel, or another Metal-internal executable route. The backend also owns Metal storage, workspaces, materialization, native bridge calls, and typed trace contributions.

```text
PlannedPartition(owner = Metal)
  -> Metal lowering/fusion/specialization
  -> MPSGraph or custom route
  -> PreparedExecutable
```

Runtime sees the prepared executable contract, not Metal graph objects or kernel-selection logic.

## Training boundary

Training owns optimizer algorithms. If Metal prepare recognizes and fuses an optimizer update, that executable remains a Metal backend concern. The training extension must not depend on Metal or introduce a `MetalOptimizerBridge`.

## Open implementation detail

Native bridge technology, supported operating systems and devices, synchronization, command queues, error translation, memory sharing, MPSGraph segmentation, custom-kernel source management, and resource cleanup remain undecided until focused tasks provide evidence.

See [Metal backend guide](../../backend-guide/metal-backend.md), [Backend-owned lowering ADR](../decisions/0002-backend-owned-lowering.md), and the [Metal master plan](../../planning/backends/metal/master-plan.md).
