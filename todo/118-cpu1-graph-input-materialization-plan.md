# 118. post-lowering input materialization plan

## Status Legend

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[deferred]` intentionally postponed
- `[superseded]` replaced by the post-lowering route-capability design

Overall status: `[~] Phase 0 complete; architecture simplified to post-lowering provider-capability planning`

Implementation tracking:

- [x] Phase 0: safety/inventory tests and current-state guards
- [superseded] Old Phase 1: graph/region `InputMaterializationRequirement` model
- [ ] Phase 1: route/provider representation refactor for CPU lowering and prepare
- [ ] Phase 2: provider input-layout capability query model
- [ ] Phase 3: post-lowering input materialization finalizer
- [ ] Phase 4: prepare-time binding and validation of planned materializations
- [ ] Phase 5: execute-time per-consumer temporary materialization
- [ ] Phase 6: first consumer slice, cpu1 loss routes
- [ ] Phase 7: verification, parity, traces, benchmarks, and docs
- [ ] Phase 8: extend route-capability materialization to index/scatter inputs
- [ ] Phase 9: extend route-capability materialization to attention inputs
- [deferred] Phase 10: global memory planner reuse for materialized input temps

This document is the working plan for plan 118. It intentionally replaces the
older graph/region-level materialization architecture with a smaller
post-lowering design.

Post-117 audit update, 2026-06-23:

- cpu1 default route is enabled through canonical platform runtime profiles,
  not hardcoded `RuntimeConfig` defaults.
- canonical profiles deliberately keep cpu1 direct/fused fallback enabled as a
  compatibility guard while broad strided/view materialization is unfinished.
- the execute-boundary storage contract exists in cpu1 through
  `backend.cpu1.storage.Cpu1StorageAccessPlan` and
  `Cpu1StorageAccessKind`.
- broad strided/view materialization is the main known reason cpu1 still needs
  the default-route fallback guard.
- plan 118 must build on the existing storage access plan where useful instead
  of adding a second generic storage accessor framework.

## Goal

Introduce explicit input materialization for selected lowered execution routes
whose concrete input contract cannot consume a source tensor/view directly.

The decision must happen after backend route/lowering selection, not in the
general graph optimizer. The planner must not special-case `cpu1` by name.
Instead, the post-lowering/finalization phase asks the selected execution
route/provider/family for its input layout capability for each concrete input.

The first implementation slice targets cpu1 loss operations routed through the
selected cpu1 direct provider:

- `CROSS_ENTROPY_LOSS_INDICES`
- dense `NLL_LOSS`
- dense `CROSS_ENTROPY_LOSS`

The same mechanism should later apply to other selected routes and input roles:

- reduction routes that lack direct support for a specific strided/layout case
- index/scatter input operands, tracked in Phase 8
- attention `q/k/v/mask` input operands, tracked in Phase 9

The target behavior is:

1. Lowering selects a concrete execution route/provider/family for an operation
   or lowered unit.
2. Post-lowering finalization builds an `InputLayoutQuery` for each concrete
   consumer input, including input role, dtype, storage family, layout, axis,
   reduction mode, and selected route.
3. The selected route's input-layout contract returns an `InputLayoutDecision`:
   `USE_DIRECT`, `MATERIALIZE_CONTIGUOUS`, or `REJECT`.
4. Finalization records explicit planned materialization only for
   `MATERIALIZE_CONTIGUOUS` decisions.
5. Prepare validates and binds the selected route plus planned effective input
   descriptors. Prepare does not invent materialization plans.
6. Runtime performs the physical copy immediately before the consumer executes,
   into a per-consumer temporary buffer.
7. Kernels consume the prepared effective input and never decide or copy
   secretly.
8. Trace metadata exposes materialization count, source, target, bytes, and
   reason.

The important architectural invariant:

```text
graph optimizer does not decide input materialization.
selected route/provider/family declares its concrete input capability.
post-lowering finalization records explicit materialization decisions.
prepare validates and binds the plan.
execute performs planned copies immediately before the consumer.
kernels do not silently materialize.
```

## Non-Goals

- Do not add graph/region-level `InputMaterializationRequirement`.
- Do not modify `OptimizedRegion` to carry input materialization requirements.
- Do not make `DefaultRegionOptimizer` decide backend input materialization.
- Do not special-case `cpu1` by name in the materialization planner.
- Do not add strided/view loss kernels in this work.
- Do not add arbitrary strided/view attention kernels in this work.
- Do not insert synthetic `CompiledNode` objects as a first implementation.
- Do not create a transitional compatibility layer that routes cpu1 consumers
  back through old `backend.cpu` kernels.
- Do not hide input materialization behind `Cpu1LossPreparer`,
  `Cpu1AttentionPreparer`, or kernel fallback logic.
- Do not introduce a generic hot-path storage accessor framework just for
  planned input materialization.
- Do not change the public `Tensor` API.
- Do not make backend residency part of the public `Tensor` API.
- Do not commit local benchmark/calibration artifacts.
- Do not commit temporary verification scratch files.

## Current State And Constraints From Current Codebase

### Post-117 route and storage contract state

Plan 117 closed the dense/default-route readiness slice. The important current
state for plan 118 is:

- cpu1 direct and fused routes are profile-enabled through canonical
  `macos-arm64` runtime profiles.
- hardcoded runtime defaults remain conservative.
- canonical runtime profiles enable fallback from unsupported cpu1 direct/fused
  units back to the old CPU route.
- that fallback is intentional until selected-route input materialization can
  explicitly handle unsupported strided/view inputs.
- `Cpu1StorageAccessPlan` is already the prepare-time representation of source
  and output access shape:
  - `DENSE_CONTIGUOUS`
  - `DENSE_WITH_OFFSET`
  - `STRIDED`
  - `BROADCAST`
  - `UNSUPPORTED`
- elementwise, fused, reductions, index/scatter, normalization, pool/conv,
  dtype, attention, and several prepared units already carry access plans.
- cpu1 kernels must not silently copy unsupported inputs. Any copy must come
  from an explicit post-lowering materialization decision and must be visible in
  trace/benchmark evidence.

This changes the implementation posture of plan 118: do not invent a graph
storage-access abstraction. Define a route/provider input capability contract
and map its decisions to explicit planned materialization metadata after the
route is known.

### Current CPU route-selection gap

The current CPU lowering shape is too coarse for the new decision point:

- `backend.cpu.lowering.CpuRegionLowerer` can currently classify lowered units
  as broad families such as `DIRECT_KERNEL`, `FUSED_NATIVE`, and `BLAS`.
- cpu1 direct selection for ordinary direct kernels still happens later in
  `BackendPrepareDispatcher`.
- therefore, a post-lowering materialization planner cannot reliably ask the
  actual selected route whether input 0 of node N is accepted directly,
  materialized, or rejected.

Required refactor:

```text
lowering/finalization must represent a selected CPU execution provider/route
before materialization planning runs.
```

The first acceptable shape is intentionally small:

- introduce a selected CPU route/provider/family descriptor for each lowered
  direct/fused/BLAS unit or node plan
- route ids must distinguish at least:
  - `legacy-cpu-direct`
  - `cpu1-direct`
  - `cpu1-fused-native`
  - `blas`
- each selected route exposes or references an input-layout contract
- `BackendPrepareDispatcher` honors the selected route instead of rediscovering
  cpu1 direct eligibility after materialization planning

This does not require a broad backend API redesign. It requires enough selected
route identity for the post-lowering finalizer to query the real route contract.

### Current cpu1 reduction strided support

cpu1 reductions are not uniformly dense-only. The current direct strided
reduction support is intentionally narrow:

- `SUM` `FLOAT32`
- `SUM` `FLOAT64`
- `MEAN` `FLOAT32`
- `MEAN` `FLOAT64`
- both `JAVA_ARRAY` and `MEMORY_SEGMENT`

This support is selected in `Cpu1ReductionPreparer` through
`SUM_F32_STRIDED_SCALAR`, `SUM_F64_STRIDED_SCALAR`,
`MEAN_F32_STRIDED_SCALAR`, and `MEAN_F64_STRIDED_SCALAR`, and dispatched by
`Cpu1ReductionKernelDispatch` into the strided sum/mean loops.

The current reduction strided gaps are:

- `BFLOAT16` `SUM`/`MEAN`
- `REDUCE_MIN`
- `REDUCE_MAX`
- `REDUCE_PROD`
- `REDUCE_ALL`
- `REDUCE_ANY`
- `ARGMAX`
- `CUMSUM`
- `SOFTMAX`
- `LOG_SOFTMAX`
- broadcast input reductions

Important policy consequence for this plan:

```text
Do not blindly materialize every strided reduction input.
If the selected route already has a direct strided path for SUM/MEAN F32/F64,
use it.
If a selected reduction route does not have direct strided support, that route's
input contract may return MATERIALIZE_CONTIGUOUS or REJECT depending on the
specific op/input role/dtype/storage/layout/axis/reduction mode.
```

Dense-with-offset inputs are a separate case. The current reduction preparer
accepts `DENSE_WITH_OFFSET` for non-softmax-like reductions, so those inputs
should not be copied just because `storageOffset != 0`. Softmax-like reductions
remain stricter and are candidates for planned contiguous materialization when
the selected route contract says so.

The strided sum/mean kernels are scalar from the Vector API perspective. They
can use `Cpu1RangeLauncher` over output work items, but they do not yet provide
a specialized parallel partial-reduction path for the single-output strided
case. That is future tuning work, not a reason to hide materialization in
prepare.

### Where contiguous insertion happens

This plan does not make `Cpu1NodePreparer` insert hidden `CONTIGUOUS` work.
There are two different moments:

1. The decision moment.
2. The copy execution moment.

The decision moment belongs after the execution route is selected and before
backend prepare binds executable artifacts:

```text
lowered unit/node + selected execution route/provider + concrete input metadata
  -> route input-layout capability query
  -> explicit planned input materialization, direct-use decision, or rejection
  -> prepare validates and binds the finalized plan
```

The physical copy happens later, at execute time, immediately before the
consumer that needs the dense input:

```text
runtime source tensor/view
  -> planned materialization copy into per-consumer dense contiguous temp
  -> selected route kernel consumes the temporary effective input
```

The first implementation deliberately does not synthesize extra `CompiledNode`
objects for `CONTIGUOUS`. It carries an explicit input materialization plan
instead. Global temp-slot reuse through the memory planner is a later follow-up.

### Current cpu1 loss support

Current local code has cpu1 loss preparer and kernels for:

- `CROSS_ENTROPY_LOSS_INDICES`
  - dense contiguous logits
  - dense contiguous `INT32`/`INT64` targets
  - `JAVA_ARRAY`
  - `MEMORY_SEGMENT`
  - `FLOAT32`, `FLOAT64`, `BFLOAT16` logits/output
- dense `NLL_LOSS`
  - dense contiguous log probabilities
  - dense contiguous dense targets
  - `JAVA_ARRAY`
  - `MEMORY_SEGMENT`
  - `FLOAT32`, `FLOAT64`, `BFLOAT16`
- dense `CROSS_ENTROPY_LOSS`
  - dense contiguous logits
  - dense contiguous dense targets
  - `JAVA_ARRAY`
  - `MEMORY_SEGMENT`
  - `FLOAT32`, `FLOAT64`, `BFLOAT16`

The dense-contiguous contract is enforced in
`src/main/java/backend/cpu1/prepare/Cpu1LossPreparer.java`.

Current guard shape:

```java
if (!logits.denseContiguousWithoutOffset()
        || !targets.denseContiguousWithoutOffset()
        || node.storageOffset() != 0
        || !node.contiguous()) {
    throw new UnsupportedOperationException(
            "cpu1 ... first version requires dense contiguous ...");
}
```

This is correct as a safety guard. The change must not remove the guard and
replace it with hidden runtime behavior. Instead, prepare should validate an
explicit effective input descriptor only when the selected route finalizer
already planned materialization.

### Current cpu1 attention support

Current local code has cpu1 direct attention preparer and kernels for:

- `SCALED_DOT_PRODUCT_ATTENTION`
  - dense contiguous no-offset `q`
  - dense contiguous no-offset `k`
  - dense contiguous no-offset `v`
  - optional dense contiguous no-offset `BOOL` mask
  - dense contiguous no-offset output
  - `JAVA_ARRAY`
  - `MEMORY_SEGMENT`
  - `FLOAT32`, `FLOAT64`, `BFLOAT16`
- `SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS`
  - dense contiguous no-offset output
  - requires the input compiled descriptor to be an actual
    `SCALED_DOT_PRODUCT_ATTENTION` node
  - reads the attention forward weights cache from runtime state
  - `JAVA_ARRAY`
  - `MEMORY_SEGMENT`
  - `FLOAT32`, `FLOAT64`, `BFLOAT16`

The dense-contiguous contract is enforced in
`src/main/java/backend/cpu1/prepare/Cpu1AttentionPreparer.java` and rechecked at
runtime in `src/main/java/backend/cpu1/kernels/linalg/attention/Cpu1AttentionLoops.java`.

This is correct as a safety guard. The change must not replace it with hidden
runtime behavior. Instead, attention should receive explicit effective dense
input descriptors only when the selected route contract already requested
materialization.

Attention is intentionally stricter than simple elementwise kernels. Arbitrary
strided inner dimensions usually make SDPA slower than explicit copy-then-dense:

```text
bad generic strided SDPA:
  every dot-product load has non-unit stride
  K/V access loses locality
  vectorization is weak or impossible
  the hot loop pays offset math per element

preferred first policy:
  selected attention route may materialize unsupported q/k/v/mask views
  existing dense attention kernel consumes effective dense inputs
  copy cost and reason are visible in trace
```

A future optimized attention phase may add narrower "view-aware dense" kernels
where only batch/head base offsets differ but each inner `depth` / `valueDim`
row remains contiguous. That is a route-capability decision, not hidden
`Cpu1AttentionPreparer` policy.

### Existing runtime hooks that matter

Useful existing hooks:

- `PreparedExecutionArtifact.allocateRuntimeState(...)`
  - can allocate run-scoped state for a prepared artifact
- `PreparedRuntimeStateAllocator.putPreparedInputTensor(...)`
  - can store run-local prepared input tensors, but only as `Tensor`
- `ExecutionContext.runtimeTensorForNodeId(...)`
  - reads real compiled node runtime tensors
- `ExecutionContext.runtimeService(...)`
  - can expose backend-neutral run-scoped services if needed
- `ExecutionState.requireNativeOutputStorage(...)`
  - allocates native storage for real node outputs
- `ExecutionContext.allocateNativeStorage(...)`
  - allocates run-owned native CPU storage, but not tied to a node output

Constraint: `PreparedRuntimeStateAllocator.putPreparedInputTensor(...)` is not
enough for `MEMORY_SEGMENT` materialized inputs. A backend-neutral prepared
input materialization buffer needs to represent both Java-array and native CPU
segment targets.

## Superseded Architecture

The following older plan items are explicitly superseded:

- `graph.compile.planning.region.InputLayoutRequirement`
- `graph.compile.planning.region.BackendInputContract`
- `graph.compile.planning.region.Cpu1BackendInputContract`
- `graph.compile.planning.region.InputMaterializationRequirement`
- `graph.compile.planning.region.RegionInputMaterializationPlanner`
- adding `inputMaterializationRequirements` to `OptimizedRegion`
- wiring materialization into `DefaultRegionOptimizer`
- making region optimization trace say `input-materialization-required`

Reason: those tasks decide materialization before the backend route/provider is
known. That is too broad and can produce the wrong answer for routes that
already support the input directly, such as current cpu1 strided `SUM/MEAN`
`FLOAT32/FLOAT64`, or for routes that should reject rather than copy.

Replacement rule:

```text
OptimizedRegion may continue to describe graph/region structure.
RegionExecutionPlan or a lowered/finalized execution artifact carries selected
route identity and planned input materialization.
```

## Architecture Decision

### Decision

Use selected-route input capabilities to decide materialization after lowering.
Runtime executes only explicit prepared materialization.

The first implementation must not insert a synthetic `CompiledNode` for
`CONTIGUOUS`. Instead, it should add a post-lowering data model that can
express:

```text
selected route R for consumer node N input i cannot use source S directly;
route R requests MATERIALIZE_CONTIGUOUS for this concrete input;
prepare validates effective dense input metadata;
execute copies S into temp T immediately before N runs.
```

### Why route capability instead of graph optimizer policy

Different selected routes can make different correct decisions for the same
logical input:

- legacy CPU direct may support a strided input directly
- cpu1 direct loss may require contiguous inputs
- cpu1 reduction may support strided `SUM/MEAN` F32/F64 directly
- BLAS may require contiguous or packed operands for specific dimensions
- fused native may have a narrower or broader input layout contract than a
  single direct kernel

A graph/region optimizer does not know these final route-specific details. The
selected route/provider does.

### Why not synthetic `CompiledNode` first

A synthetic `CompiledNode` for `CONTIGUOUS` would be tempting because existing
layout kernels already know how to materialize. It is rejected for the first
slice because it would force the compile graph, descriptor index, memory plan,
publication, step ordering, and trace identity to pretend that a user graph node
exists. The desired behavior is a selected execution-route requirement, not a
semantic graph rewrite.

Synthetic nodes may become useful later if graph-level layout canonicalization
is implemented broadly. This plan keeps the first implementation narrower and
more explicit.

### Why not hidden preparer materialization

Hidden materialization in a family preparer such as `Cpu1LossPreparer` or
`Cpu1AttentionPreparer` would make tests pass quickly, but it would violate the
runtime boundary:

- lowering/finalization trace would not explain why the copy exists
- benchmark reports would not distinguish compute from planned copy-in
- preparers would become route planners
- kernels would gain policy behavior
- future accelerator/backends would need to rediscover the same decision

A preparer may validate and bind an already planned materialization. It must
not decide to create one from scratch.

### Why no generic storage accessor framework now

The target is explicit input materialization, not general strided loss
execution. A generic accessor framework would add indirection to hot paths and
blur the direct prepared-unit style that cpu1 is moving toward. The immediate
copy needs are simple:

- read a logical source tensor using shape/strides/storage offset
- write a dense contiguous temporary buffer
- support `FLOAT32`, `FLOAT64`, `BFLOAT16`, `INT32`, `INT64`, `BOOL`
- support Java arrays and native CPU segments

A small logical-copy helper is enough.

## Capability Query Model

The concrete names below are sketches. The important part is the direction:
route/provider contracts answer per concrete lowered input.

### `InputLayoutAction`

```java
package backend.lowering.input;

public enum InputLayoutAction {
    USE_DIRECT,
    MATERIALIZE_CONTIGUOUS,
    REJECT
}
```

### `InputRole`

```java
package backend.lowering.input;

public enum InputRole {
    GENERIC,
    LOGITS,
    LOG_PROBABILITIES,
    TARGETS,
    DATA,
    INDICES,
    UPDATES,
    QUERY,
    KEY,
    VALUE,
    MASK
}
```

### `InputLayoutQuery`

```java
package backend.lowering.input;

import backend.cpu1.storage.Cpu1StorageAccessKind;
import backend.lowering.route.SelectedExecutionRoute;
import operations.Operation;
import tensor.DataType;
import tensor.StorageType;

import java.util.OptionalInt;

public record InputLayoutQuery(
        int consumerNodeId,
        Operation.OpType opType,
        int inputIndex,
        InputRole role,
        int sourceNodeId,
        DataType dataType,
        StorageType storageType,
        Cpu1StorageAccessKind accessKind,
        int[] shape,
        int[] strides,
        int storageOffset,
        OptionalInt axis,
        String reductionMode,
        SelectedExecutionRoute selectedRoute
) {
    public boolean denseContiguousNoOffset() {
        return accessKind == Cpu1StorageAccessKind.DENSE_CONTIGUOUS
                && storageOffset == 0;
    }
}
```

Notes:

- `Cpu1StorageAccessKind` is shown because this repository already has that
  classification. If a backend-neutral access kind exists by implementation
  time, use it. Do not create a duplicate if `Cpu1StorageAccessPlan` already
  provides the needed fact for CPU routes.
- The query includes route identity. Contracts can validate that they are being
  asked for the route they own.
- Axis and reduction mode are part of the query because reductions, softmax,
  gather/scatter, and attention decisions can depend on them.

### `InputLayoutDecision`

```java
package backend.lowering.input;

import tensor.DataType;

public record InputLayoutDecision(
        InputLayoutAction action,
        String reasonCode,
        DataType targetDataType,
        int[] targetShape,
        int[] targetStrides,
        int targetStorageOffset,
        String diagnostic
) {
    public static InputLayoutDecision useDirect(String reasonCode) {
        return new InputLayoutDecision(
                InputLayoutAction.USE_DIRECT,
                reasonCode,
                null,
                new int[0],
                new int[0],
                0,
                ""
        );
    }

    public static InputLayoutDecision materializeContiguous(
            String reasonCode,
            DataType dataType,
            int[] shape,
            int[] denseStrides
    ) {
        return new InputLayoutDecision(
                InputLayoutAction.MATERIALIZE_CONTIGUOUS,
                reasonCode,
                dataType,
                shape,
                denseStrides,
                0,
                ""
        );
    }

    public static InputLayoutDecision reject(String reasonCode, String diagnostic) {
        return new InputLayoutDecision(
                InputLayoutAction.REJECT,
                reasonCode,
                null,
                new int[0],
                new int[0],
                0,
                diagnostic
        );
    }
}
```

### Provider contract

```java
package backend.lowering.input;

public interface InputLayoutCapability {
    InputLayoutDecision decide(InputLayoutQuery query);
}
```

### Selected route descriptor

```java
package backend.lowering.route;

import backend.lowering.input.InputLayoutCapability;

public record SelectedExecutionRoute(
        String routeId,
        String providerId,
        String family,
        String kernelId,
        InputLayoutCapability inputLayoutCapability
) {
    public InputLayoutCapability requireInputLayoutCapability() {
        if (inputLayoutCapability == null) {
            throw new IllegalStateException("Selected route has no input layout capability: " + routeId);
        }
        return inputLayoutCapability;
    }
}
```

### Example cpu1 direct loss contract

This is a provider implementation example. The planner should not hard-code
this class or `cpu1` route names.

```java
final class Cpu1DirectLossInputLayoutCapability implements InputLayoutCapability {
    static final String REASON =
            "cpu1-direct-loss-dense-contiguous-input-contract";

    @Override
    public InputLayoutDecision decide(InputLayoutQuery query) {
        if (query.role() != InputRole.LOGITS
                && query.role() != InputRole.LOG_PROBABILITIES
                && query.role() != InputRole.TARGETS) {
            return InputLayoutDecision.useDirect("cpu1-direct-loss-non-loss-input");
        }
        if (query.denseContiguousNoOffset()) {
            return InputLayoutDecision.useDirect("cpu1-direct-loss-input-already-dense");
        }
        if (materializable(query)) {
            return InputLayoutDecision.materializeContiguous(
                    REASON,
                    query.dataType(),
                    query.shape(),
                    TensorMetadata.computeStrides(query.shape())
            );
        }
        return InputLayoutDecision.reject(
                "cpu1-direct-loss-input-not-materializable",
                "Unsupported source layout for explicit loss input materialization"
        );
    }

    private static boolean materializable(InputLayoutQuery query) {
        return switch (query.accessKind()) {
            case DENSE_WITH_OFFSET, STRIDED, BROADCAST -> true;
            case DENSE_CONTIGUOUS -> query.storageOffset() != 0;
            case UNSUPPORTED -> false;
        };
    }
}
```

### Example cpu1 reduction contract

```java
final class Cpu1DirectReductionInputLayoutCapability implements InputLayoutCapability {
    @Override
    public InputLayoutDecision decide(InputLayoutQuery query) {
        if (query.denseContiguousNoOffset()) {
            return InputLayoutDecision.useDirect("cpu1-reduction-input-dense");
        }
        if (isDirectSupportedStridedSumMean(query)) {
            return InputLayoutDecision.useDirect("cpu1-reduction-direct-strided-sum-mean");
        }
        if (canCopyThenUseDenseReduction(query)) {
            return InputLayoutDecision.materializeContiguous(
                    "cpu1-reduction-copy-then-dense-contract",
                    query.dataType(),
                    query.shape(),
                    TensorMetadata.computeStrides(query.shape())
            );
        }
        return InputLayoutDecision.reject(
                "cpu1-reduction-layout-rejected",
                "Selected reduction route cannot consume or materialize this input"
        );
    }

    private static boolean isDirectSupportedStridedSumMean(InputLayoutQuery query) {
        return query.accessKind() == Cpu1StorageAccessKind.STRIDED
                && (query.opType() == Operation.OpType.SUM
                    || query.opType() == Operation.OpType.MEAN)
                && (query.dataType() == DataType.FLOAT32
                    || query.dataType() == DataType.FLOAT64)
                && (query.storageType() == StorageType.JAVA_ARRAY
                    || query.storageType() == StorageType.MEMORY_SEGMENT);
    }
}
```

This preserves the reduction rule: direct strided support wins over blind
materialization.

### Post-lowering finalization hook

```java
final class InputMaterializationFinalizer {
    FinalizedLoweredUnit finalizeInputs(LoweredExecutionUnit unit, DescriptorIndex descriptors) {
        SelectedExecutionRoute route = unit.regionPlan().selectedRoute();
        ArrayList<PlannedInputMaterialization> materializations = new ArrayList<>();
        ArrayList<InputLayoutBinding> bindings = new ArrayList<>();

        for (LoweredInput input : unit.loweredInputs()) {
            InputLayoutQuery query = InputLayoutQueryFactory.from(unit, input, route, descriptors);
            InputLayoutDecision decision = route.requireInputLayoutCapability().decide(query);
            switch (decision.action()) {
                case USE_DIRECT -> bindings.add(InputLayoutBinding.direct(input, decision));
                case MATERIALIZE_CONTIGUOUS -> {
                    PlannedInputMaterialization plan =
                            PlannedInputMaterialization.from(query, decision);
                    materializations.add(plan);
                    bindings.add(InputLayoutBinding.materialized(input, plan, decision));
                }
                case REJECT -> throw new UnsupportedOperationException(decision.diagnostic());
            }
        }

        return unit.withFinalizedInputLayout(bindings, materializations);
    }
}
```

Fallback policy should remain explicit. If a route rejects an input and fallback
is enabled, the route selector may choose a different route before finalization
commits the plan. The final plan must describe the route that actually runs.

## Proposed Data Flow Example

Example user graph:

```text
base logits: shape [4, 8], dense contiguous
strided logits view: select/slice/permute view, shape [2, 8], non-contiguous
targets: shape [2], dense contiguous INT64
loss: CROSS_ENTROPY_LOSS_INDICES(strided logits, targets, classAxis=1)
```

Planned execution:

```text
Compiled graph:
  node 3 = strided logits view
  node 4 = targets
  node 5 = crossEntropyLossFromIndices(node 3, node 4)

Lowering:
  create lowered unit for node 5
  select route=cpu1-direct, family=direct-kernel, kernel=ce-indices

Post-lowering finalization:
  query route=cpu1-direct for node=5 input=0 role=LOGITS dtype=FLOAT32 layout=STRIDED
  decision=MATERIALIZE_CONTIGUOUS reason=cpu1-direct-loss-dense-contiguous-input-contract
  query route=cpu1-direct for node=5 input=1 role=TARGETS dtype=INT64 layout=DENSE_CONTIGUOUS
  decision=USE_DIRECT
  attach PlannedInputMaterialization:
      materializationId=region-0/unit-5/input-0/node-3
      sourceNodeId=3
      consumerNodeId=5
      inputIndex=0
      dataType=FLOAT32
      shape=[2, 8]
      targetStrides=[8, 1]
      targetStorageOffset=0
      selectedRoute=cpu1-direct

Prepare:
  Cpu1LossPreparer receives selected route and finalized input bindings
  effective logits descriptor is dense contiguous no-offset
  targets remain direct source input
  unplanned strided direct prepare still rejects

Runtime:
  Cpu1LossExecutableUnit.run(context)
    -> Cpu1InputMaterializer materializes node 3 into per-consumer temp
    -> Cpu1CrossEntropyLossIndicesLoops reads logits from materialized temp
    -> targets are read from node 4
    -> output node 5 is written by existing dense loss kernel

Trace:
  cpu1MaterializedInputCount=1
  cpu1MaterializedInputs=[route=cpu1-direct,input=0,source=3,target=DENSE_CONTIGUOUS]
  cpu1MaterializationReasons=[cpu1-direct-loss-dense-contiguous-input-contract]
```

## Exact Files Planned

This list is intentionally smaller than the older graph-region plan. It must be
rechecked at implementation time against the current source tree.

### Superseded files not to add for this design

- `src/main/java/graph/compile/planning/region/InputLayoutRequirement.java`
- `src/main/java/graph/compile/planning/region/BackendInputContract.java`
- `src/main/java/graph/compile/planning/region/Cpu1BackendInputContract.java`
- `src/main/java/graph/compile/planning/region/InputMaterializationRequirement.java`
- `src/main/java/graph/compile/planning/region/RegionInputMaterializationPlanner.java`

### Source files not expected to change for materialization decisions

- `src/main/java/graph/compile/planning/region/OptimizedRegion.java`
- `src/main/java/graph/compile/planning/region/DefaultRegionOptimizer.java`
- `src/main/java/graph/compile/planning/region/ExecutionUnitFactory.java`

If any of these files are touched during implementation, the change must be for
route/lowering plumbing already present in that layer, not to make graph/region
optimization decide input materialization.

### Likely new source files

- `src/main/java/backend/lowering/input/InputLayoutAction.java`
- `src/main/java/backend/lowering/input/InputRole.java`
- `src/main/java/backend/lowering/input/InputLayoutQuery.java`
- `src/main/java/backend/lowering/input/InputLayoutDecision.java`
- `src/main/java/backend/lowering/input/InputLayoutCapability.java`
- `src/main/java/backend/lowering/input/InputLayoutBinding.java`
- `src/main/java/backend/lowering/input/PlannedInputMaterialization.java`
- `src/main/java/backend/lowering/input/InputMaterializationFinalizer.java`
- `src/main/java/backend/lowering/route/SelectedExecutionRoute.java`
- `src/main/java/backend/lowering/route/CpuExecutionRouteSelector.java`
- `src/main/java/graph/execution/plan/PreparedInputStorageKind.java`
- `src/main/java/graph/execution/plan/PreparedInputMaterializationSpec.java`
- `src/main/java/graph/execution/plan/PreparedInputMaterializationBuffer.java`
- `src/main/java/backend/cpu1/prepare/Cpu1PreparedInputMaterialization.java`
- `src/main/java/backend/cpu1/prepare/Cpu1PreparedLossInput.java`
- `src/main/java/backend/cpu1/prepare/Cpu1LossInputResolver.java`
- `src/main/java/backend/cpu1/exec/Cpu1InputMaterializer.java`
- `src/main/java/backend/cpu1/kernels/layout/copy/Cpu1LogicalTensorCopy.java`
- `src/main/java/backend/cpu1/kernels/loss/Cpu1LossInputViews.java`

### Likely modified source files

- `src/main/java/backend/cpu/lowering/CpuRegionLowerer.java`
- `src/main/java/backend/lowering/region/RegionExecutionPlan.java`
- `src/main/java/backend/lowering/region/RegionExecutionGroup.java`
- `src/main/java/backend/prepare/LoweredRegionIndex.java`
- `src/main/java/backend/prepare/BackendPrepareContext.java`
- `src/main/java/backend/prepare/PreparedExecutionBuilder.java`
- `src/main/java/backend/prepare/BackendPrepareDispatcher.java`
- `src/main/java/graph/execution/plan/PreparedRuntimeStateAllocator.java`
- `src/main/java/graph/execution/state/RuntimeWorkspaceStore.java`
- `src/main/java/graph/execution/state/ExecutionState.java`
- `src/main/java/backend/runtime/ExecutionContext.java`
- `src/main/java/backend/cpu1/prepare/Cpu1LossPreparer.java`
- `src/main/java/backend/cpu1/prepare/Cpu1PreparedCrossEntropyLossUnit.java`
- `src/main/java/backend/cpu1/prepare/Cpu1PreparedDenseCrossEntropyLossUnit.java`
- `src/main/java/backend/cpu1/prepare/Cpu1PreparedNllLossUnit.java`
- `src/main/java/backend/cpu1/prepare/Cpu1PreparedArtifact.java`
- `src/main/java/backend/cpu1/exec/Cpu1LossExecutableUnit.java`
- `src/main/java/backend/cpu1/kernels/loss/crossentropy/Cpu1CrossEntropyLossIndicesLoops.java`
- `src/main/java/backend/cpu1/kernels/loss/crossentropy/Cpu1DenseCrossEntropyLossLoops.java`
- `src/main/java/backend/cpu1/kernels/loss/nll/Cpu1NllLossLoops.java`
- `src/main/java/backend/cpu1/trace/Cpu1TraceContributor.java`

### New tests

- `[ ]` `src/test/java/backend/lowering/input/InputLayoutCapabilityDecisionTest.java`
- `[ ]` `src/test/java/backend/cpu/lowering/CpuSelectedRouteInputMaterializationTest.java`
- `[x]` `src/test/java/backend/cpu1/Cpu1LossMaterializationExecutionContractTest.java`

### Modified tests

- `src/test/java/backend/cpu1/Cpu1CrossEntropyLossExecutionContractTest.java`
- `src/test/java/backend/cpu1/Cpu1NllLossExecutionContractTest.java`
- `src/test/java/backend/cpu1/Cpu1DenseCrossEntropyLossExecutionContractTest.java`
- `src/test/java/backend/lowering/region/RegionExecutionPlanTest.java`
- `src/test/java/backend/cpu/lowering/CpuRegionLowererTest.java`

## Phase 0: Safety/Inventory Tests And Current-State Guards

Status: `[x] complete`

Purpose: establish that dense existing paths stay green and unplanned strided
loss inputs remain rejected.

### Task 0.1: Add current-state guard for unplanned strided cpu1 loss input

Status: `[x] implemented`

What: add tests that call `Cpu1NodePreparer` directly with strided/view loss
inputs and verify that it still rejects when no materialization plan is passed.

Why: this prevents an accidental hidden materialization path inside
`Cpu1LossPreparer`.

Implemented test path:

- `src/test/java/backend/cpu1/Cpu1LossMaterializationExecutionContractTest.java`

Implemented test methods:

- `unplannedStridedCrossEntropyIndicesLogitsAreRejectedByCpu1LossPreparer`
- `unplannedStridedDenseCrossEntropyLogitsAreRejectedByCpu1LossPreparer`
- `unplannedStridedNllLogProbsAreRejectedByCpu1LossPreparer`

This test is intentionally written against the old direct prepare path. It
should remain true after all phases unless the call site passes an explicit
materialization plan selected by the post-lowering route-capability finalizer.

### Task 0.2: Inventory existing dense loss tests

Status: `[x] validated`

What: keep existing dense tests as baseline:

- `Cpu1CrossEntropyLossExecutionContractTest`
- `Cpu1NllLossExecutionContractTest`
- `Cpu1DenseCrossEntropyLossExecutionContractTest`

Why: materialization must not regress current dense `JAVA_ARRAY` or
`MEMORY_SEGMENT` behavior.

Validation:

```bash
./gradlew test --tests backend.cpu1.Cpu1CrossEntropyLossExecutionContractTest --tests backend.cpu1.Cpu1NllLossExecutionContractTest --tests backend.cpu1.Cpu1DenseCrossEntropyLossExecutionContractTest
```

Validated in the Phase 0 pass together with the new guard test:

```bash
./gradlew test --tests 'backend.cpu1.Cpu1LossMaterializationExecutionContractTest' --tests 'backend.cpu1.Cpu1CrossEntropyLossExecutionContractTest' --tests 'backend.cpu1.Cpu1NllLossExecutionContractTest' --tests 'backend.cpu1.Cpu1DenseCrossEntropyLossExecutionContractTest'
```

Result: `BUILD SUCCESSFUL`.

### Task 0.3: Add trace expectation for no materialization on dense paths

Status: `[deferred] until Phase 5 introduces real prepared input materialization`

What: update existing loss trace assertions to include:

```java
assertEquals(0, attrs.getOrDefault("cpu1MaterializedInputCount", 0));
```

Why: dense paths should remain explicit: no materialized inputs were planned,
and traces should say zero rather than omit the concept once the feature exists.

Reason for deferral: adding a zero materialization trace key before a
materialization model exists would create a trace contract for a feature that is
not yet represented in prepared artifacts. The trace key should be added in the
same phase that adds explicit planned input materialization metadata.

## Phase 1: Route/Provider Representation Refactor

Status: `[ ] not started`

Purpose: make the actual selected CPU execution route visible before input
materialization decisions are finalized.

### Task 1.1: Inventory current route selection points

Status: `[ ] not started`

What: inspect current selection for:

- `DIRECT_KERNEL`
- `FUSED_NATIVE`
- `BLAS`
- legacy CPU direct
- cpu1 direct
- cpu1 fused native

Why: the finalizer must query the route that will actually run. Current cpu1
direct selection in `BackendPrepareDispatcher` is too late for route-capability
materialization planning.

Expected output:

- list current classes/methods that select or override CPU route
- list whether each route has enough metadata to answer input-layout queries
- list where fallback from cpu1 to legacy CPU is selected

### Task 1.2: Introduce selected route identity

Status: `[ ] not started`

What: add a small selected route descriptor, for example
`SelectedExecutionRoute`.

Why: `RegionExecutionPlan` or node plans need to carry enough route identity
for finalization and prepare to agree on the route.

Required route ids:

- `legacy-cpu-direct`
- `cpu1-direct`
- `cpu1-fused-native`
- `blas`

Route identity should include:

- route id
- provider id
- lowering family
- optional kernel id
- input-layout capability contract

### Task 1.3: Move or mirror cpu1 direct route selection before prepare binds

Status: `[ ] not started`

What: refactor selection so cpu1 direct eligibility is represented in the
lowered/finalized plan before `Cpu1LossPreparer` is called.

Why: materialization planning cannot happen after prepare has already selected
cpu1 direct from scratch.

Acceptable first implementation:

- `CpuRegionLowerer` records a provisional direct route candidate.
- a CPU route selector/finalizer resolves `legacy-cpu-direct` vs `cpu1-direct`
  vs `blas`.
- `BackendPrepareDispatcher` consumes the selected route and validates it.

Non-acceptable implementation:

- `BackendPrepareDispatcher` independently decides cpu1 direct and then asks
  `Cpu1LossPreparer` to materialize rejected inputs.

### Task 1.4: Preserve explicit fallback semantics

Status: `[ ] not started`

What: if a selected route returns `REJECT` and fallback is enabled, route
selection may choose a different route before the finalized plan is committed.

Why: fallback should be route selection, not hidden prepare behavior.

Trace requirement:

- final trace should identify the selected route that actually ran
- fallback, if it occurs, should remain visible in existing route/fallback trace
  fields or a new focused field

## Phase 2: Provider Input-Layout Capability Query Model

Status: `[ ] not started`

Purpose: provide a backend-neutral decision API that selected routes implement.

### Task 2.1: Add `InputLayoutAction`

Status: `[ ] not started`

What: add an enum with:

- `USE_DIRECT`
- `MATERIALIZE_CONTIGUOUS`
- `REJECT`

Why: every input decision must be explicit and finite. A boolean
`requiresMaterialization` is too weak because some inputs should be rejected.

### Task 2.2: Add `InputLayoutQuery`

Status: `[ ] not started`

What: add a query record that captures concrete lowered input facts:

- selected route
- consumer node id
- op type
- input index
- input role
- source node id
- dtype
- storage family
- access kind/layout class
- shape
- strides
- storage offset
- axis
- reduction mode or op-specific attributes needed by route contracts

Why: decisions must be per concrete operation/input role/dtype/storage/layout
and not per backend family name.

### Task 2.3: Add `InputLayoutDecision`

Status: `[ ] not started`

What: add a decision record with:

- action
- reason code
- target dtype
- target shape
- target strides
- target storage offset
- diagnostic string for rejects

Why: finalization needs enough data to create planned materialization metadata
without asking prepare to infer target layout.

### Task 2.4: Add provider capability contract

Status: `[ ] not started`

What: selected routes expose:

```java
InputLayoutDecision decide(InputLayoutQuery query);
```

Why: materialization policy belongs to the route/provider/family that owns the
kernel contract.

### Task 2.5: Implement first route contracts

Status: `[ ] not started`

What: implement focused contracts for the first needed routes:

- legacy CPU direct: usually `USE_DIRECT` for layouts existing CPU kernels
  already support, otherwise `REJECT`
- cpu1 direct loss: `USE_DIRECT` for dense contiguous no-offset, otherwise
  `MATERIALIZE_CONTIGUOUS` for copyable layouts, otherwise `REJECT`
- cpu1 direct reduction: preserve direct strided `SUM/MEAN` F32/F64 support
  and only materialize/reject unsupported strided families
- BLAS: decide based on concrete operand layout and selected BLAS contract

Why: this proves the design is provider-driven and not a cpu1-name check inside
the planner.

## Phase 3: Post-Lowering Input Materialization Finalizer

Status: `[ ] not started`

Purpose: convert selected-route decisions into explicit finalized input
bindings and planned materialization records.

### Task 3.1: Add `InputLayoutBinding`

Status: `[ ] not started`

What: represent the effective input passed to prepare:

- direct source binding
- materialized source binding
- selected route id
- decision reason
- effective dtype/shape/strides/storage offset

Why: prepare should validate a clear effective input descriptor instead of
guessing whether a source descriptor was planned for copy.

### Task 3.2: Add `PlannedInputMaterialization`

Status: `[ ] not started`

What: add a post-lowering plan for one copy-in:

- materialization id
- selected route id
- consumer node id
- input index
- input role
- source node id
- source dtype/layout/shape/strides/storage offset
- target dtype/shape/dense strides/storage offset
- target storage kind, if already known
- reason code

Why: runtime needs executable copy metadata, and traces need stable identity.

### Task 3.3: Finalize lowered units after route selection

Status: `[ ] not started`

What: add an `InputMaterializationFinalizer` that iterates lowered inputs,
builds `InputLayoutQuery`, calls the selected route capability, and records
bindings/materializations.

Why: this is the new decision point replacing graph-region planning.

Rules:

- `USE_DIRECT` creates a direct binding and no copy plan.
- `MATERIALIZE_CONTIGUOUS` creates a materialized binding and a
  `PlannedInputMaterialization`.
- `REJECT` fails the selected route or asks route selection to choose another
  route before finalization completes.
- finalization must be deterministic.
- duplicate `(consumerNodeId, inputIndex)` materialization plans are invalid.

### Task 3.4: Attach finalized plans to lowered execution artifacts

Status: `[ ] not started`

What: add finalized input bindings and materialization plans to
`RegionExecutionPlan`, `RegionExecutionGroup`, `RegionNodePlan`, or the narrowest
existing lowered artifact that prepare already consumes.

Why: prepare must see exactly the finalized route and input plan.

Constraint:

```text
Do not attach this to OptimizedRegion.
```

### Task 3.5: Add finalization trace events

Status: `[ ] not started`

What: trace selected-route input decisions after lowering.

Suggested event shape:

```text
input-layout-decision:
  route=cpu1-direct
  consumer=5
  input=0
  role=LOGITS
  source=3
  action=MATERIALIZE_CONTIGUOUS
  reason=cpu1-direct-loss-dense-contiguous-input-contract
```

Why: fallback/materialization must be visible before execution.

## Phase 4: Prepare-Time Binding And Validation

Status: `[ ] not started`

Purpose: make prepare consume finalized route/input bindings without deciding
materialization itself.

### Task 4.1: Index finalized materializations in prepare context

Status: `[ ] not started`

What: expose lookup by `(consumerNodeId, inputIndex)` and by materialization id.

Why: consumer-specific preparers need to bind effective inputs, and executable
artifacts need to register runtime buffer specs.

### Task 4.2: Make dispatcher honor selected route

Status: `[ ] not started`

What: `BackendPrepareDispatcher` should prepare the selected route recorded in
the finalized lowered artifact.

Why: if dispatcher reselects cpu1 direct independently, the materialization plan
may describe a different route than the artifact actually executes.

### Task 4.3: Add effective prepared input handles

Status: `[ ] not started`

What: add a small prepared input handle that can represent direct or materialized
effective input.

Sketch:

```java
public record PreparedEffectiveInput(
        int consumerNodeId,
        int inputIndex,
        InputRole role,
        int sourceNodeId,
        String materializationId,
        boolean materialized,
        DataType dataType,
        int[] shape,
        int[] strides,
        int storageOffset
) {
    public boolean denseContiguousNoOffset() {
        return storageOffset == 0
                && Arrays.equals(strides, TensorMetadata.computeStrides(shape));
    }
}
```

Why: loss, index/scatter, and attention should not each invent incompatible
ways to refer to materialized temps.

### Task 4.4: Keep unplanned non-dense guards

Status: `[ ] not started`

What: preparers must continue to reject non-dense inputs unless the finalized
input binding says the effective input is materialized and dense.

Why: this preserves Phase 0 safety tests and prevents hidden copy paths.

Expected behavior:

- dense direct input without a plan passes
- planned materialized input with effective dense descriptor passes
- strided input without a plan rejects
- plan for wrong consumer/input rejects or is ignored and then direct guard
  rejects

## Phase 5: Execute-Time Per-Consumer Temporary Materialization

Status: `[ ] not started`

Purpose: execute the explicit planned copy immediately before the consumer.

### Task 5.1: Add runtime buffer spec

Status: `[ ] not started`

What: add `PreparedInputMaterializationSpec` and
`PreparedInputMaterializationBuffer` for Java-array and native CPU temp storage.

Why: cpu1 loss has both `JAVA_ARRAY` and `MEMORY_SEGMENT` variants.

### Task 5.2: Register specs from prepared artifacts

Status: `[ ] not started`

What: prepared artifacts register each planned materialized input during
`allocateRuntimeState(...)`.

Why: execution state should own run-scoped temp allocation.

### Task 5.3: Materialize immediately before consumer execution

Status: `[ ] not started`

What: executable units call the materializer before binding input views and
dispatching kernels.

Required ordering:

```text
execute consumer step:
  1. copy each planned source input into its per-consumer temp
  2. resolve effective direct/materialized views
  3. run selected route kernel
```

Why: materialized inputs are per-consumer runtime temporaries. They are not
global graph values in the first implementation.

### Task 5.4: Add logical-to-dense copy helper

Status: `[ ] not started`

What: add a focused copy helper that reads logical shape/strides/storage offset
and writes dense contiguous temp storage.

Supported first-slice dtypes:

- `FLOAT32`
- `FLOAT64`
- `BFLOAT16`
- `INT32`
- `INT64`
- `BOOL`

Supported storage:

- Java arrays
- native CPU memory segments

Why: this is the explicit copy engine for planned materialization. It is not a
generic hot-path accessor framework.

### Task 5.5: Trace copy cost

Status: `[ ] not started`

Suggested attributes:

```text
cpu1MaterializedInputCount
cpu1MaterializedInputIds
cpu1MaterializedInputSourceNodeIds
cpu1MaterializedInputIndexes
cpu1MaterializationReasons
cpu1MaterializationStorageKinds
cpu1MaterializationBytes
cpu1MaterializationElementCount
```

Why: fallback/materialization must be visible in traces and benchmark reports.

## Phase 6: First Consumer Slice, cpu1 Loss Routes

Status: `[ ] not started`

Purpose: prove the architecture on the existing dense cpu1 loss kernels.

### Task 6.1: Bind loss effective inputs

Status: `[ ] not started`

What: update `Cpu1LossPreparer` to consume finalized effective inputs or planned
materializations from the selected route plan.

Why: direct unplanned strided inputs must still reject, while planned
materialized inputs validate as effective dense inputs.

### Task 6.2: Update prepared loss units

Status: `[ ] not started`

What: replace raw input node id reads in prepared loss units with effective
input handles.

Why: runtime may need to read logits/targets/log-probs from a materialized temp
rather than directly from the source node.

### Task 6.3: Resolve direct/materialized loss views at runtime

Status: `[ ] not started`

What: add or reuse a helper that maps effective input handles to `Cpu1TensorView`
from either runtime source tensors or prepared materialization buffers.

Why: kernels should receive normal dense views and not know the policy path.

### Task 6.4: Keep dense output guards unchanged

Status: `[ ] not started`

What: loss output must remain dense contiguous no-offset for this slice.

Why: output materialization/copy-back is not part of input materialization.

## Phase 7: Verification, Parity, Benchmarks, Docs

Status: `[ ] not started`

Purpose: prove correctness across dense unchanged paths, materialized paths,
storage kinds, dtype variants, selected-route decisions, and traces.

### Task 7.1: Route capability decision tests

Status: `[ ] not started`

Test cases:

- cpu1 direct loss dense inputs return `USE_DIRECT`
- cpu1 direct loss copyable strided inputs return `MATERIALIZE_CONTIGUOUS`
- cpu1 direct loss unsupported layouts return `REJECT`
- cpu1 reduction strided `SUM/MEAN` `FLOAT32/FLOAT64` for Java array and memory
  segment return `USE_DIRECT`
- unsupported strided reduction families return either `MATERIALIZE_CONTIGUOUS`
  or `REJECT` according to selected route capability, never blanket
  materialization
- legacy CPU route does not get rewritten as cpu1 policy

### Task 7.2: Lowering/finalization tests

Status: `[ ] not started`

Modify:

- `src/test/java/backend/cpu/lowering/CpuRegionLowererTest.java`
- `src/test/java/backend/lowering/region/RegionExecutionPlanTest.java`

Test cases:

- selected route is represented before prepare
- finalizer attaches planned input materializations to lowered artifacts
- duplicate `(consumerNodeId, inputIndex)` plans are rejected
- plan for consumer outside ordered nodes is rejected
- `OptimizedRegion` is not required to carry input materialization plans

### Task 7.3: Runtime execution contract tests

Status: `[ ] not started`

File:

- `src/test/java/backend/cpu1/Cpu1LossMaterializationExecutionContractTest.java`

Test cases:

- `CROSS_ENTROPY_LOSS_INDICES` with strided logits materialized to Java array
  matches dense expected output
- `CROSS_ENTROPY_LOSS_INDICES` with strided targets materialized to Java array
  matches dense expected output
- `CROSS_ENTROPY_LOSS_INDICES` with strided logits and `INT64` targets works
- dense `NLL_LOSS` with strided log-probs materialized to Java array works
- dense `CROSS_ENTROPY_LOSS` with strided logits and targets works
- `MEMORY_SEGMENT` materialized input path works for F32 CE indices
- trace includes `cpu1MaterializedInputCount > 0`
- direct unplanned strided prepare remains rejected

### Task 7.4: Existing dense parity tests

Status: `[ ] not started`

Run and update only trace assertions if needed:

```bash
./gradlew test --tests backend.cpu1.Cpu1CrossEntropyLossExecutionContractTest --tests backend.cpu1.Cpu1NllLossExecutionContractTest --tests backend.cpu1.Cpu1DenseCrossEntropyLossExecutionContractTest
```

### Task 7.5: Benchmark smoke

Status: `[ ] not started`

Suggested benchmark scenarios:

- dense CE indices baseline
- strided logits with materialization
- strided targets with materialization
- memory segment materialization
- reduction strided `SUM/MEAN` F32/F64 direct path with zero materialization

Do not commit local calibration outputs unless intentionally updating canonical
fixtures.

### Task 7.6: Documentation

Status: `[ ] not started`

Potential docs to update after implementation:

- `docs/architecture.md`
- `docs/graph-optimizer.md`
- `docs/framework-concepts.md`
- `docs/modules.md`

Docs should explain:

- selected route/provider decides input layout capability
- post-lowering finalization records materialization decisions
- prepare validates and binds explicit plans
- runtime executes planned materialization immediately before the consumer
- trace fields expose materialization
- why this is not a synthetic graph node

## Phase 8: Extend Route-Capability Materialization To cpu1 Index/Scatter Inputs

Status: `[ ] not started`

Purpose: reuse the same selected-route capability mechanism for cpu1
index/scatter input operands that do not satisfy a selected direct-kernel
contract.

This phase is input materialization only. It does not implement output
materialization or copy-back for strided/offset scatter outputs.

### Phase 8 invariant

```text
cpu1 index/scatter kernels must not silently materialize.
cpu1 index/scatter preparer must not decide materialization by itself.
selected route capability decides USE_DIRECT, MATERIALIZE_CONTIGUOUS, or REJECT.
prepare validates and binds explicit materialization plans.
execute materializes temps before kernel run.
```

### Scope

Apply planned input materialization to selected routes for:

- `GATHER`
- `GATHER_AXIS`
- `GATHER_ND`
- `TAKE_ALONG_AXIS`
- `SCATTER_ADD`
- `SCATTER_AXIS_ADD`
- `SCATTER_ELEMENTS`
- `SCATTER_ND`

For read-only index operations, materialization may apply to:

- data/input
- indices

For scatter write operations, materialization may apply to:

- data/base
- indices
- updates

The output/write target must still satisfy the direct dense contiguous no-offset
contract in this phase.

### Why output materialization is deferred

Scatter output materialization is a different problem:

- output is a write target, not just a read input
- strided/offset output would require copy-back or scatter-back semantics
- aliasing and memory planner ownership become harder because output writes may
  overlap with source/base storage
- correctness and traceability deserve a separate plan or future phase

### Tasks

- [ ] Inventory current cpu1 index/scatter dense guards and input roles.
- [ ] Add route capability decisions for read-only index input roles.
- [ ] Add route capability decisions for scatter base/data, indices, and
  updates.
- [ ] Reuse generalized effective input handles.
- [ ] Execute planned materialization before `Cpu1IndexExecutableUnit` binds
  views.
- [ ] Add trace metadata:
  - `cpu1IndexMaterializedInputCount`
  - `cpu1IndexMaterializedInputs`
  - `cpu1IndexMaterializationReasons`
- [ ] Add focused tests for planned materialized index/scatter inputs and
  unplanned rejection.
- [ ] Keep strided/offset outputs rejected.

## Phase 9: Extend Route-Capability Materialization To cpu1 Attention Inputs

Status: `[ ] not started`

Purpose: reuse the same selected-route capability mechanism for cpu1 attention
input operands that do not satisfy the selected direct attention contract.

This phase is input materialization only. It does not implement output
materialization/copy-back for strided attention outputs, arbitrary strided inner
dimension attention kernels, or vectorized attention kernels.

### Phase 9 invariant

```text
cpu1 attention kernels must not silently materialize.
cpu1 attention preparer must not decide materialization by itself.
selected route capability decides USE_DIRECT, MATERIALIZE_CONTIGUOUS, or REJECT.
prepare validates and binds explicit materialization plans.
execute materializes q/k/v/mask temps before the dense attention kernel runs.
```

### Scope

Apply planned input materialization to selected routes for:

- `SCALED_DOT_PRODUCT_ATTENTION`

Materializable input roles:

- input 0: query `q`
- input 1: key `k`
- input 2: value `v`
- input 3: optional bool mask

Do not apply input materialization directly to
`SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS`. That op reads the runtime forward
weights cache attached to the attention output. Its output must remain dense
contiguous no-offset in this phase.

The attention output/write target must still satisfy the direct dense
contiguous no-offset contract in this phase.

### Why output materialization is deferred

Attention output materialization is a write-target problem:

- output copy-back would need explicit post-kernel writeback semantics
- aliasing with source views must be handled by memory planning
- trace must distinguish input copy-in from output copy-back
- strided output may require a different operation family than dense SDPA

### Tasks

- [ ] Inventory current cpu1 attention dense guards and input roles.
- [ ] Add selected-route decisions for `q/k/v/mask`.
- [ ] Bind planned attention inputs in prepare through generalized effective
  input handles.
- [ ] Execute materialization before attention binds views.
- [ ] Add trace metadata:
  - `cpu1AttentionMaterializedInputCount`
  - `cpu1AttentionMaterializedInputs`
  - `cpu1AttentionMaterializationReasons`
- [ ] Add focused tests for planned materialized `q/k/v/mask` inputs.
- [ ] Keep strided/offset attention outputs rejected.
- [ ] Keep `SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS` outside input
  materialization.

## Phase 10: Global Memory Planner Reuse

Status: `[deferred]`

Purpose: after correctness is proven, reduce allocation pressure and integrate
materialized input temps with global memory planning.

First implementation rule:

```text
Use per-consumer runtime temporary materialization.
Do not block the first implementation on global memory planner reuse.
```

Future model:

```text
PlannedInputMaterialization
  -> MemoryPlanner assigns temp slot when lifetime is known
  -> ExecutionState binds materialization buffer to that slot
  -> per-consumer execution reuses the assigned slot safely
```

Potential future files:

- `src/main/java/graph/compile/planning/memory/RegionValueFlowPlanner.java`
- `src/main/java/graph/compile/planning/memory/RegionBindingAllocator.java`
- `src/main/java/graph/execution/residency/RuntimeMemoryBinder.java`

## Rejected Alternatives

### Graph/region `InputMaterializationRequirement`

Rejected and superseded.

Why:

- decides before the selected route is known
- would require `OptimizedRegion` changes for backend-specific policy
- can blindly materialize inputs that the selected route supports directly
- cannot distinguish `legacy-cpu-direct`, `cpu1-direct`, `blas`, and fused
  route-specific contracts cleanly

Allowed replacement:

- selected route/provider/family capability queried after lowering
- planned materializations attached to lowered/finalized execution artifacts

### Synthetic `CompiledNode` `CONTIGUOUS`

Rejected for first slice.

Why:

- changes graph topology for a backend execution requirement
- complicates descriptor index and memory planning
- risks exposing synthetic nodes in publication/trace
- overfits a local direct-kernel need to a graph rewrite

When reconsidered:

- if graph-level layout canonicalization becomes a general optimizer feature
- if multiple backends consume the same synthetic layout node semantics

### Direct hidden preparer materialization

Rejected.

Why:

- makes copy-in invisible in lowering/finalization trace
- hides performance cost from benchmark reports
- turns preparer into a planner
- encourages kernels to accept non-contract inputs

Allowed behavior:

- preparers may bind materialization plans passed from finalized lowering
- preparers may validate effective descriptors
- preparers may reject unplanned non-contiguous inputs
- preparers must not create materialization plans from source descriptors on
  their own

### Generic storage accessor framework

Rejected for this phase.

Why:

- cpu1 prepared kernels are intentionally direct
- accessor indirection would affect hot paths
- the immediate requirement is a copy helper, not strided loss compute

Allowed behavior:

- a small logical-to-dense copy helper for planned copy-in
- future accessor work only if multiple families prove the need

### Immediate strided loss kernels

Rejected for this phase.

Why:

- kernel matrix would grow before planner plumbing is correct
- dense kernels are already validated
- materialization decision must be traceable first

Future option:

- selected route capability can later choose between:
  - `USE_DIRECT` with a strided loss kernel
  - `MATERIALIZE_CONTIGUOUS` and use the dense kernel
  - `REJECT`

## Validation Commands

Required focused commands after implementation:

```bash
./gradlew test --tests backend.lowering.input.InputLayoutCapabilityDecisionTest
./gradlew test --tests backend.cpu.lowering.CpuSelectedRouteInputMaterializationTest
./gradlew test --tests backend.cpu1.Cpu1LossMaterializationExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1CrossEntropyLossExecutionContractTest --tests backend.cpu1.Cpu1NllLossExecutionContractTest --tests backend.cpu1.Cpu1DenseCrossEntropyLossExecutionContractTest
./gradlew classes
git diff --check -- todo/118-cpu1-graph-input-materialization-plan.md
```

Additional recommended commands after Phase 8:

```bash
./gradlew test --tests backend.cpu1.Cpu1GatherExecutionContractTest
./gradlew test --tests ScatterElementsExecutionTest
./gradlew test --tests ScatterNdExecutionTest
./gradlew test --tests GatherExecutionTest
./gradlew test --tests GatherNdExecutionTest
```

## Validation Against Assignment

- [x] This document is a plan only.
- [x] It targets `todo/118-cpu1-graph-input-materialization-plan.md`.
- [x] It replaces graph/region optimizer materialization decisions with
  post-lowering selected-route capability decisions.
- [x] It explicitly marks graph-level `InputMaterializationRequirement` and
  `OptimizedRegion` materialization changes as superseded.
- [x] It does not special-case `cpu1` by name in the planner; cpu1 appears as a
  provider contract implementation example.
- [x] It includes concrete `InputLayoutQuery`, `InputLayoutDecision`, action
  enum, provider contract, selected route, and finalization sketches.
- [x] It preserves the reduction note that cpu1 directly supports strided
  `SUM/MEAN` `FLOAT32/FLOAT64` for Java array and memory segment storage.
- [x] It keeps physical copy at execute time immediately before the consumer.
- [x] It states that prepare binds/validates the plan and kernels do not decide
  or copy secretly.
- [x] It makes per-consumer runtime temporary materialization the first
  implementation.
- [x] It defers global memory planner reuse.
- [x] It includes the required CPU route/provider refactor for
  `legacy-cpu-direct`, `cpu1-direct`, `cpu1-fused-native`, and `blas`.

## Remaining Debt Or Follow-Up Work

Known follow-up after the planned implementation:

- Memory planner slot integration for input materialization temps is deferred
  to Phase 10 and should not block correctness.
- `UNKNOWN_OR_COMPLEX` source layouts should remain unsupported until a route
  contract and copy implementation explicitly cover them.
- Strided loss kernels remain a future optimization and are intentionally not
  part of this plan.
- Strided index/scatter output materialization and copy-back remain deferred to
  a separate plan or future phase.
- Attention output materialization and copy-back remain deferred.
- Parallel scatter remains deferred.
- Uniqueness analysis for duplicate-safe parallel scatter remains deferred.
- Hidden prepare/runtime fallback remains a non-goal.
- A generic hot-path storage accessor remains a non-goal.
- Benchmark calibration outputs must not be committed unless promoted to
  canonical fixtures deliberately.
