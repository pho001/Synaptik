# 118. cpu1 graph/lowering-driven input materialization plan

## Status Legend

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `[deferred]` intentionally postponed

Overall status: `[ ] not started`

Implementation tracking:

- [ ] Phase 0: safety/inventory tests and current-state guards
- [ ] Phase 1: graph region materialization requirement model
- [ ] Phase 2: propagate materialization requirements through lowering and prepare indexes
- [ ] Phase 3: runtime temporary storage model for materialized inputs
- [ ] Phase 4: cpu1 prepared input materialization execution
- [ ] Phase 5: enable cpu1 loss prepare for planned materialized inputs
- [ ] Phase 6: memory planning integration hardening
- [ ] Phase 7: verification, parity, benchmarks, and docs
- [ ] Phase 8: extend planned input materialization to cpu1 index/scatter units
- [ ] Phase 9: extend planned input materialization to cpu1 attention units

This document is intentionally detailed. It is a planning artifact only. It does
not implement source changes.

## Goal

Introduce graph/lowering-driven contiguous materialization of inputs for cpu1
operations whose direct kernels intentionally require dense contiguous no-offset
inputs.

The first implementation slice targets cpu1 loss operations:

- `CROSS_ENTROPY_LOSS_INDICES`
- dense `NLL_LOSS`
- dense `CROSS_ENTROPY_LOSS`

The same explicit materialization mechanism is also the planned home for later
cpu1 consumer families that currently reject unsupported views instead of
silently copying:

- index/scatter input operands, tracked in Phase 8
- attention `q/k/v/mask` input operands, tracked in Phase 9

The target behavior is:

1. Graph/region optimization identifies when a cpu1 consumer input violates the
   selected dense contiguous no-offset contract.
2. Lowering carries that decision in `RegionExecutionPlan`.
3. Prepare turns the lowered decision into explicit prepared input
   materialization metadata.
4. Runtime executes the prepared materialization before the cpu1 kernel.
5. The cpu1 kernel consumes the materialized dense contiguous input through an
   explicit prepared handle.
6. Trace metadata exposes that materialization happened and why.

The important architectural invariant:

```text
cpu1 kernel does not silently decide to materialize.
cpu1 preparer does not silently decide to materialize.
graph/region/lowering decides materialization and prepare/runtime only execute it.
```

## Non-Goals

- Do not add strided/view loss kernels in this work.
- Do not add arbitrary strided/view attention kernels in this work.
- Do not insert synthetic `CompiledNode` objects as a first implementation.
- Do not create a transitional compatibility layer that routes cpu1 consumers
  back through the old `backend.cpu` kernels.
- Do not hide input materialization behind `Cpu1LossPreparer` fallback logic.
- Do not hide input materialization behind `Cpu1AttentionPreparer` fallback
  logic.
- Do not introduce a generic storage accessor framework just for planned input
  materialization.
- Do not change the public `Tensor` API.
- Do not make backend residency part of the public `Tensor` API.
- Do not commit local benchmark/calibration artifacts.
- Do not modify source files as part of writing this plan.

## Current State And Constraints From Current Codebase

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
replace it with hidden runtime behavior. Instead, the guard must be taught to
accept an explicit prepared effective input descriptor only when lowering
already requested materialization.

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
input descriptors only when graph/region/lowering already requested
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
  materialize unsupported q/k/v/mask views to dense contiguous temps
  execute the existing dense attention kernel
  expose copy cost and reason in trace
```

A future optimized attention phase may add narrower "view-aware dense" kernels
where only batch/head base offsets differ but each inner `depth` / `valueDim`
row remains contiguous. That is not the generic materialization policy and does
not belong inside `Cpu1AttentionPreparer`.

### Relevant existing graph/region/lowering types

Current types that must be respected:

- `graph.compile.planning.region.ExecutionUnit`
  - has `unitId`
  - has `inputValueRefs`, `outputValueRefs`
  - has `orderedNodeIds`
  - has `requiredPreparedInputNodeIds`
  - has no input materialization plan today
- `graph.compile.planning.region.OptimizedRegion`
  - has `executionUnits`
  - has `regionValues`
  - has `materializedOutputs`
  - has no input materialization plan today
- `graph.compile.planning.region.DefaultRegionOptimizer`
  - builds units
  - derives region values
  - emits region trace events
- `graph.compile.planning.region.ExecutionUnitFactory`
  - builds structural unit records
  - should not synthesize `CompiledNode` entries
- `graph.compile.planning.value.GraphValueRef`
  - currently supports only `GraphValueKind.NODE`
  - this plan does not extend it as the first step
- `backend.cpu.lowering.CpuRegionLowerer`
  - converts `ExecutionUnit` into `LoweredExecutionUnit`
  - creates `RegionExecutionPlan`
  - currently maps `GraphValueRef.NODE` back to execution input node ids
- `backend.lowering.region.RegionExecutionPlan`
  - currently has ordered nodes, inputs, outputs, node plans, groups, cost,
    decision, backend payload
  - must become the lowering-visible carrier of input materialization
- `backend.prepare.BackendPrepareContext`
  - owns prepare-time indexes
  - publishes selected backend plans and lowered regions
  - can be extended with an input materialization lookup
- `backend.prepare.PreparedExecutionBuilder`
  - uses lowered CPU fused and specialized units today
  - does not yet use `cpuLoweredUnitForAnchor`/direct lowered CPU units for
    single-op cpu1 loss
- `backend.cpu1.prepare`, `backend.cpu1.exec`, `backend.cpu1.kernels.loss`,
  `backend.cpu1.kernels.layout`
  - own cpu1 prepare/execution/loss/layout implementation

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
enough for `MEMORY_SEGMENT` materialized inputs. A backend-neutral prepared input
materialization buffer needs to represent both Java-array and native CPU segment
targets.

## Architecture Decision

### Decision

Use graph/lowering to decide materialization. cpu1 runtime executes only explicit
prepared materialization.

The first implementation must not insert a synthetic `CompiledNode` for
`CONTIGUOUS`. Instead, it should add a data model that can express:

```text
consumer node N input i requires DENSE_CONTIGUOUS_NO_OFFSET;
source node S is not dense contiguous no-offset;
materialize S into temp input T before executing N.
```

### Why not synthetic `CompiledNode` first

A synthetic `CompiledNode` for `CONTIGUOUS` would be tempting because existing
layout kernels already know how to materialize. It is rejected for the first
slice because it would force the compile graph, descriptor index, memory plan,
publication, step ordering, and trace identity to pretend that a user graph node
exists. The desired behavior is a backend execution requirement for a selected
region/unit, not a semantic graph rewrite.

Synthetic nodes may become useful later if graph-level layout canonicalization
is implemented broadly. This plan keeps the first implementation narrower and
more explicit.

### Why not hidden cpu1 preparer materialization

Hidden materialization in a family preparer such as `Cpu1LossPreparer` or
`Cpu1AttentionPreparer` would make tests pass quickly, but it would violate the
runtime boundary:

- graph/lowering trace would not explain why the copy exists
- benchmark reports would not distinguish loss compute from planned copy-in
- cpu1 kernels would gain policy behavior
- future accelerator/backends would need to rediscover the same decision

A cpu1 preparer may validate and bind an already planned materialization. It
must not decide to create one from scratch.

### Why no generic storage accessor framework now

The target is loss input materialization, not general strided loss execution.
A generic accessor framework would add indirection to cpu1 hot paths and blur
the direct prepared-unit style that cpu1 is moving toward. The immediate copy
needs are simple:

- read a logical source tensor using shape/strides/storage offset
- write a dense contiguous temporary buffer
- support `FLOAT32`, `FLOAT64`, `BFLOAT16`, `INT32`, `INT64`, `BOOL`
- support Java arrays and native CPU segments

A small logical-copy helper is enough.

### Why not immediate strided loss kernels

Strided loss kernels would multiply kernel variants for:

- logits/log-probs layout
- target layout
- storage kind
- dtype
- axis position
- reduction mode

This is premature. Current dense kernels are simple and validated. The first
correct architecture is to make the materialization decision explicit and
visible. Strided kernels can be added later as an alternate lowering decision.

### Why not immediate arbitrary strided attention kernels

Generic strided attention is even more likely to be a poor default hot path than
generic strided loss:

- `Q @ K^T` performs a dot product for every `(query, key)` pair
- softmax needs a scratch row over keys
- `weights @ V` reads every value row for every query row
- mask handling and all-masked rows add control flow
- vectorized dot products require contiguous inner `depth` loads
- vectorized value accumulation requires contiguous `valueDim` loads

For arbitrary view layouts, the kernel would pay offset math inside the hottest
loops and would often lose vectorization. The first policy should therefore be:

```text
if q/k/v/mask are dense contiguous no-offset:
  use cpu1 dense attention directly
else:
  graph/lowering may plan explicit input materialization
  cpu1 attention consumes the materialized dense effective input
```

Later, lowering may choose a specialized view-aware attention kernel for the
restricted case where the inner compute rows stay contiguous and only batch/head
base offsets differ. That is an optimization decision, not the baseline
materialization policy.

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

Region optimizer:
  unit cpu-region-0-unit-5 consumes node-3 and node-4
  Cpu1BackendInputContract says loss input 0 and input 1 require
  DENSE_CONTIGUOUS_NO_OFFSET
  node-3 descriptor is STRIDED_VIEW
  create InputMaterializationRequirement:
      consumerNodeId=5
      inputIndex=0
      sourceValueRef=node-3
      requiredLayout=DENSE_CONTIGUOUS_NO_OFFSET

Lowering:
  RegionInputMaterializationPlan:
      materializationId=cpu-region-0/unit-5/input-0/node-3
      sourceNodeId=3
      consumerNodeId=5
      inputIndex=0
      dataType=FLOAT32
      shape=[2, 8]
      targetStrides=[8, 1]
      targetStorageOffset=0

Prepare:
  Cpu1LossPreparer receives RegionInputMaterializationPlan
  effective logits descriptor is dense contiguous no-offset
  prepared logits input stores:
      sourceNodeId=3
      materializationId=cpu-region-0/unit-5/input-0/node-3
      materialized=true
  targets remain direct source input

Runtime:
  Cpu1LossExecutableUnit.run(context)
    -> Cpu1InputMaterializer materializes node 3 into temp contiguous input
    -> Cpu1CrossEntropyLossIndicesLoops reads logits from materialized temp
    -> targets are read from node 4
    -> output node 5 is written by existing dense loss kernel

Trace:
  cpu1LossMaterializedInputCount=1
  cpu1LossMaterializedInputs=[input=0,source=3,layout=STRIDED_VIEW,target=DENSE_CONTIGUOUS_NO_OFFSET]
  cpu1LossMaterializationReason=cpu1-loss-dense-contiguous-input-contract
```

## Exact Files Planned

### New source files

- `src/main/java/graph/compile/planning/region/InputLayoutRequirement.java`
- `src/main/java/graph/compile/planning/region/BackendInputContract.java`
- `src/main/java/graph/compile/planning/region/Cpu1BackendInputContract.java`
- `src/main/java/graph/compile/planning/region/InputMaterializationRequirement.java`
- `src/main/java/graph/compile/planning/region/RegionInputMaterializationPlanner.java`
- `src/main/java/backend/lowering/region/RegionInputMaterializationPlan.java`
- `src/main/java/graph/execution/plan/PreparedInputStorageKind.java`
- `src/main/java/graph/execution/plan/PreparedInputMaterializationSpec.java`
- `src/main/java/graph/execution/plan/PreparedInputMaterializationBuffer.java`
- `src/main/java/backend/cpu1/prepare/Cpu1PreparedInputMaterialization.java`
- `src/main/java/backend/cpu1/prepare/Cpu1PreparedLossInput.java`
- `src/main/java/backend/cpu1/prepare/Cpu1LossInputResolver.java`
- `src/main/java/backend/cpu1/exec/Cpu1InputMaterializer.java`
- `src/main/java/backend/cpu1/kernels/layout/copy/Cpu1LogicalTensorCopy.java`
- `src/main/java/backend/cpu1/kernels/loss/Cpu1LossInputViews.java`

### Modified source files

- `src/main/java/graph/compile/planning/region/OptimizedRegion.java`
- `src/main/java/graph/compile/planning/region/DefaultRegionOptimizer.java`
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

- `src/test/java/graph/compile/planning/region/Cpu1LossInputMaterializationPlanningTest.java`
- `src/test/java/backend/cpu1/Cpu1LossMaterializationExecutionContractTest.java`

### Modified tests

- `src/test/java/backend/cpu1/Cpu1CrossEntropyLossExecutionContractTest.java`
- `src/test/java/backend/cpu1/Cpu1NllLossExecutionContractTest.java`
- `src/test/java/backend/cpu1/Cpu1DenseCrossEntropyLossExecutionContractTest.java`
- `src/test/java/backend/lowering/region/RegionExecutionPlanTest.java`
- `src/test/java/backend/cpu/lowering/CpuRegionLowererTest.java`

## Phase 0: Safety/Inventory Tests And Current-State Guards

Status: `[ ] not started`

Purpose: establish that dense existing paths stay green and unplanned strided
loss inputs remain rejected.

### Task 0.1: Add current-state guard for unplanned strided cpu1 loss input

Status: `[ ] not started`

What: add a test that calls `Cpu1NodePreparer` directly with a strided/view loss
input and verifies that it still rejects when no materialization plan is passed.

Why: this prevents an accidental hidden materialization path inside
`Cpu1LossPreparer`.

Proposed test path:

- `src/test/java/backend/cpu1/Cpu1LossMaterializationExecutionContractTest.java`

Initial test methods:

```java
@Test
void unplannedStridedCrossEntropyIndicesInputIsRejectedByCpu1LossPreparer() {
    Tensor base = new Tensor(new float[]{
            1.0f, 2.0f, 3.0f,
            4.0f, 5.0f, 6.0f,
            7.0f, 8.0f, 9.0f,
            10.0f, 11.0f, 12.0f
    }, new int[]{4, 3}, null, "unplannedBaseLogits", DataType.FLOAT32);
    Tensor logits = base.slice(0, 0, 4, 2);
    Tensor targets = new Tensor(new int[]{2, 1}, new int[]{2}, null, "unplannedTargets", DataType.INT32);
    Tensor loss = logits.crossEntropyLossFromIndices(targets, 1, LossReduction.MEAN);
    Fixture fixture = fixture(loss);

    UnsupportedOperationException thrown = assertThrows(
            UnsupportedOperationException.class,
            () -> new Cpu1NodePreparer().prepare(
                    fixture.node(),
                    fixture.descriptorIndex(),
                    Cpu1PrepareConfig.scalarSingleThread()
            )
    );

    assertTrue(thrown.getMessage().contains("requires dense contiguous"));
}
```

This test is intentionally written against the old direct prepare path. It
should remain true after all phases unless the call site passes an explicit
materialization plan.

### Task 0.2: Inventory existing dense loss tests

Status: `[ ] not started`

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

### Task 0.3: Add trace expectation for no materialization on dense paths

Status: `[ ] not started`

What: update existing loss trace assertions to include:

```java
assertEquals(0, attrs.getOrDefault("cpu1LossMaterializedInputCount", 0));
```

Why: dense paths should remain explicit: no materialized inputs were planned,
and traces should say zero rather than omit the concept once the feature exists.

## Phase 1: Graph Region Materialization Requirement Model

Status: `[ ] not started`

Purpose: represent input layout requirements in graph/region planning without
creating synthetic nodes.

### Task 1.1: Add `InputLayoutRequirement`

Status: `[ ] not started`

What: add a small immutable requirement describing the required input layout.

Why: cpu1 loss currently needs dense contiguous no-offset inputs, but future
backends may need different input contracts. A named value object keeps that
decision out of cpu1 kernels.

File:

- `src/main/java/graph/compile/planning/region/InputLayoutRequirement.java`

Complete proposed code:

```java
package graph.compile.planning.region;

import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.LayoutClass;
import tensor.TensorMetadata;

import java.util.Arrays;
import java.util.Objects;

/**
 * Backend input layout requirement discovered during region optimization.
 */
public record InputLayoutRequirement(
        Kind kind,
        LayoutClass requiredLayoutClass,
        boolean requireZeroStorageOffset,
        boolean requireDenseStrides,
        String reasonCode
) {
    public enum Kind {
        AS_IS,
        DENSE_CONTIGUOUS_NO_OFFSET
    }

    public InputLayoutRequirement {
        kind = kind == null ? Kind.AS_IS : kind;
        requiredLayoutClass = requiredLayoutClass == null ? LayoutClass.UNKNOWN_OR_COMPLEX : requiredLayoutClass;
        reasonCode = reasonCode == null ? "" : reasonCode;
        if (kind == Kind.AS_IS) {
            requiredLayoutClass = LayoutClass.UNKNOWN_OR_COMPLEX;
            requireZeroStorageOffset = false;
            requireDenseStrides = false;
        }
    }

    public static InputLayoutRequirement asIs() {
        return new InputLayoutRequirement(
                Kind.AS_IS,
                LayoutClass.UNKNOWN_OR_COMPLEX,
                false,
                false,
                ""
        );
    }

    public static InputLayoutRequirement denseContiguousNoOffset(String reasonCode) {
        return new InputLayoutRequirement(
                Kind.DENSE_CONTIGUOUS_NO_OFFSET,
                LayoutClass.DENSE_CONTIGUOUS,
                true,
                true,
                reasonCode
        );
    }

    public boolean requiresMaterialization(CompiledTensorDescriptor descriptor) {
        return !satisfiedBy(descriptor);
    }

    public boolean satisfiedBy(CompiledTensorDescriptor descriptor) {
        Objects.requireNonNull(descriptor, "descriptor cannot be null");
        if (kind == Kind.AS_IS) {
            return true;
        }
        if (requiredLayoutClass != LayoutClass.UNKNOWN_OR_COMPLEX
                && descriptor.layoutClass() != requiredLayoutClass) {
            return false;
        }
        if (requireZeroStorageOffset && descriptor.storageOffset() != 0) {
            return false;
        }
        if (requireDenseStrides
                && !Arrays.equals(descriptor.strides(), TensorMetadata.computeStrides(descriptor.shape()))) {
            return false;
        }
        return true;
    }

    public int[] effectiveTargetStrides(int[] shape) {
        if (kind == Kind.AS_IS) {
            return shape == null ? new int[0] : TensorMetadata.computeStrides(shape);
        }
        if (kind == Kind.DENSE_CONTIGUOUS_NO_OFFSET) {
            return TensorMetadata.computeStrides(shape == null ? new int[0] : shape);
        }
        throw new IllegalStateException("Unhandled input layout requirement kind: " + kind);
    }
}
```

### Task 1.2: Add `BackendInputContract`

Status: `[ ] not started`

What: add a graph-region interface for target/input contracts.

Why: `DefaultRegionOptimizer` should ask a target policy for requirements rather
than hard-code cpu1 loss operation details throughout optimizer code.

File:

- `src/main/java/graph/compile/planning/region/BackendInputContract.java`

Complete proposed code:

```java
package graph.compile.planning.region;

import graph.CompiledNode;

import java.util.Collections;
import java.util.List;

/**
 * Region-time backend input contract.
 */
public interface BackendInputContract {
    List<InputLayoutRequirement> inputRequirementsFor(
            ExecutionUnit unit,
            CompiledNode consumer,
            RegionOptimizationContext context
    );

    static BackendInputContract none() {
        return (unit, consumer, context) -> {
            int inputCount = consumer == null ? 0 : consumer.inputIds().size();
            return Collections.nCopies(inputCount, InputLayoutRequirement.asIs());
        };
    }
}
```

### Task 1.3: Add `Cpu1BackendInputContract`

Status: `[ ] not started`

What: add the first target-specific input contract. It should only describe
cpu1 loss dense input requirements.

Why: cpu1 loss dense kernels support current dense inputs but require any
strided/view source to be materialized before compute.

File:

- `src/main/java/graph/compile/planning/region/Cpu1BackendInputContract.java`

Complete proposed code:

```java
package graph.compile.planning.region;

import graph.CompiledNode;
import operations.Operation;

import java.util.Collections;
import java.util.List;

/**
 * cpu1-specific graph input contract for region planning.
 */
public final class Cpu1BackendInputContract implements BackendInputContract {
    public static final String LOSS_DENSE_CONTIGUOUS_REASON =
            "cpu1-loss-dense-contiguous-input-contract";

    private static final InputLayoutRequirement DENSE_CONTIGUOUS_NO_OFFSET =
            InputLayoutRequirement.denseContiguousNoOffset(LOSS_DENSE_CONTIGUOUS_REASON);

    @Override
    public List<InputLayoutRequirement> inputRequirementsFor(
            ExecutionUnit unit,
            CompiledNode consumer,
            RegionOptimizationContext context
    ) {
        if (consumer == null || consumer.operation() == null) {
            return List.of();
        }
        Operation.OpType opType = consumer.operation().opType();
        return switch (opType) {
            case CROSS_ENTROPY_LOSS_INDICES, NLL_LOSS, CROSS_ENTROPY_LOSS ->
                    repeated(consumer.inputIds().size(), DENSE_CONTIGUOUS_NO_OFFSET);
            default -> repeated(consumer.inputIds().size(), InputLayoutRequirement.asIs());
        };
    }

    private static List<InputLayoutRequirement> repeated(
            int count,
            InputLayoutRequirement requirement
    ) {
        if (count <= 0) {
            return List.of();
        }
        return Collections.nCopies(count, requirement);
    }
}
```

### Task 1.4: Add `InputMaterializationRequirement`

Status: `[ ] not started`

What: add a region-level record describing one planned source-input to
temporary dense input requirement.

Why: this record is the graph/region decision. It is not an executable copy and
does not imply a cpu1 storage kind.

File:

- `src/main/java/graph/compile/planning/region/InputMaterializationRequirement.java`

Complete proposed code:

```java
package graph.compile.planning.region;

import graph.compile.planning.value.GraphValueRef;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Region-level planned input materialization.
 */
public record InputMaterializationRequirement(
        String requirementId,
        String unitId,
        int consumerNodeId,
        int inputIndex,
        GraphValueRef sourceValueRef,
        InputLayoutRequirement layoutRequirement,
        DataType dataType,
        int[] shape,
        int[] sourceStrides,
        int sourceStorageOffset,
        int[] targetStrides,
        long elementCount,
        String reasonCode
) {
    public InputMaterializationRequirement {
        if (requirementId == null || requirementId.isBlank()) {
            throw new IllegalArgumentException("requirementId cannot be blank");
        }
        if (unitId == null || unitId.isBlank()) {
            throw new IllegalArgumentException("unitId cannot be blank");
        }
        if (consumerNodeId < 0 || inputIndex < 0) {
            throw new IllegalArgumentException("consumerNodeId and inputIndex must be non-negative");
        }
        sourceValueRef = Objects.requireNonNull(sourceValueRef, "sourceValueRef cannot be null");
        layoutRequirement = Objects.requireNonNull(layoutRequirement, "layoutRequirement cannot be null");
        dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        shape = shape == null ? new int[0] : shape.clone();
        sourceStrides = sourceStrides == null ? new int[0] : sourceStrides.clone();
        targetStrides = targetStrides == null ? new int[0] : targetStrides.clone();
        if (sourceStorageOffset < 0) {
            throw new IllegalArgumentException("sourceStorageOffset cannot be negative");
        }
        if (elementCount < 0L) {
            throw new IllegalArgumentException("elementCount cannot be negative");
        }
        reasonCode = reasonCode == null ? "" : reasonCode;
    }

    public int sourceNodeId() {
        return sourceValueRef.nodeId();
    }

    public String tempValueId() {
        return requirementId;
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] sourceStrides() {
        return sourceStrides.clone();
    }

    @Override
    public int[] targetStrides() {
        return targetStrides.clone();
    }

    @Override
    public String toString() {
        return "InputMaterializationRequirement{"
                + "requirementId='" + requirementId + '\''
                + ", unitId='" + unitId + '\''
                + ", consumerNodeId=" + consumerNodeId
                + ", inputIndex=" + inputIndex
                + ", sourceValueRef=" + sourceValueRef.valueId()
                + ", layoutRequirement=" + layoutRequirement.kind()
                + ", dataType=" + dataType
                + ", shape=" + Arrays.toString(shape)
                + ", reasonCode='" + reasonCode + '\''
                + '}';
    }
}
```

### Task 1.5: Add `RegionInputMaterializationPlanner`

Status: `[ ] not started`

What: scan optimized execution units and create requirements for inputs that
violate the target contract.

Why: this is the first place where the decision exists. It keeps cpu1 loss
materialization out of kernel runtime and out of `Cpu1LossPreparer`.

File:

- `src/main/java/graph/compile/planning/region/RegionInputMaterializationPlanner.java`

Complete proposed code:

```java
package graph.compile.planning.region;

import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.LayoutClass;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.value.GraphValueRef;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Builds explicit input materialization requirements for optimized region units.
 */
public final class RegionInputMaterializationPlanner {
    private final BackendInputContract inputContract;

    public RegionInputMaterializationPlanner(BackendInputContract inputContract) {
        this.inputContract = Objects.requireNonNull(inputContract, "inputContract cannot be null");
    }

    public static RegionInputMaterializationPlanner cpu1Defaults() {
        return new RegionInputMaterializationPlanner(new Cpu1BackendInputContract());
    }

    public List<InputMaterializationRequirement> plan(
            Partition partition,
            RegionOptimizationContext context,
            List<ExecutionUnit> units
    ) {
        Objects.requireNonNull(partition, "partition cannot be null");
        Objects.requireNonNull(context, "context cannot be null");
        if (partition.target() != PartitionTarget.CPU || units == null || units.isEmpty()) {
            return List.of();
        }
        ArrayList<InputMaterializationRequirement> out = new ArrayList<>();
        for (ExecutionUnit unit : units) {
            collectUnitRequirements(unit, context, out);
        }
        return List.copyOf(out);
    }

    private void collectUnitRequirements(
            ExecutionUnit unit,
            RegionOptimizationContext context,
            ArrayList<InputMaterializationRequirement> out
    ) {
        if (unit == null) {
            return;
        }
        for (int consumerNodeId : unit.orderedNodeIds()) {
            CompiledNode consumer = context.compiledNode(consumerNodeId);
            if (consumer == null || consumer.operation() == null) {
                continue;
            }
            List<InputLayoutRequirement> requirements =
                    inputContract.inputRequirementsFor(unit, consumer, context);
            for (int inputIndex = 0; inputIndex < consumer.inputIds().size(); inputIndex++) {
                InputLayoutRequirement requirement = inputIndex < requirements.size()
                        ? requirements.get(inputIndex)
                        : InputLayoutRequirement.asIs();
                if (requirement.kind() == InputLayoutRequirement.Kind.AS_IS) {
                    continue;
                }
                int sourceNodeId = consumer.inputIds().get(inputIndex);
                CompiledNode source = context.compiledNode(sourceNodeId);
                if (source == null) {
                    continue;
                }
                CompiledTensorDescriptor sourceDescriptor = CompiledTensorDescriptorBuilder.fromNode(source);
                if (!requirement.requiresMaterialization(sourceDescriptor)) {
                    continue;
                }
                if (!materializableByLogicalCopy(sourceDescriptor)) {
                    continue;
                }
                out.add(requirementFor(unit, consumer, inputIndex, sourceDescriptor, requirement));
            }
        }
    }

    private InputMaterializationRequirement requirementFor(
            ExecutionUnit unit,
            CompiledNode consumer,
            int inputIndex,
            CompiledTensorDescriptor source,
            InputLayoutRequirement layoutRequirement
    ) {
        String id = unit.unitId()
                + ":node-" + consumer.id()
                + ":input-" + inputIndex
                + ":source-node-" + source.nodeId();
        return new InputMaterializationRequirement(
                id,
                unit.unitId(),
                consumer.id(),
                inputIndex,
                GraphValueRef.node(source.nodeId()),
                layoutRequirement,
                source.dataType(),
                source.shape(),
                source.strides(),
                source.storageOffset(),
                layoutRequirement.effectiveTargetStrides(source.shape()),
                source.logicalElementCount(),
                layoutRequirement.reasonCode()
        );
    }

    private static boolean materializableByLogicalCopy(CompiledTensorDescriptor descriptor) {
        if (descriptor.logicalElementCount() > Integer.MAX_VALUE) {
            return false;
        }
        return switch (descriptor.layoutClass()) {
            case DENSE_CONTIGUOUS -> false;
            case DENSE_WITH_OFFSET, STRIDED_VIEW, BROADCAST_ZERO_STRIDE -> true;
            case UNKNOWN_OR_COMPLEX -> false;
        };
    }
}
```

Important note: `UNKNOWN_OR_COMPLEX` is not accepted in the first slice. That
keeps the implementation honest. If a future tensor layout needs support, add a
test and a deliberate copy implementation.

### Task 1.6: Extend `OptimizedRegion`

Status: `[ ] not started`

What: add `inputMaterializationRequirements` to the optimized region.

Why: lowering needs to see a region-level materialization decision without
modifying `GraphValueRef` or creating synthetic nodes.

File:

- `src/main/java/graph/compile/planning/region/OptimizedRegion.java`

Complete proposed record shape:

```java
public record OptimizedRegion(
        String regionId,
        ExecutionRegionKind regionKind,
        Partition sourcePartition,
        PartitionTarget target,
        List<ExecutionUnit> executionUnits,
        List<RegionValue> regionValues,
        List<GraphValueRef> materializedOutputs,
        List<InputMaterializationRequirement> inputMaterializationRequirements,
        RegionOptimizationTrace trace
) {
    public OptimizedRegion {
        if (regionId == null || regionId.isBlank()) {
            throw new IllegalArgumentException("regionId cannot be blank");
        }
        if (sourcePartition == null || target == null) {
            throw new IllegalArgumentException("sourcePartition and target cannot be null");
        }
        regionKind = regionKind == null ? sourcePartition.regionKind() : regionKind;
        executionUnits = List.copyOf(executionUnits == null ? List.of() : executionUnits);
        regionValues = List.copyOf(regionValues == null ? List.of() : regionValues);
        materializedOutputs = List.copyOf(materializedOutputs == null ? List.of() : materializedOutputs);
        inputMaterializationRequirements = List.copyOf(
                inputMaterializationRequirements == null ? List.of() : inputMaterializationRequirements
        );
        trace = trace == null ? RegionOptimizationTrace.empty() : trace;
    }

    public OptimizedRegion(
            String regionId,
            ExecutionRegionKind regionKind,
            Partition sourcePartition,
            PartitionTarget target,
            List<ExecutionUnit> executionUnits,
            List<RegionValue> regionValues,
            List<GraphValueRef> materializedOutputs,
            RegionOptimizationTrace trace
    ) {
        this(
                regionId,
                regionKind,
                sourcePartition,
                target,
                executionUnits,
                regionValues,
                materializedOutputs,
                List.of(),
                trace
        );
    }

    public OptimizedRegion(
            String regionId,
            Partition sourcePartition,
            PartitionTarget target,
            List<ExecutionUnit> executionUnits,
            List<RegionValue> regionValues,
            List<GraphValueRef> materializedOutputs,
            RegionOptimizationTrace trace
    ) {
        this(
                regionId,
                sourcePartition == null ? null : sourcePartition.regionKind(),
                sourcePartition,
                target,
                executionUnits,
                regionValues,
                materializedOutputs,
                List.of(),
                trace
        );
    }
}
```

### Task 1.7: Wire planner into `DefaultRegionOptimizer`

Status: `[ ] not started`

What: after units are built and before the final `OptimizedRegion` is returned,
run `RegionInputMaterializationPlanner`.

Why: this ensures the decision is part of region optimization trace and not
added later by cpu1 prepare.

File:

- `src/main/java/graph/compile/planning/region/DefaultRegionOptimizer.java`

Complete replacement for the tail of `optimize(...)`:

```java
UnitBuildResult unitBuild = buildUnits(partition, context);
List<ExecutionUnit> units = unitBuild.units();
List<InputMaterializationRequirement> inputMaterializations =
        RegionInputMaterializationPlanner.cpu1Defaults().plan(partition, context, units);
List<RegionValue> regionValues = partition.values().stream()
        .map(value -> toRegionValue(value, partition, context, units))
        .toList();

ArrayList<String> traceEvents = new ArrayList<>();
traceEvents.add("units=" + units.size());
traceEvents.add("target=" + partition.target().name());
traceEvents.add("regionKind=" + partition.regionKind().name());
traceEvents.add("plannerStrategy=" + partition.plannerStrategy().name());
traceEvents.add("inputMaterializations=" + inputMaterializations.size());
for (InputMaterializationRequirement requirement : inputMaterializations) {
    traceEvents.add("input-materialization-required:"
            + "unit=" + requirement.unitId()
            + ",consumerNode=" + requirement.consumerNodeId()
            + ",inputIndex=" + requirement.inputIndex()
            + ",source=" + requirement.sourceValueRef().valueId()
            + ",layout=" + requirement.layoutRequirement().kind().name()
            + ",reason=" + requirement.reasonCode());
}
traceEvents.addAll(unitBuild.traceEvents());
RegionOptimizationTrace trace = new RegionOptimizationTrace(traceEvents);

return new OptimizedRegion(
        partition.partitionId(),
        partition,
        partition.target(),
        units,
        regionValues,
        partition.requiredMaterializedValueRefs(),
        inputMaterializations,
        trace
);
```

## Phase 2: Propagate Materialization Requirements Through Lowering

Status: `[ ] not started`

Purpose: make the region decision available to prepare and trace through
`RegionExecutionPlan`.

### Task 2.1: Add `RegionInputMaterializationPlan`

Status: `[ ] not started`

What: add a lowering-level immutable plan.

Why: lowering artifacts should not expose graph-only records directly to cpu1
prepare. This record is backend-neutral enough for `RegionExecutionPlan`, but
contains concrete node ids and dense target layout metadata.

File:

- `src/main/java/backend/lowering/region/RegionInputMaterializationPlan.java`

Complete proposed code:

```java
package backend.lowering.region;

import graph.compile.planning.region.InputLayoutRequirement;
import graph.compile.planning.region.InputMaterializationRequirement;
import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Lowered region plan for one prepared input materialization.
 */
public record RegionInputMaterializationPlan(
        String materializationId,
        String unitId,
        int consumerNodeId,
        int inputIndex,
        int sourceNodeId,
        InputLayoutRequirement.Kind requiredLayoutKind,
        DataType dataType,
        int[] shape,
        int[] sourceStrides,
        int sourceStorageOffset,
        int[] targetStrides,
        long elementCount,
        RegionStorageContract storageContract,
        String reasonCode
) {
    public RegionInputMaterializationPlan {
        if (materializationId == null || materializationId.isBlank()) {
            throw new IllegalArgumentException("materializationId cannot be blank");
        }
        if (unitId == null || unitId.isBlank()) {
            throw new IllegalArgumentException("unitId cannot be blank");
        }
        if (consumerNodeId < 0 || inputIndex < 0 || sourceNodeId < 0) {
            throw new IllegalArgumentException("node ids and inputIndex must be non-negative");
        }
        requiredLayoutKind = Objects.requireNonNull(requiredLayoutKind, "requiredLayoutKind cannot be null");
        dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        shape = shape == null ? new int[0] : shape.clone();
        sourceStrides = sourceStrides == null ? new int[0] : sourceStrides.clone();
        targetStrides = targetStrides == null ? new int[0] : targetStrides.clone();
        if (sourceStorageOffset < 0) {
            throw new IllegalArgumentException("sourceStorageOffset cannot be negative");
        }
        if (elementCount < 0L || elementCount > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("elementCount must fit int for first implementation: " + elementCount);
        }
        storageContract = storageContract == null ? RegionStorageContract.CPU_ARRAY : storageContract;
        reasonCode = reasonCode == null ? "" : reasonCode;
    }

    public static RegionInputMaterializationPlan from(InputMaterializationRequirement requirement) {
        Objects.requireNonNull(requirement, "requirement cannot be null");
        return new RegionInputMaterializationPlan(
                requirement.tempValueId(),
                requirement.unitId(),
                requirement.consumerNodeId(),
                requirement.inputIndex(),
                requirement.sourceNodeId(),
                requirement.layoutRequirement().kind(),
                requirement.dataType(),
                requirement.shape(),
                requirement.sourceStrides(),
                requirement.sourceStorageOffset(),
                requirement.targetStrides(),
                requirement.elementCount(),
                RegionStorageContract.CPU_ARRAY,
                requirement.reasonCode()
        );
    }

    public int elementCountAsInt() {
        return Math.toIntExact(elementCount);
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] sourceStrides() {
        return sourceStrides.clone();
    }

    @Override
    public int[] targetStrides() {
        return targetStrides.clone();
    }

    @Override
    public String toString() {
        return "RegionInputMaterializationPlan{"
                + "materializationId='" + materializationId + '\''
                + ", consumerNodeId=" + consumerNodeId
                + ", inputIndex=" + inputIndex
                + ", sourceNodeId=" + sourceNodeId
                + ", requiredLayoutKind=" + requiredLayoutKind
                + ", dataType=" + dataType
                + ", shape=" + Arrays.toString(shape)
                + ", reasonCode='" + reasonCode + '\''
                + '}';
    }
}
```

### Task 2.2: Extend `RegionExecutionPlan`

Status: `[ ] not started`

What: add `inputMaterializationPlans`.

Why: prepare already receives region plans through lowered execution units.
This is the cleanest carrier.

File:

- `src/main/java/backend/lowering/region/RegionExecutionPlan.java`

Complete canonical field addition:

```java
public record RegionExecutionPlan(
        String regionId,
        PartitionTarget target,
        LoweringFamily loweringFamily,
        int anchorNodeId,
        List<Integer> orderedNodeIds,
        List<Integer> externalInputNodeIds,
        List<Integer> boundaryOutputNodeIds,
        List<RegionNodePlan> nodePlans,
        List<RegionExecutionGroup> executionGroups,
        List<RegionInputMaterializationPlan> inputMaterializationPlans,
        RegionCost cost,
        RegionDecision decision,
        RegionBackendPayload backendPayload
) implements LoweredUnitArtifact {
    public RegionExecutionPlan {
        regionId = regionId == null ? "" : regionId;
        if (regionId.isBlank()) {
            throw new IllegalArgumentException("regionId cannot be blank");
        }
        target = target == null ? PartitionTarget.NONE : target;
        loweringFamily = Objects.requireNonNull(loweringFamily, "loweringFamily cannot be null");
        if (anchorNodeId < 0) {
            throw new IllegalArgumentException("anchorNodeId must be >= 0");
        }
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        requireUnique(orderedNodeIds, "orderedNodeIds");
        if (!orderedNodeIds.contains(anchorNodeId)) {
            throw new IllegalArgumentException("orderedNodeIds must contain anchorNodeId=" + anchorNodeId);
        }
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        boundaryOutputNodeIds = List.copyOf(boundaryOutputNodeIds == null ? List.of() : boundaryOutputNodeIds);
        requireUnique(boundaryOutputNodeIds, "boundaryOutputNodeIds");
        requireSubset(boundaryOutputNodeIds, orderedNodeIds, "boundaryOutputNodeIds");
        nodePlans = List.copyOf(nodePlans == null ? List.of() : nodePlans);
        executionGroups = List.copyOf(executionGroups == null ? List.of() : executionGroups);
        inputMaterializationPlans = List.copyOf(
                inputMaterializationPlans == null ? List.of() : inputMaterializationPlans
        );
        validateNodePlans(nodePlans, orderedNodeIds);
        validateExecutionGroups(executionGroups, orderedNodeIds);
        validateInputMaterializationPlans(inputMaterializationPlans, orderedNodeIds);
        cost = cost == null ? RegionCost.ofWork(0L) : cost;
        decision = decision == null ? RegionDecision.selected(loweringFamily.id(), "selected") : decision;
        backendPayload = backendPayload == null ? EmptyRegionPayload.INSTANCE : backendPayload;
    }

    public RegionExecutionPlan(
            String regionId,
            PartitionTarget target,
            LoweringFamily loweringFamily,
            int anchorNodeId,
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            List<Integer> boundaryOutputNodeIds,
            List<RegionNodePlan> nodePlans,
            List<RegionExecutionGroup> executionGroups,
            RegionCost cost,
            RegionDecision decision,
            RegionBackendPayload backendPayload
    ) {
        this(
                regionId,
                target,
                loweringFamily,
                anchorNodeId,
                orderedNodeIds,
                externalInputNodeIds,
                boundaryOutputNodeIds,
                nodePlans,
                executionGroups,
                List.of(),
                cost,
                decision,
                backendPayload
        );
    }
}
```

Complete new validation helper:

```java
private static void validateInputMaterializationPlans(
        List<RegionInputMaterializationPlan> inputMaterializationPlans,
        List<Integer> orderedNodeIds
) {
    Set<String> ids = new LinkedHashSet<>();
    for (RegionInputMaterializationPlan plan : inputMaterializationPlans) {
        if (plan == null) {
            throw new IllegalArgumentException("inputMaterializationPlans cannot contain null");
        }
        if (!orderedNodeIds.contains(plan.consumerNodeId())) {
            throw new IllegalArgumentException("inputMaterializationPlans contain consumer outside orderedNodeIds: "
                    + plan.consumerNodeId());
        }
        if (!ids.add(plan.materializationId())) {
            throw new IllegalArgumentException("Duplicate input materialization id=" + plan.materializationId());
        }
    }
}
```

### Task 2.3: Map requirements in `CpuRegionLowerer`

Status: `[ ] not started`

What: filter region requirements by `unitId`, convert them to
`RegionInputMaterializationPlan`, and attach them to the region execution plan.

Why: lowering is the boundary where region planning becomes backend execution
metadata.

File:

- `src/main/java/backend/cpu/lowering/CpuRegionLowerer.java`

Complete method additions:

```java
private List<RegionInputMaterializationPlan> inputMaterializationPlans(
        ExecutionUnit unit,
        LoweringRequest request
) {
    if (request.region().inputMaterializationRequirements().isEmpty()) {
        return List.of();
    }
    return request.region().inputMaterializationRequirements().stream()
            .filter(requirement -> unit.unitId().equals(requirement.unitId()))
            .map(RegionInputMaterializationPlan::from)
            .toList();
}

private List<String> tempValueIds(List<RegionInputMaterializationPlan> plans) {
    if (plans == null || plans.isEmpty()) {
        return List.of();
    }
    return plans.stream()
            .map(RegionInputMaterializationPlan::materializationId)
            .distinct()
            .toList();
}
```

Complete `regionPlan(...)` local changes:

```java
List<RegionInputMaterializationPlan> inputMaterializationPlans =
        inputMaterializationPlans(unit, request);

RegionExecutionGroup group = new RegionExecutionGroup(
        unit.unitId() + "-group-0",
        orderedNodeIds,
        executionKind,
        physicalKernel,
        externalInputNodeIds,
        boundaryOutputNodeIds.isEmpty() ? List.of(anchorNodeId) : boundaryOutputNodeIds,
        tempValueIds(inputMaterializationPlans),
        storageContract,
        inputMaterializationPlans.isEmpty()
                ? "cpu-lowered-unit"
                : "cpu-lowered-unit-with-input-materialization"
);
return new RegionExecutionPlan(
        request.region().regionId() + "/" + unit.unitId(),
        graph.compile.planning.partition.PartitionTarget.CPU,
        family,
        anchorNodeId,
        orderedNodeIds,
        externalInputNodeIds,
        boundaryOutputNodeIds.isEmpty() ? List.of(anchorNodeId) : boundaryOutputNodeIds,
        nodePlans,
        List.of(group),
        inputMaterializationPlans,
        RegionCost.ofWork(unit.estimatedWork()),
        inputMaterializationPlans.isEmpty()
                ? RegionDecision.selected(family.id(), "cpu-lowered-unit")
                : RegionDecision.selected(family.id(), "cpu-lowered-unit-with-input-materialization"),
        payload
);
```

### Task 2.4: Index direct CPU lowered units and materialization plans in prepare

Status: `[ ] not started`

What: extend `LoweredRegionIndex` with:

- direct CPU unit by start node id
- materialization plans by `(consumerNodeId, inputIndex)`

Why: `PreparedExecutionBuilder` currently consumes CPU fused and specialized
lowered units but not ordinary lowered direct CPU units. cpu1 loss materialization
must come from a lowered unit, so the direct unit has to be prepared through a
lowered path.

File:

- `src/main/java/backend/prepare/LoweredRegionIndex.java`

Complete new key helper:

```java
private static long materializationKey(int consumerNodeId, int inputIndex) {
    return ((long) consumerNodeId << Integer.SIZE) ^ (inputIndex & 0xffffffffL);
}
```

Complete new fields:

```java
private final Map<Integer, LoweredExecutionUnit> cpuDirectUnitsByStart;
private final Map<Long, RegionInputMaterializationPlan> inputMaterializationsByConsumerInput;
```

Complete publish additions inside `publishCpuRegion(...)`:

```java
if (unit.loweringFamily() == LoweringFamily.FUSED_NATIVE) {
    cpuFusedUnitsByStart.put(unit.orderedNodeIds().getFirst(), unit);
    publishInputMaterializations(plan);
    continue;
}
if (isCpuSpecializedUnit(plan)) {
    cpuSpecializedUnitsByStart.put(unit.orderedNodeIds().getFirst(), unit);
} else if (plan != null) {
    cpuDirectUnitsByStart.put(unit.orderedNodeIds().getFirst(), unit);
}
if (plan != null) {
    publishInputMaterializations(plan);
}
```

Complete new methods:

```java
private void publishInputMaterializations(RegionExecutionPlan plan) {
    if (plan == null || plan.inputMaterializationPlans().isEmpty()) {
        return;
    }
    for (RegionInputMaterializationPlan materialization : plan.inputMaterializationPlans()) {
        inputMaterializationsByConsumerInput.put(
                materializationKey(materialization.consumerNodeId(), materialization.inputIndex()),
                materialization
        );
    }
}

LoweredExecutionUnit cpuDirectUnitForStart(int nodeId) {
    return cpuDirectUnitsByStart.get(nodeId);
}

RegionInputMaterializationPlan inputMaterializationFor(int consumerNodeId, int inputIndex) {
    return inputMaterializationsByConsumerInput.get(materializationKey(consumerNodeId, inputIndex));
}

List<RegionInputMaterializationPlan> inputMaterializationsForConsumer(int consumerNodeId) {
    if (inputMaterializationsByConsumerInput.isEmpty()) {
        return List.of();
    }
    return inputMaterializationsByConsumerInput.values().stream()
            .filter(plan -> plan.consumerNodeId() == consumerNodeId)
            .sorted(java.util.Comparator.comparingInt(RegionInputMaterializationPlan::inputIndex))
            .toList();
}
```

### Task 2.5: Expose prepare-context lookup

Status: `[ ] not started`

What: add pass-through methods on `BackendPrepareContext`.

Why: cpu1 loss preparer should receive materialization plans through prepare
context/lowered plan, not discover them itself.

File:

- `src/main/java/backend/prepare/BackendPrepareContext.java`

Complete method additions:

```java
public LoweredExecutionUnit cpuDirectUnitForStart(int nodeId) {
    return loweredRegionIndex.cpuDirectUnitForStart(nodeId);
}

public RegionInputMaterializationPlan inputMaterializationFor(int consumerNodeId, int inputIndex) {
    return loweredRegionIndex.inputMaterializationFor(consumerNodeId, inputIndex);
}

public List<RegionInputMaterializationPlan> inputMaterializationsForConsumer(int consumerNodeId) {
    return loweredRegionIndex.inputMaterializationsForConsumer(consumerNodeId);
}
```

### Task 2.6: Teach `PreparedExecutionBuilder` direct CPU lowered step

Status: `[ ] not started`

What: add a branch after CPU specialized and before accelerator regions:

```java
LoweredExecutionUnit directCpuUnit = context.cpuDirectUnitForStart(node.id());
if (directCpuUnit != null) {
    addPreparedRegionStep(
            prepareCpuDirectStep(directCpuUnit, context, dispatcher),
            context,
            program.forwardBoundaryNodeId(),
            executionSteps,
            forwardSteps,
            backwardSteps,
            coveredNodeIds
    );
    continue;
}
```

Why: without this, a direct single-op loss lowered region is ignored by prepare
and the ordinary node path cannot see `RegionExecutionPlan.inputMaterializationPlans()`.

File:

- `src/main/java/backend/prepare/PreparedExecutionBuilder.java`

Complete new helper:

```java
private static PreparedExecutionStep prepareCpuDirectStep(
        LoweredExecutionUnit directUnit,
        BackendPrepareContext context,
        BackendPrepareDispatcher dispatcher
) {
    var regionPlan = requireBoundaryStepNode(
            directUnit.requireRegionPlan(),
            context,
            "CPU direct"
    );
    int outputNodeId = representativeBoundaryNodeId(regionPlan);
    if (directUnit.orderedNodeIds().isEmpty() || directUnit.orderedNodeIds().getLast() != outputNodeId) {
        throw new IllegalStateException("CPU direct prepared step output must be the last ordered node. unit="
                + directUnit.unitId() + ", outputNodeId=" + outputNodeId
                + ", orderedNodeIds=" + directUnit.orderedNodeIds());
    }
    CompiledNode outputNode = context.compiledNode(outputNodeId);
    CompiledNodeExecutionMetadata metadata = dispatcher.prepareCpuDirectStep(outputNode, directUnit, context);
    return new PreparedExecutionStep(
            outputNode,
            metadata,
            regionPlan.orderedNodeIds(),
            regionPlan.boundaryOutputNodeIds()
    );
}
```

### Task 2.7: Add dispatcher route for direct cpu1 loss

Status: `[ ] not started`

What: add `BackendPrepareDispatcher.prepareCpuDirectStep(...)`.

Why: direct lowered CPU loss with input materialization must be prepared through
`Cpu1LossPreparer`. Other direct units can remain on existing CPU prepare.

File:

- `src/main/java/backend/prepare/BackendPrepareDispatcher.java`

Complete field addition:

```java
private final backend.cpu1.prepare.Cpu1LossPreparer cpu1LossPreparer;
```

Constructor addition:

```java
this.cpu1LossPreparer = new backend.cpu1.prepare.Cpu1LossPreparer();
```

Complete method:

```java
public CompiledNodeExecutionMetadata prepareCpuDirectStep(
        CompiledNode outputNode,
        LoweredExecutionUnit loweredUnit,
        BackendPrepareContext context
) {
    Objects.requireNonNull(outputNode, "outputNode cannot be null");
    Objects.requireNonNull(loweredUnit, "loweredUnit cannot be null");
    Objects.requireNonNull(context, "context cannot be null");
    RegionExecutionPlan regionPlan = loweredUnit.requireRegionPlan();
    Operation operation = outputNode.operation();
    if (operation != null && backend.cpu1.prepare.Cpu1LossPreparer.isLossOp(operation.opType())) {
        Cpu1PreparedArtifact artifact = cpu1LossPreparer.prepare(
                outputNode,
                context.descriptorIndex(),
                Cpu1PrepareConfig.automatic(runtimeConfig, Runtime.getRuntime().availableProcessors()),
                regionPlan.inputMaterializationPlans()
        );
        return new CompiledNodeExecutionMetadata(
                ComputeBackend.CPU,
                null,
                outputNode.inputIds(),
                artifact,
                InputResidencyRequirement.none(),
                OutputResidencyEffect.cpuCurrentPreserveNative()
        );
    }
    return cpuPreparer.prepareAsCpu(outputNode, context);
}
```

Reason for `InputResidencyRequirement.none()`: materialized cpu1 inputs perform
their exact source read requirement during `Cpu1InputMaterializer`. Non-materialized
loss inputs still call `requireCpuReadable` or `requireNativeReadable` in the
loss input view helper. This avoids pre-run generic CPU-array materialization
when a `MEMORY_SEGMENT` path was selected.

## Phase 3: Runtime Temporary Storage Model For Materialized Inputs

Status: `[ ] not started`

Purpose: support run-scoped temporary materialized inputs without pretending
they are graph nodes and without adding cpu1-specific methods to
`ExecutionContext`.

### Task 3.1: Add storage kind enum for prepared input buffers

Status: `[ ] not started`

File:

- `src/main/java/graph/execution/plan/PreparedInputStorageKind.java`

Complete proposed code:

```java
package graph.execution.plan;

/**
 * Storage family for run-scoped prepared input materialization buffers.
 */
public enum PreparedInputStorageKind {
    JAVA_ARRAY,
    NATIVE_CPU
}
```

### Task 3.2: Add `PreparedInputMaterializationSpec`

Status: `[ ] not started`

What: immutable run-state allocation spec for one materialized input.

Why: prepared artifacts need to allocate materialized input buffers per run, but
`RuntimeWorkspaceStore` needs a backend-neutral spec.

File:

- `src/main/java/graph/execution/plan/PreparedInputMaterializationSpec.java`

Complete proposed code:

```java
package graph.execution.plan;

import tensor.DataType;

import java.util.Arrays;
import java.util.Objects;

/**
 * Run-state allocation spec for one prepared input materialization buffer.
 */
public record PreparedInputMaterializationSpec(
        String materializationId,
        int ownerNodeId,
        int inputIndex,
        DataType dataType,
        int[] shape,
        int elementCount,
        PreparedInputStorageKind storageKind,
        String label
) {
    public PreparedInputMaterializationSpec {
        if (materializationId == null || materializationId.isBlank()) {
            throw new IllegalArgumentException("materializationId cannot be blank");
        }
        if (ownerNodeId < 0 || inputIndex < 0) {
            throw new IllegalArgumentException("ownerNodeId and inputIndex must be non-negative");
        }
        dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        shape = shape == null ? new int[0] : shape.clone();
        if (elementCount < 0) {
            throw new IllegalArgumentException("elementCount cannot be negative");
        }
        storageKind = storageKind == null ? PreparedInputStorageKind.JAVA_ARRAY : storageKind;
        label = label == null ? materializationId : label;
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public String toString() {
        return "PreparedInputMaterializationSpec{"
                + "materializationId='" + materializationId + '\''
                + ", ownerNodeId=" + ownerNodeId
                + ", inputIndex=" + inputIndex
                + ", dataType=" + dataType
                + ", shape=" + Arrays.toString(shape)
                + ", elementCount=" + elementCount
                + ", storageKind=" + storageKind
                + '}';
    }
}
```

### Task 3.3: Add `PreparedInputMaterializationBuffer`

Status: `[ ] not started`

What: runtime buffer object that can hold either Java-array tensor storage or
native CPU storage.

Why: cpu1 loss has both `JAVA_ARRAY` and `MEMORY_SEGMENT` variants. The temp
model must support both without a cpu1-specific `ExecutionContext` method.

File:

- `src/main/java/graph/execution/plan/PreparedInputMaterializationBuffer.java`

Complete proposed code:

```java
package graph.execution.plan;

import tensor.DataType;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.Objects;

/**
 * Run-scoped buffer backing one prepared input materialization.
 */
public final class PreparedInputMaterializationBuffer {
    private final PreparedInputMaterializationSpec spec;
    private final Tensor tensor;
    private final NativeTensorStorage nativeStorage;

    public PreparedInputMaterializationBuffer(
            PreparedInputMaterializationSpec spec,
            Tensor tensor,
            NativeTensorStorage nativeStorage
    ) {
        this.spec = Objects.requireNonNull(spec, "spec cannot be null");
        this.tensor = Objects.requireNonNull(tensor, "tensor cannot be null");
        this.nativeStorage = nativeStorage;
        if (spec.storageKind() == PreparedInputStorageKind.NATIVE_CPU && nativeStorage == null) {
            throw new IllegalArgumentException("nativeStorage is required for NATIVE_CPU materialized inputs");
        }
        if (tensor.getDataType() != spec.dataType()) {
            throw new IllegalArgumentException("materialized input tensor dtype mismatch. tensor="
                    + tensor.getDataType() + ", spec=" + spec.dataType());
        }
        if (tensor.getFlatDataSize() != spec.elementCount()) {
            throw new IllegalArgumentException("materialized input tensor size mismatch. tensor="
                    + tensor.getFlatDataSize() + ", spec=" + spec.elementCount());
        }
    }

    public PreparedInputMaterializationSpec spec() {
        return spec;
    }

    public Tensor tensor() {
        return tensor;
    }

    public NativeTensorStorage nativeStorage() {
        if (nativeStorage == null) {
            throw new IllegalStateException("Materialized input " + spec.materializationId()
                    + " is not backed by native CPU storage.");
        }
        return nativeStorage;
    }

    public boolean nativeBacked() {
        return nativeStorage != null;
    }

    public DataType dataType() {
        return spec.dataType();
    }
}
```

### Task 3.4: Extend `PreparedRuntimeStateAllocator`

Status: `[ ] not started`

What: add a method for materialized input specs.

Why: specs must be registered during `PreparedExecutionArtifact.allocateRuntimeState`.

File:

- `src/main/java/graph/execution/plan/PreparedRuntimeStateAllocator.java`

Complete method addition:

```java
/**
 * Registers a run-local prepared input materialization spec.
 *
 * @param spec materialization buffer spec
 */
default void putPreparedInputMaterializationSpec(PreparedInputMaterializationSpec spec) {
}
```

### Task 3.5: Extend `RuntimeWorkspaceStore`, `ExecutionState`, and `ExecutionContext`

Status: `[ ] not started`

What:

- `RuntimeWorkspaceStore` stores materialization specs by id.
- `ExecutionState` lazily creates `PreparedInputMaterializationBuffer`.
- `ExecutionContext` exposes a backend-neutral lookup:
  `preparedInputMaterializationFor(String materializationId)`.

Why: allocation of native CPU temp storage needs access to the run-owned native
allocator, which exists after `ExecutionState` is created. Lazy allocation keeps
the model compatible with per-run native pool configuration.

Files:

- `src/main/java/graph/execution/state/RuntimeWorkspaceStore.java`
- `src/main/java/graph/execution/state/ExecutionState.java`
- `src/main/java/backend/runtime/ExecutionContext.java`

Complete `ExecutionContext` method addition:

```java
public PreparedInputMaterializationBuffer preparedInputMaterializationFor(String materializationId) {
    if (executionState == null) {
        throw new IllegalStateException("ExecutionContext does not carry per-run ExecutionState.");
    }
    return executionState.preparedInputMaterializationFor(materializationId);
}
```

Important: this method imports only `graph.execution.plan.PreparedInputMaterializationBuffer`.
It does not import cpu1 classes.

Complete `ExecutionState` method outline:

```java
private final Map<String, PreparedInputMaterializationBuffer> preparedInputBuffers = new HashMap<>();

public PreparedInputMaterializationBuffer preparedInputMaterializationFor(String materializationId) {
    PreparedInputMaterializationSpec spec = workspaceStore.preparedInputMaterializationSpec(materializationId);
    return preparedInputBuffers.computeIfAbsent(spec.materializationId(), ignored -> allocatePreparedInput(spec));
}

private PreparedInputMaterializationBuffer allocatePreparedInput(PreparedInputMaterializationSpec spec) {
    Tensor tensor = createDenseRuntimeTensor(spec);
    NativeTensorStorage nativeStorage = spec.storageKind() == PreparedInputStorageKind.NATIVE_CPU
            ? allocateNativeStorage(spec.dataType(), spec.elementCount(), spec.label())
            : null;
    return new PreparedInputMaterializationBuffer(spec, tensor, nativeStorage);
}
```

Complete dense tensor allocation helper:

```java
private static Tensor createDenseRuntimeTensor(PreparedInputMaterializationSpec spec) {
    int elements = spec.elementCount();
    int[] shape = spec.shape();
    String label = spec.label();
    return switch (spec.dataType()) {
        case FLOAT32 -> new Tensor(new float[elements], shape, null, label, DataType.FLOAT32);
        case FLOAT64 -> new Tensor(new double[elements], shape, null, label, DataType.FLOAT64);
        case BFLOAT16 -> new Tensor(new short[elements], shape, null, label, DataType.BFLOAT16);
        case BOOL -> new Tensor(new byte[elements], shape, null, label, DataType.BOOL);
        case INT32 -> new Tensor(new int[elements], shape, null, label, DataType.INT32);
        case INT64 -> new Tensor(new long[elements], shape, null, label, DataType.INT64);
    };
}
```

## Phase 4: cpu1 Prepared Input Materialization Execution

Status: `[ ] not started`

Purpose: execute lowered materialization plans immediately before cpu1 loss
kernels.

### Task 4.1: Add `Cpu1PreparedInputMaterialization`

Status: `[ ] not started`

File:

- `src/main/java/backend/cpu1/prepare/Cpu1PreparedInputMaterialization.java`

Complete proposed code:

```java
package backend.cpu1.prepare;

import backend.cpu1.storage.Cpu1StorageKind;
import backend.lowering.region.RegionInputMaterializationPlan;
import graph.execution.plan.PreparedInputMaterializationSpec;
import graph.execution.plan.PreparedInputStorageKind;
import tensor.DataType;

import java.util.Objects;

/**
 * cpu1 prepared copy-in step for one materialized loss input.
 */
public record Cpu1PreparedInputMaterialization(
        String materializationId,
        int ownerNodeId,
        int inputIndex,
        int sourceNodeId,
        DataType dataType,
        int[] shape,
        int[] sourceStrides,
        int sourceStorageOffset,
        int[] targetStrides,
        int elementCount,
        Cpu1StorageKind targetStorageKind,
        String reasonCode
) {
    public Cpu1PreparedInputMaterialization {
        if (materializationId == null || materializationId.isBlank()) {
            throw new IllegalArgumentException("materializationId cannot be blank");
        }
        if (ownerNodeId < 0 || inputIndex < 0 || sourceNodeId < 0) {
            throw new IllegalArgumentException("node ids and inputIndex must be non-negative");
        }
        dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        shape = shape == null ? new int[0] : shape.clone();
        sourceStrides = sourceStrides == null ? new int[0] : sourceStrides.clone();
        targetStrides = targetStrides == null ? new int[0] : targetStrides.clone();
        if (sourceStorageOffset < 0 || elementCount < 0) {
            throw new IllegalArgumentException("sourceStorageOffset and elementCount must be non-negative");
        }
        targetStorageKind = targetStorageKind == null ? Cpu1StorageKind.JAVA_ARRAY : targetStorageKind;
        reasonCode = reasonCode == null ? "" : reasonCode;
    }

    public static Cpu1PreparedInputMaterialization from(
            RegionInputMaterializationPlan plan,
            Cpu1StorageKind targetStorageKind
    ) {
        return new Cpu1PreparedInputMaterialization(
                plan.materializationId(),
                plan.consumerNodeId(),
                plan.inputIndex(),
                plan.sourceNodeId(),
                plan.dataType(),
                plan.shape(),
                plan.sourceStrides(),
                plan.sourceStorageOffset(),
                plan.targetStrides(),
                plan.elementCountAsInt(),
                targetStorageKind,
                plan.reasonCode()
        );
    }

    public PreparedInputMaterializationSpec toRuntimeSpec() {
        return new PreparedInputMaterializationSpec(
                materializationId,
                ownerNodeId,
                inputIndex,
                dataType,
                shape,
                elementCount,
                targetStorageKind == Cpu1StorageKind.MEMORY_SEGMENT
                        ? PreparedInputStorageKind.NATIVE_CPU
                        : PreparedInputStorageKind.JAVA_ARRAY,
                "cpu1-materialized-input-node-" + ownerNodeId + "-input-" + inputIndex
        );
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] sourceStrides() {
        return sourceStrides.clone();
    }

    @Override
    public int[] targetStrides() {
        return targetStrides.clone();
    }
}
```

### Task 4.2: Add `Cpu1LogicalTensorCopy`

Status: `[ ] not started`

What: shared logical-to-dense copy helper for cpu1 materialization.

Why: existing `Cpu1LayoutKernelSupport` has useful logic, but it is tied to
`Cpu1PreparedLayoutUnit` and currently rejects `INT32`/`INT64` in some paths.
Loss materialization needs integer target support for CE indices.

File:

- `src/main/java/backend/cpu1/kernels/layout/copy/Cpu1LogicalTensorCopy.java`

Complete proposed code:

```java
package backend.cpu1.kernels.layout.copy;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.storage.Cpu1StorageKind;
import tensor.DataType;
import tensor.dtype.TensorDTypeOps;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Small cpu1 logical tensor copy helper used by explicit input materialization.
 */
public final class Cpu1LogicalTensorCopy {
    private Cpu1LogicalTensorCopy() {
    }

    public static void copyLogicalToDense(
            Cpu1TensorView source,
            Cpu1TensorView target,
            DataType dataType
    ) {
        if (source == null || target == null) {
            throw new IllegalArgumentException("source and target cannot be null");
        }
        if (dataType == null) {
            throw new IllegalArgumentException("dataType cannot be null");
        }
        if (source.elementCount() != target.elementCount()) {
            throw new IllegalArgumentException("Materialized input copy size mismatch. source="
                    + source.elementCount() + ", target=" + target.elementCount());
        }
        int[] shape = source.shape();
        int[] denseStrides = denseStrides(shape);
        for (int linear = 0; linear < source.elementCount(); linear++) {
            int sourceOffset = source.storageOffset() + logicalOffset(linear, shape, source.strides(), denseStrides);
            int targetOffset = target.storageOffset() + linear;
            copyElement(source, sourceOffset, target, targetOffset, dataType);
        }
        target.markStorageModified();
    }

    private static void copyElement(
            Cpu1TensorView source,
            int sourceOffset,
            Cpu1TensorView target,
            int targetOffset,
            DataType dataType
    ) {
        switch (dataType) {
            case FLOAT32 -> writeF32(target, targetOffset, readF32(source, sourceOffset));
            case FLOAT64 -> writeF64(target, targetOffset, readF64(source, sourceOffset));
            case BFLOAT16 -> writeBf16(target, targetOffset, readBf16(source, sourceOffset));
            case BOOL -> writeBool(target, targetOffset, readBool(source, sourceOffset));
            case INT32 -> writeI32(target, targetOffset, readI32(source, sourceOffset));
            case INT64 -> writeI64(target, targetOffset, readI64(source, sourceOffset));
        }
    }

    private static float readF32(Cpu1TensorView view, int offset) {
        return view.storageKind() == Cpu1StorageKind.JAVA_ARRAY
                ? view.float32Array()[offset]
                : view.segment().get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    private static void writeF32(Cpu1TensorView view, int offset, float value) {
        if (view.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            view.float32Array()[offset] = value;
        } else {
            view.segment().set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
        }
    }

    private static double readF64(Cpu1TensorView view, int offset) {
        return view.storageKind() == Cpu1StorageKind.JAVA_ARRAY
                ? view.float64Array()[offset]
                : view.segment().get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    private static void writeF64(Cpu1TensorView view, int offset, double value) {
        if (view.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            view.float64Array()[offset] = value;
        } else {
            view.segment().set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
        }
    }

    private static short readBf16(Cpu1TensorView view, int offset) {
        return view.storageKind() == Cpu1StorageKind.JAVA_ARRAY
                ? view.bfloat16Array()[offset]
                : view.segment().get(JAVA_SHORT, (long) offset * Short.BYTES);
    }

    private static void writeBf16(Cpu1TensorView view, int offset, short value) {
        if (view.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            view.bfloat16Array()[offset] = value;
        } else {
            view.segment().set(JAVA_SHORT, (long) offset * Short.BYTES, value);
        }
    }

    private static byte readBool(Cpu1TensorView view, int offset) {
        return view.storageKind() == Cpu1StorageKind.JAVA_ARRAY
                ? view.boolArray()[offset]
                : view.segment().get(JAVA_BYTE, offset);
    }

    private static void writeBool(Cpu1TensorView view, int offset, byte value) {
        byte normalized = value == 0 ? (byte) 0 : (byte) 1;
        if (view.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            view.boolArray()[offset] = normalized;
        } else {
            view.segment().set(JAVA_BYTE, offset, normalized);
        }
    }

    private static int readI32(Cpu1TensorView view, int offset) {
        return view.storageKind() == Cpu1StorageKind.JAVA_ARRAY
                ? view.int32Array()[offset]
                : view.segment().get(JAVA_INT, (long) offset * Integer.BYTES);
    }

    private static void writeI32(Cpu1TensorView view, int offset, int value) {
        if (view.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            view.int32Array()[offset] = value;
        } else {
            view.segment().set(JAVA_INT, (long) offset * Integer.BYTES, value);
        }
    }

    private static long readI64(Cpu1TensorView view, int offset) {
        return view.storageKind() == Cpu1StorageKind.JAVA_ARRAY
                ? view.int64Array()[offset]
                : view.segment().get(JAVA_LONG, (long) offset * Long.BYTES);
    }

    private static void writeI64(Cpu1TensorView view, int offset, long value) {
        if (view.storageKind() == Cpu1StorageKind.JAVA_ARRAY) {
            view.int64Array()[offset] = value;
        } else {
            view.segment().set(JAVA_LONG, (long) offset * Long.BYTES, value);
        }
    }

    private static int logicalOffset(int linear, int[] shape, int[] strides, int[] denseStrides) {
        int remaining = linear;
        int offset = 0;
        for (int dim = 0; dim < shape.length; dim++) {
            int coordinate = denseStrides[dim] == 0 ? 0 : remaining / denseStrides[dim];
            remaining = denseStrides[dim] == 0 ? remaining : remaining % denseStrides[dim];
            offset += coordinate * strides[dim];
        }
        return offset;
    }

    private static int[] denseStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride = Math.multiplyExact(stride, shape[i]);
        }
        return strides;
    }
}
```

Note: the import of `TensorDTypeOps` can be removed if unused after final
implementation. The code above does not convert BF16 through float; it copies
raw bits, which is correct for materialization.

### Task 4.3: Add `Cpu1InputMaterializer`

Status: `[ ] not started`

File:

- `src/main/java/backend/cpu1/exec/Cpu1InputMaterializer.java`

Complete proposed code:

```java
package backend.cpu1.exec;

import backend.cpu1.kernels.layout.copy.Cpu1LogicalTensorCopy;
import backend.cpu1.prepare.Cpu1PreparedInputMaterialization;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import graph.execution.plan.PreparedInputMaterializationBuffer;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

import java.util.List;

/**
 * Executes explicit cpu1 prepared input materializations.
 */
public final class Cpu1InputMaterializer {
    private Cpu1InputMaterializer() {
    }

    public static void materializeAll(
            List<Cpu1PreparedInputMaterialization> materializations,
            ExecutionContext context
    ) {
        if (materializations == null || materializations.isEmpty()) {
            return;
        }
        for (Cpu1PreparedInputMaterialization materialization : materializations) {
            materialize(materialization, context);
        }
    }

    public static void materialize(
            Cpu1PreparedInputMaterialization materialization,
            ExecutionContext context
    ) {
        if (materialization == null) {
            throw new IllegalArgumentException("materialization cannot be null");
        }
        if (context == null) {
            throw new IllegalArgumentException("context cannot be null");
        }
        PreparedInputMaterializationBuffer target =
                context.preparedInputMaterializationFor(materialization.materializationId());
        Cpu1TensorView source = sourceView(materialization, context);
        Cpu1TensorView destination = destinationView(materialization, target);
        Cpu1LogicalTensorCopy.copyLogicalToDense(source, destination, materialization.dataType());
        if (target.nativeBacked()) {
            target.nativeStorage().markModified();
        }
    }

    private static Cpu1TensorView sourceView(
            Cpu1PreparedInputMaterialization materialization,
            ExecutionContext context
    ) {
        Tensor sourceTensor = context.runtimeTensorForNodeId(materialization.sourceNodeId());
        if (materialization.targetStorageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            NativeTensorStorage nativeInput = context.requireNativeReadable(
                    materialization.sourceNodeId(),
                    CpuMaterializationReason.CPU_CONSUMER
            );
            return Cpu1TensorView.fromNativeStorage(sourceTensor, nativeInput);
        }
        context.requireCpuReadable(materialization.sourceNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        return Cpu1TensorView.fromTensor(sourceTensor);
    }

    private static Cpu1TensorView destinationView(
            Cpu1PreparedInputMaterialization materialization,
            PreparedInputMaterializationBuffer target
    ) {
        if (materialization.targetStorageKind() == Cpu1StorageKind.MEMORY_SEGMENT) {
            return Cpu1TensorView.fromNativeStorage(target.tensor(), target.nativeStorage());
        }
        return Cpu1TensorView.fromTensor(target.tensor());
    }
}
```

### Task 4.4: Integrate materialization into `Cpu1LossExecutableUnit`

Status: `[ ] not started`

What: add a prepared materialization list and run it before the selected loss
kernel.

Why: this is the explicit execution of the lowered decision.

File:

- `src/main/java/backend/cpu1/exec/Cpu1LossExecutableUnit.java`

Complete field addition:

```java
private final List<Cpu1PreparedInputMaterialization> inputMaterializations;
```

Complete constructor pattern:

```java
public Cpu1LossExecutableUnit(
        Cpu1PreparedCrossEntropyLossUnit preparedCrossEntropyLossUnit,
        List<Cpu1PreparedInputMaterialization> inputMaterializations
) {
    if (preparedCrossEntropyLossUnit == null) {
        throw new IllegalArgumentException("preparedCrossEntropyLossUnit cannot be null");
    }
    this.preparedCrossEntropyLossUnit = preparedCrossEntropyLossUnit;
    this.preparedDenseCrossEntropyLossUnit = null;
    this.preparedNllLossUnit = null;
    this.crossEntropyKernel = preparedCrossEntropyLossUnit.kernel();
    this.denseCrossEntropyKernel = null;
    this.nllKernel = null;
    this.inputMaterializations = List.copyOf(inputMaterializations == null ? List.of() : inputMaterializations);
}
```

Complete `run(...)` replacement:

```java
@Override
public void run(ExecutionContext context) {
    Cpu1InputMaterializer.materializeAll(inputMaterializations, context);
    if (preparedCrossEntropyLossUnit != null) {
        crossEntropyKernel.run(preparedCrossEntropyLossUnit, context);
        return;
    }
    if (preparedDenseCrossEntropyLossUnit != null) {
        denseCrossEntropyKernel.run(preparedDenseCrossEntropyLossUnit, context);
        return;
    }
    nllKernel.run(preparedNllLossUnit, context);
}
```

## Phase 5: Enable cpu1 Loss Prepare For Planned Materialized Inputs

Status: `[ ] not started`

Purpose: bind effective loss inputs to direct source node ids or materialized
temp ids.

### Task 5.1: Add `Cpu1PreparedLossInput`

Status: `[ ] not started`

File:

- `src/main/java/backend/cpu1/prepare/Cpu1PreparedLossInput.java`

Complete proposed code:

```java
package backend.cpu1.prepare;

import tensor.DataType;

import java.util.Objects;

/**
 * Prepared cpu1 loss input. It may refer to a real source node or a materialized temp.
 */
public record Cpu1PreparedLossInput(
        int sourceNodeId,
        int inputIndex,
        String materializationId,
        boolean materialized,
        DataType dataType,
        int[] shape,
        int[] strides,
        int storageOffset
) {
    public Cpu1PreparedLossInput {
        if (sourceNodeId < 0 || inputIndex < 0) {
            throw new IllegalArgumentException("sourceNodeId and inputIndex must be non-negative");
        }
        if (materialized && (materializationId == null || materializationId.isBlank())) {
            throw new IllegalArgumentException("materialized inputs require materializationId");
        }
        materializationId = materializationId == null ? "" : materializationId;
        dataType = Objects.requireNonNull(dataType, "dataType cannot be null");
        shape = shape == null ? new int[0] : shape.clone();
        strides = strides == null ? new int[0] : strides.clone();
        if (storageOffset < 0) {
            throw new IllegalArgumentException("storageOffset cannot be negative");
        }
    }

    public static Cpu1PreparedLossInput direct(
            int sourceNodeId,
            int inputIndex,
            DataType dataType,
            int[] shape,
            int[] strides,
            int storageOffset
    ) {
        return new Cpu1PreparedLossInput(
                sourceNodeId,
                inputIndex,
                "",
                false,
                dataType,
                shape,
                strides,
                storageOffset
        );
    }

    public static Cpu1PreparedLossInput materialized(
            int sourceNodeId,
            int inputIndex,
            String materializationId,
            DataType dataType,
            int[] shape,
            int[] denseStrides
    ) {
        return new Cpu1PreparedLossInput(
                sourceNodeId,
                inputIndex,
                materializationId,
                true,
                dataType,
                shape,
                denseStrides,
                0
        );
    }

    @Override
    public int[] shape() {
        return shape.clone();
    }

    @Override
    public int[] strides() {
        return strides.clone();
    }
}
```

### Task 5.2: Add `Cpu1LossInputResolver`

Status: `[ ] not started`

What: map descriptor inputs plus lowered materialization plans to
`Cpu1PreparedLossInput`.

Why: this keeps effective input resolution centralized and keeps
`Cpu1LossPreparer` contract checks readable.

File:

- `src/main/java/backend/cpu1/prepare/Cpu1LossInputResolver.java`

Complete proposed code:

```java
package backend.cpu1.prepare;

import backend.cpu1.storage.Cpu1StorageKind;
import backend.lowering.region.RegionInputMaterializationPlan;
import graph.compile.descriptor.CompiledTensorDescriptor;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import tensor.TensorMetadata;

import java.util.List;

/**
 * Resolves cpu1 loss inputs to direct or explicitly materialized effective inputs.
 */
final class Cpu1LossInputResolver {
    private Cpu1LossInputResolver() {
    }

    static Cpu1PreparedLossInput resolve(
            int consumerNodeId,
            int inputIndex,
            int sourceNodeId,
            CompiledTensorDescriptorIndex descriptorIndex,
            List<RegionInputMaterializationPlan> plans
    ) {
        CompiledTensorDescriptor descriptor = descriptorIndex.byNodeId(sourceNodeId);
        RegionInputMaterializationPlan materialization = findPlan(consumerNodeId, inputIndex, sourceNodeId, plans);
        if (materialization == null) {
            return Cpu1PreparedLossInput.direct(
                    sourceNodeId,
                    inputIndex,
                    descriptor.dataType(),
                    descriptor.shape(),
                    descriptor.strides(),
                    descriptor.storageOffset()
            );
        }
        return Cpu1PreparedLossInput.materialized(
                sourceNodeId,
                inputIndex,
                materialization.materializationId(),
                materialization.dataType(),
                materialization.shape(),
                TensorMetadata.computeStrides(materialization.shape())
        );
    }

    static List<Cpu1PreparedInputMaterialization> preparedMaterializations(
            List<RegionInputMaterializationPlan> plans,
            Cpu1StorageKind storageKind
    ) {
        if (plans == null || plans.isEmpty()) {
            return List.of();
        }
        return plans.stream()
                .map(plan -> Cpu1PreparedInputMaterialization.from(plan, storageKind))
                .toList();
    }

    private static RegionInputMaterializationPlan findPlan(
            int consumerNodeId,
            int inputIndex,
            int sourceNodeId,
            List<RegionInputMaterializationPlan> plans
    ) {
        if (plans == null || plans.isEmpty()) {
            return null;
        }
        for (RegionInputMaterializationPlan plan : plans) {
            if (plan.consumerNodeId() == consumerNodeId
                    && plan.inputIndex() == inputIndex
                    && plan.sourceNodeId() == sourceNodeId) {
                return plan;
            }
        }
        return null;
    }
}
```

### Task 5.3: Add loss input view helper

Status: `[ ] not started`

File:

- `src/main/java/backend/cpu1/kernels/loss/Cpu1LossInputViews.java`

Complete proposed code:

```java
package backend.cpu1.kernels.loss;

import backend.cpu1.exec.Cpu1TensorView;
import backend.cpu1.prepare.Cpu1PreparedLossInput;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionContext;
import graph.execution.plan.PreparedInputMaterializationBuffer;
import tensor.Tensor;
import tensor.storage.NativeTensorStorage;

/**
 * Resolves direct or materialized cpu1 loss inputs at runtime.
 */
public final class Cpu1LossInputViews {
    private Cpu1LossInputViews() {
    }

    public static Cpu1TensorView arrayView(Cpu1PreparedLossInput input, ExecutionContext context) {
        if (input.materialized()) {
            PreparedInputMaterializationBuffer buffer =
                    context.preparedInputMaterializationFor(input.materializationId());
            return Cpu1TensorView.fromTensor(buffer.tensor());
        }
        context.requireCpuReadable(input.sourceNodeId(), CpuMaterializationReason.CPU_CONSUMER);
        Tensor tensor = context.runtimeTensorForNodeId(input.sourceNodeId());
        return Cpu1TensorView.fromTensor(tensor);
    }

    public static Cpu1TensorView segmentView(Cpu1PreparedLossInput input, ExecutionContext context) {
        if (input.materialized()) {
            PreparedInputMaterializationBuffer buffer =
                    context.preparedInputMaterializationFor(input.materializationId());
            return Cpu1TensorView.fromNativeStorage(buffer.tensor(), buffer.nativeStorage());
        }
        NativeTensorStorage nativeInput = context.requireNativeReadable(
                input.sourceNodeId(),
                CpuMaterializationReason.CPU_CONSUMER
        );
        Tensor tensor = context.runtimeTensorForNodeId(input.sourceNodeId());
        return Cpu1TensorView.fromNativeStorage(tensor, nativeInput);
    }
}
```

### Task 5.4: Extend `Cpu1LossPreparer`

Status: `[ ] not started`

What: add overload accepting lowered materialization plans. Existing direct
prepare method delegates with an empty list.

Why: direct tests and dense existing callers continue to work, while graph
lowering can pass explicit plans.

File:

- `src/main/java/backend/cpu1/prepare/Cpu1LossPreparer.java`

Complete public overloads:

```java
public Cpu1PreparedArtifact prepare(
        CompiledNode node,
        CompiledTensorDescriptorIndex descriptorIndex,
        Cpu1PrepareConfig config
) {
    return prepare(node, descriptorIndex, config, List.of());
}

public Cpu1PreparedArtifact prepare(
        CompiledNode node,
        CompiledTensorDescriptorIndex descriptorIndex,
        Cpu1PrepareConfig config,
        List<RegionInputMaterializationPlan> inputMaterializationPlans
) {
    Objects.requireNonNull(node, "node cannot be null");
    Objects.requireNonNull(config, "config cannot be null");
    inputMaterializationPlans = List.copyOf(inputMaterializationPlans == null ? List.of() : inputMaterializationPlans);
    Operation operation = Objects.requireNonNull(node.operation(), "node operation cannot be null");
    return switch (operation.opType()) {
        case CROSS_ENTROPY_LOSS -> prepareDenseCrossEntropyLoss(
                node,
                descriptorIndex,
                config,
                operation,
                inputMaterializationPlans
        );
        case CROSS_ENTROPY_LOSS_INDICES -> prepareCrossEntropyLossIndices(
                node,
                descriptorIndex,
                config,
                operation,
                inputMaterializationPlans
        );
        case NLL_LOSS -> prepareNllLoss(
                node,
                descriptorIndex,
                config,
                operation,
                inputMaterializationPlans
        );
        default -> throw new UnsupportedOperationException("cpu1 loss preparer does not support "
                + operation.opType());
    };
}
```

Complete effective descriptor helper:

```java
private static EffectiveLossInput effectiveInput(Cpu1PreparedLossInput input) {
    return new EffectiveLossInput(
            input.dataType(),
            input.shape(),
            input.strides(),
            input.storageOffset(),
            input.materialized()
    );
}

private record EffectiveLossInput(
        DataType dataType,
        int[] shape,
        int[] strides,
        int storageOffset,
        boolean materialized
) {
    private EffectiveLossInput {
        shape = shape == null ? new int[0] : shape.clone();
        strides = strides == null ? new int[0] : strides.clone();
    }

    int rank() {
        return shape.length;
    }

    long logicalElementCount() {
        long count = 1L;
        for (int dimension : shape) {
            count = Math.multiplyExact(count, dimension);
        }
        return count;
    }

    boolean denseContiguousWithoutOffset() {
        return storageOffset == 0 && java.util.Arrays.equals(strides, tensor.TensorMetadata.computeStrides(shape));
    }
}
```

Contract change: existing `requireContract`, `requireNllContract`, and
`requireDenseCrossEntropyContract` should accept `EffectiveLossInput` for inputs
instead of raw `CompiledTensorDescriptor`. They must still require
`denseContiguousWithoutOffset()` on the effective input. That means:

- direct strided input without plan is rejected
- planned materialized input passes because the effective input is dense
- output still must be `node.storageOffset() == 0 && node.contiguous()`

### Task 5.5: Update prepared loss unit fields

Status: `[ ] not started`

What: replace raw input node id fields with `Cpu1PreparedLossInput` fields.

Why: materialized temp inputs do not have compiled node ids.

Files:

- `Cpu1PreparedCrossEntropyLossUnit.java`
- `Cpu1PreparedDenseCrossEntropyLossUnit.java`
- `Cpu1PreparedNllLossUnit.java`

Example complete field and accessor change for CE indices:

```java
private final Cpu1PreparedLossInput logitsInput;
private final Cpu1PreparedLossInput targetsInput;

public Cpu1PreparedCrossEntropyLossUnit(
        int nodeId,
        Cpu1PreparedLossInput logitsInput,
        Cpu1PreparedLossInput targetsInput,
        Operation.OpType opType,
        DataType logitsDataType,
        DataType targetDataType,
        Cpu1StorageKind storageKind,
        Cpu1CrossEntropyKernelId kernelId,
        int classAxis,
        int axisSize,
        int axisStride,
        int groupCount,
        int[] logitsShape,
        int[] targetShape,
        LossReduction reduction,
        Integer ignoreIndex,
        Cpu1LaunchConfig launchConfig,
        Cpu1LaunchPolicy launchPolicy,
        Cpu1ScratchBufferSpec scratchBufferSpec
) {
    if (nodeId < 0) {
        throw new IllegalArgumentException("nodeId cannot be negative");
    }
    requirePositive(axisSize, "axisSize");
    requirePositive(axisStride, "axisStride");
    requirePositive(groupCount, "groupCount");
    this.nodeId = nodeId;
    this.logitsInput = Objects.requireNonNull(logitsInput, "logitsInput cannot be null");
    this.targetsInput = Objects.requireNonNull(targetsInput, "targetsInput cannot be null");
    this.opType = Objects.requireNonNull(opType, "opType cannot be null");
    this.logitsDataType = Objects.requireNonNull(logitsDataType, "logitsDataType cannot be null");
    this.targetDataType = Objects.requireNonNull(targetDataType, "targetDataType cannot be null");
    this.storageKind = Objects.requireNonNull(storageKind, "storageKind cannot be null");
    this.kernelId = Objects.requireNonNull(kernelId, "kernelId cannot be null");
    this.kernel = Cpu1CrossEntropyKernelDispatch.kernelFor(kernelId);
    this.classAxis = classAxis;
    this.axisSize = axisSize;
    this.axisStride = axisStride;
    this.groupCount = groupCount;
    this.logitsShape = Objects.requireNonNull(logitsShape, "logitsShape cannot be null").clone();
    this.targetShape = Objects.requireNonNull(targetShape, "targetShape cannot be null").clone();
    this.reduction = Objects.requireNonNull(reduction, "reduction cannot be null");
    this.ignoreIndex = ignoreIndex;
    this.launchConfig = Objects.requireNonNull(launchConfig, "launchConfig cannot be null");
    this.launchPolicy = Objects.requireNonNull(launchPolicy, "launchPolicy cannot be null");
    this.scratchBufferSpec = Objects.requireNonNull(scratchBufferSpec, "scratchBufferSpec cannot be null");
    if (classAxis < 0 || classAxis >= this.logitsShape.length) {
        throw new IllegalArgumentException("classAxis out of bounds: " + classAxis
                + " for logits shape " + Arrays.toString(this.logitsShape));
    }
}

public Cpu1PreparedLossInput logitsInput() {
    return logitsInput;
}

public Cpu1PreparedLossInput targetsInput() {
    return targetsInput;
}

public int logitsNodeId() {
    return logitsInput.sourceNodeId();
}

public int targetsNodeId() {
    return targetsInput.sourceNodeId();
}
```

The `logitsNodeId()`/`targetsNodeId()` accessors can stay only as diagnostic
source-node accessors. Runtime input reads should use `logitsInput()` and
`targetsInput()`.

### Task 5.6: Update loss loops to resolve direct/materialized inputs

Status: `[ ] not started`

What: replace local helper reads in loss loops:

```java
Cpu1TensorView logits = inputArrayView(unit.logitsNodeId(), context);
```

with:

```java
Cpu1TensorView logits = Cpu1LossInputViews.arrayView(unit.logitsInput(), context);
```

Why: kernels must read from the prepared effective input, not blindly from the
compiled source node id.

Files:

- `Cpu1CrossEntropyLossIndicesLoops.java`
- `Cpu1DenseCrossEntropyLossLoops.java`
- `Cpu1NllLossLoops.java`

Complete method example:

```java
public static void runF32I32DenseArray(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
    Cpu1TensorView logits = Cpu1LossInputViews.arrayView(unit.logitsInput(), context);
    Cpu1TensorView targets = Cpu1LossInputViews.arrayView(unit.targetsInput(), context);
    Cpu1TensorView output = outputArrayView(unit, context);
    runF32(unit, context, logits.float32Array(), targets.int32Array(), null, logits.storageOffset(),
            targets.storageOffset(), output);
}
```

Complete segment example:

```java
public static void runF32I32DenseSegment(Cpu1PreparedCrossEntropyLossUnit unit, ExecutionContext context) {
    Cpu1TensorView logits = Cpu1LossInputViews.segmentView(unit.logitsInput(), context);
    Cpu1TensorView targets = Cpu1LossInputViews.segmentView(unit.targetsInput(), context);
    NativeTensorStorage nativeOutput = outputSegmentStorage(unit, context);
    Cpu1TensorView output = Cpu1TensorView.fromNativeStorage(context.runtimeTensorForNodeId(unit.nodeId()), nativeOutput);
    runF32Segment(unit, context, logits.segment(), targets.segment(), false, logits.storageOffset(),
            targets.storageOffset(), output, nativeOutput);
}
```

Apply the same complete substitution to all dense-array and dense-segment
entrypoints in the three loss loop classes.

### Task 5.7: Allocate materialized input runtime specs from `Cpu1PreparedArtifact`

Status: `[ ] not started`

What: `Cpu1PreparedArtifact.allocateRuntimeState(...)` must register materialized
input specs in addition to scratch buffers.

Why: runtime buffers must exist before execution.

File:

- `src/main/java/backend/cpu1/prepare/Cpu1PreparedArtifact.java`

Complete addition:

```java
private final List<Cpu1PreparedInputMaterialization> inputMaterializations;
```

Complete helper:

```java
private void allocateInputMaterializations(PreparedRuntimeStateAllocator allocator) {
    if (inputMaterializations == null || inputMaterializations.isEmpty()) {
        return;
    }
    for (Cpu1PreparedInputMaterialization materialization : inputMaterializations) {
        allocator.putPreparedInputMaterializationSpec(materialization.toRuntimeSpec());
    }
}
```

Complete `allocateRuntimeState(...)` replacement:

```java
@Override
public void allocateRuntimeState(int nodeId, PreparedRuntimeStateAllocator allocator) {
    if (allocator == null) {
        return;
    }
    allocateInputMaterializations(allocator);
    Cpu1ScratchBufferSpec spec = scratchBufferSpec();
    if (spec.isEmpty()) {
        return;
    }
    allocator.putWorkspace(nodeId, Cpu1ScratchBuffer.allocate(spec));
}
```

### Task 5.8: Trace materialized inputs

Status: `[ ] not started`

What: include planned materialization metadata in cpu1 loss traces.

Why: fallback/materialization must be visible in traces and benchmark reports.

File:

- `src/main/java/backend/cpu1/trace/Cpu1TraceContributor.java`

Trace attributes:

```text
cpu1LossMaterializedInputCount
cpu1LossMaterializedInputIds
cpu1LossMaterializedInputSourceNodeIds
cpu1LossMaterializedInputIndexes
cpu1LossMaterializationReasons
cpu1LossMaterializationStorageKind
```

Complete helper:

```java
private static void addLossMaterializationAttrs(
        LinkedHashMap<String, Object> attrs,
        List<Cpu1PreparedInputMaterialization> materializations
) {
    List<Cpu1PreparedInputMaterialization> planned =
            materializations == null ? List.of() : materializations;
    attrs.put("cpu1LossMaterializedInputCount", planned.size());
    attrs.put("cpu1LossMaterializedInputIds", planned.stream()
            .map(Cpu1PreparedInputMaterialization::materializationId)
            .toList());
    attrs.put("cpu1LossMaterializedInputSourceNodeIds", planned.stream()
            .map(Cpu1PreparedInputMaterialization::sourceNodeId)
            .toList());
    attrs.put("cpu1LossMaterializedInputIndexes", planned.stream()
            .map(Cpu1PreparedInputMaterialization::inputIndex)
            .toList());
    attrs.put("cpu1LossMaterializationReasons", planned.stream()
            .map(Cpu1PreparedInputMaterialization::reasonCode)
            .toList());
    attrs.put("cpu1LossMaterializationStorageKind", planned.stream()
            .map(materialization -> materialization.targetStorageKind().name())
            .distinct()
            .toList());
}
```

This helper requires prepared units or artifact trace contribution to pass the
materialization list. If keeping trace contribution on `Cpu1PreparedArtifact`,
pass `inputMaterializations` from the artifact into `Cpu1TraceContributor`.

## Phase 6: Memory Planning Integration Hardening

Status: `[ ] not started`

Purpose: after correctness is proven, remove unnecessary per-execute allocation
pressure and prepare integration with memory planner slots.

### Task 6.1: Avoid repeated Java temp allocations inside a run

Status: `[ ] not started`

What: ensure `ExecutionState.preparedInputMaterializationFor(...)` caches buffers
per materialization id for the whole run.

Why: a loss step must not allocate a new temp on every call to
`Cpu1LossInputViews`.

### Task 6.2: Reuse native temp storage through existing native CPU pool

Status: `[ ] not started`

What: allocate native temp buffers through `ExecutionContext.allocateNativeStorage`
or the underlying `ExecutionState` native allocator.

Why: this lets current native pool policies apply without adding cpu1-specific
allocator logic.

### Task 6.3: Future memory planner slot integration

Status: `[deferred]`

What: integrate input materialization temps with `MemoryPlan`/region slot
planning.

Why deferred: first implementation should prove correctness and trace
visibility. Slot reuse is a performance hardening step.

Planned future model:

```text
InputMaterializationRequirement
  -> RegionInputMaterializationPlan
  -> MemoryPlanner assigns temp slot when lifetime is known
  -> ExecutionState binds temp materialization buffer to that slot
```

Future files:

- `src/main/java/graph/compile/planning/memory/RegionValueFlowPlanner.java`
- `src/main/java/graph/compile/planning/memory/RegionBindingAllocator.java`
- `src/main/java/graph/execution/residency/RuntimeMemoryBinder.java`

### Task 6.4: Trace and benchmark copy cost

Status: `[ ] not started`

What: add copy byte count and input count to trace.

Why: benchmark reports need to show when loss execution includes materialization
copy-in.

Suggested attributes:

```text
cpu1LossMaterializationBytes
cpu1LossMaterializationElementCount
cpu1LossMaterializationInputCount
```

## Phase 7: Verification, Parity, Benchmarks, Docs

Status: `[ ] not started`

Purpose: prove correctness across dense unchanged paths, strided materialized
paths, storage kinds, dtype variants, and traces.

### Task 7.1: Graph/region planning tests

Status: `[ ] not started`

File:

- `src/test/java/graph/compile/planning/region/Cpu1LossInputMaterializationPlanningTest.java`

Test cases:

- strided logits for `CROSS_ENTROPY_LOSS_INDICES` creates one input
  materialization requirement
- strided targets for `CROSS_ENTROPY_LOSS_INDICES` creates one input
  materialization requirement
- strided logits and targets creates two deterministic requirements
- dense contiguous inputs create zero requirements
- `NLL_LOSS` dense target/logProbs views create requirements
- dense `CROSS_ENTROPY_LOSS` view inputs create requirements
- requirement trace includes `input-materialization-required`

### Task 7.2: Lowering tests

Status: `[ ] not started`

Modify:

- `src/test/java/backend/cpu/lowering/CpuRegionLowererTest.java`
- `src/test/java/backend/lowering/region/RegionExecutionPlanTest.java`

Test cases:

- `RegionExecutionPlan` accepts and exposes `inputMaterializationPlans`
- duplicate materialization ids are rejected
- consumer outside `orderedNodeIds` is rejected
- `CpuRegionLowerer` maps `InputMaterializationRequirement` to
  `RegionInputMaterializationPlan`
- execution group `tempValueIds` includes materialization ids

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
- trace includes `cpu1LossMaterializedInputCount > 0`
- direct unplanned strided prepare remains rejected

### Task 7.4: Existing dense parity tests

Status: `[ ] not started`

Run and update only trace assertions if needed:

```bash
./gradlew test --tests backend.cpu1.Cpu1CrossEntropyLossExecutionContractTest --tests backend.cpu1.Cpu1NllLossExecutionContractTest --tests backend.cpu1.Cpu1DenseCrossEntropyLossExecutionContractTest
```

### Task 7.5: Benchmark smoke

Status: `[ ] not started`

What: add benchmark/report coverage only after correctness passes.

Suggested benchmark scenarios:

- dense CE indices baseline
- strided logits with materialization
- strided targets with materialization
- memory segment materialization

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

- graph/lowering decides materialization
- cpu1 executes prepared materialization
- trace fields that expose materialization
- why this is not a synthetic graph node

## Phase 8: Extend Planned Input Materialization To cpu1 Index/Scatter Units

Status: `[ ] not started`

Purpose: reuse the same graph/lowering -> prepare -> runtime temporary
materialization mechanism for cpu1 index/scatter input operands that do not
satisfy the dense contiguous no-offset cpu1 direct-kernel contract.

This belongs in plan 118 because index/scatter materialization is the same
architectural feature as loss input materialization: graph/region/lowering
decides that a consumer input must be dense contiguous no-offset, prepare binds
that explicit plan, and execute materializes a run-local temp before the kernel
runs. The follow-up should not create a second index/scatter-specific policy
path.

This phase is input materialization only. It does not implement output
materialization or copy-back for strided/offset scatter outputs.

### Phase 8 invariant

```text
cpu1 index/scatter kernels must not silently materialize.
cpu1 index/scatter preparer must not decide materialization by itself.
graph/region/lowering decides materialization.
prepare validates and binds explicit materialization plans.
execute materializes temps before kernel run.
```

Dense contiguous/no-offset cpu1 kernels remain unchanged. Planned
materialization supplies effective dense inputs to those kernels. The current
dense direct rejection remains the guard when no explicit materialization plan
exists.

### Scope

Apply planned input materialization to these cpu1 index/scatter operations:

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

Scatter output materialization is a different problem than input
materialization:

- output is a write target, not just a read input
- strided/offset output would require copy-back or scatter-back semantics after
  the kernel
- aliasing and memory planner ownership become harder because output writes may
  overlap with source/base storage
- correctness and traceability deserve a separate plan or future phase

This phase should therefore keep rejecting strided/offset outputs unless a later
plan introduces explicit output materialization and copy-back semantics.

### Data flow examples

Example 1: materialize gathered data input.

```text
user graph:
  permuted = permute(data)
  out = gather(permuted, indices)

planning:
  consumer=GATHER input=0 source=permuted requires DENSE_CONTIGUOUS_NO_OFFSET
  graph/region creates materialization plan for (outNodeId, inputIndex=0)

execute:
  Cpu1InputMaterializer copies permuted logical view into a dense temp
  GATHER reads data from the materialized dense temp
  indices remain direct if already dense
```

Example 2: materialize gathered indices.

```text
user graph:
  stridedIndices = slice(indicesBase)
  out = gather(data, stridedIndices)

planning:
  consumer=GATHER input=1 source=stridedIndices requires DENSE_CONTIGUOUS_NO_OFFSET

execute:
  data remains direct if dense
  indices are materialized before GATHER binds views
```

Example 3: materialize scatter base and updates only.

```text
user graph:
  out = scatterElements(stridedBase, indices, stridedUpdates, axis)

planning:
  materialization can be planned for base/data input and updates input
  output materialization is not planned in Phase 8

execute:
  base/data and updates may be effective dense inputs
  output must still be dense contiguous no-offset
  strided/offset output is rejected by the existing direct guard
```

Example 4: materialize `SCATTER_ND` indices.

```text
user graph:
  stridedIndices = permute(indices)
  out = scatterNd(data, stridedIndices, updates)

planning:
  consumer=SCATTER_ND input=1 source=stridedIndices requires
  DENSE_CONTIGUOUS_NO_OFFSET

execute:
  Cpu1InputMaterializer materializes indices
  SCATTER_ND reads dense effective indices
```

### Expected data model extensions

Generalize the planned materialization lookup by `(consumerNodeId, inputIndex)`
so it is not loss-specific. Loss and index/scatter prepare code should use the
same prepare-context lookup shape.

Proposed planning shape:

```java
record InputMaterializationKey(int consumerNodeId, int inputIndex) {
}

Optional<RegionInputMaterializationPlan> plannedInputMaterialization(
        int consumerNodeId,
        int inputIndex
) {
    return plannedInputMaterialization(new InputMaterializationKey(
            consumerNodeId,
            inputIndex
    ));
}
```

`Cpu1PreparedIndexUnit` should receive effective input descriptors/views or
prepared input handles analogous to loss. If Phase 5 creates a generalized
prepared input handle, reuse it rather than creating an index-only duplicate.

Proposed prepared-unit shape:

```java
record Cpu1PreparedIndexInput(
        int inputIndex,
        int sourceNodeId,
        CompiledTensorDescriptor directDescriptor,
        Optional<Cpu1PreparedInputMaterialization> materialization
) {
    boolean materialized() {
        return materialization.isPresent();
    }
}
```

`Cpu1IndexExecutableUnit` should run materialization before binding kernel
views, then resolve direct/materialized views per input.

Proposed execution shape:

```java
void execute(ExecutionContext context) {
    Cpu1MaterializedInputs materializedInputs =
            Cpu1InputMaterializer.materializeAll(context, preparedInputMaterializations);

    Cpu1IndexInputViews views = Cpu1IndexInputResolver.resolve(
            context,
            preparedInputs,
            materializedInputs
    );

    dispatch.run(context, views);
}
```

Trace metadata should identify index/scatter materialization separately from
loss while sharing the same underlying input materialization trace model.

Proposed trace fields:

```text
cpu1IndexMaterializedInputCount=2
cpu1IndexMaterializedInputs=[
  input=0,source=7,op=SCATTER_ELEMENTS,target=DENSE_CONTIGUOUS_NO_OFFSET,
  input=2,source=9,op=SCATTER_ELEMENTS,target=DENSE_CONTIGUOUS_NO_OFFSET
]
cpu1IndexMaterializationReason=cpu1-index-dense-contiguous-input-contract
```

### Task 8.1: Inventory current cpu1 index/scatter dense guards

Status: `[ ] not started`

What: inspect `Cpu1IndexPreparer`, `Cpu1PreparedIndexUnit`,
`Cpu1IndexExecutableUnit`, index/scatter dispatch, and dense guard helpers for
all scoped operations.

Why: identify exactly which input roles are currently required to be dense
contiguous no-offset and where unplanned strided inputs are rejected.

Expected output:

- list each scoped op
- map input index to role name
- record current guard location
- record whether output/write target is already guarded separately

### Task 8.2: Extend materialization requirement selector to index/scatter input roles

Status: `[ ] not started`

What: extend the graph/region input contract from loss-only to include
index/scatter input roles.

Why: materialization must still be selected before prepare. `Cpu1IndexPreparer`
must not infer materialization from descriptors on its own.

Proposed selector shape:

```java
return switch (opType) {
    case GATHER, GATHER_AXIS, GATHER_ND, TAKE_ALONG_AXIS ->
            indexReadRequirements(consumer.inputIds().size());
    case SCATTER_ADD, SCATTER_AXIS_ADD, SCATTER_ELEMENTS, SCATTER_ND ->
            scatterInputRequirements(opType, consumer.inputIds().size());
    default ->
            repeated(consumer.inputIds().size(), InputLayoutRequirement.asIs());
};
```

The selector should mark only read inputs. It must not mark the output/write
target for materialization in Phase 8.

### Task 8.3: Extend prepare context lookup usage from loss to index units

Status: `[ ] not started`

What: make cpu1 index prepare code ask the generalized prepare context for
planned input materializations by `(consumerNodeId, inputIndex)`.

Why: the prepare layer should validate and bind explicit lowering decisions
without owning the materialization policy.

Expected behavior:

- direct dense input without a plan remains accepted
- non-dense input with a matching plan is accepted as an effective dense input
- non-dense input without a matching plan remains rejected
- plan for the wrong consumer/input is ignored or rejected

### Task 8.4: Introduce or reuse prepared index input handles

Status: `[ ] not started`

What: introduce a prepared index input handle/resolver, or reuse the generalized
prepared input handle if Phase 5 created one.

Why: the executable unit needs a single explicit way to resolve each input to
either the direct runtime tensor view or the materialized temp view.

Do not introduce an index-specific wrapper if a generalized prepared input
handle already exists and fits the exact need.

### Task 8.5: Execute materialization before `Cpu1IndexExecutableUnit` binds views

Status: `[ ] not started`

What: call `Cpu1InputMaterializer.materializeAll(...)` before index/scatter view
binding and dispatch.

Why: dense kernels should see effective dense input views and should not know
whether those views came from original storage or a materialized temp.

Required ordering:

```text
Cpu1IndexExecutableUnit.execute(context)
  1. materialize all planned input temps for this unit
  2. resolve direct/materialized views per input
  3. validate output/write target remains direct dense contiguous no-offset
  4. dispatch existing dense cpu1 index/scatter kernel
```

### Task 8.6: Add trace metadata for materialized index inputs

Status: `[ ] not started`

What: add trace attributes that count and describe index/scatter materialized
inputs.

Why: fallback and copy-in cost must be visible in traces and benchmark reports,
matching the plan 118 rule that materialization is explicit.

Suggested attributes:

- `cpu1IndexMaterializedInputCount`
- `cpu1IndexMaterializedInputs`
- `cpu1IndexMaterializationReason`

### Task 8.7: Tests for read-only index materialization

Status: `[ ] not started`

What: add focused tests proving read-only index operations accept planned
materialized inputs and still reject unplanned non-dense inputs.

Coverage:

- `permute(data).gather(indices)` materializes data before `GATHER`
- `data.gather(stridedIndices)` materializes indices before `GATHER`
- `GATHER_AXIS`, `GATHER_ND`, and `TAKE_ALONG_AXIS` coverage for at least one
  strided data/input case and one strided indices case where applicable
- dense direct paths still use zero materialized inputs
- unplanned strided inputs still fail the dense direct guard

### Task 8.8: Tests for scatter input/update/index materialization

Status: `[ ] not started`

What: add focused scatter tests for planned materialization of base/data,
indices, and updates, while explicitly still rejecting strided outputs.

Coverage:

- `stridedBase.scatterElements(indices, stridedUpdates, axis)` materializes
  base/data and updates as inputs
- `scatterNd(data, stridedIndices, updates)` materializes indices
- `SCATTER_ADD`, `SCATTER_AXIS_ADD`, `SCATTER_ELEMENTS`, and `SCATTER_ND`
  include representative input/update/index materialization coverage
- strided/offset output remains rejected in this phase
- duplicate-index behavior remains unchanged from existing dense direct kernels

### Task 8.9: Benchmarks and regression checks

Status: `[ ] not started`

What: run focused correctness and smoke benchmark checks for dense direct and
planned materialized index/scatter paths.

Why: materialization should be visible as copy-in overhead without regressing
the dense direct cpu1 hot path.

Do not commit local benchmark/calibration artifacts unless intentionally
promoting canonical fixtures.

## Phase 9: Extend Planned Input Materialization To cpu1 Attention Units

Status: `[ ] not started`

Purpose: reuse the same graph/lowering -> prepare -> runtime temporary
materialization mechanism for cpu1 attention input operands that do not satisfy
the dense contiguous no-offset direct attention contract.

This belongs in plan 118 because attention materialization is a policy decision:
graph/region/lowering decides whether the copy is worth it, prepare binds the
explicit plan, and execute materializes run-local temps before the dense
attention kernel runs. `Cpu1AttentionPreparer` and
`Cpu1AttentionLoops` must not silently decide to copy unsupported views.

This phase is input materialization only. It does not implement output
materialization/copy-back for strided attention outputs, arbitrary strided inner
dimension attention kernels, or vectorized attention kernels.

### Phase 9 invariant

```text
cpu1 attention kernels must not silently materialize.
cpu1 attention preparer must not decide materialization by itself.
graph/region/lowering decides materialization.
prepare validates and binds explicit materialization plans.
execute materializes q/k/v/mask temps before the dense attention kernel runs.
```

Dense contiguous/no-offset cpu1 attention kernels remain unchanged. Planned
materialization supplies effective dense inputs to those kernels. The current
dense direct rejection remains the guard when no explicit materialization plan
exists.

### Scope

Apply planned input materialization to:

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

Attention output materialization is a write-target problem, not an input-copy
problem:

- output copy-back would need explicit post-kernel writeback semantics
- aliasing with source views must be handled by memory planning
- trace must distinguish input copy-in from output copy-back
- strided output may require a different operation family than dense SDPA

This phase therefore keeps rejecting strided/offset attention outputs unless a
later plan introduces explicit output materialization and copy-back semantics.

### Data flow examples

Example 1: materialize a transposed query view.

```text
user graph:
  qView = permute(qBase, ...)
  out = scaledDotProductAttention(qView, k, v)

planning:
  consumer=SDPA input=0 source=qView requires DENSE_CONTIGUOUS_NO_OFFSET
  graph/region creates materialization plan for (outNodeId, inputIndex=0)

execute:
  Cpu1InputMaterializer copies qView logical order into a dense temp
  SDPA reads q from the materialized dense temp
  k/v remain direct if already dense
```

Example 2: materialize an attention mask.

```text
user graph:
  maskView = slice(maskBase, ...)
  out = scaledDotProductAttention(q, k, v, maskView)

planning:
  consumer=SDPA input=3 source=maskView requires DENSE_CONTIGUOUS_NO_OFFSET

execute:
  maskView is copied as BOOL raw bytes into a dense temp
  SDPA uses the existing dense mask path
```

Example 3: reject strided output.

```text
user graph:
  outView = attention result written into a strided target

planning:
  Phase 9 does not create output materialization/copy-back plans

execute:
  prepare rejects because SDPA output is not dense contiguous no-offset
```

### Task 9.1: Inventory current cpu1 attention dense guards

Status: `[ ] not started`

What: inspect `Cpu1AttentionPreparer`, `Cpu1PreparedAttentionUnit`,
`Cpu1AttentionExecutableUnit`, and `Cpu1AttentionLoops`.

Why: map exactly which descriptor/runtime guards must start accepting explicit
effective dense materialized inputs while preserving unplanned rejection.

Expected output:

- map input indexes to role names: q/k/v/mask
- record prepare guard location for each input role
- record runtime guard location for each input role
- record output guard location and confirm output materialization remains out
  of scope

### Task 9.2: Extend materialization requirement selector to attention input roles

Status: `[ ] not started`

What: extend the graph/region input contract selector to include
`SCALED_DOT_PRODUCT_ATTENTION`.

Why: attention materialization must be selected before prepare.
`Cpu1AttentionPreparer` must not infer materialization from descriptors on its
own.

Proposed selector shape:

```java
return switch (opType) {
    case SCALED_DOT_PRODUCT_ATTENTION ->
            attentionInputRequirements(consumer.inputIds().size());
    default ->
            existingRequirements(opType, consumer);
};
```

Rules:

- mark q/k/v as `DENSE_CONTIGUOUS_NO_OFFSET`
- mark mask as `DENSE_CONTIGUOUS_NO_OFFSET` only when present
- do not mark attention output
- do not mark `SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS` input, because that op
  consumes a cache relationship, not the logical attention output data

### Task 9.3: Bind planned attention inputs in prepare

Status: `[ ] not started`

What: make cpu1 attention prepare ask the generalized prepare context for
planned input materializations by `(consumerNodeId, inputIndex)`.

Why: prepare should validate and bind explicit lowering decisions without
owning materialization policy.

Expected behavior:

- direct dense q/k/v/mask without a plan remains accepted
- non-dense q/k/v/mask with a matching plan is accepted as an effective dense
  input
- non-dense q/k/v/mask without a matching plan remains rejected
- plan for the wrong consumer/input is ignored or rejected
- output remains direct dense contiguous no-offset

### Task 9.4: Reuse generalized prepared input handles for attention

Status: `[ ] not started`

What: use the same prepared input handle/resolver created for loss/index instead
of introducing an attention-specific materialization wrapper.

Why: attention should share the explicit input materialization mechanism. The
only attention-specific part is the role mapping and shape/dtype validation.

Proposed prepared-unit shape:

```java
record Cpu1PreparedAttentionInput(
        int inputIndex,
        String role,
        int sourceNodeId,
        CompiledTensorDescriptor directDescriptor,
        Optional<Cpu1PreparedInputMaterialization> materialization
) {
    boolean materialized() {
        return materialization.isPresent();
    }
}
```

Do not introduce this record if the generalized prepared input handle already
contains the same information with clear naming.

### Task 9.5: Execute materialization before attention binds views

Status: `[ ] not started`

What: call `Cpu1InputMaterializer.materializeAll(...)` before attention runtime
binds q/k/v/mask views.

Why: dense attention loops should see effective dense input views and should
not know whether those views came from original storage or materialized temps.

Required ordering:

```text
Cpu1AttentionExecutableUnit.execute(context)
  1. materialize all planned input temps for this attention unit
  2. resolve direct/materialized q/k/v/mask views
  3. validate output remains direct dense contiguous no-offset
  4. dispatch existing dense cpu1 attention kernel
```

### Task 9.6: Add trace metadata for materialized attention inputs

Status: `[ ] not started`

What: add trace attributes that count and describe materialized attention
inputs.

Why: copy-in cost must be visible in traces and benchmark reports.

Suggested attributes:

- `cpu1AttentionMaterializedInputCount`
- `cpu1AttentionMaterializedInputs`
- `cpu1AttentionMaterializationReason`

Example:

```text
cpu1AttentionMaterializedInputCount=2
cpu1AttentionMaterializedInputs=[
  input=0,role=q,source=12,target=DENSE_CONTIGUOUS_NO_OFFSET,
  input=3,role=mask,source=15,target=DENSE_CONTIGUOUS_NO_OFFSET
]
cpu1AttentionMaterializationReason=cpu1-attention-dense-contiguous-input-contract
```

### Task 9.7: Attention materialization tests

Status: `[ ] not started`

What: add focused tests proving attention accepts planned materialized inputs
and still rejects unplanned non-dense inputs.

Coverage:

- strided/permuted q materialized before SDPA
- strided/permuted k materialized before SDPA
- strided/sliced v materialized before SDPA
- strided/sliced mask materialized before masked SDPA
- multiple materialized attention inputs in one SDPA node
- dense direct attention path still uses zero materialized inputs
- unplanned strided q/k/v/mask still fails the dense direct guard
- strided/offset attention output remains rejected
- `SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS` continues to require a cached forward
  attention output and is not treated as an input-materialization consumer

### Task 9.8: Attention materialization benchmarks

Status: `[ ] not started`

What: add benchmark smoke coverage after correctness passes.

Suggested scenarios:

- dense direct SDPA baseline
- q materialization + dense SDPA
- k/v materialization + dense SDPA
- mask materialization + dense masked SDPA
- compare copy-then-dense against any future view-aware dense attention kernel

Do not commit local benchmark/calibration artifacts unless intentionally
promoting canonical fixtures.

## Rejected Alternatives

### Synthetic `CompiledNode` `CONTIGUOUS`

Rejected for first slice.

Why:

- changes graph topology for a backend execution requirement
- complicates descriptor index and memory planning
- risks exposing synthetic nodes in publication/trace
- overfits a local cpu1 loss need to a graph rewrite

When reconsidered:

- if graph-level layout canonicalization becomes a general optimizer feature
- if multiple backends consume the same synthetic layout node semantics

### Direct hidden cpu1 preparer materialization

Rejected.

Why:

- makes copy-in invisible in lowering trace
- hides performance cost from benchmark reports
- turns preparer into a planner
- encourages kernels to accept non-contract inputs

Allowed behavior:

- cpu1 family preparers may bind materialization plans passed from lowering
- they may validate effective descriptors
- they may reject unplanned non-contiguous inputs
- they must not create materialization plans from source descriptors on their
  own

### Generic storage accessor framework

Rejected for this phase.

Why:

- cpu1 prepared kernels are intentionally direct
- accessor indirection would affect hot paths
- the immediate requirement is a copy helper, not strided loss compute

Allowed behavior:

- a small `Cpu1LogicalTensorCopy` helper for planned copy-in
- future accessor work only if multiple families prove the need

### Immediate strided loss kernels

Rejected for this phase.

Why:

- kernel matrix would grow before planner plumbing is correct
- dense kernels are already validated
- materialization decision must be traceable first

Future option:

- graph/lowering can later choose between:
  - `MATERIALIZE_INPUT`
  - `KEEP_STRIDED_AND_USE_STRIDED_LOSS_KERNEL`

## Validation Commands

Required focused commands:

```bash
./gradlew test --tests graph.compile.planning.region.Cpu1LossInputMaterializationPlanningTest
./gradlew test --tests backend.cpu1.Cpu1LossMaterializationExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1CrossEntropyLossExecutionContractTest --tests backend.cpu1.Cpu1NllLossExecutionContractTest --tests backend.cpu1.Cpu1DenseCrossEntropyLossExecutionContractTest
./gradlew test --tests backend.cpu1.Cpu1GatherExecutionContractTest
./gradlew test --tests ScatterElementsExecutionTest
./gradlew test --tests ScatterNdExecutionTest
./gradlew test --tests GatherExecutionTest
./gradlew test --tests GatherNdExecutionTest
./gradlew classes
git diff --check
```

Additional recommended commands after Phase 2:

```bash
./gradlew test --tests backend.cpu.lowering.CpuRegionLowererTest --tests backend.lowering.region.RegionExecutionPlanTest
```

## Validation Against Assignment

- [x] This document is a plan only.
- [x] It targets `todo/118-cpu1-graph-input-materialization-plan.md`.
- [x] It keeps materialization decision in graph/region/lowering.
- [x] It does not propose hidden cpu1 kernel fallback.
- [x] It does not use synthetic `CompiledNode` as the first step.
- [x] It respects current named types:
  - `ExecutionUnit`
  - `OptimizedRegion`
  - `DefaultRegionOptimizer`
  - `ExecutionUnitFactory`
  - `GraphValueRef`
  - `CpuRegionLowerer`
  - `RegionExecutionPlan`
  - `BackendPrepareContext`
  - `PreparedExecutionBuilder`
  - cpu1 prepare/execution/loss/layout packages
- [x] It includes phases, tasks, status tracking, planned file paths, and proposed code.

## Remaining Debt Or Follow-Up Work

Known follow-up after the planned implementation:

- Memory planner slot integration for input materialization temps is deferred to
  Phase 6 and should not block correctness.
- `UNKNOWN_OR_COMPLEX` source layouts remain unsupported in the first slice.
- Existing `ExecutionContext.cpu1ScratchBufferForNodeId(...)` is already a
  cpu1-specific method; this plan avoids adding more cpu1 temp APIs there, but a
  later cleanup could move scratch lookup behind cpu1-local helpers.
- Strided loss kernels remain a future optimization and are intentionally not
  part of this plan.
- Strided index/scatter output materialization and copy-back remain deferred to
  a separate plan or future phase.
- Parallel scatter remains deferred.
- Uniqueness analysis for duplicate-safe parallel scatter remains deferred.
- Hidden prepare/runtime fallback remains a non-goal.
- A generic hot-path storage accessor remains a non-goal.
- Benchmark calibration outputs must not be committed unless promoted to
  canonical fixtures deliberately.
