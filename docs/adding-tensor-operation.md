<!-- generated-by: gsd-doc-writer -->
# Adding A Tensor Operation

Navigation: [Index](index.md#recommended-reading-paths) | [Development](development.md#adding-tensor-ops) | [Tensor API](tensor-api.md#operation-catalog) | [Modules](modules.md#operations-primitive-semantic-descriptors) | [Graph Optimizer](graph-optimizer.md#cse) | [Testing](testing.md#targeted-test-patterns)

Chapters: [Purpose](#purpose) | [Mental Model](#mental-model) | [Choose The Operation Kind](#choose-the-operation-kind) | [Source Map](#source-map) | [Implementation Checklist](#implementation-checklist) | [Worked Example: Unary Floating Operation](#worked-example-unary-floating-operation) | [Descriptor Layer](#descriptor-layer) | [Tensor Builder Layer](#tensor-builder-layer) | [Public API Layer](#public-api-layer) | [CPU Kernel Layer](#cpu-kernel-layer) | [Autograd](#autograd) | [Broadcasting Shape And DType Rules](#broadcasting-shape-and-dtype-rules) | [Optimizer Integration](#optimizer-integration) | [Fusion And Accelerator Integration](#fusion-and-accelerator-integration) | [Testing Matrix](#testing-matrix) | [Common Mistakes](#common-mistakes)

This guide explains how to add a new public tensor operation to Synaptik without breaking the compile, optimize, prepare, and execute pipeline. It is implementation-grounded: every layer named here exists in the current source tree.

## Table Of Contents

- [Purpose](#purpose)
- [Mental Model](#mental-model)
- [Choose The Operation Kind](#choose-the-operation-kind)
- [Source Map](#source-map)
- [Implementation Checklist](#implementation-checklist)
- [Worked Example: Unary Floating Operation](#worked-example-unary-floating-operation)
- [Descriptor Layer](#descriptor-layer)
- [Tensor Builder Layer](#tensor-builder-layer)
- [Public API Layer](#public-api-layer)
- [CPU Kernel Layer](#cpu-kernel-layer)
- [Autograd](#autograd)
- [Broadcasting Shape And DType Rules](#broadcasting-shape-and-dtype-rules)
- [Optimizer Integration](#optimizer-integration)
- [Fusion And Accelerator Integration](#fusion-and-accelerator-integration)
- [Testing Matrix](#testing-matrix)
- [Common Mistakes](#common-mistakes)

## Purpose

A Synaptik tensor operation is not just one Java method. A real operation usually needs:

- a semantic operation id in `Operation.OpType`
- an immutable descriptor under `operations.*`
- a graph-building method under `tensor.ops.*`
- public facades in `TensorOps` and often `Tensor`
- a CPU kernel and resolver entry
- autograd logic if differentiable
- CSE signature support if the descriptor has parameters
- optional fusion or accelerator support
- tests for graph construction, compiled execution, gradients, dtype/layout behavior, and optimizer interaction

If any layer is skipped, the operation may appear to work eagerly as a graph node but fail later during `compute()`, graph optimization, fusion, CSE, or backend preparation.

## Mental Model

The operation pipeline is layered:

```text
User call
  Tensor.square()
    -> TensorOps.square(tensor)
      -> tensor.ops.unary.SquareOp.build(tensor)
        -> operations.elementwise.unary.square descriptor
        -> TensorPrimitiveBuilder creates a graph Tensor node
          -> CompiledGraph snapshots Operation.OpType.SQUARE
            -> optimizer may rewrite/CSE/partition/fuse
              -> backend prepare resolves CpuSquareKernel
                -> ComputeEngine executes prepared CPU or accelerator step
```

The descriptor says what the node means. The tensor builder says how to construct graph shape, dtype, inputs, and backward behavior. The backend kernel says how to compute concrete bytes.

## Choose The Operation Kind

Before adding files, classify the operation. This controls where it belongs and which contracts apply.

| Kind | Examples | Main builder package | Descriptor package | CPU kernel pattern |
|---|---|---|---|---|
| Unary elementwise | `neg`, `exp`, `sqrt`, `relu` | `tensor.ops.unary` | `operations.elementwise.unary` | `backend.cpu.kernels.elementwise.unary` |
| Binary elementwise | `add`, `mul`, `min`, comparisons | `tensor.ops.binary`, `tensor.ops.compare`, `tensor.ops.bool` | `operations.elementwise.binary`, `operations.elementwise.compare`, `operations.elementwise.logical` | `backend.cpu.kernels.elementwise.binary`, `compare`, `logical` |
| Ternary/select | `where` | `tensor.ops.select` | `operations.elementwise.where` | `backend.cpu.kernels.elementwise.where` |
| Reduction | `sum`, `mean`, `max(dim)` | `tensor.ops.reduction` | `operations.reduction` | `backend.cpu.kernels.reduction` |
| Layout/view | `reshape`, `permute`, `expand`, `contiguous` | `tensor.ops.layout`, `tensor.ops.index` | `operations.layout`, `operations.index` | `backend.cpu.kernels.layout` |
| Linear algebra | `matmul`, `linear`, attention | `tensor.ops.linalg` | `operations.linalg` | `backend.cpu.kernels.linalg` |
| NN special op | `conv2d`, pooling, norms, losses | family-specific `tensor.ops.*` | `operations.nn.*` or `operations.loss` | family-specific CPU kernel package |

Use the existing family closest to the operation. Do not create a new top-level package unless the operation family is genuinely new.

There is one important exception to the "new operation" flow: some public methods are ergonomic compositions, not primitive operations. In that case the right implementation is a method in `Tensor`, a static facade in `TensorOps`, a builder/helper in `tensor.ops.*`, tests, and docs, but no new `Operation.OpType`.

Current examples:

- `Tensor.stack(axis, ...)` composes `expandDims` and `concat`.
- `Tensor.unstack(axis)` composes `select` across the selected axis.
- `Tensor.take(axis, int[])` builds an index tensor and delegates to `gatherAxis` semantics.
- masked `sum` and masked `mean` compose `where`, reductions, valid-count handling, and division.
- masked cross entropy composes log-softmax, target selection/reduction, masking, and valid-count normalization.

This composition-first path keeps the semantic graph clean. It avoids creating a specialized kernel just because a higher-level consumer framework wants a convenient public method.

## Source Map

| Responsibility | Source |
|---|---|
| Operation id and fusable flag | [`Operation.java`](../src/main/java/operations/Operation.java) |
| Public static facade | [`TensorOps.java`](../src/main/java/tensor/TensorOps.java) |
| Public instance facade | [`Tensor.java`](../src/main/java/tensor/Tensor.java) |
| Primitive graph construction | [`TensorPrimitiveBuilder.java`](../src/main/java/tensor/internal/TensorPrimitiveBuilder.java) |
| Binary broadcasting planner | [`TensorBroadcastOps.java`](../src/main/java/tensor/TensorBroadcastOps.java), [`BroadcastPlanner.java`](../src/main/java/tensor/layout/BroadcastPlanner.java) |
| DType helpers | [`TensorDTypes.java`](../src/main/java/tensor/dtype/TensorDTypes.java), [`DataType.java`](../src/main/java/tensor/DataType.java) |
| CPU kernel resolver | [`CpuKernelRegistry.java`](../src/main/java/backend/cpu/kernels/CpuKernelRegistry.java) |
| CPU prepare | [`CpuNodePreparer.java`](../src/main/java/backend/cpu/prepare/CpuNodePreparer.java) |
| CSE parameter signatures | [`CommonSubexpressionEliminationRule.java`](../src/main/java/graph/optimizer/simplify/CommonSubexpressionEliminationRule.java) |
| CPU fused planning/codegen | [`backend/cpu/fused`](../src/main/java/backend/cpu/fused) |
| Metal allowlist and lowering | [`MetalPartitionSupport.java`](../src/main/java/backend/metal/lowering/MetalPartitionSupport.java), [`AcceleratorSubgraphLowerer.java`](../src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java) |
| Operation documentation | [`tensor-api.md`](tensor-api.md#operation-catalog) |
| Development/source hygiene tests | [`SourceTreeHygieneTest.java`](../src/test/java/SourceTreeHygieneTest.java) |

## Implementation Checklist

Use this checklist for any new operation:

1. Define the semantic operation id in `Operation.OpType`.
2. Choose the correct `OpArityClass`.
3. Set the `fusable` flag only when the operation is safe for generic elementwise fusion.
4. Add an immutable operation descriptor class under `src/main/java/operations/<family>/`.
5. Store only semantic parameters in the descriptor, not tensors, arrays that can be mutated externally, runtime config, or backend state.
6. Add a builder method in the correct `src/main/java/tensor/ops/<family>/` class.
7. Validate nulls, ranks, dtype, axis ranges, broadcast compatibility, and option values in the builder.
8. Build the result through `TensorPrimitiveBuilder`.
9. Attach backward logic with `TensorInternalAccess.setGradientRule(...)` and `GradientContext.accumulate(...)` if the operation participates in autograd.
10. Add a static facade method in `TensorOps`.
11. Add an instance method in `Tensor` when the operation should be fluent.
12. Add a CPU kernel and register it in `CpuKernelRegistry`.
13. Update `CpuNodePreparer` only if the operation needs workspace, prepared metadata, caches, or a special compute plan.
14. Update CSE `parameterKey(...)` if the descriptor carries any semantic parameter.
15. Update fusion/accelerator allowlists only after the operation is actually implemented and tested in those paths.
16. Add operation-level documentation in `tensor-api.md`.
17. Add tests for forward execution, compiled execution, invalid inputs, gradients, dtype behavior, and optimizer behavior.
18. Run focused tests and source hygiene checks.

## Worked Example: Unary Floating Operation

This section uses a hypothetical `square(x)` operation to show the end-to-end shape. The repository already has ways to express square as `x.mul(x)` or optimized `pow(2)`, so this is a teaching example rather than a recommendation to add a duplicate operation.

Expected user behavior:

```java
Tensor x = new Tensor(
        new double[]{-2.0, 3.0},
        new int[]{2},
        null,
        "x",
        DataType.FLOAT64
);
// x = [-2, 3]

Tensor y = x.square();
// y = x^2
// y = [4, 9]

Tensor loss = y.sumAll();
loss.compute(ComputeOptions.training());
// upstream gradient from sumAll is [1, 1]
// dy/dx = 2*x = [-4, 6]
// x.grad = [-4, 6]
```

For this operation:

- arity: unary
- shape: same as input
- dtype: same floating dtype as input
- backward formula: `d square(x) / dx = 2 * x`
- CPU kernel: one input array, one output array
- fusable: yes if the fused scalar/vector paths implement it

## Descriptor Layer

Add an operation id in [`Operation.java`](../src/main/java/operations/Operation.java):

```java
SQUARE(OpArityClass.ELEMENT_WISE, true),
```

The second argument is the generic fusion flag. Set it to `true` only if the operation is safe inside the current elementwise fusion system. If you are adding a descriptor before fusion support exists, use `false` first; it is better to miss a fusion than to generate a wrong fused kernel.

Add the descriptor class under the matching family:

```java
package operations.elementwise.unary;

import operations.Operation;

/**
 * Squares each element of a floating tensor.
 */
public final class square implements Operation {
    @Override
    public OpType opType() {
        return OpType.SQUARE;
    }

    @Override
    public String getExpression() {
        return "square";
    }
}
```

Important descriptor rules:

- Descriptor class names currently follow the repository's lowercase operation style, for example `add`, `pow`, and `sum`.
- Descriptors are immutable semantic objects.
- Descriptors may store parameters such as exponent, axis, `keepDims`, convolution options, or input shape.
- If storing arrays or option objects, use defensive copies or immutable option records where the existing pattern requires it.
- Descriptors must not store `Tensor` instances, runtime tensors, backend kernels, native handles, calibration profiles, or mutable execution state.

Parameterized example from current code: [`pow.java`](../src/main/java/operations/elementwise/unary/pow.java) stores `double exponent` and `float exponentF32`. CSE and kernels read those values later.

## Tensor Builder Layer

Add a concrete graph-building class in the operation family package. For a unary floating op this would be `src/main/java/tensor/ops/unary/SquareOp.java`:

```java
public static Tensor square(Tensor input) {
    UnarySupport.requireNumeric(input, "square");

    Operation op = new square();
    Tensor out = TensorPrimitiveBuilder.unary(
            input,
            op,
            "square",
            TensorDTypes.requireFloating(input.getDataType())
    );
    TensorInternalAccess.setGradientRule(out, context -> {
        Tensor outGrad = out.getGradient();
        if (outGrad == null || !input.getRequiresGrad()) {
            return;
        }
        context.accumulate(input, outGrad.mul(input).mul(2.0));
    });
    return out;
}
```

What each line does:

| Step | Why it exists |
|---|---|
| `UnarySupport.requireNumeric(input, "square")` | Fails early for null or non-floating input instead of letting a backend fail later. |
| `new square()` | Creates the immutable semantic descriptor that compile/prepare/backend code will see. |
| `TensorPrimitiveBuilder.unary(...)` | Creates a derived graph tensor with one predecessor and the descriptor attached. |
| `TensorDTypes.requireFloating(input.getDataType())` | Preserves the established dtype rule for floating unary operations. |
| `setGradientRule(...)` | Adds typed Java-level autograd graph construction. |
| `outGrad.mul(input).mul(2.0)` | Builds the gradient as normal tensor operations, so it can itself be compiled and optimized. |

Do not directly mutate `Tensor.prevTensors`, operation fields, graph ids, or storage arrays. `TensorPrimitiveBuilder` is the supported construction path.

## Public API Layer

Add a static facade to [`TensorOps.java`](../src/main/java/tensor/TensorOps.java):

```java
public static Tensor square(Tensor input) {
    return SquareOp.build(input);
}
```

Add a fluent instance facade to [`Tensor.java`](../src/main/java/tensor/Tensor.java) when the operation is meant for public users:

```java
public Tensor square() {
    return TensorOps.square(this);
}
```

Use both only when both APIs make sense. Some operations are intentionally static or factory-like because they need several inputs or options. For example, `TensorOps.conv2d(...)` has option-heavy overloads, while common unary/binary operations are natural instance methods.

After adding public API, update [Tensor API](tensor-api.md#operation-catalog) with:

- signature
- purpose
- dtype and shape contract
- invalid input behavior
- autograd formula
- concrete input/output example with values
- edge cases

## CPU Kernel Layer

Every operation that can survive optimization into execution needs backend support. For CPU execution:

1. Add a kernel class under the correct package.
2. Implement the relevant `CpuKernel` dtype entry points.
3. Reuse the family executor when possible.
4. Register the singleton in [`CpuKernelRegistry.java`](../src/main/java/backend/cpu/kernels/CpuKernelRegistry.java).

For a unary elementwise op, follow existing kernels such as [`CpuNegKernel.java`](../src/main/java/backend/cpu/kernels/elementwise/unary/CpuNegKernel.java), [`CpuPowKernel.java`](../src/main/java/backend/cpu/kernels/elementwise/unary/CpuPowKernel.java), or [`CpuReluKernel.java`](../src/main/java/backend/cpu/kernels/elementwise/unary/CpuReluKernel.java).

Minimal scalar shape:

```java
public final class CpuSquareKernel implements CpuKernel, UnaryElementwiseKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseUnaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseUnaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseUnaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public double applyF64(double value) {
        return value * value;
    }

    @Override
    public float applyF32(float value) {
        return value * value;
    }

    @Override
    public float applyBF16(float value) {
        return value * value;
    }
}
```

Then register it:

```java
private static final CpuSquareKernel SQUARE = new CpuSquareKernel();

public static CpuKernel resolve(Operation.OpType type) {
    return switch (type) {
        case SQUARE -> SQUARE;
        // existing cases...
    };
}
```

For performance-sensitive elementwise ops, consider the direct/vector methods used by existing kernels:

- `supportsVectorF64()` / `applyVectorF64(...)`
- `supportsVectorF32()` / `applyVectorF32(...)`
- `supportsDirectF64()` / `runDirectF64(...)`
- `supportsDirectF32()` / `runDirectF32(...)`
- `supportsDirectBF16()` / `runDirectBF16(...)`

Do not add per-operation dispatch thresholds inside the kernel hot loop. Dispatch policy belongs in the planners and calibrated runtime config, not in ad hoc kernel conditionals.

## Autograd

Autograd is usually attached in the tensor builder, not the backend kernel. The backend computes the forward primitive; the builder defines how to construct backward graph nodes.

For `square(x)`:

```text
y = x^2
dL/dx = dL/dy * 2*x
```

The backward builder:

```java
TensorInternalAccess.setGradientRule(out, context -> {
    Tensor outGrad = out.getGradient();
    if (outGrad == null || !input.getRequiresGrad()) {
        return;
    }
    context.accumulate(input, outGrad.mul(input).mul(2.0));
});
```

For broadcasted binary operations, gradients must be reduced back to the original input shapes. Existing binary ops use:

```java
TensorBroadcastOps.sumToShape(outGrad, first.getShape())
```

Concrete broadcast gradient example:

```java
Tensor x = new Tensor(
        new double[]{1.0, 2.0, 3.0, 4.0},
        new int[]{2, 2},
        null,
        "x",
        DataType.FLOAT64
);
// x = [[1, 2],
//      [3, 4]]

Tensor b = new Tensor(
        new double[]{10.0, 20.0},
        new int[]{2},
        null,
        "b",
        DataType.FLOAT64
);
// b = [10, 20]

Tensor y = x.add(b);
// b is broadcast to [[10, 20],
//                    [10, 20]]
// y = [[11, 22],
//      [13, 24]]

Tensor loss = y.sumAll();
// upstream dL/dy = [[1, 1],
//                  [1, 1]]
// dL/db before sumToShape has shape [2, 2]
// sumToShape(..., [2]) gives b.grad = [2, 2]
```

## Broadcasting Shape And DType Rules

Use existing helpers:

| Situation | Helper |
|---|---|
| Unary floating dtype | `TensorDTypes.requireFloating(input.getDataType())` |
| Binary floating dtype promotion | `TensorDTypes.promoteFloating(first.getDataType(), second.getDataType())` |
| Binary broadcast output shape | `TensorBroadcastOps.planBinary(first, second)` |
| Ternary `where` broadcast | Existing `WhereOp.build(...)` pattern |
| Reduction output shape | Existing reduction operation builders, `ReductionSupport`, and descriptors |
| Layout/view shape and strides | Existing layout operation builders and `LayoutSupport` |

Do not reimplement broadcasting by hand. The planner already handles aligned dimensions and emits consistent errors such as `Broadcast mismatch at dim ...`.

For binary elementwise ops, the normal construction pattern is:

```java
BroadcastPlan plan = TensorBroadcastOps.planBinary(first, second);
Operation op = new someBinaryOp(plan);
Tensor out = TensorPrimitiveBuilder.binary(
        first,
        second,
        plan.outShape(),
        op,
        "opLabel",
        TensorDTypes.promoteFloating(first.getDataType(), second.getDataType()),
        null
);
```

## Optimizer Integration

### CSE

If the operation has no parameters and only depends on input ids, op type, shape, dtype, and backend, the default CSE signature may be enough. If the operation has semantic parameters, update `CommonSubexpressionEliminationRule.parameterKey(...)`.

Examples that need parameter signatures:

| Operation | Parameter that affects semantics |
|---|---|
| `POW` | exponent |
| `MUL_SCALAR` | scalar |
| `CLAMP_MIN` / `CLAMP_MAX` | threshold |
| `SUM` / `MEAN` | dimension and `keepDims` |
| `SOFTMAX` / `LOG_SOFTMAX` | dimension |
| `RESHAPE` / `PERMUTE` / `EXPAND` | target shape or axes |
| `CONV2D` / pooling | options and shape metadata |

If you forget this, CSE can merge two nodes that look structurally similar but are semantically different. For example, `x.pow(2)` and `x.pow(3)` must not collapse into one node.

### AR rewrites

If the operation is an algebraic simplification target or source, update the rewrite rules under `src/main/java/graph/optimizer/rewrite`. Examples already covered by current code include canonicalization around sigmoid, relu, pow, linear, conv2d DAG lowering, and related patterns.

### Memory planning

Most ordinary operations need no memory planner change. Update memory planning only when the operation:

- creates views or aliases
- has special workspace requirements
- saves forward values for backward
- has non-standard output ownership
- interacts with region outputs or layout boundaries

The public entry point is `MemoryPlanner`, but the implementation is split by responsibility. View and alias ownership belongs in `TensorLifetimePlanner`; reusable tensor interval rules belong in `ReusableIntervalBuilder`; slot packing belongs in `ReusableSlotAllocator`; workspace-sensitive binding exclusions belong in `RuntimeMemoryBindingPolicyPlanner`; region value, binding, and handoff changes belong in `RegionValueFlowPlanner`, `RegionBindingAllocator`, and `RegionHandoffPlanner`.

## Fusion And Accelerator Integration

### CPU fusion

Setting `Operation.OpType(..., true)` makes the op eligible for generic elementwise fusion, but that is not sufficient by itself. The CPU fused path must know how to interpret or generate the operation.

Check these areas before enabling fusion:

- interpreted fused execution: `backend/cpu/fused/exec/InterpretedPreparedFusedExecutable.java`
- scalar ASM expression emission: `backend/cpu/fused/codegen/FusedScalarExpressionEmitter.java`
- vector ASM expression emission: `backend/cpu/fused/codegen/FusedVectorExpressionEmitter.java`
- vector helper methods: `backend/cpu/fused/codegen/FusedVectorOps.java`
- scalar helper methods: `backend/cpu/fused/codegen/FusedScalarOps.java`
- cost classification: `backend/cpu/fused/optimize/FusedCostModel.java`
- attribute extraction for parameterized ops: `backend/cpu/fused/codegen/FusedNodeAttributes.java`

If those paths do not support the op, either add support and tests or mark the op non-fusable until it is safe.

### Metal/CUDA accelerator lowering

Accelerator support is a separate decision. Do not add a new op to Metal/CUDA allowlists just because CPU supports it.

For Metal, check:

- [`MetalPartitionSupport.java`](../src/main/java/backend/metal/lowering/MetalPartitionSupport.java) for planner legality
- [`AcceleratorSubgraphLowerer.java`](../src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java) for DAG node mapping and scalar parameter encoding
- [`synaptik_apple_mps_stub.m`](../src/main/native/apple/synaptik_apple_mps_stub.m) for the native Objective-C implementation
- [Metal Backend: Supported Operations And DTypes](metal-backend.md#supported-operations-and-dtypes) for dtype boundaries
- [Metal Backend: Native Buffer ABI](metal-backend.md#native-buffer-abi) for buffer contracts
- [Metal Backend: Fallbacks And Failure Modes](metal-backend.md#fallbacks-and-failure-modes) for runtime fallback behavior

Current Metal compute/output dtype support is `FLOAT32` only, with `BOOL` allowed only for selected predicate input roles. A new CPU op should stay CPU-only until Metal semantics, dtype behavior, native code, and tests are all in place.

## Testing Matrix

At minimum, add focused tests that prove the operation works through the real compile/execute path.

| Test type | What to verify |
|---|---|
| Graph construction | `tensor.getOperation().opType()` is the new op, shape/dtype are correct, inputs are correct. |
| Direct compiled execution | `CompiledGraph.compile(...).prepare(...).execute(...)` or `compute()` returns expected values. |
| DTypes | Supported dtypes work; unsupported dtypes fail early with a clear exception. |
| Broadcasting | Broadcast-compatible inputs produce expected shape and values; incompatible shapes fail. |
| Gradients | `.backward()` / training compute produces expected gradients, including broadcast reduction. |
| CSE | Parameterized operations with different parameters do not collapse; identical operations can collapse when safe. |
| Fusion | If fusable, fused and non-fused profiles produce the same output and gradient. |
| Layout | Non-contiguous, permuted, expanded, or storage-offset inputs behave correctly or fail by design. |
| Source hygiene | New files live in the current package structure and do not revive legacy paths. |

Useful commands:

```bash
./gradlew test --no-daemon --tests AllOpsTest
./gradlew test --no-daemon --tests DataTypeExecutionCoverageTest
./gradlew test --no-daemon --tests BroadcastContractMatrixTest
./gradlew test --no-daemon --tests CommonSubexpressionEliminationRuleTest
./gradlew test --no-daemon --tests SourceTreeHygieneTest
```

For a new special operation, add a family-specific execution test similar to existing files such as:

- [`SoftmaxExecutionTest.java`](../src/test/java/SoftmaxExecutionTest.java)
- [`LogSoftmaxExecutionTest.java`](../src/test/java/LogSoftmaxExecutionTest.java)
- [`GatherExecutionTest.java`](../src/test/java/GatherExecutionTest.java)
- [`TakeAlongAxisExecutionTest.java`](../src/test/java/TakeAlongAxisExecutionTest.java)
- [`NllLossExecutionTest.java`](../src/test/java/NllLossExecutionTest.java)

For performance-sensitive ops, add benchmark coverage only after correctness tests pass. Benchmark must measure; it should not write calibration or autotune results.

## Common Mistakes

| Mistake | Consequence | Correct approach |
|---|---|---|
| Adding only a `Tensor` method | Graph builds may compile, but backend execution fails because no descriptor/kernel exists. | Add every layer from descriptor to CPU resolver. |
| Adding `OpType` but no CSE parameter key | Parameterized nodes can be merged incorrectly. | Update `parameterKey(...)` for every semantic parameter. |
| Marking an op fusable before fused execution supports it | FUSE can produce a plan the backend cannot execute correctly. | Keep `fusable=false` until interpreted and ASM fused paths are implemented and tested. |
| Computing gradients with detached Java arrays | Backward graph cannot be optimized or differentiated consistently. | Build gradients from tensor operations. |
| Reimplementing broadcasting manually | Shape behavior diverges from the rest of the framework. | Use `TensorBroadcastOps.planBinary(...)` and `sumToShape(...)`. |
| Letting backend kernels validate public API shape rules | Errors happen too late and may be backend-specific. | Validate user-facing contracts in `tensor.ops.*`. |
| Adding Metal allowlist support without native implementation | Planner can select a region that cannot execute. | Add native lowering/shim/test first, then allowlist. |
| Forgetting docs | Users see the method but not dtype, shape, gradient, or examples. | Update [Tensor API: Operation Catalog](tensor-api.md#operation-catalog) and [Examples: Running Examples](examples.md#running-examples) where appropriate. |
