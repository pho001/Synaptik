# Operations (src/main/java/operations)

## Purpose

The `operations` package contains operation descriptors used by:

- tensor graph nodes
- optimizer reasoning via `opType()`
- backend kernel dispatch via `opType()`
- fused-cluster descriptors

Important architectural point:

- `Operation` classes are descriptors
- they are not the primary place where forward math or autodiff logic lives

Forward graph construction and backward wiring are mostly defined in:

- [src/main/java/tensor/TensorBinaryOps.java](../tensor/TensorBinaryOps.java)
- [src/main/java/tensor/TensorUnaryOps.java](../tensor/TensorUnaryOps.java)
- [src/main/java/tensor/TensorReduceOps.java](../tensor/TensorReduceOps.java)
- [src/main/java/tensor/TensorLayoutOps.java](../tensor/TensorLayoutOps.java)
- [src/main/java/tensor/TensorMatMulOps.java](../tensor/TensorMatMulOps.java)

## Main Components

- Base interface:
  - [src/main/java/operations/Operation.java](../operations/Operation.java)
- Fused descriptor:
  - [src/main/java/operations/FusedOperation.java](../operations/FusedOperation.java)
  - [src/main/java/operations/FusedOperationFactory.java](../operations/FusedOperationFactory.java)

Current descriptor classes in this package:

- `add`
- `sub`
- `mul`
- `div`
- `min`
- `max`
- `matmul`
- `neg`
- `inv`
- `log`
- `exp`
- `fastExp`
- `tanh`
- `fastTanh`
- `pow`
- `sqrt`
- `mulScalar`
- `sum`
- `relu`
- `sigmoid`
- `contiguous`
- `reshape`
- `permute`
- `expandDims`
- `squeeze`
- `noop`
- `FusedOperation`

Not every descriptor is necessarily exposed as a direct public `Tensor` instance method.

The user-facing source of truth for the public tensor surface is:

- [src/main/java/tensor/API.md](../tensor/API.md)

## What an Operation Descriptor Should Contain

In the current architecture an operation descriptor should usually contain only:

- `opType()`
- small immutable descriptor state if needed
  - broadcast plan
  - reduction dimension
  - scalar exponent
  - scalar multiplier
  - reshape/permute metadata
- optional human-readable expression string

It should usually not contain:

- backend preference logic
- runtime compiled executable instances
- generic math fallback implementations for normal execution
- execution caches

Those concerns live elsewhere:

- tensor helper classes build graph nodes and backward lambdas
- backend kernels execute ops
- compiled/prepared graph layers own runtime metadata

## How to Add a New Operation

This is the current end-to-end checklist for adding a new operation correctly.

### 1. Decide the operation family

First decide where the operation belongs:

- unary
- binary
- reduction
- layout/view-like
- n-ary / special-case op

This determines which tensor helper class should build it.

Examples:

- unary -> `TensorUnaryOps`
- binary -> `TensorBinaryOps`
- reduction -> `TensorReduceOps`
- layout -> `TensorLayoutOps`
- matmul -> `TensorMatMulOps`
- fused descriptor path -> `FusedOperationFactory`

### 2. Add or update `Operation.OpType`

File:
- [src/main/java/operations/Operation.java](../operations/Operation.java)

Tasks:

1. add the new enum constant to `OpType`
2. set correct metadata on that enum
   - category
   - fusable/non-fusable
3. verify any switch statements over `OpType` still cover the new value

Why:

- optimizer rules
- backend kernel registries
- planner dispatch
- fused logic

all reason over `opType()`, not over Java class names.

### 3. Add the operation descriptor class

Create a descriptor in `src/main/java/operations/`.

Typical minimal shape:

```java
package operations;

public final class relu implements Operation {
    @Override
    public OpType opType() {
        return OpType.RELU;
    }

    @Override
    public String getExpression() {
        return "relu";
    }
}
```

If the operation needs descriptor parameters, store only immutable descriptor state.

Examples:

- scalar parameter
- broadcast plan
- reduction axis
- transpose/permute axes

### 4. Add graph-building support in tensor helpers

This is the most important step. The public `Tensor` API is not created by the descriptor class itself.

You must add the operation to the correct tensor helper:

- `TensorUnaryOps`
- `TensorBinaryOps`
- `TensorReduceOps`
- `TensorLayoutOps`
- `TensorNaryOps`
- `TensorMatMulOps`

Tasks:

1. build the output tensor
2. assign the descriptor instance
3. set correct output dtype
4. attach backward lambda if gradients are supported

Typical shape:

```java
Operation op = new relu();
Tensor out = new Tensor(input.getShape(), List.of(input), op, "relu");
out.setDataType(TensorDataTypeUtil.unary(input));
out.setBackwardFunction(() -> {
    // gradient logic
});
return out;
```

### 5. Expose it on `Tensor` if it should be public

File:
- [src/main/java/tensor/Tensor.java](../tensor/Tensor.java)

Add the public instance method only if this operation should be part of the user-facing tensor surface.

Example:

```java
public Tensor relu() {
    return TensorOps.relu(this);
}
```

If the operation is internal-only or not yet intended as public API, do not add the `Tensor` method.

### 6. Add forwarding method to `TensorOps`

File:
- [src/main/java/tensor/TensorOps.java](../tensor/TensorOps.java)

Add a simple dispatcher method:

```java
public static Tensor relu(Tensor input) {
    return TensorUnaryOps.relu(input);
}
```

This keeps `Tensor` small and keeps actual graph-building in the specialized helper class.

### 7. Add backend kernel support

For CPU support you must:

1. implement or extend the CPU kernel
2. register it in `CpuKernelRegistry`
3. ensure planner/backend dispatch can resolve the op

Relevant places:

- [src/main/java/backend/kernels/cpu/](../backend/kernels/cpu)
- [src/main/java/backend/registry/CpuKernelRegistry.java](../backend/registry/CpuKernelRegistry.java)
- [src/main/java/backend/CPUBackend.java](../backend/CPUBackend.java)

Depending on the op, you may also need:

- broadcast handling
- non-contiguous handling
- reduction hints
- matmul hints
- dtype-specific F16/F32/F64 loops

### 8. Decide if the op is fusable

If the op is element-wise and should participate in fusion:

1. mark it correctly in `Operation.OpType`
2. ensure fusion rules accept it
3. ensure fused codegen emitters support it

Relevant files:

- [src/main/java/graph/optimizer/rules/FuseElementWiseRule.java](../graph/optimizer/rules/FuseElementWiseRule.java)
- [src/main/java/graph/codegen/FusedScalarExpressionEmitter.java](../graph/codegen/FusedScalarExpressionEmitter.java)
- [src/main/java/graph/codegen/FusedVectorExpressionEmitter.java](../graph/codegen/FusedVectorExpressionEmitter.java)
- [src/main/java/graph/codegen/HFusedOperationGenerator.java](../graph/codegen/HFusedOperationGenerator.java)

If the op is not fusable, make that explicit.

### 9. Add tests

At minimum add:

1. forward correctness test
2. backward correctness test if gradients are supported
3. dtype coverage if relevant
4. broadcasting / non-contiguous coverage if relevant
5. fused equivalence coverage if the op is fusable

Good existing references:

- [src/test/java/AllOpsTest.java](../../../test/java/AllOpsTest.java)
- [src/test/java/DataTypeExecutionCoverageTest.java](../../../test/java/DataTypeExecutionCoverageTest.java)
- [src/test/java/BroadcastBinaryOpsTest.java](../../../test/java/BroadcastBinaryOpsTest.java)
- [src/test/java/FusedExecutionModesTest.java](../../../test/java/FusedExecutionModesTest.java)

### 10. Update documentation

If the new op becomes public on `Tensor`, update:

- [src/main/java/tensor/API.md](../tensor/API.md)
- [src/main/java/tensor/README.md](../tensor/README.md) if needed
- root [README.md](../../../README.md) if it changes the project-level operation catalog materially

## Practical Examples by Operation Type

### Simple unary op

Required changes usually touch:

- `Operation.OpType`
- `operations/<op>.java`
- `TensorUnaryOps`
- `TensorOps`
- `Tensor`
- CPU kernel + registry
- tests
- docs

### Simple binary op

Required changes usually touch:

- `Operation.OpType`
- `operations/<op>.java`
- `TensorBinaryOps`
- `TensorOps`
- `Tensor`
- CPU kernel + registry
- broadcast support behavior
- tests
- docs

### Reduction op

Required changes usually touch:

- `Operation.OpType`
- `operations/<op>.java`
- `TensorReduceOps`
- `TensorOps`
- `Tensor`
- dedicated reduction backend path
- tests for axis/all-reduce behavior
- docs

### Layout op

Required changes usually touch:

- descriptor
- `TensorLayoutOps`
- `TensorOps`
- `Tensor`
- backend materialization / view semantics
- shape/stride tests
- docs

## Common Mistakes

- adding a descriptor class but forgetting to expose it through tensor helpers
- adding a `Tensor` method but forgetting backend kernel registry support
- matching by class name instead of `opType()`
- putting runtime compiled state into the descriptor
- forgetting fused emitter support for fusable ops
- forgetting dtype-specific behavior and tests
- documenting an operation because the descriptor exists, even though it is not public on `Tensor`
