# Operations (src/main/java/operations)

## Contents

- [Purpose](#purpose)
- [Main Components](#main-components)
- [What an Operation Descriptor Should Contain](#what-an-operation-descriptor-should-contain)
- [Operation Strategy](#operation-strategy)
- [Current Strategy Matrix](#current-strategy-matrix)
  - [Composition-Only Surface](#composition-only-surface)
  - [Specialized-Only Primitives](#specialized-only-primitives)
  - [Surface + Specialized Primitive (`both`)](#surface--specialized-primitive-both)
- [Lowering Policy](#lowering-policy)
- [How to Add a New Operation](#how-to-add-a-new-operation)
- [Practical Examples by Operation Type](#practical-examples-by-operation-type)
  - [Simple unary op](#simple-unary-op)
  - [Simple binary op](#simple-binary-op)
  - [Compare op](#compare-op)
  - [Select op](#select-op)
  - [Logical bool op](#logical-bool-op)
  - [Reduction op](#reduction-op)
  - [Layout op](#layout-op)
- [Common Mistakes](#common-mistakes)

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
- `greaterThan`
- `greaterOrEqual`
- `lessThan`
- `lessOrEqual`
- `equalTo`
- `notEqualTo`
- `where`
- `gather`
- `gatherGrad`
- `takeAlongAxis`
- `takeAlongAxisGrad`
- `scatterAdd`
- `conv2d`
- `conv2dBackwardInput`
- `conv2dBackwardWeight`
- `maxPool2d`
- `maxPool2dBackwardInput`
- `avgPool2d`
- `avgPool2dBackwardInput`
- `logicalAnd`
- `logicalOr`
- `logicalNot`
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
- `mean`
- `softmax`
- `logSoftmax`
- `nllLoss`
- `crossEntropyLoss`
- `reduceAll`
- `reduceAny`
- `reduceMin`
- `reduceMax`
- `reduceMinGrad`
- `reduceMaxGrad`
- `relu`
- `clampMin`
- `clampMax`
- `abs`
- `sigmoid`
- `contiguous`
- `reshape`
- `expand`
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

Additional current descriptor notes:

- comparison descriptors (`greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`, `notEqualTo`)
  - represent numeric-input compare ops
  - produce `BOOL` tensors
- `where`
  - is a select descriptor
  - consumes one `BOOL` condition plus two numeric branches
  - returns promoted numeric output
- logical bool descriptors (`logicalAnd`, `logicalOr`, `logicalNot`)
  - are `BOOL`-only ops
  - are nondifferentiable
- `gather`
  - is currently a deliberately narrow indexing primitive
  - `INT32` is now the preferred index dtype
  - numeric floating tensors with integral values are still accepted as a transitional compatibility mode
  - it is introduced mainly as the minimal base for future index-target loss family work

Those concerns live elsewhere:

- tensor helper classes build graph nodes and backward lambdas
- backend kernels execute ops
- compiled/prepared graph layers own runtime metadata

## Operation Strategy

Before adding a new user-facing tensor operation, decide which of these three architectural buckets it belongs to:

- `composition-only`
  - exposed as API ergonomics on `Tensor`
  - represented in the graph as composition of existing primitives
  - no dedicated canonical primitive is added
- `specialized-only`
  - exists as a first-class graph/backend primitive
  - has dedicated runtime meaning
  - should not be modeled as a permanent derived sugar graph if the specialized semantics are the real contract
- `both`
  - may exist as ergonomic surface and also as a specialized runtime primitive
  - optimizer may lower the ergonomic/composed form into the specialized primitive
  - only valid if semantics are exactly preserved

Decision rule:

- choose `composition-only` when the operation is mostly API sugar and does not need:
  - unique gradient semantics
  - dedicated planner/backend behavior
  - hot-path specialization
- choose `specialized-only` when the operation:
  - has its own semantic contract
  - has its own gradient/tie policy
  - needs dedicated memory/runtime behavior
  - is not naturally expressible without creating an artificial graph
- choose `both` when:
  - the public ergonomic form is useful
  - but a dedicated primitive gives real execution benefit
  - and the lowering from surface form to primitive does not change semantics

## Current Strategy Matrix

### Composition-Only Surface

These should stay as derived tensor helpers unless a strong runtime reason appears later:

- `minimum(second)`
- `maximum(second)`
- future piecewise helpers such as:
  - `step`
  - `isPositive`
  - `isNegative`
  - `isNonNegative`
  - `isNonPositive`
  - `signMask`

Reasoning:

- they are naturally expressed through compare/select algebra
- they do not currently require their own backend/planner contract
- keeping them derived prevents primitive-set bloat

Important note:

- `minimum/maximum` are intentionally not aliases for specialized `min/max`
- their semantics are compare/select-based and follow `where(...)` branch behavior on ties
- this is different from specialized `min/max` contracts

### Specialized-Only Primitives

These are canonical graph/runtime primitives and should remain first-class:

- arithmetic/runtime core:
  - `ADD`
  - `SUB`
  - `MUL`
  - `DIV`
  - `MUL_SCALAR`
  - `POW`
  - `NEG`
  - `INV`
  - `LOG`
  - `EXP`
  - `FAST_EXP`
  - `TANH`
  - `FAST_TANH`
  - `SQRT`
- compare/select/bool core:
  - `GT`
  - `GE`
  - `LT`
  - `LE`
  - `EQ`
  - `NE`
  - `WHERE`
  - `LOGICAL_AND`
  - `LOGICAL_OR`
  - `LOGICAL_NOT`
- indexing core:
  - `GATHER`
  - `GATHER_GRAD`
  - `TAKE_ALONG_AXIS`
  - `TAKE_ALONG_AXIS_GRAD`
  - `SCATTER_ADD`
- spatial core:
  - `CONV2D`
  - `CONV2D_BACKWARD_INPUT`
  - `CONV2D_BACKWARD_WEIGHT`
  - `MAX_POOL2D`
  - `MAX_POOL2D_BACKWARD_INPUT`
  - `AVG_POOL2D`
  - `AVG_POOL2D_BACKWARD_INPUT`
- reductions:
  - `SUM`
  - `MEAN`
  - `SOFTMAX`
  - `LOG_SOFTMAX`
  - `NLL_LOSS`
  - `CROSS_ENTROPY_LOSS`
  - `REDUCE_MIN`
  - `REDUCE_MAX`
  - `REDUCE_ALL`
  - `REDUCE_ANY`
  - `REDUCE_MIN_GRAD`
  - `REDUCE_MAX_GRAD`
  - `MIN_GRAD`
  - `MAX_GRAD`
- linear algebra:
  - `MATMUL`
- layout / storage semantics:
  - `CONTIGUOUS`
  - `RESHAPE`
  - `EXPAND`
  - `PERMUTE`
  - `EXPAND_DIMS`
  - `SQUEEZE`
- execution/runtime anchors:
  - `NOOP`
  - `FUSED`

Reasoning:

- these ops already carry real backend/planner meaning
- many of them have dedicated runtime kernels, reduction policies, memory behavior, or alias/materialization semantics
- they form the current canonical primitive set the optimizer reasons over

### Surface + Specialized Primitive (`both`)

These ops are strong candidates for both a public surface and a dedicated canonical/runtime primitive:

- `relu`
- `clamp`
- `clampMin`
- `clampMax`
- `abs`

Reasoning:

- all of them can be expressed compositionally
- all of them are plausible hot-path ops where specialization can reduce graph size and improve runtime behavior

Current status:

- `relu` already exists as a specialized primitive
- `clampMin` and `clampMax` now exist as specialized primitives
- `clamp(...)` remains composition over those specialized one-sided primitives
- optimizer lowering may normalize compare/select forms into `clampMin` / `clampMax` when semantics match exactly
- `abs` now exists as a specialized primitive with an explicit backward contract (`sign(x)`, `0` at `x == 0`)
- there is intentionally no automatic lowering from compare/select forms to `abs` yet, because the subgradient at `0` must stay explicit and consistent

Future likely `both` candidates:

- `abs`
- `leakyRelu`
- `gelu`
- `silu`
- `hardSigmoid`
- `hardSwish`

## Lowering Policy

The optimizer may lower a compositional surface into a specialized primitive only when semantics are identical.

Examples:

- allowed:
  - `where(x > 0, x, 0)` -> `RELU`
- not automatically allowed:
  - `where(a < b, a, b)` -> specialized `MIN`
  - `where(a > b, a, b)` -> specialized `MAX`

Why:

- specialized `min/max` carry different semantics than compare/select-based `minimum/maximum`, especially around gradient/tie behavior

This means:

- `Tensor` surface ergonomics do not automatically define canonical graph form
- canonical graph form is the smaller primitive set with explicit runtime meaning
- optimizer lowering is semantic-preserving normalization, not arbitrary graph shortening

## How to Add a New Operation

This is the current end-to-end checklist for adding a new operation correctly.

### 1. Decide the operation family

First decide where the operation belongs:

- unary
- binary
- comparison
- select
- logical bool
- reduction
- layout/view-like
- n-ary / special-case op

This determines which tensor helper class should build it.

Before doing that, also decide whether the operation belongs to:

- `composition-only`
- `specialized-only`
- `both`

using the strategy rules above.

Examples:

- unary -> `TensorUnaryOps`
- binary -> `TensorBinaryOps`
- comparison -> `TensorCompareOps`
- select -> `TensorSelectOps`
- logical bool -> `TensorBoolOps`
- reduction -> `TensorReduceOps`
- layout -> `TensorLayoutOps`
- spatial / convolution -> `TensorConvOps`
- spatial / pooling -> `TensorPoolOps`
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

Only add a new `OpType` when the operation is intended to be a canonical primitive.
Do not add new primitive enum values for pure API sugar that should remain compositional.

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
- where-broadcast/shape metadata if needed by the op family
- reduction axis
- transpose/permute axes

### 4. Add graph-building support in tensor helpers

This is the most important step. The public `Tensor` API is not created by the descriptor class itself.

You must add the operation to the correct tensor helper:

- `TensorUnaryOps`
- `TensorBinaryOps`
- `TensorCompareOps`
- `TensorSelectOps`
- `TensorBoolOps`
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

For compare/select/bool ops the dtype contract is different and must be explicit:

- compare ops
  - numeric inputs
  - `BOOL` output
- `where`
  - `BOOL` condition
  - numeric branches
  - promoted numeric output
- logical bool ops
  - `BOOL` input/output only

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
- where-style multi-input broadcast handling
- non-contiguous handling
- reduction hints
- matmul hints
- dtype-specific F16/F32/F64 loops
- `BOOL` execution path if the op is logical or compare/select related

### 8. Decide if the op is fusable

If the op should participate in fusion, first ask which kind of thing it is.

Fused compute algebra currently accepts:

- unary numeric ops
- binary numeric ops
- compare ops
- logical ops
- `where`

It intentionally does **not** accept:

- layout/view transforms as fused compute nodes
- indexing ops
- reductions
- `matmul`
- losses
- special grad kernels

Layout/view transforms such as:

- `select`
- `reshape`
- `expand`
- `permute`
- `expandDims`
- `squeeze`

are handled as fused external input access metadata, not as fused compute nodes.

This distinction is extremely important.

#### Fused compute algebra

These ops compute a new value for each logical output element:

- `add`
- `mul`
- `relu`
- compare ops
- logical ops
- `where`

Example:

```java
Tensor out = Tensor.where(a.greaterThan(b), x, y).relu();
// compare -> bool intermediate
// where   -> numeric value chosen from x or y
// relu    -> final numeric output
```

This is valid fused compute algebra because every step is still:

- one logical output space
- one local per-element computation

#### Fused access algebra

These ops do not compute a new arithmetic value.
They only change how logical coordinates read from backing storage:

- `select`
- `reshape`
- `expand`
- `permute`
- `expandDims`
- `squeeze`

Example:

```java
Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base");
Tensor view = base.select(0, 1);
Tensor out = view.relu().exp();
```

Fused interpretation:

```java
// base:
// [[1, 2, 3],
//  [4, 5, 6]]
//
// view = [4, 5, 6]
// out  = exp(relu(view))
```

The fused cluster keeps only:

- `RELU`
- `EXP`

The `select(...)` part is absorbed into external input metadata:

- backing tensor = `base`
- `storageOffset = 3`
- effective strides = `[1]`

#### Barriers

These ops must stop fused cluster growth:

- `gather`
- `takeAlongAxis`
- `scatterAdd`
- reductions
- `matmul`
- losses
- special grad kernels

Example:

```java
Tensor out = base.gather(indices, 1).relu().exp();
```

Correct fused behavior:

- `gather(...)` stays outside fused compute algebra
- only `relu().exp()` may fuse above it

That is intentional.
`gather` is not a static access transform and not a same-output-space local compute op.

So if the op belongs to fused compute algebra:

1. mark it correctly in `Operation.OpType`
2. ensure fusion rules accept it
3. ensure fused codegen emitters support it

Relevant files:

- [src/main/java/graph/optimizer/rules/FuseElementWiseRule.java](../graph/optimizer/rules/FuseElementWiseRule.java)
- [src/main/java/graph/optimizer/fusion/FusedAccessResolver.java](../graph/optimizer/fusion/FusedAccessResolver.java)
- [src/main/java/graph/codegen/FusedScalarExpressionEmitter.java](../graph/codegen/FusedScalarExpressionEmitter.java)
- [src/main/java/graph/codegen/FusedVectorExpressionEmitter.java](../graph/codegen/FusedVectorExpressionEmitter.java)
- [src/main/java/graph/codegen/FusedExternalInputPlan.java](../graph/codegen/FusedExternalInputPlan.java)
- [src/main/java/graph/codegen/FusedNodePlan.java](../graph/codegen/FusedNodePlan.java)
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

### Compare op

Required changes usually touch:

- `Operation.OpType`
- `operations/<op>.java`
- `TensorCompareOps`
- `TensorOps`
- `Tensor`
- CPU kernel + registry
- binary broadcast support
- prepared-input type contract if input/output dtypes differ
- tests
- docs

Special contract:

- numeric inputs
- `BOOL` output
- nondifferentiable

### Select op

Required changes usually touch:

- `Operation.OpType`
- `operations/<op>.java`
- `TensorSelectOps`
- `TensorOps`
- `Tensor`
- dedicated 3-input broadcast plan/helper
- CPU kernel + registry
- prepared-input type contract
- backward wiring
- tests
- docs

Special contract:

- one `BOOL` condition input
- numeric branch inputs
- promoted numeric output

### Logical bool op

Required changes usually touch:

- `Operation.OpType`
- `operations/<op>.java`
- `TensorBoolOps`
- `TensorOps`
- `Tensor`
- CPU kernel + registry
- `BOOL` broadcast behavior for binary bool ops
- tests
- docs

Special contract:

- `BOOL`-only input/output
- nondifferentiable

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
- forgetting `BOOL` / mixed dtype contract for compare/select families
- forcing compare/select through the same numeric prepared-input assumptions as normal arithmetic ops
- documenting an operation because the descriptor exists, even though it is not public on `Tensor`
