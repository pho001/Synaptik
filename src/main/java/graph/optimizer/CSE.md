# CSE Stage

`CSE` is the structural common subexpression elimination stage.

Its job is simple:

- detect tensors that compute the same thing
- keep one representative
- redirect the duplicates to that representative

It is a graph-structure pass, not a value-equality pass.

## Entry Point

- implementation:
  - [cse/CommonSubexpressionEliminationRule.java](./cse/CommonSubexpressionEliminationRule.java)
- config:
  - [../../config/optimizer/CseConfig.java](../../config/optimizer/CseConfig.java)

Default presets:

- training defaults:
  - `strictSafety = true`
- inference defaults:
  - `strictSafety = false`

## Core Algorithm

The pass walks the graph once in topological order.

For each tensor:

1. rewrite already replaced inputs
2. build a structural signature
3. if the signature already exists:
   - replace the tensor with the existing representative
4. otherwise:
   - remember the tensor as the representative of that signature

At the end it rebuilds a clean reachable topological closure.

## What Counts As "The Same"

The signature includes:

- `opType`
- op-specific parameters
- recursively resolved input signatures

And, in strict mode, also safety-sensitive properties such as:

- `requiresGrad`
- resolved backend
- output shape
- backward flag

So the pass is not trying to prove numerical equality.
It is only merging nodes that are structurally the same under the current safety mode.

## Leaf Semantics

Leaf handling is where most accidental CSE bugs usually happen.
The current rule is intentionally conservative.

### Trainable leaf tensors

If a tensor:

- has no operation, and
- `requiresGrad == true`

then it is identity-based.

So two separate trainable parameters with the same numeric values are not merged.

### Scalar constant leaves

If a tensor:

- has no operation
- does not require gradients
- contains exactly one scalar value

then it can be signatured structurally by:

- scalar value / bit pattern
- shape
- dtype-sensitive representation

So repeated scalar constants such as `0.0`, `1.0`, or `-1.0` may be merged.

### Other leaves

Other leaves remain identity-based.

This includes:

- non-scalar constants
- ordinary leaf inputs
- layout/view leaves that arrive in the optimizer graph as leaves

## Commutativity

The rule currently treats only these ops as commutative:

- `ADD`
- `MUL`

So:

```text
add(a, b)
```

and:

```text
add(b, a)
```

are considered the same expression by CSE.

The same applies to `mul(a, b)` and `mul(b, a)`.

## Explicit Non-Goals

`CSE` does not:

- reorder arbitrary arithmetic for equivalence proofs
- fold constants
- simplify `x + 0`
- fuse elementwise chains
- reason about runtime cost

Those belong to:

- `AR` for algebraic cleanup
- `FUSE` for elementwise fusion

## Examples

### Example 1: repeated scalar subexpression

Graph:

```text
t1 = x.mul(2.0)
t2 = x.mul(2.0)
y  = t1.add(t2)
```

If `t1` and `t2` are structurally identical, `CSE` can keep only one of them and rewrite the graph to:

```text
t1 = x.mul(2.0)
y  = t1.add(t1)
```

### Example 2: trainable leaves are not merged

Suppose:

```text
w1 = parameter([1.0, 2.0])
w2 = parameter([1.0, 2.0])
```

Even though values and shape match, if both are trainable leaves they remain separate because their identity matters for autograd semantics.

## Interaction With Forward/Backward Graphs

The optimizer runs over the joint compile-time graph.
That means `CSE` can in principle see both:

- forward nodes
- backward nodes

The backward flag is therefore part of the safety story.
The stage should not accidentally merge expressions across places where phase or gradient semantics matter.

## What CSE Refuses To Merge

The implementation explicitly avoids treating some nodes as ordinary merge candidates.
Important examples include:

- already fused nodes
- certain special structural cases where identity or descriptor state matters more than apparent surface similarity

When in doubt, the pass prefers not to merge rather than risk semantic drift.
