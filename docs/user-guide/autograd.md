# Request gradients (planned workflow)

## Outcome

This guide explains how automatic differentiation (autograd) will become part of graph
compilation. `CompileMode` is current standalone declarative configuration, but no current compile
aggregate or compiler consumes it. Compiler autograd, gradient publication, and the runnable
training lifecycle remain planned.

The current value construction is runnable metadata only:

```java
import io.github.pho001.synaptik.config.compile.CompileMode;

CompileMode graphScope = CompileMode.FORWARD_AND_BACKWARD;
```

`graphScope` requests future compiler-owned autograd and combined forward/backward compile-time
graph work. Constructing the enum value does not capture a graph, construct gradients, publish
them, prepare a schedule, or run training. `TRAINING_STEP` likewise records a future graph-scope
direction without adding an optimizer or optimizer-update graph.

## Planned flow

```text
forward expression
  -> compile mode requests backward work
  -> compiler builds backward graph
  -> combined graph optimization and planning
  -> prepare and run
  -> publish requested gradients
```

The compiler owns global derivative rules and backward graph construction. A concrete backend does not traverse the whole forward graph to derive gradients; it prepares only the partitions assigned to it.

## Numerical interpretation

For the scalar expression `y = x × x` at `x = 3`, the forward value is `y = 9` and the derivative is `dy/dx = 2 × x = 6`. This explains the mathematical goal only. Synaptik does not yet expose an API that constructs or runs this example.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| A backend builds the global backward graph | Autograd ownership is misplaced. | Keep derivative expansion in compiler passes. |
| Gradient data is stored in runtime device state on `Tensor` | Public tensor state and residency were mixed. | Publish through the planned binding and runtime mechanisms. |
| A forward-only compile exposes gradients | The compile mode did not request backward work. | Select `FORWARD_AND_BACKWARD` or a supported future `TRAINING_STEP` path once a compiler consumer exists. |

## Related documentation

- [Training graph](../architecture/training-graph.md)
- [Autograd strategy note](../design/notes/autograd-strategy.md)
- [Training API status](../api/training-api.md)
