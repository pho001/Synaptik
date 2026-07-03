# Train a model (planned workflow)

## Outcome

This guide explains the intended first training lifecycle. No training session, optimizer, compiler autograd, or executable runtime API is implemented yet.

## Planned initial workflow

1. Mark selected public tensor state as trainable parameters.
2. Build a forward expression and loss.
3. Compile a forward-and-backward graph.
4. Prepare and run it to publish parameter gradients.
5. Apply a backend-independent optimizer step.

```text
forward/backward run -> published gradients -> optimizer.step()
```

For a simple stochastic-gradient-descent update with parameter `w = 2.0`, gradient `g = 0.5`, and learning rate `0.1`, the intended mathematical update is `w - 0.1 × 0.5 = 1.95`. This arithmetic illustrates optimizer ownership; it is not a runnable Synaptik call.

## Boundaries

The training extension owns the optimizer algorithm. A concrete backend owns any fused CPU, Metal, or CUDA implementation selected during prepare. Training must not import concrete backend modules.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| A `MetalOptimizerBridge` appears in training | Backend execution leaked into the extension. | Put the Metal route in Metal prepare/kernels. |
| Training constructs backend partitions | Compiler and planning ownership leaked. | Let compiler/planning produce ownership and partitions. |
| The conceptual optimizer defaults are treated as stable | No training API contract exists yet. | Wait for focused training tasks and reference Javadoc. |

## Related documentation

- [Training API status](../api/training-api.md)
- [Autograd](autograd.md)
- [Training graph](../architecture/training-graph.md)
- [Training master plan](../planning/extensions/training/master-plan.md)
