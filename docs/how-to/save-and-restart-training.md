# Save a model and start fresh training after load

## Outcome and prerequisites

Save current trained values, close the original Model, load an independently owned Model, verify
predictions and configure a **new** training session. Use JDK26 and the installed NN artifacts from
[the learning project setup](../examples/README.md). The `save-load` mode supplies the complete
commented procedure; use its scratch output location, not a valuable existing file.

A **parameter** is stored model state that training may change. An **optimizer** computes those
changes, and its state can include more than the current parameter values. A **checkpoint**
must be understood by its stated contents: saving this model is not saving every fact needed to
continue a training process. **Detached** predictions own copied bytes; **owned** loaded state
has a new responsible owner rather than borrowing the original Model's storage.

## Procedure

1. Train the learning Model or run its save/load mode. The learned weight is two; the intended
   function is `y=2x`.
2. Save with `Model.save` while the Model and its state are valid. The parent directory must exist.
   Successful publication atomically replaces an existing target; choose that path deliberately.
3. Close the original Model. Retain a detached prediction if you want to compare it afterward.
4. Load with `Model.load`. Supply fresh input tensors matching the saved full signature, including
   shape, layout, dtype and gradient eligibility. This learning signature is static `[2,1]`.
5. Compute to verify learned values. New input `[2,4]` should produce `[4,8]` without the original
   Model or its Engine.
6. If further training is intended, configure a fresh target signature, objective builder and
   optimizer, then call `train`. Explain that as a new session, not resumed history.

The [canonical source](../examples/README.md) is the runnable example. The
[checkpoint reference](../reference/checkpoint/graph.md) documents admission and version selection.

## What crosses the file boundary

| State | Saved by owned Model graph persistence? |
|---|---|
| Admitted source operations and signatures | Yes |
| Current named parameter/buffer/constant values and identities needed for reconstruction | Yes |
| Generated code, prepared schedule or native handles | No |
| Training objective, optimizer configuration/moments/step or accumulation state | No |
| Dataset position, epoch, random stream or Java callback | No |

An **epoch** is a complete pass through the training dataset. Dataset position is progress within
that pass; a **random stream** supplies successive random choices. Optimizer **moments** are
running summaries of gradients used by algorithms such as Adam. Gradient **accumulation** is a
pending sum awaiting an update. Those are independent pieces of progress, not quantities
recoverable from the learned weight alone.

A model file is not a Java object dump. Source persistence and backend execution have separate
admission: a graph may be executable but outside the bounded wire inventory, or load successfully
and still require a compatible execution backend. Use public load APIs rather than inspecting a
magic value to decide how to execute it.

## In-memory recovery is a different workflow

Training has detached in-memory state snapshots and strict transactional restore. Recreating a
compatible session and restoring its state requires caller coordination of parameters, input
progress and the objective. That is not an implied durable cross-owner resume format. See the
[Training reference](../reference/training/sessions.md) and the existing
[in-memory recovery procedure](../user-guide/training.md).

## Common errors and useful variation

| Symptom | Cause to check | Action |
|---|---|---|
| Save rejects an otherwise working model | The source is outside the selected wire slice | Use the persistence inventory; do not assume every Tensor operation is serializable |
| Missing-parent or I/O failure | Destination setup/permissions | Create an application-owned parent and retry explicitly |
| Fresh input rejected after load | Full saved signature does not match | Recreate the required descriptor, not just the element count |
| `train` rejects after load | No new training configuration | Configure the objective and optimizer on the loaded Model |
| Predictions match, later optimization differs | Optimizer history was not saved | Treat this as new training, or use a supported caller-coordinated in-memory restore |

Useful variation: save before the first compute. It should still encode admitted source and
values; saving is not a request to create an Engine. The teaching file does not certify arbitrary
convolution/Scan geometries or optimizer continuation.
