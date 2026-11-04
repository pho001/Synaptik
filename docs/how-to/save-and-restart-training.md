# Save a [model](../glossary.md#term-model) and start fresh training after load

## Outcome and prerequisites

Save current trained values, close the original [Model](../glossary.md#term-model), load an independently [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions), verify
[predictions](../glossary.md#term-prediction) and configure a **new** [training session](../glossary.md#training-session--trainingsession). Use JDK26 and the installed [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [artifacts](../glossary.md#term-artifact) from
[the learning project setup](../examples/README.md). The `save-load` mode supplies the complete
commented procedure; use its scratch output location, not a valuable existing file.

A **[parameter](../glossary.md#term-parameter)** is stored model state that training may change. An **[optimizer](../glossary.md#term-optimizer)** computes those
changes, and its state can include more than the current parameter values. A **[checkpoint](../glossary.md#term-checkpoint)**
must be understood by its stated contents: saving this model is not saving every fact needed to
continue a training process. **[Detached](../glossary.md#detached-model-results)** predictions own copied bytes; **owned** loaded state
has a new responsible [owner](../glossary.md#term-owner) rather than [borrowing](../glossary.md#term-borrowing) the original Model's storage.

## Procedure

1. Train the learning Model or run its save/load mode. The learned weight is two; the intended
   function is <span data-glossary-navigation="true"> ([run glossary](../glossary.md#term-run); [weight glossary](../glossary.md#term-weight))</span>`y=2x`.
2. Save with `Model.save` while the Model and its state are valid. The parent directory must exist.
   Successful publication atomically replaces an existing target; choose that path deliberately.<span data-glossary-navigation="true"> ([publication glossary](../glossary.md#term-publication); [target glossary](../glossary.md#term-target))</span>
3. Close the original [Model](../glossary.md#term-model). Retain a [detached](../glossary.md#detached-model-results) [prediction](../glossary.md#term-prediction) if you want to compare it afterward.
4. Load with `Model.load`. Supply fresh input tensors matching the saved full signature, including
   shape, layout, dtype and gradient eligibility. This learning signature is static <span data-glossary-navigation="true"> ([tensors glossary](../glossary.md#tensor); [signature glossary](../glossary.md#term-signature); [shape glossary](../glossary.md#shape); [layout glossary](../glossary.md#layout); [dtype glossary](../glossary.md#data-type--datatype); [gradient glossary](../glossary.md#term-gradient))</span>`[2,1]`.
5. Compute to verify learned values. New input `[2,4]` should produce `[4,8]` without the original
   Model or its Engine.<span data-glossary-navigation="true"> ([Engine glossary](../glossary.md#term-engine))</span>
6. If further training is intended, configure a fresh target signature, objective builder and
   optimizer, then call <span data-glossary-navigation="true"> ([objective glossary](../glossary.md#term-objective); [optimizer glossary](../glossary.md#term-optimizer))</span>`train`. Explain that as a new session, not resumed history.

The [canonical source](../examples/README.md)<span data-glossary-navigation="true"> ([canonical glossary](../glossary.md#term-canonical-form))</span> is the runnable example. The
[checkpoint reference](../reference/checkpoint/graph.md)<span data-glossary-navigation="true"> ([checkpoint glossary](../glossary.md#term-checkpoint))</span> documents [admission](../glossary.md#term-admission) and version selection.

## What crosses the file boundary

| State | Saved by [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions) [graph](../glossary.md#graph) [persistence](../glossary.md#term-persistence)? |
|---|---|
| Admitted source operations and [signatures](../glossary.md#term-signature) | Yes |
| Current named [parameter](../glossary.md#term-parameter)/[buffer](../glossary.md#term-buffer)/[constant](../glossary.md#term-constant) values and [identities](../glossary.md#term-identity-meanings) needed for reconstruction | Yes |
| Generated code, [prepared schedule](../glossary.md#prepared-schedule--preparedschedule) or native handles | No |
| Training [objective](../glossary.md#term-objective), [optimizer](../glossary.md#term-optimizer) configuration/moments/step or accumulation state | No |
| [Dataset](../glossary.md#term-dataset) position, [epoch](../glossary.md#term-epoch-meanings), [random stream](../glossary.md#graph-rng-state--graphrngstate) or Java [callback](../glossary.md#term-callback) | No |

An **epoch** is a complete pass through the training dataset. Dataset position is progress within
that pass; a **random stream** supplies successive random choices. Optimizer **moments** are
running summaries of [gradients](../glossary.md#term-gradient) used by algorithms such as [Adam](../glossary.md#adaptive-optimizer--adam-adamw). Gradient **accumulation** is a
pending sum awaiting an update. Those are independent pieces of progress, not quantities
recoverable from the learned [weight](../glossary.md#term-weight) alone.

A [model](../glossary.md#term-model) file is not a Java object dump. Source persistence and [backend](../glossary.md#backend) execution have separate
[admission](../glossary.md#term-admission): a graph may be executable but outside the bounded wire inventory, or load successfully
and still require a compatible execution backend. Use public load APIs rather than inspecting a
magic value to decide how to execute it.

## In-memory recovery is a different workflow

Training has [detached](../glossary.md#detached-model-results) in-memory state snapshots and strict transactional restore. Recreating a
compatible session and restoring its state requires caller coordination of [parameters](../glossary.md#term-parameter), input
progress and the [objective](../glossary.md#term-objective). That is not an implied durable cross-[owner](../glossary.md#term-owner) [resume](../glossary.md#schema-serialization-and-checkpoint) format. See the
[Training reference](../reference/training/sessions.md) and the existing
[in-memory recovery procedure](../user-guide/training.md).

## Common errors and useful variation

| Symptom | Cause to check | Action |
|---|---|---|
| Save rejects an otherwise working [model](../glossary.md#term-model) | The source is outside the selected wire [slice](../glossary.md#slice) | Use the [persistence](../glossary.md#term-persistence) inventory; do not assume every [Tensor](../glossary.md#tensor) [operation](../glossary.md#operation) is serializable |
| Missing-parent or I/O failure | Destination setup/permissions | Create an application-owned parent and retry explicitly |
| Fresh input rejected after load | Full saved [signature](../glossary.md#term-signature) does not match | Recreate the required [descriptor](../glossary.md#term-descriptor), not just the [element](../glossary.md#term-element) count |
| `train` rejects after load | No new training configuration | Configure the [objective](../glossary.md#term-objective) and [optimizer](../glossary.md#term-optimizer) on the loaded Model |
| [Predictions](../glossary.md#term-prediction) match, later optimization differs | [Optimizer history](../glossary.md#term-optimizer-history) was not saved | Treat this as new training, or use a supported caller-coordinated in-memory restore |

Useful variation: save before the first compute. It should still encode admitted source and
values; saving is not a request to create an [Engine](../glossary.md#term-engine). The teaching file does not certify arbitrary
[convolution](../glossary.md#term-channels-first-convolution-layers)/[Scan](../glossary.md#term-cumulative-scan) geometries or optimizer continuation.
