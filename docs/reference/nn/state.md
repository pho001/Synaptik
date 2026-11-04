# [Parameter](../../glossary.md#term-parameter) updates, state dictionaries and [Engine](../../glossary.md#term-engine) coordination

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)

A **[binding](../../glossary.md#term-binding)** is the [Tensor](../../glossary.md#tensor) currently held by a named parameter or [buffer](../../glossary.md#term-buffer) wrapper. Replacing a
binding changes later reads of that wrapper, not expressions already built from its old Tensor.
A **state dictionary** maps complete names to current Tensor references; it is a shallow [snapshot](../../glossary.md#term-snapshot)
of bindings, not [detached value](../../glossary.md#detached-model-results) bytes, a saved file or [optimizer history](../../glossary.md#term-optimizer-history).

<a id="current-nn-parameter-update-contract"></a>
## Current [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [parameter](../../glossary.md#term-parameter) update contract

A normal [module](../../glossary.md#term-module) declaration accepts a floating [Tensor](../../glossary.md#tensor) with `requiresGrad == true`. A concrete
input-dependent module may instead reserve private names and validators, but no incomplete or
nullable `Parameter` wrapper is exposed. Its reservation occupies the direct namespace and state
order immediately. Until its complete group is published by layer-local forward initialization or
strict state load, direct/recursive parameter discovery and state export fail with the first
unbound qualified path.

A final
`Parameter` wrapper privately retains the declaration-time exact [data type](../../glossary.md#data-type--datatype) and structural [Shape](../../glossary.md#shape).
A generic downstream consumer can obtain that wrapper from `Module.parametersRecursively()` and
call `Parameter.replace(value)`. Replacement requires a non-null Tensor with the same exact data
type, a structurally equal Shape, and `requiresGrad == true`; a failed [validation](../../glossary.md#term-validation) leaves the
previous exact [binding](../../glossary.md#term-binding) current.

A successful replacement preserves the wrapper, local name, and recursive discovery path while
installing the exact supplied Tensor reference. Tensor [identity](../../glossary.md#term-identity-meanings), [layout](../../glossary.md#layout), [host storage](../../glossary.md#host-storage),
[provenance](../../glossary.md#provenance), and label are deliberately outside compatibility and may differ. A Tensor obtained
before replacement, and expressions already constructed from it, remain unchanged.

This is one [mutable](../../glossary.md#term-mutable) binding [operation](../../glossary.md#operation), not an [optimizer step](../../glossary.md#optimizer-and-stochastic-gradient-descent). It is not [thread-safe](../../glossary.md#term-thread-safe) and provides
no version, update sequencing, or cross-binding consistency guarantee. Callers must coordinate it
with forward construction and other work that requires a consistent binding [view](../../glossary.md#term-view-meanings). [`Buffer`](../../glossary.md#term-buffer) has no
corresponding public wrapper update operation; module subclasses retain only the protected
direct-buffer replacement contract for their own state transitions.

<a id="current-nn-state-dictionary-contract"></a>
## Current [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) state-dictionary contract

`Module.stateDictionary()` exports one complete [module](../../glossary.md#term-module) tree as an [immutable](../../glossary.md#term-immutable), encounter-ordered
`StateDictionary`. Every `StateEntry` contains a relative dot-separated path, a `StateKind` of
[`PARAMETER`](../../glossary.md#term-parameter) or [`BUFFER`](../../glossary.md#term-buffer), and the exact current [Tensor](../../glossary.md#tensor) reference. Export visits each module's direct
parameters, then direct buffers, then child subtrees in child-registration order. It creates a
shallow value [snapshot](../../glossary.md#term-snapshot): later wrapper replacement does not change an earlier dictionary, but no
Tensor is copied, evaluated, materialized, [detached](../../glossary.md#detached-model-results), or transferred to a new [owner](../../glossary.md#term-owner).

`Module.loadStateDictionary(dictionary)` is strict and path-keyed, so candidate entry order need
not match export order. The candidate must contain exactly the complete [target](../../glossary.md#term-target) path set. Before
installing anything, load checks missing paths, unexpected paths, then each target's kind, exact
[data type](../../glossary.md#data-type--datatype), and structural [Shape](../../glossary.md#shape); a parameter also requires `requiresGrad == true`. A buffer uses
the target's current data type and Shape as its [schema](../../glossary.md#term-schema) and deliberately ignores [gradient](../../glossary.md#term-gradient)
eligibility because buffer role, not a Tensor flag, excludes it from [optimizer](../../glossary.md#term-optimizer) discovery. After
all checks pass, the exact candidate Tensor references are installed through existing wrappers or
newly prepared reserved-parameter wrappers in target traversal order.

Reserved parameter paths participate in that complete target set. Strict load validates their
retained layer-specific schema, prepares real wrappers for the entire candidate tree, and publishes
each complete reserved group without invoking its initializer, creating a generator, copying a
Tensor, or consuming random draws. An equivalent initialized automatic and [eager](../../glossary.md#term-eager-evaluation) [`Linear`](../../glossary.md#term-linear-meanings) use the
same [`weight`](../../glossary.md#term-weight)/optional [`bias`](../../glossary.md#term-bias) paths, kinds, exact types, and Shapes, so their state dictionaries are
compatible when their inferred and explicit [feature](../../glossary.md#term-feature) widths match. The same rule covers automatic
and explicit [RNN](../../glossary.md#term-recurrent-neural-network-rnn)/[GRU](../../glossary.md#term-gated-recurrent-unit-gru)/[LSTM](../../glossary.md#term-long-short-term-memory-lstm) cells through `inputWeight`, `hiddenWeight`, and optional `bias`; packed
gate order and exact Shapes remain cell-owned schema.

Supplied-table and initialized [`Embedding`](../../glossary.md#embedding) instances use the same single `weight` path, Parameter
kind, exact floating type, positive [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) Shape, and gradient-eligibility schema. Their state
dictionaries are therefore compatible when those exact table facts match. Loading or replacement
installs the complete exact candidate table; no row receives [padding](../../glossary.md#padding)-specific treatment.

Ordinary [validation](../../glossary.md#term-validation) failure changes no [binding](../../glossary.md#term-binding). This validate-before-install guarantee assumes
caller-coordinated access: module state export and load are not [thread-safe](../../glossary.md#term-thread-safe), linearizable,
synchronized, or simultaneously visible to racing readers. Tensors and expressions retained
before a successful load remain unchanged; later wrapper reads and forward construction observe
the new bindings.

The dictionary is an in-memory module-state boundary, not a persistent [checkpoint](../../glossary.md#term-checkpoint). It contains no
bytes, files, [codec](../../glossary.md#term-codec), format version, migration rules, evaluated Tensor values, [optimizer state](../../glossary.md#term-optimizer-state),
[`TrainingSession`](../../glossary.md#training-session--trainingsession) state, [graph](../../glossary.md#graph) random-number-generator state, [compiler](../../glossary.md#term-compiler) [artifact](../../glossary.md#term-artifact), [prepared
execution](../../glossary.md#prepared-execution--preparedexecution), [runtime](../../glossary.md#term-runtime) state, or [backend](../../glossary.md#backend) state. The current Training workflow can coordinate this
module-owned state with its separately owned `TrainingState`. By contrast, the current
[model checkpoint API](../checkpoint/models.md)<span data-glossary-navigation="true"> ([model glossary](../../glossary.md#term-model); [API glossary](../../glossary.md#term-application-programming-interface-api))</span> saves, inspects, and loads fully bound NN
parameter and buffer values through a file with [atomic live publication](../../glossary.md#artifact-envelope-and-atomic-publication); it does not persist
`TrainingState` or provide [exact training continuation](../../glossary.md#exact-training-continuation). A future Training Checkpoint adapter may
coordinate [persistence](../../glossary.md#term-persistence) of both state owners.

For example, a root module with parameter `weight`, buffer `step`, and child `encoder` containing
parameter `scale` and buffer `runningMean` exports this order:

```text
weight                    PARAMETER
step                      BUFFER
encoder.scale             PARAMETER
encoder.runningMean       BUFFER
```

Loading those same four paths in another list order succeeds when every binding is compatible;
omitting `encoder.runningMean` fails before `weight` or any other target changes. This example
demonstrates ordering, path [identity](../../glossary.md#term-identity-meanings), and [atomic](../../glossary.md#resource-transaction-and-rollback) validation only—it does not serialize or execute
Tensor values.

<a id="current-engine-coordination"></a>
## Current [Engine](../../glossary.md#term-engine) coordination

The one-shot `Engine.backward(objective, targets, maximumTotalBytes)` remains useful when a caller
wants [detached](../../glossary.md#detached-model-results) [gradients](../../glossary.md#term-gradient) without mutation or reuse. [`TrainingSession`](../../glossary.md#training-session--trainingsession) instead supplies the
[parameter](../../glossary.md#term-parameter) mapping, one reusable [prepared execution](../../glossary.md#prepared-execution--preparedexecution), accumulation, [optimizer state](../../glossary.md#term-optimizer-state), parameter
mutation, counters, [snapshot](../../glossary.md#term-snapshot)/restore, and close behavior defined above. Both use the same ordinary
Engine [compile](../../glossary.md#compile)/[prepare](../../glossary.md#prepare)/[run](../../glossary.md#term-run)/[materialization](../../glossary.md#materialization) authority.

```text
TrainingSession.open
  -> compile forward + first gradients once
  -> prepare one Engine PreparedExecution once
  -> repeated run or runValues / materialize / stage / commit
```

Schedulers, [mixed precision](../../glossary.md#term-mixed-precision) beyond the stated parameter domain, [persistence](../../glossary.md#term-persistence) for [exact training
continuation](../../glossary.md#exact-training-continuation), and compiled/[fused](../../glossary.md#specialization-and-fusion) [optimizer](../../glossary.md#term-optimizer) updates remain planned. A later compiled optimizer
[graph](../../glossary.md#graph) requires a separate architecture decision. Choosing a fused [Metal](../../glossary.md#term-metal) [kernel](../../glossary.md#kernel) for any future
optimizer update belongs to Metal prepare; a `MetalOptimizerBridge` inside Training would
reverse the required [dependency direction](../../glossary.md#java-platform-and-build-tools).

<a id="related-contracts"></a>
## Related contracts

Old [model](../../glossary.md#term-model) recovery examples use an explicit `:extensions:nn` [dependency](../../glossary.md#term-dependency); generic [Checkpoint](../../glossary.md#term-checkpoint)
alone no longer exports [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context). The [generic graph API](../checkpoint/graph.md)<span data-glossary-navigation="true"> ([graph glossary](../../glossary.md#graph); [API glossary](../../glossary.md#term-application-programming-interface-api))</span> stores source/value
state only, not [optimizer](../../glossary.md#term-optimizer)/session state, and adds no fresh-value [TrainingSession](../../glossary.md#training-session--trainingsession) support.

- [Training graph](../../architecture/training-graph.md)<span data-glossary-navigation="true"> ([Training graph glossary](../../glossary.md#training-graph))</span>
- [Autograd guide](../../user-guide/autograd.md)<span data-glossary-navigation="true"> ([Autograd glossary](../../glossary.md#autograd))</span>
- [Training guide](../../user-guide/training.md)
- [Training master plan](../../planning/extensions/training/master-plan.md)

---

[Previous topic](recurrent-layers.md)
