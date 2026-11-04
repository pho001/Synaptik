# Parameter updates, state dictionaries and Engine coordination

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)

<a id="current-nn-parameter-update-contract"></a>
## Current NN parameter update contract

A normal module declaration accepts a floating Tensor with `requiresGrad == true`. A concrete
input-dependent module may instead reserve private names and validators, but no incomplete or
nullable `Parameter` wrapper is exposed. Its reservation occupies the direct namespace and state
order immediately. Until its complete group is published by layer-local forward initialization or
strict state load, direct/recursive parameter discovery and state export fail with the first
unbound qualified path.

A final
`Parameter` wrapper privately retains the declaration-time exact data type and structural Shape.
A generic downstream consumer can obtain that wrapper from `Module.parametersRecursively()` and
call `Parameter.replace(value)`. Replacement requires a non-null Tensor with the same exact data
type, a structurally equal Shape, and `requiresGrad == true`; a failed validation leaves the
previous exact binding current.

A successful replacement preserves the wrapper, local name, and recursive discovery path while
installing the exact supplied Tensor reference. Tensor identity, layout, host storage,
provenance, and label are deliberately outside compatibility and may differ. A Tensor obtained
before replacement, and expressions already constructed from it, remain unchanged.

This is one mutable binding operation, not an optimizer step. It is not thread-safe and provides
no version, update sequencing, or cross-binding consistency guarantee. Callers must coordinate it
with forward construction and other work that requires a consistent binding view. `Buffer` has no
corresponding public wrapper update operation; module subclasses retain only the protected
direct-buffer replacement contract for their own state transitions.

<a id="current-nn-state-dictionary-contract"></a>
## Current NN state-dictionary contract

`Module.stateDictionary()` exports one complete module tree as an immutable, encounter-ordered
`StateDictionary`. Every `StateEntry` contains a relative dot-separated path, a `StateKind` of
`PARAMETER` or `BUFFER`, and the exact current Tensor reference. Export visits each module's direct
parameters, then direct buffers, then child subtrees in child-registration order. It creates a
shallow value snapshot: later wrapper replacement does not change an earlier dictionary, but no
Tensor is copied, evaluated, materialized, detached, or transferred to a new owner.

`Module.loadStateDictionary(dictionary)` is strict and path-keyed, so candidate entry order need
not match export order. The candidate must contain exactly the complete target path set. Before
installing anything, load checks missing paths, unexpected paths, then each target's kind, exact
data type, and structural Shape; a parameter also requires `requiresGrad == true`. A buffer uses
the target's current data type and Shape as its schema and deliberately ignores gradient
eligibility because buffer role, not a Tensor flag, excludes it from optimizer discovery. After
all checks pass, the exact candidate Tensor references are installed through existing wrappers or
newly prepared reserved-parameter wrappers in target traversal order.

Reserved parameter paths participate in that complete target set. Strict load validates their
retained layer-specific schema, prepares real wrappers for the entire candidate tree, and publishes
each complete reserved group without invoking its initializer, creating a generator, copying a
Tensor, or consuming random draws. An equivalent initialized automatic and eager `Linear` use the
same `weight`/optional `bias` paths, kinds, exact types, and Shapes, so their state dictionaries are
compatible when their inferred and explicit feature widths match. The same rule covers automatic
and explicit RNN/GRU/LSTM cells through `inputWeight`, `hiddenWeight`, and optional `bias`; packed
gate order and exact Shapes remain cell-owned schema.

Supplied-table and initialized `Embedding` instances use the same single `weight` path, Parameter
kind, exact floating type, positive rank-two Shape, and gradient-eligibility schema. Their state
dictionaries are therefore compatible when those exact table facts match. Loading or replacement
installs the complete exact candidate table; no row receives padding-specific treatment.

Ordinary validation failure changes no binding. This validate-before-install guarantee assumes
caller-coordinated access: module state export and load are not thread-safe, linearizable,
synchronized, or simultaneously visible to racing readers. Tensors and expressions retained
before a successful load remain unchanged; later wrapper reads and forward construction observe
the new bindings.

The dictionary is an in-memory module-state boundary, not a persistent checkpoint. It contains no
bytes, files, codec, format version, migration rules, evaluated Tensor values, optimizer state,
`TrainingSession` state, graph random-number-generator state, compiler artifact, prepared
execution, runtime state, or backend state. The current Training workflow can coordinate this
module-owned state with its separately owned `TrainingState`. By contrast, the current
[model checkpoint API](../checkpoint/models.md) saves, inspects, and loads fully bound NN
parameter and buffer values through a file with atomic live publication; it does not persist
`TrainingState` or provide exact training continuation. A future Training Checkpoint adapter may
coordinate persistence of both state owners.

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
demonstrates ordering, path identity, and atomic validation only—it does not serialize or execute
Tensor values.

<a id="current-engine-coordination"></a>
## Current Engine coordination

The one-shot `Engine.backward(objective, targets, maximumTotalBytes)` remains useful when a caller
wants detached gradients without mutation or reuse. `TrainingSession` instead supplies the
parameter mapping, one reusable prepared execution, accumulation, optimizer state, parameter
mutation, counters, snapshot/restore, and close behavior defined above. Both use the same ordinary
Engine compile/prepare/run/materialization authority.

```text
TrainingSession.open
  -> compile forward + first gradients once
  -> prepare one Engine PreparedExecution once
  -> repeated run or runValues / materialize / stage / commit
```

Schedulers, mixed precision beyond the stated parameter domain, persistence for exact training
continuation, and compiled/fused optimizer updates remain planned. A later compiled optimizer
graph requires a separate architecture decision. Choosing a fused Metal kernel for any future
optimizer update belongs to Metal prepare; a `MetalOptimizerBridge` inside Training would
reverse the required dependency direction.

<a id="related-contracts"></a>
## Related contracts

Old model recovery examples use an explicit `:extensions:nn` dependency; generic Checkpoint
alone no longer exports NN. The [generic graph API](../checkpoint/graph.md) stores source/value
state only, not optimizer/session state, and adds no fresh-value TrainingSession support.

- [Training graph](../../architecture/training-graph.md)
- [Autograd guide](../../user-guide/autograd.md)
- [Training guide](../../user-guide/training.md)
- [Training master plan](../../planning/extensions/training/master-plan.md)

---

[Previous topic](recurrent-layers.md)
