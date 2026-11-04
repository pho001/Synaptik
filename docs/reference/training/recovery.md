# Caller-coordinated in-memory recovery

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/training/src/main/java/io/github/pho001/synaptik/training/TrainingSession.java)

Exact continuation means taking the same next training input with all state that can affect the
step restored consistently. Model values alone omit the optimizer's history and the caller's
progress. An **unpublished candidate** is a private replacement assembled and validated before the
caller makes it active; discarding a failed candidate is not a built-in cross-owner rollback.

```mermaid
flowchart TD
    accTitle: Validate a private recovery candidate before replacing the active holder
    accDescr: Saved model values, training state, modes and contexts, and caller progress feed one private candidate. Failure discards the candidate. Only successful validation followed by retirement of the old session permits caller-controlled publication; retirement failure aborts publication.
    M["Model values"] --> C["Private replacement tree and session"]
    S["TrainingState: parameters and optimizer history"] --> C
    X["Modes, forward contexts, next input and external state"] --> C
    C --> V{"All participant checks pass?"}
    V -- No --> D["Close and discard candidate"]
    V -- Yes --> R["Retire quiescent old session"]
    R --> P["On success: caller publishes replacement holder"]
    R --> F["On failure: abort publication; old session may be unusable"]
```

Read every incoming branch as required information, not fields automatically stored in a model
file. The caller supplies exclusion throughout; the diagram is not a durable checkpoint adapter
or a promise that a failed old-session close can be reversed.

<a id="caller-coordinated-in-memory-recovery"></a>
## Caller-coordinated in-memory recovery

An exact next-input handoff is a caller-owned workflow across NN (including the relocated
`nn.checkpoint.ModelCheckpoint` model-only API),
Training, and any external input or random-state owners. At a quiescent successful-run boundary,
the caller keeps a detached `TrainingState`, a complete model artifact containing parameter and
persistent-buffer values, every relevant module's local mode, the effective `ForwardContext`
used at each mode-sensitive objective-construction site, fixed graph-construction inputs, and a
typed next-input cursor with all affecting external state. `Module.stateDictionary()` is shallow:
its Tensor references are not detached value evidence. Neither `TrainingState` counters nor
`ModelCheckpoint.inspect` can reconstruct the next input or expose model value bytes.

Ordinary `ModelCheckpoint.save` captures the actual quiescent trained tree's parameter and buffer
leaves on the validated CPU route, without a caller-created expression view. Checkpoint internally
selects raw-preserving movement after full preflight; it neither rebinds that tree nor guarantees
generic leaf-output support in Engine. Keep source storage live and accessible through save.

The supported integration fixture follows this sequence under caller-owned exclusion:

1. Check participant completeness and every fact available from detached state without changing
   the published run. Create a **distinct, unpublished**, fully bound NN tree.
2. Call `ModelCheckpoint.load` on that private tree. Load rereads and validates the artifact even
   if it was previously inspected. The fixture also directly saves this loaded heap tree, compares
   complete artifact bytes, and loads it again before native copying; this is a model round-trip
   check, not a required recovery stage. Restore local modes parent first, then child overrides
   for a mixed-mode tree.
3. Use `TrainingParameterStorage.copyToSharedNative(privateTree.trainingParameters(), privateArena)` before building
   the replacement objective or opening a session. Loaded parameters are heap-backed and
   Engine-executable, but `TrainingSession.open` requires writable, shareable native parameter
   storage. Keep the Arena alive through session close.
4. Build the fixed objective from the tree's **final Tensor bindings** and recorded effective
   forward contexts, then call `TrainingSession.open`. Before `restore`, compare the exact stable
   path set and canonical logical `parameterBytes()` of `privateSession.state().parameters()`
   with the saved `TrainingState`. These fresh views start at position zero and contain
   big-endian represented bytes; raw native-order NN carrier bytes are not comparable.
5. Call `privateSession.restore(savedState)` for its own optimizer, schema, counter, slot, and
   payload validation. Validate the staged typed cursor and external-state handoff against the
   candidate, then retire the quiescent old session and publish one replacement holder under
   caller exclusion. On ordinary failure, close and discard the private session and Arena
   without publishing them. A failure while retiring the old session aborts publication, but
   cannot promise that old session is still runnable.

For example, the [CPU private-restore fixture](../../../testing/integration-tests/src/test/java/io/github/pho001/synaptik/testing/integration/PrivateTrainingRestoreIntegrationTest.java)
captures after two successful runs with a pending gradient, optimizer momentum, a persistent
buffer, mixed local modes, and an evaluation `ForwardContext` that differs from a child's training
mode. Its typed cursor names input position `2` and carries the external random state used to
derive that input. The restored candidate runs that same next input and matches an uninterrupted
CPU control in all detached Training fields, buffer value, modes, effective context, and caller
cursor. An equal-schema artifact with different parameter values is rejected at the canonical
byte comparison; private load, copy, open, restore, and late cursor failures do not swap the
published holder. This is evidence for that bounded same-route in-memory protocol, not a
production coordinator or a general transaction.

The model artifact contains no `TrainingState`, mode, cursor, or external-state codec. A durable
`TrainingCheckpoint` adapter is not implemented. Missing next-input or affecting external state,
training-mode BatchNorm without step-aligned running-buffer proof, and a changing per-run graph
random stream are outside this exact-continuation claim. A fixed
`GraphRngState.initial(key, counter)` expression can be rebuilt for the new objective but is not
advanced by the current session; cross-device or cross-route bitwise replay is not promised.


---

[Previous topic](optimizers.md) · [Next topic](../nn/composition.md)
