# Training API

## Purpose and implementation status

This reference records the implemented neural-network parameter boundary consumed by future
training and the remaining planned training concepts without inventing optimizer APIs.
`extensions/nn` now provides module-owned `Parameter` and `Buffer` declarations, recursive
parameter discovery, one public schema-validated parameter replacement capability, and strict
in-memory module-tree state export/load. It also provides the narrow `UnaryTensorModule` subtype
and immutable numeric-child `Sequential` composition for modules whose complete forward signature
is exactly `Tensor forward(Tensor)`, three explicit-state recurrent cells, and one statically
packed vanilla-RNN sequence container.
`extensions/training`, public gradient publication, optimizer behavior, and prepared execution
are not implemented. The compiler now provides a public immutable functional-gradient request
value and a bounded package-private one/two-stage reverse-mode integration path.

Training will own backend-independent optimizer algorithms and session concepts. The compiler will own global automatic differentiation (autograd), while concrete backends will own any backend-specific lowering or fused optimizer route.

The accepted compiler design builds one or two reverse-mode stages with existing public Tensor
operations before one combined forward/backward capture. Model task 0025 and Compiler tasks 0004
and 0006 are Complete.
Public Tensors gain no gradient/backward lifecycle state. The current internal `TRAINING_STEP`
mode uses the same `FunctionalGradientRequest` contract as `FORWARD_AND_BACKWARD`; it adds no
optimizer update. Runtime gradient delivery, optimizer behavior, preparation, and execution
remain planned.

Package-private `GraphCompiler` currently returns mode-neutral `GraphCompilation`. A
`TRAINING_STEP` result may carry the same combined forward/backward graph as
`FORWARD_AND_BACKWARD`; a `FORWARD_ONLY` result has no BACKWARD nodes and empty gradient results.
This internal graph-stage result is not the later `CompileArtifacts` aggregate or a training
session/result type.

The current compiler request supports one or two bounded stages, exact forward or first-stage-
gradient output references, aligned explicit seeds or scalar default seeds, ordered identity-
unique targets, and ERROR/ZERO disconnected behavior. The compiler preflights each complete
selected slice before creating formulas, captures forward and all derivative roots once, and
retains per-node derivative order beside unchanged graph phase. These facts do not expose a
public training workflow, deliver a gradient at runtime, choose a parameter update, select a
backend, prepare a schedule, or execute training.

## Current NN unary composition contract

`Module` remains the general owner of named state, children, and train/eval mode and has no
universal forward method. `UnaryTensorModule` is the narrower public subtype for a module whose
complete forward contract accepts one non-null Tensor and returns one non-null Tensor. Current
`Linear`, `LayerNorm`, and `Embedding` layers participate. `BatchNorm` does not, because it
requires an explicit `ForwardContext` and may transition running-statistic buffers. `Dropout`
does not, because it requires both an explicit context and caller-threaded graph random-number-
generator state and returns a result carrying output plus next state.

`Sequential` accepts one `List<? extends UnaryTensorModule>`, snapshots it, and permanently owns
the exact children under decimal names `0`, `1`, and so on. It passes the exact input reference to
child `0`, passes each exact child result to the next child once, and returns the exact final
result. Empty composition returns the exact input. A later failure preserves already constructed
prefix expressions and suppresses the remaining suffix; the container performs no pipeline-wide
validation, rollback, caching, flattening, fusion, compilation, or execution.

For example, a sequence containing a `Linear` child followed by a `LayerNorm` child exposes state
paths such as `0.weight`, `0.bias`, `1.scale`, and `1.bias` and composes declaratively as:

```text
input reference
  -> child 0 Linear.forward(input)
  -> child 1 LayerNorm.forward(exact child-0 result)
  -> exact child-1 result
```

Inherited recursive mode propagation and state-dictionary export/load follow those numeric child
paths. The sequence does not create a training session, choose an optimizer, publish gradients,
or imply that any Tensor expression has been executed.

## Current NN recurrent composition contract

`RnnCell`, `GruCell`, and `LstmCell` each describe one recurrent step and extend `Module`
directly. RNN and gated recurrent unit (GRU) cells accept input plus hidden state and return one
next-hidden Tensor. The long short-term memory (LSTM) cell accepts input, hidden state, and cell
state and returns both next states. None retains caller-threaded recurrent state, and none fits the
one-input contract of `UnaryTensorModule` or `Sequential`.

`RnnSequence` is the current cell-specific sequence container. It permanently owns one exact
`RnnCell` child and accepts:

```java
RnnSequenceForwardResult result = sequence.forward(input, initialHidden, lengths);
```

Here `input` must have a fully static time-major Shape `[time, batch, inputSize]`,
`initialHidden` must have Shape `[batch, hiddenSize]`, and `lengths` is a Java `long[]` with one
value in `[0, time]` per original batch row. The method defensively copies the array before using
it and retains neither copy. Callers must coordinate any array mutation that could race with that
copy.

The copied lengths determine an **active batch** at each time step: original batch rows whose
length exceeds that step, kept in ascending original order. The sequence statically constructs
one compact batched cell expression for every non-empty step. It never examines Tensor values to
decide activity, so an all-zero input row remains active whenever its explicit length includes the
step.

For example, lengths `[5, 3, 1]` produce five packed outputs with active batch extents
`[3, 2, 2, 1, 1]`:

```text
time step          0      1      2      3      4
active rows      0,1,2   0,1    0,1     0      0
active extent      3      2      2      1      1
```

There are five Java cell calls—one per non-empty time step—and their compact Shapes represent
nine logical recurrent row applications rather than fifteen dense padded rows. Each
`packedOutputs().get(t)` is the exact cell Tensor for that compact step. `finalHidden()` restores
one row per original batch entry: row 0 comes from step 4, row 1 from step 2, and row 2 from step
0. A zero-length row instead uses its row from `initialHidden`; if all lengths are zero, the
packed list is empty and `finalHidden()` is the exact `initialHidden` reference.

This is static Tensor-expression construction, not numerical execution. It proves that padded
logical rows are absent from the constructed cell operands; it does not prove compiler capture,
gradient support, backend lowering, physical kernel skipping, or execution. Runtime Tensor
lengths or masks cannot currently choose the number of steps or active-batch Shapes. Applying a
dense `WHERE` after a full-batch cell would still construct padded cell work. A future dynamic
form therefore needs a genuine Model recurrent scan or control-flow contract. Current cumulative
sum and product scans have fixed associative bodies and are not that primitive. Cell-specific GRU
and LSTM sequence containers remain planned rather than implemented.

## Current NN parameter update contract

A module declaration accepts a floating Tensor with `requiresGrad == true`. Its final
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
all checks pass, the exact candidate Tensor references are installed through the existing stable
wrappers in target traversal order.

Ordinary validation failure changes no binding. This validate-before-install guarantee assumes
caller-coordinated access: module state export and load are not thread-safe, linearizable,
synchronized, or simultaneously visible to racing readers. Tensors and expressions retained
before a successful load remain unchanged; later wrapper reads and forward construction observe
the new bindings.

The dictionary is an in-memory module-state boundary, not a persistent checkpoint. It contains no
bytes, files, codec, format version, migration rules, evaluated Tensor values, optimizer state,
`TrainingSession` state, graph random-number-generator state, compiler artifact, prepared
execution, runtime state, or backend state. A future training workflow may coordinate module state
with separately owned optimizer or session state, and a future persistence adapter may encode
materialized values, but neither API exists now.

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

## Planned concepts

- `extensions/nn` currently owns `Parameter` and `Buffer` declarations plus strict in-memory
  module-tree state export/load. Training will consume discovered `Parameter` wrappers through
  their bounded replacement capability, while `ParameterGroup` will describe optimizer-group
  settings.
- `Optimizer` implementations such as SGD, Adam, and AdamW will define mathematical updates without importing CPU, Metal, or CUDA modules.
- `TrainingSession` and `TrainingStep` will coordinate forward/backward execution, gradient publication, and optimizer updates through shared lifecycle contracts.

No optimizer signatures, default hyperparameters, update sequencing, gradient-to-parameter
mapping, persistent checkpoint format, or optimizer exception types are stable yet. They will be
defined by focused extension tasks after model, compiler, runtime, and publication contracts
exist. The current NN-owned replacement and state-dictionary rules do not imply those future
training or persistence contracts.

## Planned public initial flow

```text
compile forward + backward graph
  -> prepare owned partitions
  -> run forward/backward
  -> publish gradients
  -> optimizer.step()
```

The initial optimizer step is backend-agnostic. A later architecture version may compile optimizer updates into the graph, but that direction requires the architecture update described by the contract.

## Boundary example

Adam computes moment estimates and a parameter update from gradients. That mathematical algorithm belongs to training. Choosing a fused Metal kernel for the update belongs to Metal prepare. A `MetalOptimizerBridge` inside training would reverse the required dependency direction and is not a supported design.

## Related contracts

- [Training graph](../architecture/training-graph.md)
- [Autograd guide](../user-guide/autograd.md)
- [Training guide](../user-guide/training.md)
- [Training master plan](../planning/extensions/training/master-plan.md)
