# Unary and declarative composition

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)

**Unary** means the complete forward boundary has one Tensor input and one Tensor output.
A Sequential passes that output to the next child. **Declarative** composition records a supported
layer structure as data rather than arbitrary Java callbacks. Reconstructing that description
creates the structure and state placeholders, not the trained parameter values or an executable.

<a id="current-nn-unary-composition-contract"></a>
## Current NN unary composition contract

`Module` remains the general owner of named state, children, and train/eval mode and has no
universal forward method. `UnaryTensorModule` is the narrower public subtype for a module whose
complete forward contract accepts one non-null Tensor and returns one non-null Tensor. Current
`Linear`, `LayerNorm`, `Embedding`, `Conv1d`, `Conv2d`, `Conv3d`, and closed `Activation`
layers participate.
`BatchNorm` does not, because it
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

<a id="current-nn-declarative-composition-contract"></a>
## Current NN declarative composition contract

`ModelSpecification` is immutable semantic data for a closed Sequential tree, not a
callback, value snapshot or wire format. Its public nested `Step` permits exactly
`Sequence`, `LinearStep`, `LayerNormStep`, `EmbeddingStep`, `Conv1dStep`, `Conv2dStep`,
`Conv3dStep` and `ActivationStep`; `ActivationKind` is RELU/TANH/SIGMOID. `AliasGroup`
and `StateDescriptor` carry identity paths and parameter schema. Lists are defensively
immutable. The constructor accepts a Sequence root and canonical alias groups;
`describe(Sequential)`, `root()`, `aliases()`, `stateDescriptors()` and
`outputDescriptor(TensorDescriptor)` are the complete declared method surface.

`DeclarativeModel` is a final `Model<Tensor, Tensor>` with no public constructor.
`reconstruct(specification)` returns a fully bound evaluation-mode descriptor-only
skeleton: one storage-/provenance-free parameter Tensor per alias group or singleton,
no initializer or weight-sized array. It directly owns numeric root children, so
paths such as `0.weight` and `1.0.scale` need no wrapper prefix. Inherited strict load
retains exact candidate references; `describe()` recomputes current identity groups
after legal replacement/load joins or splits. Equal values and physical overlaps do
not imply aliases. The inherited state API is the only value-installation bridge.

Six parameterized families expose one typed `specification()` snapshot each; public
`Activation(ActivationKind)` adds the same snapshot and unary `forward`. Linear also
adds `fromState(Tensor weight, Optional<Tensor> bias, boolean automaticInputGuards)`:
the non-null optional is empty for no bias. It validates the complete supplied group,
retains exact references and never initializes. `true` retains automatic exact-type,
positive static feature, bound-width and Java-array guards; `false` retains supplied
Model promotion/contraction. Existing constructors and lazy first binding are unchanged.

All four floating parameter types are supported. Embedding accepts INT32/INT64 at
any rank and appends its width. Linear and LayerNorm retain compatible leading
Dimensions/ranks; convolutions retain exact input type, rank-specific channels-first
geometry and compatible variable batch/spatial Dimensions. LayerNorm retains exact
epsilon bits/type and Model promotion; incompatible promoted arithmetic fails without
epsilon conversion. `outputDescriptor(input)` and `forward(input)` check the supplied
composition using ordinary Model operations, not a generic solver or a proof of an
admissible input. Metadata checks do not evaluate values or admit backend execution.

Nulls fail with `NullPointerException`; invalid schemas, unsupported modules, aliases or input
composition fail with `IllegalArgumentException`; unbound description fails with
`IllegalStateException`. Checked arithmetic and Tensor-ID
exhaustion retain existing failures. Trees permit 4096 occurrences/depth 64 (root 1).
Caller quiescence owns consistency. Training's duplicate/overlap rejection and the
open-session binding-replacement ban remain unchanged. Custom callbacks, contextual
and recurrent modules, persistent encoding and stored-model file reconstruction are
outside this capability. Follow the [complete guide](../../user-guide/declarative-models.md)
for exact paths, canonical group order and the describe → reconstruct → strict-load example.


---

[Previous topic](convolution.md) · [Next topic](recurrent-layers.md)
