# [Unary](../../glossary.md#term-unary) and declarative composition

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../extensions/nn/src/main/java/io/github/pho001/synaptik/nn/module/Model.java)

**Unary** means the complete forward boundary has one [Tensor](../../glossary.md#tensor) input and one Tensor output.
A Sequential passes that output to the next child. **Declarative** composition records a supported
layer structure as data rather than arbitrary Java callbacks. Reconstructing that description
creates the structure and state placeholders, not the trained [parameter](../../glossary.md#term-parameter) values or an executable.

<a id="current-nn-unary-composition-contract"></a>
## Current [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) [unary](../../glossary.md#term-unary) composition contract

[`Module`](../../glossary.md#term-module) remains the general [owner](../../glossary.md#term-owner) of named state, children, and train/[eval mode](../../glossary.md#inference-and-evaluation) and has no
universal forward method. `UnaryTensorModule` is the narrower public subtype for a module whose
complete forward contract accepts one non-null [Tensor](../../glossary.md#tensor) and returns one non-null Tensor. Current
[`Linear`](../../glossary.md#term-linear-meanings), `LayerNorm`, [`Embedding`](../../glossary.md#embedding), `Conv1d`, [`Conv2d`](../../glossary.md#convolution--conv2d), [`Conv3d`](../../glossary.md#conv3d--ncdhw), and closed [`Activation`](../../glossary.md#term-activation)
layers participate.
`BatchNorm` does not, because it
requires an explicit `ForwardContext` and may transition running-[statistic](../../glossary.md#term-statistic) [buffers](../../glossary.md#term-buffer). [`Dropout`](../../glossary.md#dropout--inverted-dropout)
does not, because it requires both an explicit context and caller-threaded [graph](../../glossary.md#graph) random-number-
generator state and returns a result carrying output plus next state.

`Sequential` accepts one `List<? extends UnaryTensorModule>`, snapshots it, and permanently owns
the exact children under decimal names `0`, `1`, and so on. It passes the exact input reference to
child `0`, passes each exact child result to the next child once, and returns the exact final
result. Empty composition returns the exact input. A later failure preserves already constructed
prefix expressions and suppresses the remaining suffix; the container performs no pipeline-wide
[validation](../../glossary.md#term-validation), [rollback](../../glossary.md#term-rollback), caching, flattening, [fusion](../../glossary.md#term-fusion), [compilation](../../glossary.md#compiler-engine-and-execution-stage-names), or execution.

For example, a sequence containing a `Linear` child followed by a `LayerNorm` child exposes state
paths such as `0.weight`, `0.bias`, `1.scale`, and `1.bias` and composes declaratively as:

```text
input reference
  -> child 0 Linear.forward(input)
  -> child 1 LayerNorm.forward(exact child-0 result)
  -> exact child-1 result
```

Inherited recursive mode propagation and state-dictionary export/load follow those numeric child
paths. The sequence does not create a [training session](../../glossary.md#training-session--trainingsession), choose an [optimizer](../../glossary.md#term-optimizer), publish [gradients](../../glossary.md#term-gradient),
or imply that any Tensor expression has been executed.

<a id="current-nn-declarative-composition-contract"></a>
## Current [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) declarative composition contract

`ModelSpecification` is [immutable](../../glossary.md#term-immutable) semantic data for a closed Sequential tree, not a
[callback](../../glossary.md#term-callback), value [snapshot](../../glossary.md#term-snapshot) or [wire format](../../glossary.md#term-wire-format). Its public nested `Step` permits exactly
`Sequence`, `LinearStep`, `LayerNormStep`, `EmbeddingStep`, `Conv1dStep`, `Conv2dStep`,
`Conv3dStep` and `ActivationStep`; `ActivationKind` is RELU/[TANH](../../glossary.md#term-tanh)/[SIGMOID](../../glossary.md#term-sigmoid). `AliasGroup`
and `StateDescriptor` [carry](../../glossary.md#term-carry) [identity](../../glossary.md#term-identity-meanings) paths and [parameter](../../glossary.md#term-parameter) [schema](../../glossary.md#term-schema). Lists are defensively
immutable. The constructor accepts a Sequence root and [canonical](../../glossary.md#term-canonical-form) [alias](../../glossary.md#aliasing-and-arena) groups;
`describe(Sequential)`, `root()`, `aliases()`, `stateDescriptors()` and
`outputDescriptor(TensorDescriptor)` are the complete declared method surface.

`DeclarativeModel` is a final `Model<Tensor, Tensor>` with no public constructor.
`reconstruct(specification)` returns a fully bound [evaluation](../../glossary.md#term-evaluation)-mode [descriptor](../../glossary.md#term-descriptor)-only
skeleton: one storage-/[provenance](../../glossary.md#provenance)-free parameter [Tensor](../../glossary.md#tensor) per alias group or [singleton](../../glossary.md#term-singleton),
no initializer or [weight](../../glossary.md#term-weight)-sized array. It directly owns numeric root children, so
paths such as `0.weight` and `1.0.scale` need no wrapper prefix. Inherited strict load
retains exact candidate references; `describe()` recomputes current identity groups
after legal replacement/load joins or splits. Equal values and physical overlaps do
not imply aliases. The inherited state [API](../../glossary.md#term-application-programming-interface-api) is the only value-installation bridge.

Six parameterized families expose one typed `specification()` snapshot each; public
`Activation(ActivationKind)` adds the same snapshot and [unary](../../glossary.md#term-unary) `forward`. [Linear](../../glossary.md#term-linear-meanings) also
adds `fromState(Tensor weight, Optional<Tensor> bias, boolean automaticInputGuards)`:
the non-null optional is empty for no [bias](../../glossary.md#term-bias). It validates the complete supplied group,
retains exact references and never initializes. `true` retains automatic exact-type,
positive static [feature](../../glossary.md#term-feature), bound-width and Java-array guards; `false` retains supplied
[Model](../../glossary.md#term-model) promotion/[contraction](../../glossary.md#contraction-dimension). Existing constructors and [lazy](../../glossary.md#term-lazy-evaluation) first [binding](../../glossary.md#term-binding) are unchanged.

All four floating parameter types are supported. [Embedding](../../glossary.md#embedding) accepts [INT32](../../glossary.md#data-type--datatype)/[INT64](../../glossary.md#data-type--datatype) at
any [rank](../../glossary.md#term-rank) and appends its width. Linear and LayerNorm retain compatible leading
[Dimensions](../../glossary.md#dimension)/ranks; convolutions retain exact input type, rank-specific channels-first
geometry and compatible variable [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)/[spatial](../../glossary.md#term-spatial-axes) Dimensions. LayerNorm retains exact
[epsilon](../../glossary.md#epsilon) bits/type and Model promotion; incompatible promoted arithmetic fails without
epsilon conversion. `outputDescriptor(input)` and `forward(input)` check the supplied
composition using ordinary Model operations, not a generic solver or a proof of an
admissible input. [Metadata](../../glossary.md#term-metadata) checks do not evaluate values or admit [backend](../../glossary.md#backend) execution.

Nulls fail with `NullPointerException`; invalid schemas, unsupported [modules](../../glossary.md#term-module), aliases or input
composition fail with `IllegalArgumentException`; unbound description fails with
`IllegalStateException`. Checked arithmetic and Tensor-ID
exhaustion retain existing failures. Trees permit 4096 occurrences/depth 64 (root 1).
Caller [quiescence](../../glossary.md#term-quiescence) owns consistency. Training's duplicate/overlap rejection and the
open-session binding-replacement ban remain unchanged. Custom callbacks, contextual
and recurrent modules, persistent encoding and stored-model file reconstruction are
outside this capability. Follow the [complete guide](../../user-guide/declarative-models.md)
for exact paths, canonical group order and the describe → reconstruct → strict-load example.


---

[Previous topic](convolution.md) · [Next topic](recurrent-layers.md)
