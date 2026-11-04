# Reconstruct a closed neural-network [model](../glossary.md#term-model)

## Outcome

Describe a bound [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) `Sequential`, reconstruct its configuration and shared-[parameter](../glossary.md#term-parameter)
[identities](../glossary.md#term-identity-meanings), then load values through the existing strict state boundary.
A **[descriptor](../glossary.md#term-descriptor)-only skeleton** is a fully bound [model](../glossary.md#term-model) whose parameter [Tensors](../glossary.md#tensor) have
type and [Shape](../glossary.md#shape) [metadata](../glossary.md#term-metadata) but no storage or values. It is a load [target](../glossary.md#term-target), not yet a
value-loaded executable model.

```text
bound Sequential -> immutable specification -> descriptor skeleton -> strict value load
```

The specification records configuration and [identity](../glossary.md#term-identity-meanings) sharing, not parameter values,
initializers, modes, executable plans or persistent bytes.
Read the arrows left to right: reconstruction establishes [bindings](../glossary.md#term-binding) and paths; strict
loading supplies the exact value Tensor references before ordinary execution.

**NN** means [neural network](../glossary.md#neural-network-module-parameter-buffer-and-forward-context): a parameterized composition of numerical operations.
**Sequential** applies its layers in order. A **parameter** is model state that training may
change; a **binding** connects a layer's required state position to a particular Tensor.
Here **closed** means reconstruction accepts a fixed supported inventory of layer kinds, not
that the source is a resource whose `close()` method has [run](../glossary.md#term-run). **Bound** means its state
requirements are established; it does not necessarily [mean](../glossary.md#term-mean) numerical values are present.

## Prerequisites and concrete inputs

Use Java 26 with the [NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) and [Model](../glossary.md#term-model) [modules](../glossary.md#term-module). Keep the source and candidate [quiescent](../glossary.md#term-quiescent):
do not replace [parameters](../glossary.md#term-parameter) or load state concurrently with description or forward
construction. The following complete example uses two [FLOAT32](../glossary.md#term-float32) 2-to-2 projections
with the same [weight](../glossary.md#term-weight) [Tensor](../glossary.md#tensor) and a RELU between them. Source initialization is
explicit; reconstruction does not initialize weights.

A **projection** here maps two input [features](../glossary.md#term-feature) to two output features by a [Linear](../glossary.md#term-linear-meanings) calculation.
An **[activation](../glossary.md#term-activation)** adds a transformation between layers; RELU keeps positive values and replaces
nonpositive values with zero. **Shared parameters** [mean](../glossary.md#term-mean) both paths use the exact same Tensor,
not separate copies that happen to contain equal numbers. “Quiescent” means no concurrent
state changes while the description or expression is being constructed.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.nn.layers.Activation;
import io.github.pho001.synaptik.nn.layers.Linear;
import io.github.pho001.synaptik.nn.module.DeclarativeModel;
import io.github.pho001.synaptik.nn.module.ModelSpecification;
import io.github.pho001.synaptik.nn.module.ModelSpecification.ActivationKind;
import io.github.pho001.synaptik.nn.module.Sequential;
import java.util.List;
import java.util.Optional;
import java.util.Random;

public final class ReconstructExample {
    public static void main(String[] args) {
        Linear first = new Linear(2, 2, false, DataType.FLOAT32, new Random(7));
        Linear second = new Linear(first.weight().value());
        Sequential source = new Sequential(List.of(
                first, new Activation(ActivationKind.RELU), second));

        ModelSpecification specification = ModelSpecification.describe(source);
        DeclarativeModel candidate = DeclarativeModel.reconstruct(specification);
        System.out.println(specification.aliases().getFirst().paths());
        System.out.println(candidate.parametersRecursively().get("0.weight")
                .value().hostStorage().isEmpty());

        candidate.loadStateDictionary(source.stateDictionary());
        System.out.println(candidate.parametersRecursively().get("0.weight").value()
                == candidate.parametersRecursively().get("2.weight").value());
        TensorDescriptor output = specification.outputDescriptor(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(5, 2), Optional.empty(), false));
        System.out.println(output.shape().knownElementCount().orElseThrow());
    }
}
```

Expected output:

```text
[0.weight, 2.weight]
true
true
10
```

Description borrows the source without reparenting it or changing mode. Reconstruction
owns independent layer instances under direct numeric root names and starts in
[evaluation](../glossary.md#term-evaluation) mode. The first `true` proves the skeleton has no weight storage; the
second proves strict loading retained the shared source reference at both paths.
The output [descriptor](../glossary.md#term-descriptor) has [Shape](../glossary.md#shape) `[5, 2]`, hence ten [elements](../glossary.md#term-element), without evaluation.
Changed compatible [batches](../glossary.md#term-batch-machine-learning-and-hyperparameters) and leading ranks are not a saved fixed [signature](../glossary.md#term-signature).

This example's state dictionary is a shallow [binding](../glossary.md#term-binding) [snapshot](../glossary.md#term-snapshot): loading borrows its
exact Tensors, not [detached](../glossary.md#detached-model-results) copies of their values. A [persistence](../glossary.md#term-persistence) [owner](../glossary.md#term-owner) may instead
load fresh value Tensors through the same fully bound [target](../glossary.md#term-target). Once values are loaded,
construct a forward expression and use the ordinary
[compile, prepare and run workflow](running-models.md#compile-and-prepare-once)<span data-glossary-navigation="true"> ([compile glossary](../glossary.md#compile); [prepare glossary](../glossary.md#prepare); [run glossary](../glossary.md#term-run))</span>.

A **shallow snapshot** copies the associations between names and Tensor references, not the
underlying numerical bytes. A **detached snapshot** instead owns independent value bytes.
Reconstruction and strict value loading here are therefore different from acquiring the private
copied state in the [owned learning Model](../tutorials/first-model.md).

## [Pool3d](../glossary.md#pool3d--ncdhw-pooling) in the fluent [Tensor](../glossary.md#tensor) [Model](../glossary.md#term-model) builder

The separate fluent `ModelBuilder` supports `maxPool3d(kd,kh,kw)` and
`averagePool3d(kd,kh,kw)` on [rank](../glossary.md#term-rank)-five [NCDHW](../glossary.md#conv3d--ncdhw) input. [Kernel](../glossary.md#kernel) extents are positive and ordered
depth, height, width; [stride](../glossary.md#term-stride) equals kernel, [padding](../glossary.md#padding) is zero, [dilation](../glossary.md#term-dilation) is one and mode is floor.
These steps delegate to ordinary Model Tensor operations, not new [pooling](../glossary.md#term-pooling) mathematics.
They do not add Pool3d to the closed `ModelSpecification` layer inventory below.

For an existing builder declared with [FLOAT32](../glossary.md#term-float32) volume input `[2,1,4,8,8]`, a conceptual chain is:

```java
builder.conv3d(2, 1, 1, 1)
        .maxPool3d(2, 2, 2)
        .averagePool3d(1, 2, 2)
        .flatten()
        .linear(3);
```

The pools produce `[2,2,2,4,4]` then `[2,2,2,2,2]`; flatten produces `[2,16]` and [Linear](../glossary.md#term-linear-meanings)
produces `[2,3]`, preserving [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters) rows. Qualified static [CPU](../glossary.md#term-cpu) FLOAT32/[FLOAT64](../glossary.md#term-float64) and [Metal](../glossary.md#term-metal) FLOAT32
support first-order training. Nonpositive [kernels](../glossary.md#kernel) or incompatible input rank fail Model guards.
Save/load uses existing [owned Model](../glossary.md#model-module-buffer-and-scan-distinctions) APIs and `graph-model.v6` when Pool3d occurs, even only
inside an otherwise-qualified [Scan body](../glossary.md#backpropagation-through-time-and-recurrent-state). Load restores values, not [optimizer](../glossary.md#term-optimizer)/session history;
configure training anew. Direct Tensor cold sources additionally retain full typed Pool3d
stride, symmetric padding, dilation and floor/ceil geometry beyond these builder defaults.

## Compatibility, sharing and failures

Supported steps are nested Sequential, [Linear](../glossary.md#term-linear-meanings), affine LayerNorm, [Embedding](../glossary.md#embedding),
Conv1d/2d/3d and closed RELU/[TANH](../glossary.md#term-tanh)/[SIGMOID](../glossary.md#term-sigmoid) [activations](../glossary.md#term-activation). [Parameter](../glossary.md#term-parameter) types are [FLOAT64](../glossary.md#term-float64),
[FLOAT32](../glossary.md#term-float32), [BFLOAT16](../glossary.md#term-bfloat16) and [FLOAT16](../glossary.md#term-float16); Embedding accepts [INT32](../glossary.md#data-type--datatype) or [INT64](../glossary.md#data-type--datatype) indices of any [rank](../glossary.md#term-rank).
Each layer retains its ordinary [operation](../glossary.md#operation) semantics and full attributes.
Linear preserves compatible leading [Dimensions](../glossary.md#dimension) and ranks; LayerNorm preserves compatible
leading Dimensions and its normalized trailing [Shape](../glossary.md#shape). Embedding appends the table width
to the index Shape and returns the table type. Convolutions retain [NCW](../glossary.md#term-ncw)/[NCHW](../glossary.md#term-nchw)/[NCDHW](../glossary.md#conv3d--ncdhw) ranks
3/4/5, positive static matching channels, and compatible variable [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters)/[spatial](../glossary.md#term-spatial-axes) Dimensions.
[Stride](../glossary.md#term-stride), symmetric [padding](../glossary.md#padding), [dilation](../glossary.md#term-dilation), groups and W/HW/DHW [kernel](../glossary.md#kernel) order are retained.

Supplied Linear preserves [Model](../glossary.md#term-model) promotion; automatic Linear and convolutions require
the exact configured input type. LayerNorm retains exact [epsilon](../glossary.md#epsilon) type and bits:
FLOAT16 parameters use FLOAT32 epsilon, otherwise epsilon matches the parameter type.
Input promotion can therefore make a retained epsilon incompatible and fail.
No implicit input cast or epsilon conversion occurs.
`Linear.fromState(weight, Optional<Tensor> bias, automaticInputGuards)` binds supplied
references without initialization: `true` preserves automatic exact-type, positive static
[feature](../glossary.md#term-feature), bound-width and Java-array-count guards; `false` preserves supplied-state
promotion/[contraction](../glossary.md#contraction-dimension) semantics. [Eager](../glossary.md#term-eager-evaluation) initialized Linear is not the automatic form.

State paths have no `root` or `model` prefix. A Linear followed by a nested LayerNorm
uses `0.weight`, optional `0.bias`, then `1.0.scale`, `1.0.bias`. State [descriptors](../glossary.md#term-descriptor)
follow numeric [Module](../glossary.md#term-module) traversal, not lexical sorting: LayerNorm declares scale then [bias](../glossary.md#term-bias);
the other parameterized families declare [weight](../glossary.md#term-weight) then optional bias. They are parameter-only,
fully static and [gradient](../glossary.md#term-gradient)-eligible. `outputDescriptor(input)` checks composition for that
specific input, not whether some input exists or every input is accepted; its storage-free
Model [metadata](../glossary.md#term-metadata) operations may consume [Tensor](../glossary.md#tensor) identifiers but neither evaluate values nor
choose a [backend](../glossary.md#backend). Empty roots return the exact input descriptor.

[Alias](../glossary.md#aliasing-and-arena) groups [mean](../glossary.md#term-mean) exact Tensor [identity](../glossary.md#term-reference-identity), not equal values or overlapping storage.
Description recomputes them after legal parameter replacement or strict loading,
including joins and splits. Training's duplicate/overlapping-parameter rejection
is unchanged, and state replacement remains forbidden during an open session.

Unbound layers fail description with `IllegalStateException`; bind the complete
state first rather than asking description to [run](../glossary.md#term-run) forward. Custom modules and
incompatible inputs fail with `IllegalArgumentException`. [Canonical](../glossary.md#term-canonical-form) alias groups
must contain distinct existing [schema](../glossary.md#term-schema)-matching paths in state traversal order,
with disjoint memberships. Trees are limited to 4096 occurrences including root
and depth 64 with root depth one; reused records still count as separate occurrences.
Groups themselves follow first-member order. Invalid schemas or noncanonical groups fail
before a specification is returned; null [arguments](../glossary.md#term-argument) fail with `NullPointerException`.
Checked arithmetic [overflow](../glossary.md#term-overflow) and Tensor identifier exhaustion retain their existing failures.
Tree limits protect cold description/reconstruction, not [runtime](../glossary.md#term-runtime) shape sizes or backend capacity.

## Boundaries

No arbitrary `Model.define` [capture](../glossary.md#term-capture-meanings), branches, recurrence, BatchNorm, [Dropout](../glossary.md#dropout--inverted-dropout),
wire [codec](../glossary.md#term-codec), file I/O, [optimizer state](../glossary.md#term-optimizer-state) or [runtime](../glossary.md#term-runtime) runner is provided. [Metadata](../glossary.md#term-metadata) checks
do not admit a [backend route](../glossary.md#backend-route) or widen the
[qualified Conv3d training domain](../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary)<span data-glossary-navigation="true"> ([Conv3d glossary](../glossary.md#conv3d--ncdhw))</span>.
Empty trees return the exact input semantically, without a generic zero-[operation](../glossary.md#operation) [Engine](../glossary.md#term-engine) guarantee.
[NN](../glossary.md#neural-network-module-parameter-buffer-and-forward-context) reconstruction itself still has no codec or file I/O. Downstream [Checkpoint](../glossary.md#term-checkpoint) now offers
[whole-model save and automatic load](../reference/checkpoint/models.md#whole-model-save-and-automatic-load)
for this same closed [matrix](../glossary.md#term-matrix-meanings): it stores [canonical](../glossary.md#term-canonical-form) configuration, exact [identity](../glossary.md#term-identity-meanings) groups and [detached](../glossary.md#detached-model-results)
values, then returns an independent [heap](../glossary.md#term-heap)-backed [evaluation](../glossary.md#term-evaluation) `DeclarativeModel`. Its load requires
neither an Engine, [Arena](../glossary.md#term-arena), manual skeleton nor caller configuration [digest](../glossary.md#term-digest). This differs from the
unchanged values-only [API](../glossary.md#term-application-programming-interface-api), which needs a caller-provided fully bound [target](../glossary.md#term-target) and expected digests.
Use the [stored-model inference example](running-models.md#load-a-saved-whole-model)<span data-glossary-navigation="true"> ([inference glossary](../glossary.md#term-inference))</span> for that workflow;
the shallow state-dictionary example above deliberately does not detach source values.

See the [NN declarative contract](../architecture/contracts/extensions-training.md#closed-declarative-models)
for the owning boundary, the [specification/skeleton glossary](../glossary.md#declarative-model-specification-and-descriptor-only-skeleton)
and the [module topology glossary](../glossary.md#functional-model-and-model-topology)
for the distinction between module [ownership](../glossary.md#term-ownership-meanings) and a [Tensor](../glossary.md#tensor) computation [graph](../glossary.md#graph).

The retained old file [facade](../glossary.md#term-facade) is now `io.github.pho001.synaptik.nn.checkpoint.ModelCheckpoint`;
use an explicit NN [dependency](../glossary.md#term-dependency). `model.v1` and `whole-model.v1` keep their flexible declarative
behavior and readers. The [generic graph API](../reference/checkpoint/graph.md) is a separate cold
static-[signature](../glossary.md#term-signature) capability, not an automatic conversion of these models. The separate
[owned Model workflow](running-models.md#owned-static-model-workflow)<span data-glossary-navigation="true"> ([owned Model glossary](../glossary.md#model-module-buffer-and-scan-distinctions))</span> adds defineTensor/graph and
load/loadGraph for the five-operation source subset. Generic and DeclarativeModel forward remain
flexible; their new tensor methods are unsupported and close is a no-op. To opt in explicitly,
construct [prototype](../glossary.md#term-prototype) outputs and name their reachable [leaves](../glossary.md#term-leaf) for [Model](../glossary.md#term-model).graph. Do not reinterpret
an old whole-model.v1 file as a new static graph [artifact](../glossary.md#term-artifact).
