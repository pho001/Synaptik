# Reconstruct a closed neural-network model

## Outcome

Describe a bound NN `Sequential`, reconstruct its configuration and shared-parameter
identities, then load values through the existing strict state boundary.
A **descriptor-only skeleton** is a fully bound model whose parameter Tensors have
type and Shape metadata but no storage or values. It is a load target, not yet a
value-loaded executable model.

```text
bound Sequential -> immutable specification -> descriptor skeleton -> strict value load
```

The specification records configuration and identity sharing, not parameter values,
initializers, modes, executable plans or persistent bytes.
Read the arrows left to right: reconstruction establishes bindings and paths; strict
loading supplies the exact value Tensor references before ordinary execution.

**NN** means neural network: a parameterized composition of numerical operations.
**Sequential** applies its layers in order. A **parameter** is model state that training may
change; a **binding** connects a layer's required state position to a particular Tensor.
Here **closed** means reconstruction accepts a fixed supported inventory of layer kinds, not
that the source is a resource whose `close()` method has run. **Bound** means its state
requirements are established; it does not necessarily mean numerical values are present.

## Prerequisites and concrete inputs

Use Java 26 with the NN and Model modules. Keep the source and candidate quiescent:
do not replace parameters or load state concurrently with description or forward
construction. The following complete example uses two FLOAT32 2-to-2 projections
with the same weight Tensor and a RELU between them. Source initialization is
explicit; reconstruction does not initialize weights.

A **projection** here maps two input features to two output features by a Linear calculation.
An **activation** adds a transformation between layers; RELU keeps positive values and replaces
nonpositive values with zero. **Shared parameters** mean both paths use the exact same Tensor,
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
evaluation mode. The first `true` proves the skeleton has no weight storage; the
second proves strict loading retained the shared source reference at both paths.
The output descriptor has Shape `[5, 2]`, hence ten elements, without evaluation.
Changed compatible batches and leading ranks are not a saved fixed signature.

This example's state dictionary is a shallow binding snapshot: loading borrows its
exact Tensors, not detached copies of their values. A persistence owner may instead
load fresh value Tensors through the same fully bound target. Once values are loaded,
construct a forward expression and use the ordinary
[compile, prepare and run workflow](running-models.md#compile-and-prepare-once).

A **shallow snapshot** copies the associations between names and Tensor references, not the
underlying numerical bytes. A **detached snapshot** instead owns independent value bytes.
Reconstruction and strict value loading here are therefore different from acquiring the private
copied state in the [owned learning Model](../tutorials/first-model.md).

## Pool3d in the fluent Tensor Model builder

The separate fluent `ModelBuilder` supports `maxPool3d(kd,kh,kw)` and
`averagePool3d(kd,kh,kw)` on rank-five NCDHW input. Kernel extents are positive and ordered
depth, height, width; stride equals kernel, padding is zero, dilation is one and mode is floor.
These steps delegate to ordinary Model Tensor operations, not new pooling mathematics.
They do not add Pool3d to the closed `ModelSpecification` layer inventory below.

For an existing builder declared with FLOAT32 volume input `[2,1,4,8,8]`, a conceptual chain is:

```java
builder.conv3d(2, 1, 1, 1)
        .maxPool3d(2, 2, 2)
        .averagePool3d(1, 2, 2)
        .flatten()
        .linear(3);
```

The pools produce `[2,2,2,4,4]` then `[2,2,2,2,2]`; flatten produces `[2,16]` and Linear
produces `[2,3]`, preserving batch rows. Qualified static CPU FLOAT32/FLOAT64 and Metal FLOAT32
support first-order training. Nonpositive kernels or incompatible input rank fail Model guards.
Save/load uses existing owned Model APIs and `graph-model.v6` when Pool3d occurs, even only
inside an otherwise-qualified Scan body. Load restores values, not optimizer/session history;
configure training anew. Direct Tensor cold sources additionally retain full typed Pool3d
stride, symmetric padding, dilation and floor/ceil geometry beyond these builder defaults.

## Compatibility, sharing and failures

Supported steps are nested Sequential, Linear, affine LayerNorm, Embedding,
Conv1d/2d/3d and closed RELU/TANH/SIGMOID activations. Parameter types are FLOAT64,
FLOAT32, BFLOAT16 and FLOAT16; Embedding accepts INT32 or INT64 indices of any rank.
Each layer retains its ordinary operation semantics and full attributes.
Linear preserves compatible leading Dimensions and ranks; LayerNorm preserves compatible
leading Dimensions and its normalized trailing Shape. Embedding appends the table width
to the index Shape and returns the table type. Convolutions retain NCW/NCHW/NCDHW ranks
3/4/5, positive static matching channels, and compatible variable batch/spatial Dimensions.
Stride, symmetric padding, dilation, groups and W/HW/DHW kernel order are retained.

Supplied Linear preserves Model promotion; automatic Linear and convolutions require
the exact configured input type. LayerNorm retains exact epsilon type and bits:
FLOAT16 parameters use FLOAT32 epsilon, otherwise epsilon matches the parameter type.
Input promotion can therefore make a retained epsilon incompatible and fail.
No implicit input cast or epsilon conversion occurs.
`Linear.fromState(weight, Optional<Tensor> bias, automaticInputGuards)` binds supplied
references without initialization: `true` preserves automatic exact-type, positive static
feature, bound-width and Java-array-count guards; `false` preserves supplied-state
promotion/contraction semantics. Eager initialized Linear is not the automatic form.

State paths have no `root` or `model` prefix. A Linear followed by a nested LayerNorm
uses `0.weight`, optional `0.bias`, then `1.0.scale`, `1.0.bias`. State descriptors
follow numeric Module traversal, not lexical sorting: LayerNorm declares scale then bias;
the other parameterized families declare weight then optional bias. They are parameter-only,
fully static and gradient-eligible. `outputDescriptor(input)` checks composition for that
specific input, not whether some input exists or every input is accepted; its storage-free
Model metadata operations may consume Tensor identifiers but neither evaluate values nor
choose a backend. Empty roots return the exact input descriptor.

Alias groups mean exact Tensor identity, not equal values or overlapping storage.
Description recomputes them after legal parameter replacement or strict loading,
including joins and splits. Training's duplicate/overlapping-parameter rejection
is unchanged, and state replacement remains forbidden during an open session.

Unbound layers fail description with `IllegalStateException`; bind the complete
state first rather than asking description to run forward. Custom modules and
incompatible inputs fail with `IllegalArgumentException`. Canonical alias groups
must contain distinct existing schema-matching paths in state traversal order,
with disjoint memberships. Trees are limited to 4096 occurrences including root
and depth 64 with root depth one; reused records still count as separate occurrences.
Groups themselves follow first-member order. Invalid schemas or noncanonical groups fail
before a specification is returned; null arguments fail with `NullPointerException`.
Checked arithmetic overflow and Tensor identifier exhaustion retain their existing failures.
Tree limits protect cold description/reconstruction, not runtime shape sizes or backend capacity.

## Boundaries

No arbitrary `Model.define` capture, branches, recurrence, BatchNorm, Dropout,
wire codec, file I/O, optimizer state or runtime runner is provided. Metadata checks
do not admit a backend route or widen the
[qualified Conv3d training domain](../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary).
Empty trees return the exact input semantically, without a generic zero-operation Engine guarantee.
NN reconstruction itself still has no codec or file I/O. Downstream Checkpoint now offers
[whole-model save and automatic load](../reference/checkpoint/models.md#whole-model-save-and-automatic-load)
for this same closed matrix: it stores canonical configuration, exact identity groups and detached
values, then returns an independent heap-backed evaluation `DeclarativeModel`. Its load requires
neither an Engine, Arena, manual skeleton nor caller configuration digest. This differs from the
unchanged values-only API, which needs a caller-provided fully bound target and expected digests.
Use the [stored-model inference example](running-models.md#load-a-saved-whole-model) for that workflow;
the shallow state-dictionary example above deliberately does not detach source values.

See the [NN declarative contract](../architecture/contracts/extensions-training.md#closed-declarative-models)
for the owning boundary, the [specification/skeleton glossary](../glossary.md#declarative-model-specification-and-descriptor-only-skeleton)
and the [module topology glossary](../glossary.md#functional-model-and-model-topology)
for the distinction between module ownership and a Tensor computation graph.

The retained old file facade is now `io.github.pho001.synaptik.nn.checkpoint.ModelCheckpoint`;
use an explicit NN dependency. `model.v1` and `whole-model.v1` keep their flexible declarative
behavior and readers. The [generic graph API](../reference/checkpoint/graph.md) is a separate cold
static-signature capability, not an automatic conversion of these models. The separate
[owned Model workflow](running-models.md#owned-static-model-workflow) adds defineTensor/graph and
load/loadGraph for the five-operation source subset. Generic and DeclarativeModel forward remain
flexible; their new tensor methods are unsupported and close is a no-op. To opt in explicitly,
construct prototype outputs and name their reachable leaves for Model.graph. Do not reinterpret
an old whole-model.v1 file as a new static graph artifact.
