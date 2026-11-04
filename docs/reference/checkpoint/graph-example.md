# Cold [graph](../../glossary.md#graph) [persistence](../../glossary.md#term-persistence) example and failures

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/checkpoint/src/main/java/io/github/pho001/synaptik/checkpoint/graph/GraphCheckpoint.java)

**Cold** means that this workflow rebuilds the description of a computation without running it.
A stored input [prototype](../../glossary.md#term-prototype) specifies what a future input must look like, not an earlier [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)'s
values. Rebinding substitutes a compatible new [Tensor](../../glossary.md#tensor) into that description; numerical execution
is a later, explicit step.

<a id="complete-cold-example"></a>
## Complete cold example

This Java 26 method needs only [Checkpoint](../../glossary.md#term-checkpoint) and its exported [Model](../../glossary.md#term-model) [dependency](../../glossary.md#term-dependency). The caller supplies
a file path in an existing trusted directory. Its [FLOAT32](../../glossary.md#term-float32) [scalar](../../glossary.md#term-scalar) [weight](../../glossary.md#term-weight) is `2`, and the symbolic
output is `input * weight`. It copies state, saves before any execution, loads, then substitutes a
fresh scalar `3`. The returned [Tensor](../../glossary.md#tensor) represents `3 * 2`; it has not been evaluated.

```java
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.checkpoint.graph.GraphCheckpoint;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

Tensor roundtrip(Path file) throws IOException {
    var descriptor = new TensorDescriptor(DataType.FLOAT32, Shape.of(), Optional.empty(), true);
    Tensor input = TensorFactory.create(descriptor);
    Tensor weight = TensorFactory.scalar(2.0f, Optional.empty(), true);
    var source = new GraphCheckpoint.Source(List.of(input), List.of(input.mul(weight)), List.of(
            new GraphCheckpoint.Leaf("weight", GraphCheckpoint.Role.PARAMETER, weight)));
    var acquired = GraphCheckpoint.copy(source);
    var limits = new ArtifactLimits(100_000, 10_000, 10, 32, 10_000, 90_000);
    GraphCheckpoint.save(file, acquired, limits);
    var loaded = GraphCheckpoint.load(file, limits);
    // Match full prototype metadata, including unresolved layout and requiresGrad.
    Tensor fresh = TensorFactory.create(descriptor);
    fresh.replaceHostStorage(TensorFactory.scalar(3.0f, Optional.empty(), true)
            .hostStorage().orElseThrow());
    return GraphCheckpoint.rebind(loaded, List.of(fresh)).getFirst();
}
```

Copy creates storage-free [prototypes](../../glossary.md#term-prototype) and copies each state [identity](../../glossary.md#term-identity-meanings) once, then rebuilds expressions
against the copies; the original weight/storage remain unchanged and are not retained. Load already
owns fresh ordinary [heap](../../glossary.md#term-heap) state, so it needs no second [ownership](../../glossary.md#term-ownership-meanings) copy or caller [arena](../../glossary.md#term-arena). The stored
input is a [descriptor](../../glossary.md#term-descriptor), not the scalar `3` or any previous [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters). [Rebind](../../glossary.md#capture-graph-ports-and-reconstruction) does not inspect state
storage. Produced and repeated replacements keep their upstream [provenance](../../glossary.md#provenance); repeated output
references remain possible and ordinary [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) still rejects duplicate outputs.

To evaluate, explicitly use the existing [Engine](../../glossary.md#term-engine) [API](../../glossary.md#term-application-programming-interface-api) with its own dependency and [lifecycle](../../glossary.md#lifecycle). The
integration tests cover [Linear](../../glossary.md#term-linear-meanings) [MATMUL](../../glossary.md#matrix-multiplication--matmul)/PERMUTE/ADD and each [activation](../../glossary.md#term-activation) with two fresh inputs and
ordinary forward/[autograd](../../glossary.md#autograd) equivalence. Owned [MLP](../../glossary.md#term-multilayer-perceptron-mlp) training additionally compares updates against
an independent Tensor/Engine [oracle](../../glossary.md#term-test-oracle), not just another restored Model.
Replacing the fresh scalar with a [vector](../../glossary.md#term-vector), resolved-[layout](../../glossary.md#layout) scalar or no-[gradient](../../glossary.md#term-gradient) scalar rejects
because the entire input descriptor must match. No automatic recapture/recompile is promised.

<a id="storage-bounds-and-failures"></a>
## Storage, bounds and failures

Named [leaves](../../glossary.md#term-leaf) must be [host](../../glossary.md#term-host)-readable, live and current-thread-accessible, even for empty values.
Present leaf [layouts](../../glossary.md#layout) must exactly equal zero-offset non-[view](../../glossary.md#term-view-meanings) [contiguous](../../glossary.md#term-contiguous) geometry; absent layouts
retain their absence and use checked [canonical](../../glossary.md#term-canonical-form) geometry. Intermediate PERMUTE views are [metadata](../../glossary.md#term-metadata).
Read-only [heap](../../glossary.md#term-heap)/native sources are accepted. Logical [lanes](../../glossary.md#term-lane) are native-order, unaligned integer-width
accesses; [signed zero](../../glossary.md#term-signed-zero) and [NaN payload](../../glossary.md#nan-infinity-and-signed-zero)/sign bits remain exact. [BOOL](../../glossary.md#data-type--datatype) bytes must be 0/1. Excess
capacity is ignored and distinct overlapping [Tensors](../../glossary.md#tensor) are copied independently. Labels/native
[owners](../../glossary.md#term-owner), [batches](../../glossary.md#term-batch-machine-learning-and-hyperparameters), compiled plans, [device](../../glossary.md#term-device) state and Training state are not persisted.

Callers exclude mutation, storage rebinding and closure for each whole [operation](../../glossary.md#operation). Copy/save order
is metadata, checked geometry and save bounds, all host [admission](../../glossary.md#term-admission), BOOL [validation](../../glossary.md#term-validation)/raw copying,
complete result/image, then file [publication](../../glossary.md#term-publication). Save uses the unchanged force/revalidate/single
[atomic](../../glossary.md#resource-transaction-and-rollback)-move policy, not [authentication](../../glossary.md#term-authentication) or portable power-[loss](../../glossary.md#loss) [durability](../../glossary.md#term-durability). A failed move can have
provider-dependent [target](../../glossary.md#term-target) state; inspect the target. No partial source result is returned.

Save/load require explicit `ArtifactLimits`, with no defaults. All six limits and individual
[payload](../../glossary.md#artifact-envelope-and-atomic-publication) ceilings apply before encoding or bulk copies. For [schema](../../glossary.md#term-schema) bytes D, unique state bytes B
and [identity](../../glossary.md#term-identity-meanings) count S: [manifest](../../glossary.md#term-manifest) M=72+58*S, aggregate D+B, file 48+M+D+B, entries 1+S; kind/key
ceiling must be at least 14. These are encoded bounds, not peak-heap promises. Copy/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction) have
no file ceiling; copy still preflights each payload against `Integer.MAX_VALUE - 8`.

Nulls fail in declaration/list order. Invalid source, unsupported geometry, BOOL or bounds raise
`IllegalArgumentException`; checked sizing [overflow](../../glossary.md#term-overflow) raises `ArithmeticException`, retaining existing
[Compiler](../../glossary.md#term-compiler) wrapping. Absent/dead/inaccessible state raises `IllegalStateException`. Rebind checks
source metadata before replacement count/full [descriptors](../../glossary.md#term-descriptor), without a value gate. Load validates
complete [framing](../../glossary.md#term-framing)/wire/keys/descriptors/[aliases](../../glossary.md#aliasing-and-arena)/lengths/BOOL, then descriptor-only Compiler source
[preflight](../../glossary.md#term-preflight), host import, reconstruction and recapture before publication. Stored semantic/overflow
failures become `ArtifactFormatException` with causes; I/O stays `IOException`; Tensor identifier
exhaustion and VM failures propagate.

<a id="related-api"></a>
## Related [API](../../glossary.md#term-application-programming-interface-api)

- [Opaque artifact file API and publication](artifacts.md)<span data-glossary-navigation="true"> ([Opaque glossary](../../glossary.md#term-opaque); [artifact glossary](../../glossary.md#term-artifact); [publication glossary](../../glossary.md#term-publication))</span>
- [Retained NN model checkpoint API](models.md)<span data-glossary-navigation="true"> ([NN model glossary](../../glossary.md#model-module-buffer-and-scan-distinctions); [checkpoint glossary](../../glossary.md#term-checkpoint))</span>
- [Generic graph checkpoint source glossary](../../glossary.md#generic-graph-checkpoint-source)
- [Accepted ownership and migration decision](../../design/decisions/0039-portable-graph-model-persistence.md)<span data-glossary-navigation="true"> ([ownership glossary](../../glossary.md#term-ownership-meanings))</span>

---

[Previous topic](graph-formats.md)
