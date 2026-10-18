# Generic graph checkpoint API

## Purpose and mental model

`GraphCheckpoint` saves a bounded ordinary Tensor expression and named host values without
executing it. Copy/load return ordinary symbolic Tensors and private mutable heap state; rebind
substitutes inputs symbolically. None creates an Engine, prepared execution or new Model hierarchy.
The [owning contract](../architecture/contracts/extensions-training.md#generic-graph-source-and-values)
defines the capability and [fixed wire](../architecture/contracts/extensions-training.md#canonical-graph-model-version-1).

```text
borrowed Source -> copy -> private state + reconstructed ordinary provenance
borrowed Source -> save -> graph-model.v1 -> load -> private heap state + ordinary provenance
Source + replacement inputs -> rebind -> ordinary outputs -> explicit caller Engine
```

The arrows distinguish cold acquisition/persistence/substitution from explicit execution.
Use only `:extensions:checkpoint` (physically `modules/checkpoint`); it exports Model, not Compiler,
NN or Engine. Old `nn.checkpoint.ModelCheckpoint` lives in NN and retains flexible declarative
`model.v1`/`whole-model.v1` readers, defaults and failures. Those files are not silently converted.

## Inputs, outputs and identity

`Source(inputs, outputs, leaves)` snapshots non-null lists in declaration order, borrowing Tensors.
Inputs are ordered unique provenance-free prototypes, possibly empty or unused. Their storage is
ignored and never saved. Outputs are nonempty and distinct. Every reachable non-input leaf must
have a `Leaf(name, role, tensor)` registration; unreachable named identities reject.

Roles are explicit: PARAMETER requires floating `requiresGrad=true`; BUFFER/CONSTANT retain their
declared gradient eligibility. Multiple names may alias one exact Tensor with the same role.
Equal values or physically overlapping distinct Tensors never establish aliases. Names are strict
UTF-8 nonblank dot segments, without normalization, duplicates or prefix collisions. Registration
order is retained independently of first source encounter and input signature order.

Source admission is static FORWARD ADD/MUL/NEG/MATMUL and rank-two PERMUTE[1,0], with exact full
descriptors, sharing and occurrences. All seven raw types persist where Model permits them;
integral/BOOL NEG is not supported. No optimizations, layout closure or implicit family extensions
occur. Zero-node graphs are valid data, not a new Engine execution promise. No backend rank-16
ceiling is imposed by persistence.

## Complete cold example

This Java 26 method needs only Checkpoint and its exported Model dependency. The caller supplies
a file path in an existing trusted directory. Its FLOAT32 scalar weight is `2`, and the symbolic
output is `input * weight`. It copies state, saves before any execution, loads, then substitutes a
fresh scalar `3`. The returned Tensor represents `3 * 2`; it has not been evaluated.

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

Copy creates storage-free prototypes and copies each state identity once, then rebuilds expressions
against the copies; the original weight/storage remain unchanged and are not retained. Load already
owns fresh ordinary heap state, so it needs no second ownership copy or caller arena. The stored
input is a descriptor, not the scalar `3` or any previous batch. Rebind does not inspect state
storage. Produced and repeated replacements keep their upstream provenance; repeated output
references remain possible and ordinary compilation still rejects duplicate outputs.

To evaluate, explicitly use the existing Engine API with its own dependency and lifecycle. The
integration test covers Linear MATMUL/PERMUTE/ADD, two fresh inputs and ordinary autograd equivalence.
Replacing the fresh scalar with a vector, resolved-layout scalar or no-gradient scalar rejects
because the entire input descriptor must match. No automatic recapture/recompile is promised.

## Storage, bounds and failures

Named leaves must be host-readable, live and current-thread-accessible, even for empty values.
Present leaf layouts must exactly equal zero-offset non-view contiguous geometry; absent layouts
retain their absence and use checked canonical geometry. Intermediate PERMUTE views are metadata.
Read-only heap/native sources are accepted. Logical lanes are native-order, unaligned integer-width
accesses; signed zero and NaN payload/sign bits remain exact. BOOL bytes must be 0/1. Excess
capacity is ignored and distinct overlapping Tensors are copied independently. Labels/native
owners, batches, compiled plans, device state and Training state are not persisted.

Callers exclude mutation, storage rebinding and closure for each whole operation. Copy/save order
is metadata, checked geometry and save bounds, all host admission, BOOL validation/raw copying,
complete result/image, then file publication. Save uses the unchanged force/revalidate/single
atomic-move policy, not authentication or portable power-loss durability. A failed move can have
provider-dependent target state; inspect the target. No partial source result is returned.

Save/load require explicit `ArtifactLimits`, with no defaults. All six limits and individual
payload ceilings apply before encoding or bulk copies. For schema bytes D, unique state bytes B
and identity count S: manifest M=72+58*S, aggregate D+B, file 48+M+D+B, entries 1+S; kind/key
ceiling must be at least 14. These are encoded bounds, not peak-heap promises. Copy/rebind have
no file ceiling; copy still preflights each payload against `Integer.MAX_VALUE - 8`.

Nulls fail in declaration/list order. Invalid source, unsupported geometry, BOOL or bounds raise
`IllegalArgumentException`; checked sizing overflow raises `ArithmeticException`, retaining existing
Compiler wrapping. Absent/dead/inaccessible state raises `IllegalStateException`. Rebind checks
source metadata before replacement count/full descriptors, without a value gate. Load validates
complete framing/wire/keys/descriptors/aliases/lengths/BOOL, then descriptor-only Compiler source
preflight, host import, reconstruction and recapture before publication. Stored semantic/overflow
failures become `ArtifactFormatException` with causes; I/O stays `IOException`; Tensor identifier
exhaustion and VM failures propagate.

## Related API

- [Opaque artifact file API and publication](checkpoint-artifact-api.md)
- [Retained NN model checkpoint API](model-checkpoint-api.md)
- [Generic graph checkpoint source glossary](../glossary.md#generic-graph-checkpoint-source)
- [Accepted ownership and migration decision](../design/decisions/0039-portable-graph-model-persistence.md)
