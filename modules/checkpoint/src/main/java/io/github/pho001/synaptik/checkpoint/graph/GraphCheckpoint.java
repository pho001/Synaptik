package io.github.pho001.synaptik.checkpoint.graph;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFile;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.model.tensor.Tensor;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * Cold persistence and symbolic substitution for a bounded ordinary Tensor source.
 * This facade never executes, selects a backend, owns an arena or retains a cross-call graph.
 * Supported source occurrences are static ADD, MUL, NEG, MATMUL and rank-two PERMUTE[1,0].
 * Named host values use native-order logical lanes; all represented bits are preserved.
 * Callers exclude mutation, storage rebinding and closure throughout each operation.
 * Old flexible NN model readers are separate APIs, not implicitly converted static graphs.
 */
public final class GraphCheckpoint {
    private GraphCheckpoint() { }

    /**
     * Borrowed ordinary expressions and explicit leaf roles, not an executable or second graph IR.
     * Record construction snapshots containers only; operations validate source semantics anew.
     * @param inputs ordered unique provenance-free prototypes, possibly empty or unused
     * @param outputs nonempty distinct ordered source roots
     * @param leaves ordered named reachable non-input leaves; exact identities may have aliases
     */
    public record Source(List<Tensor> inputs, List<Tensor> outputs, List<Leaf> leaves) {
        /**
         * Snapshots each list in declaration order without reading or acquiring Tensor storage.
         * @param inputs non-null prototype list without nulls; borrowed elements
         * @param outputs non-null root list without nulls; borrowed elements
         * @param leaves non-null registration list without nulls; borrowed elements
         * @throws NullPointerException if a list or element is null, in declaration/list order
         */
        public Source {
            inputs = List.copyOf(Objects.requireNonNull(inputs, "inputs"));
            outputs = List.copyOf(Objects.requireNonNull(outputs, "outputs"));
            leaves = List.copyOf(Objects.requireNonNull(leaves, "leaves"));
        }
    }

    /**
     * One exact state registration; aliases share Tensor identity and role, never merely bytes.
     * @param name strict UTF-8 name with nonblank dot segments, no normalization or prefix collision
     * @param role explicit semantic role, not inferred from gradient eligibility
     * @param tensor provenance-free reachable non-input Tensor borrowed without storage ownership
     */
    public record Leaf(String name, Role role, Tensor tensor) {
        /**
         * Checks required components in declaration order; operations validate names and roles.
         * @param name non-null exact registration name
         * @param role non-null explicit state role
         * @param tensor non-null borrowed Tensor
         * @throws NullPointerException if a component is null, in declaration order
         */
        public Leaf {
            Objects.requireNonNull(name, "name");
            Objects.requireNonNull(role, "role");
            Objects.requireNonNull(tensor, "tensor");
        }
    }

    /** Explicit state meaning, independent of storage location and BUFFER/CONSTANT gradients. */
    public enum Role {
        /** Floating gradient-eligible trainable state. */
        PARAMETER,
        /** Persistent non-parameter state, preserving its declared gradient eligibility. */
        BUFFER,
        /** Named fixed-source state, preserving its declared gradient eligibility. */
        CONSTANT
    }

    /**
     * Acquires private mutable heap state once per exact identity and rebuilds ordinary provenance.
     * Input prototypes become storage-free; no original state graph or storage is retained.
     * Metadata and geometry precede all storage admission, then BOOL checks and raw copies.
     * @param source non-null borrowed source, quiescent with live accessible canonical host state
     * @return non-null independent source preserving descriptors, roles, aliases and root order
     * @throws NullPointerException if source is null
     * @throws IllegalArgumentException if source, canonical geometry or JVM payload bounds fail
     * @throws ArithmeticException if checked sizing overflows, unless wrapped by Compiler admission
     * @throws IllegalStateException if state is absent/dead/inaccessible or Tensor IDs are exhausted
     */
    public static Source copy(Source source) {
        GraphSource captured = GraphSource.capture(Objects.requireNonNull(source, "source"));
        return captured.copy();
    }

    /**
     * Saves graph-model.v1 without executing or persisting prototype input storage.
     * Complete metadata/bounds/storage/value/image validation precedes ArtifactFile publication:
     * forced and revalidated sibling file, then one atomic move, with no non-atomic fallback.
     * This is atomic live visibility, not authentication or portable power-loss durability.
     * @param target non-null target in an existing trusted directory, no concurrent writer
     * @param source non-null borrowed source, quiescent throughout capture and publication
     * @param limits non-null six encoded ceilings, not peak-heap bounds; no implicit defaults
     * @throws NullPointerException if an argument is null, checked in declaration order
     * @throws IllegalArgumentException if source, values or any encoded bound is invalid
     * @throws ArithmeticException if checked sizing overflows, retaining Compiler wrapping
     * @throws IllegalStateException if state is absent/dead/inaccessible or IDs are exhausted
     * @throws IOException if file publication or revalidation fails; inspect uncertain move targets
     */
    public static void save(Path target, Source source, ArtifactLimits limits) throws IOException {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(limits, "limits");
        ArtifactFile.save(target, GraphArtifactCodec.encode(GraphSource.capture(source), limits), limits);
    }

    /**
     * Rereads and fully validates a graph file before importing state and publishing expressions.
     * The returned ordinary mutable heap values need no arena or second ownership copy.
     * No Engine, saved batch, labels, compiled artifacts or runtime state is restored.
     * @param source non-null quiescent regular artifact path, not a symlink
     * @param limits non-null explicit encoded ceilings
     * @return non-null detached source with storage-free prototypes and alias-preserving heap state
     * @throws NullPointerException if source or limits is null, in declaration order
     * @throws IOException if I/O, framing, wire, semantic validation or bounds fail; stored semantic
     *     and overflow failures are ArtifactFormatException with their causes
     * @throws IllegalStateException if Tensor identifier space is exhausted
     */
    public static Source load(Path source, ArtifactLimits limits) throws IOException {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(limits, "limits");
        return GraphArtifactCodec.decode(ArtifactFile.read(source, limits), limits);
    }

    /**
     * Reconstructs symbolic outputs with explicit descriptor-compatible input substitutions.
     * Source metadata is checked before replacements; no state storage is accessed. Produced and
     * repeated substitutions retain opaque upstream provenance and may give repeated outputs,
     * which ordinary compile still rejects. Named state remains borrowed and unchanged.
     * @param source non-null source with valid metadata, independent of storage readability
     * @param inputs non-null replacements in declared input order, with exact full descriptors
     * @return non-null immutable ordered ordinary outputs, potentially repeated exact references
     * @throws NullPointerException if source, inputs or a replacement is null
     * @throws IllegalArgumentException if source metadata, replacement arity or descriptors fail
     * @throws IllegalStateException if Tensor identifiers are exhausted
     */
    public static List<Tensor> rebind(Source source, List<Tensor> inputs) {
        return GraphSource.capture(Objects.requireNonNull(source, "source")).rebind(inputs);
    }
}
