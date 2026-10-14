package io.github.pho001.synaptik.checkpoint.model;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFile;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.nn.module.Module;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Objects;

/**
 * Public version-one model-state persistence boundary. The caller recreates the complete NN
 * module and supplies fingerprints of its versioned configuration and optional input artifact;
 * Java model code, optimizer state, and training continuation are not persisted. Save and load
 * require caller-owned quiescence of module bindings and storage for their entire calls. This
 * facade uses only public Engine materialization and delegates installation to NN's strict
 * whole-tree loader. It has no cross-call cache or live resource after return.
 */
public final class ModelCheckpoint {
    private ModelCheckpoint() { }

    /**
     * Materializes one complete state snapshot and publishes its v1 artifact through a forced,
     * revalidated sibling file and one atomic move. A pre-move failure leaves the prior target
     * untouched; a failed move has provider-dependent target state and must be inspected. A
     * successful move gives atomic live visibility, not portable rename power-loss durability.
     * Fully bound leaf state, including loaded heap state on the supported CPU route, requires no
     * caller expression wrappers. After complete metadata and bounds preflight, capture internally
     * selects raw-preserving contiguous movement for leaves and retains existing expressions for
     * one bounded public Engine computation. Save does not replace source bindings or descriptors;
     * every state path retains its own payload even when materialization shares an original Tensor.
     * This does not guarantee generic leaf-output support in Engine or expand backend capability.
     * The aggregate payload ceiling counts schema plus all paths, including aliases, not transient
     * Engine workspace, defensive copies, or peak heap. Borrowed source storage must remain live,
     * accessible, and free from conflicting mutation until this synchronous call returns.
     *
     * @param target non-null path in an existing trusted local directory without concurrent writers
     * @param module non-null fully bound quiescent NN tree to snapshot once
     * @param engine non-null open Engine used for bounded host materialization; unused for empty state
     * @param configDigest non-null exact 32-byte SHA-256 of caller-defined versioned configuration
     * @param inputDigest optional exact 32-byte input-artifact SHA-256, or null for absent
     * @param limits non-null file, schema, entry, and aggregate payload ceilings
     * @throws IOException if publication or file validation fails
     * @throws IllegalArgumentException if fingerprints, state metadata, or bounds are invalid
     * @throws IllegalStateException if state is unbound or Engine materialization fails
     * @throws ArithmeticException if checked Shape or artifact-size arithmetic overflows
     * @throws NullPointerException if a required argument is null
     */
    public static void save(Path target, Module module, Engine engine, byte[] configDigest,
                            byte[] inputDigest, ArtifactLimits limits) throws IOException {
        Objects.requireNonNull(target, "target");
        ModelArtifactCandidate candidate = ModelArtifactCapture.capture(
                module, engine, configDigest, inputDigest, limits);
        // Do not publish a payload that a subsequent typed inspection would reject.
        ModelArtifactCodec.decode(candidate.envelope(), limits);
        ArtifactFile.save(target, candidate.envelope(), limits);
    }

    /**
     * Reads and validates the complete bounded artifact, including model schema, exact payload
     * correspondence, checksums, and canonical BOOL bytes, before returning detached metadata.
     * Its result remains usable after the file changes; a later load independently rereads it.
     *
     * @param source non-null path to a quiescent regular file, not a symbolic link
     * @param limits non-null resource ceilings for the complete file and model payloads
     * @return immutable typed inspection with defensive digests and ordered descriptors
     * @throws IOException if the file cannot be read or is malformed, unsupported, or over limit
     * @throws NullPointerException if an argument is null
     */
    public static ModelCheckpointInspection inspect(Path source, ArtifactLimits limits)
            throws IOException {
        ModelArtifactCandidate candidate = read(source, limits);
        return new ModelCheckpointInspection(candidate.configDigest(), candidate.inputDigest(),
                candidate.schemaFingerprint(), candidate.states().stream()
                .map(state -> new ModelCheckpointInspection.State(state.path(), state.kind(),
                        state.type(), state.shape(), state.length())).toList());
    }

    /**
     * Rereads a complete v1 artifact, checks exact expected digest presence and bytes, and
     * reconstructs native-order heap-backed dense leaf Tensors before one NN strict installation
     * attempt. A complete target snapshot is required, so lazy/unbound groups are rejected even
     * though NN's loader can initialize them. Target Tensor identity defines alias groups: each
     * group's stored path values must be byte-identical, then share one reconstructed Tensor.
     * Ordinary validation, alias, or reconstruction failure leaves all target bindings unchanged.
     * This cannot verify source alias identity, which v1 does not store; the caller's versioned
     * configuration fingerprint must cover intended sharing.
     *
     * @param source non-null path to a quiescent regular v1 artifact
     * @param target non-null fully bound quiescent recreated NN tree
     * @param expectedConfigDigest non-null exact 32-byte expected configuration SHA-256
     * @param expectedInputDigest optional exact 32-byte expected input-artifact SHA-256, or null
     * @param limits non-null complete artifact and model-payload ceilings
     * @throws IOException if the file is inaccessible, corrupt, malformed, or over limit
     * @throws IllegalArgumentException if digests, paths, descriptors, or alias values mismatch
     * @throws IllegalStateException if target state is unbound or Tensor identifiers are exhausted
     * @throws NullPointerException if a required argument is null
     */
    public static void load(Path source, Module target, byte[] expectedConfigDigest,
                            byte[] expectedInputDigest, ArtifactLimits limits) throws IOException {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(expectedConfigDigest, "expectedConfigDigest");
        if (expectedConfigDigest.length != 32 || expectedInputDigest != null
                && expectedInputDigest.length != 32)
            throw new IllegalArgumentException("SHA-256 digest must be 32 bytes");
        ModelArtifactCandidate candidate = read(source, limits);
        if (!Arrays.equals(expectedConfigDigest, candidate.configDigest())
                || !Arrays.equals(expectedInputDigest, candidate.inputDigest()))
            throw new IllegalArgumentException("Model checkpoint fingerprint mismatch");
        // NN's loader permits reserved lazy groups; v1 explicitly requires a complete target.
        var current = target.stateDictionary();
        target.loadStateDictionary(ModelStateRestoration.reconstruct(candidate, current));
    }

    private static ModelArtifactCandidate read(Path source, ArtifactLimits limits) throws IOException {
        return ModelArtifactCodec.decode(ArtifactFile.read(
                Objects.requireNonNull(source, "source"), Objects.requireNonNull(limits, "limits")), limits);
    }
}
