package io.github.pho001.synaptik.checkpoint.model;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFile;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.nn.module.Module;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Public version-one model-state persistence boundary. The caller recreates the complete NN
 * module and supplies fingerprints of its versioned configuration and optional input artifact;
 * Java model code, optimizer state, and training continuation are not persisted. Capture, save,
 * and load require caller-owned quiescence of module bindings and storage for their entire calls. This
 * facade uses only public Engine materialization and delegates installation to NN's strict
 * whole-tree loader. It has no cross-call cache or live resource after return.
 */
public final class ModelCheckpoint {
    private ModelCheckpoint() { }

    /**
     * Captures one complete neural-network (NN) state snapshot as a detached, semantically
     * validated {@code model.v1} envelope, without file I/O or changing source bindings.
     * After metadata and bounds preflight, one bounded public Engine computation materializes
     * unique Tensor identities; empty state makes no Engine call. Each path retains its own
     * exact canonical logical bits, including signed zeros and NaN sign/payload bits. Fully bound
     * leaves need no caller expression wrappers on supported routes; capture does not widen
     * Engine or backend capability. Borrowed storage must remain live, accessible, and quiescent
     * throughout this synchronous call. The result survives source Arena and Engine closure.
     * Aggregate limits count schema plus every aliased path, not workspace, copies, or peak heap.
     *
     * @param module non-null fully bound NN tree, quiescent for the entire call
     * @param engine non-null Engine, open when materialization is needed; unused for empty state
     * @param configDigest non-null exact 32-byte SHA-256 of caller-defined versioned configuration
     * @param inputDigest optional exact 32-byte input-artifact SHA-256, or null for absent
     * @param limits non-null complete file-equivalent, manifest, entry, name, and payload ceilings
     * @return fully validated immutable envelope with no live Tensor, Engine, or storage handles
     * @throws ArtifactFormatException if final semantic decode rejects the captured payloads
     * @throws IllegalArgumentException if fingerprints, metadata, or preflight bounds are invalid
     * @throws IllegalStateException if state is unbound or materialization violates its contract
     * @throws ArithmeticException if checked Shape or artifact-size arithmetic overflows
     * @throws NullPointerException if a required argument is null
     * @throws RuntimeException if Engine computation or copying reports another unchecked failure
     */
    public static ArtifactEnvelope capture(Module module, Engine engine, byte[] configDigest,
                                           byte[] inputDigest, ArtifactLimits limits)
            throws ArtifactFormatException {
        ModelArtifactCandidate candidate = ModelArtifactCapture.capture(
                module, engine, configDigest, inputDigest, limits);
        // Do not return a payload that a subsequent typed inspection would reject.
        return ModelArtifactCodec.decode(candidate.envelope(), limits).envelope();
    }

    /**
     * Independently validates a detached model envelope and returns its ordered descriptors and
     * defensive digests. Validation includes kind, schema, exact ordinal payload correspondence,
     * canonical BOOL bytes, and every artifact limit, including manifest and file-equivalent size.
     * No encoded file checksum exists to check here. The result carries no resource handle or
     * trusted-validation credential: byte access and load validate their supplied envelope anew.
     *
     * @param envelope non-null detached opaque envelope; never modified
     * @param limits non-null complete file-equivalent and model-payload ceilings
     * @return immutable detached typed inspection in model metadata order, never null
     * @throws ArtifactFormatException if the model is malformed, unsupported, or over any limit
     * @throws NullPointerException if an argument is null
     */
    public static ModelCheckpointInspection inspect(ArtifactEnvelope envelope, ArtifactLimits limits)
            throws ArtifactFormatException {
        return inspection(ModelArtifactCodec.decode(envelope, limits));
    }

    /**
     * Independently validates a complete model envelope and exposes exact canonical logical bytes
     * for every parameter and buffer, keyed by exact Unicode NN path in model metadata order,
     * not sorted path order. No numeric conversion or NaN normalization occurs. The map structure
     * and payload bytes are immutable, but buffer position, limit, and byte order remain mutable.
     * Every path and call receives a fresh independent read-only view at position zero, with
     * exact payload capacity/limit and BIG_ENDIAN order; cursor changes cannot affect other views.
     * Values have detached lifetime and retain no source storage or Engine handles. Limits include
     * manifest and complete file-equivalent size and schema plus every aliased path; they do not
     * bound defensive copies, workspace, or peak heap. A prior inspection is never trusted.
     *
     * @param envelope non-null detached model image; never modified
     * @param limits non-null ceilings independently applied to the entire model envelope
     * @return non-null unmodifiable insertion-ordered map with fresh read-only byte views
     * @throws ArtifactFormatException if kind, schema, payloads, BOOL bytes, or bounds are invalid
     * @throws NullPointerException if an argument is null
     */
    public static Map<String, ByteBuffer> canonicalStateBytes(ArtifactEnvelope envelope,
                                                             ArtifactLimits limits)
            throws ArtifactFormatException {
        ModelArtifactCandidate candidate = ModelArtifactCodec.decode(envelope, limits);
        Map<String, ByteBuffer> bytes = new LinkedHashMap<>();
        for (int i = 0; i < candidate.states().size(); i++) {
            bytes.put(candidate.states().get(i).path(),
                    candidate.envelope().entries().get(i + 1).payload().order(ByteOrder.BIG_ENDIAN));
        }
        return Collections.unmodifiableMap(bytes);
    }

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
     * @param engine non-null Engine, open when materialization is needed; unused for empty state
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
        ArtifactFile.save(target, capture(module, engine, configDigest, inputDigest, limits), limits);
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
        return inspection(read(source, limits));
    }

    /**
     * Projects already validated metadata into detached public descriptors and defensive digests;
     * this projection does not establish a credential for later operations.
     *
     * @param candidate non-null fully decoded model candidate
     * @return immutable inspection of this candidate, with no live resource references
     */
    private static ModelCheckpointInspection inspection(ModelArtifactCandidate candidate) {
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
     * The target must not belong to an open Training session, whose captured bindings would
     * otherwise be invalidated. This restriction is caller-enforced; load does not inspect Training.
     *
     * @param source non-null path to a quiescent regular v1 artifact
     * @param target non-null fully bound quiescent recreated NN tree with no open Training session
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
        validateLoadArguments(target, expectedConfigDigest, expectedInputDigest);
        install(read(source, limits), target, expectedConfigDigest, expectedInputDigest);
    }

    /**
     * Independently validates a complete detached model and expected digest presence/bytes before
     * reconstructing all values and attempting one NN strict whole-tree installation. The target
     * must be fully bound, quiescent, and not owned by an open Training session. Its Tensor identity
     * defines aliases: every path in a group must have identical stored bytes, then receives one
     * fresh native-order heap-backed dense leaf. A group with a parameter is gradient-eligible;
     * buffer-only groups are not. Raw bits are preserved without numeric conversion. Source alias
     * identity is not stored; the caller configuration digest must cover intended sharing.
     * Ordinary validation, alias, or reconstruction failure leaves every binding unchanged.
     * Neither prior inspection nor canonical-byte access authorizes this load; all limits and
     * model semantics are checked again. No Engine, file I/O, or cross-owner transaction is used.
     * The open-Training-session restriction is caller-enforced; load does not inspect Training.
     *
     * @param envelope non-null detached model image; never modified
     * @param target non-null fully bound recreated NN tree, quiescent for the entire call and
     *     not owned by an open Training session
     * @param expectedConfigDigest non-null exact 32-byte expected configuration SHA-256
     * @param expectedInputDigest optional exact 32-byte input-artifact SHA-256, or null for absent
     * @param limits non-null complete file-equivalent and model-payload ceilings
     * @throws ArtifactFormatException if the model is malformed, unsupported, or over any limit
     * @throws IllegalArgumentException if digests, paths, descriptors, or alias values mismatch
     * @throws IllegalStateException if target state is unbound or Tensor identifiers are exhausted
     * @throws NullPointerException if a required argument is null
     */
    public static void load(ArtifactEnvelope envelope, Module target, byte[] expectedConfigDigest,
                            byte[] expectedInputDigest, ArtifactLimits limits)
            throws ArtifactFormatException {
        validateLoadArguments(target, expectedConfigDigest, expectedInputDigest);
        install(ModelArtifactCodec.decode(envelope, limits), target,
                expectedConfigDigest, expectedInputDigest);
    }

    /**
     * Checks target presence before expected digest presence and lengths, preserving the file
     * facade's argument-failure ordering before any read or semantic decode.
     *
     * @param target non-null recreated tree, not inspected or mutated here
     * @param expectedConfigDigest non-null expected 32-byte SHA-256
     * @param expectedInputDigest optional expected 32-byte SHA-256, or null for absent
     * @throws NullPointerException if target or expectedConfigDigest is null
     * @throws IllegalArgumentException if either present digest is not 32 bytes
     */
    private static void validateLoadArguments(Module target, byte[] expectedConfigDigest,
                                               byte[] expectedInputDigest) {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(expectedConfigDigest, "expectedConfigDigest");
        if (expectedConfigDigest.length != 32 || expectedInputDigest != null
                && expectedInputDigest.length != 32)
            throw new IllegalArgumentException("SHA-256 digest must be 32 bytes");
    }

    /**
     * Checks exact digest presence/bytes and reconstructs a complete target-compatible dictionary
     * before the sole strict NN installation attempt. The temporary shallow target snapshot
     * borrows its existing Tensor references only for reconstruction; nothing live is retained
     * after return. Earlier validation/reconstruction failures leave all target bindings unchanged.
     *
     * @param candidate non-null fully decoded candidate under the caller's limits
     * @param target non-null fully bound quiescent tree
     * @param expectedConfigDigest non-null expected 32-byte SHA-256
     * @param expectedInputDigest optional expected 32-byte SHA-256, or null for absent
     * @throws IllegalArgumentException if digests, paths, descriptors, or alias bytes mismatch
     * @throws IllegalStateException if target state is unbound or Tensor identifiers are exhausted
     */
    private static void install(ModelArtifactCandidate candidate, Module target,
                                byte[] expectedConfigDigest, byte[] expectedInputDigest) {
        if (!Arrays.equals(expectedConfigDigest, candidate.configDigest())
                || !Arrays.equals(expectedInputDigest, candidate.inputDigest()))
            throw new IllegalArgumentException("Model checkpoint fingerprint mismatch");
        // NN's loader permits reserved lazy groups; v1 explicitly requires a complete target.
        var current = target.stateDictionary();
        target.loadStateDictionary(ModelStateRestoration.reconstruct(candidate, current));
    }

    /**
     * Reads and validates current file bytes, then decodes their model meaning without retaining
     * the file or trusting an earlier inspection. Source presence precedes limits presence checks.
     *
     * @param source non-null quiescent regular artifact path, not a symbolic link
     * @param limits non-null complete file and model-payload ceilings
     * @return fully decoded detached candidate, never null
     * @throws IOException if file access, framing, integrity, model semantics, or bounds fail
     * @throws NullPointerException if source or limits is null
     */
    private static ModelArtifactCandidate read(Path source, ArtifactLimits limits) throws IOException {
        return ModelArtifactCodec.decode(ArtifactFile.read(
                Objects.requireNonNull(source, "source"), Objects.requireNonNull(limits, "limits")), limits);
    }
}
