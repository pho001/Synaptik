package io.github.pho001.synaptik.nn.checkpoint;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFile;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.nn.module.DeclarativeModel;
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
 * Public version-one NN model persistence boundary, retained in NN after the generic split.
 * Opaque artifacts still come from Checkpoint; this facade retains the old readers and flexible
 * declarative semantics without forwarding from the former package or converting to graph-model.v1.
 * Values-only operations require a recreated NN
 * target and caller configuration digest. Whole-model operations save a closed supported Sequential
 * or DeclarativeModel and automatically load independent heap-backed evaluation state; configuration
 * identity is computed from the canonical definition. Java callbacks, optimizer state, and training
 * continuation are not persisted. Whole-model roots are recognized only as
 * {@link io.github.pho001.synaptik.nn.module.Sequential} or {@link DeclarativeModel}; arbitrary
 * Java model definitions, custom children and recurrent layers are unsupported. Values-only
 * {@code model.v1} and whole-model {@code whole-model.v1} are distinct kinds: old target-load and
 * inspection operations do not infer topology or inspect the outer whole-model envelope.
 * Capture, save and target-load require caller-owned quiescence of module bindings and storage
 * for their entire calls. Whole-model description and values capture may each read state; there
 * is no combined snapshot or transaction. This
 * facade uses only public Engine materialization and delegates installation to NN's strict
 * whole-tree loader. It has no cross-call cache or live resource after return.
 *
 * <p>Whole-model load uses no Engine or caller Arena and returns independent heap-backed evaluation
 * state only after complete validation. Parameters remain gradient-eligible: evaluation mode does
 * not construct autograd or promise a no-gradient execution route. No backend, device state or
 * compiled/prepared plan is saved. Semantic empty trees gain no generic zero-operation Engine
 * guarantee. Configuration SHA-256 binds exact definition bytes, not authenticity; any external
 * input identity must already have been verified by the application.</p>
 */
public final class ModelCheckpoint {
    private ModelCheckpoint() { }

    /**
     * Saves a supported whole model with default encoded limits and absent input identity.
     * Uses 256 MiB file/aggregate, 4 MiB manifest, 16384 entries, 64-byte names and 64 MiB payloads.
     * Definition metadata is always bounded to one MiB; these are not peak-heap limits.
     * The exact limits tuple is {@code (268435456L, 4194304, 16384, 64, 67108864, 268435456L)};
     * one MiB is 1048576 bytes. NN's 4096-occurrence/depth-64 tree bounds still apply.
     * @param target non-null publication path in an existing trusted local directory
     * @param module non-null fully bound Sequential or DeclarativeModel, with bindings/storage
     *     live, accessible and quiescent throughout
     * @param engine non-null Engine, open if values need materialization; unused for empty state
     * @throws IOException if capture validation or atomic file publication fails
     * @throws IllegalArgumentException if source configuration or encoded bounds are invalid
     * @throws IllegalStateException if state is unbound or Engine materialization fails
     * @throws ArithmeticException if checked capture arithmetic overflows
     * @throws NullPointerException if a required argument is null
     */
    public static void save(Path target, Module module, Engine engine) throws IOException {
        save(target, module, engine, WholeModelComposition.DEFAULT_LIMITS);
    }

    /**
     * Saves a whole model with caller encoded bounds and absent input identity. Description and
     * public values capture may each read state; caller quiescence spans both, not a transaction.
     * Capture completes before filesystem work; publication retains the existing atomic-move policy.
     * A successful move provides atomic live visibility, not portable power-loss durability.
     * Inspect a target after a failed move because its state is provider-dependent.
     * @param target non-null publication path in an existing trusted local directory
     * @param module non-null fully bound Sequential or DeclarativeModel, with bindings/storage
     *     live, accessible and quiescent throughout
     * @param engine non-null Engine, open for nonempty state; borrowed only during this call
     * @param limits non-null encoded ceilings, not heap/workspace bounds
     * @throws IOException if capture validation or file publication fails
     * @throws IllegalArgumentException if source or preflight bounds are invalid
     * @throws IllegalStateException if source is unbound or Engine materialization fails
     * @throws ArithmeticException if checked capture arithmetic overflows
     * @throws NullPointerException if a required argument is null
     */
    public static void save(Path target, Module module, Engine engine, ArtifactLimits limits) throws IOException {
        Objects.requireNonNull(target, "target");
        ArtifactFile.save(target, captureWholeModel(module, engine, limits), limits);
    }

    /**
     * Saves a supported whole model bound to application-verified input identity. Checkpoint does
     * not read or authenticate that input artifact. Entire-call source quiescence is caller-owned;
     * description and values capture are separate reads. Captured bytes outlive all source resources.
     * Publication uses the same force/revalidation/one-atomic-move policy as values-only save,
     * not portable power-loss durability. Inspect the target after an uncertain failed move.
     * @param target non-null publication path in a trusted existing directory
     * @param module non-null fully bound Sequential or DeclarativeModel with live, accessible,
     *     quiescent bindings and storage
     * @param engine non-null Engine, open for nonempty state; unused for empty state
     * @param input non-null application-verified SHA-256 input identity
     * @param limits non-null complete encoded ceilings, not peak-heap bounds
     * @throws IOException if capture validation or atomic publication fails
     * @throws IllegalArgumentException if source or preflight bounds are invalid
     * @throws IllegalStateException if source is unbound or Engine materialization fails
     * @throws ArithmeticException if checked capture arithmetic overflows
     * @throws NullPointerException if a required argument is null
     */
    public static void save(Path target, Module module, Engine engine, InputArtifactFingerprint input,
                            ArtifactLimits limits) throws IOException {
        Objects.requireNonNull(target, "target");
        ArtifactFile.save(target, captureWholeModel(module, engine, input, limits), limits);
    }

    /**
     * Automatically loads a supported whole model using default encoded ceilings and requiring
     * absent input identity. Values-only files require the existing target-load API instead.
     * Default limits are the same as {@link #save(Path, Module, Engine)}. Stored input presence
     * raises {@link ArtifactFormatException}; it is never treated as verified identity by default.
     * @param source non-null quiescent regular file, not a symlink
     * @return non-null independent heap-backed evaluation model with gradient-eligible parameters
     * @throws IOException if access, framing, definition, values, limits or expected absence fail
     * @throws IllegalStateException if Tensor identifiers are exhausted
     * @throws NullPointerException if source is null
     */
    public static DeclarativeModel load(Path source) throws IOException {
        return load(source, WholeModelComposition.DEFAULT_LIMITS);
    }

    /**
     * Loads a private descriptor skeleton and exact heap values, returning only after complete
     * validation. Loading performs no Engine work, RNG initialization, autograd or compilation;
     * it retains no native handle or live file and requires no caller Arena.
     * Stored input-identity presence is rejected unless explicitly expected through the typed overload.
     * @param source non-null quiescent regular whole-model file
     * @param limits non-null complete encoded ceilings
     * @return non-null independent heap-backed evaluation model with gradient-eligible parameters,
     *     surviving file/source closure
     * @throws IOException if access or stored kind, metadata, values, identity or bounds are invalid
     * @throws IllegalStateException if Tensor identifiers are exhausted
     * @throws NullPointerException if a required argument is null
     */
    public static DeclarativeModel load(Path source, ArtifactLimits limits) throws IOException {
        return loadWholeModel(ArtifactFile.read(Objects.requireNonNull(source, "source"),
                Objects.requireNonNull(limits, "limits")), limits);
    }

    /**
     * Automatically loads only when stored input presence and bytes match verified expected identity.
     * Hashes bind configuration, not authenticity; the application verifies the input artifact.
     * @param source non-null quiescent regular whole-model file
     * @param input non-null verified expected SHA-256 input identity; presence is required
     * @param limits non-null complete encoded ceilings, not peak-heap bounds
     * @return independent non-null heap-backed evaluation model with gradient-eligible parameters,
     *     no live Engine or file handle, and no caller Arena lifetime
     * @throws IOException if access, stored data, bounds or expected input identity fail
     * @throws IllegalStateException if Tensor identifiers are exhausted
     * @throws NullPointerException if a required argument is null
     */
    public static DeclarativeModel load(Path source, InputArtifactFingerprint input, ArtifactLimits limits)
            throws IOException {
        Objects.requireNonNull(source, "source");
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(limits, "limits");
        return loadWholeModel(ArtifactFile.read(source, limits), input, limits);
    }

    /**
     * Captures a closed whole model with absent input identity, without filesystem work. Current
     * aliases mean exact Tensor identity, never merely equal bytes. Unsupported/unbound sources and
     * all encoded limits fail before Engine admission or bulk value copying. Description and values
     * capture may read bindings separately under caller quiescence; no combined snapshot is promised.
     * Empty semantic trees need no Engine call but gain no generic zero-operation execution route.
     * @param module non-null fully bound Sequential or DeclarativeModel with live, accessible,
     *     quiescent bindings and storage throughout
     * @param engine non-null borrowed Engine, open when nonempty values need materialization
     * @param limits non-null complete encoded ceilings; fixed definition ceiling is one MiB
     * @return fully coherent detached immutable whole-model envelope, with no live source resource
     * @throws ArtifactFormatException if captured definition/value coherence is invalid
     * @throws IllegalArgumentException if source or preflight limits are invalid
     * @throws IllegalStateException if source is unbound or Engine materialization fails
     * @throws ArithmeticException if checked capture arithmetic overflows
     * @throws NullPointerException if a required argument is null
     */
    public static ArtifactEnvelope captureWholeModel(Module module, Engine engine, ArtifactLimits limits)
            throws ArtifactFormatException {
        return WholeModelComposition.capture(module, engine, null, limits);
    }

    /**
     * Captures a whole model and verified input identity through the same bounded public values
     * boundary. No input-artifact parsing or authentication is performed. Quiescence spans separate
     * description/value reads; state and resources are neither mutated nor retained.
     * @param module non-null fully bound Sequential or DeclarativeModel with live, accessible,
     *     quiescent bindings/storage throughout
     * @param engine non-null borrowed Engine; unused for empty state
     * @param input non-null owner-verified input SHA-256 identity, stored as present
     * @param limits non-null encoded ceilings, not peak-heap bounds; definition at most one MiB
     * @return non-null detached immutable coherent envelope surviving source/Engine closure
     * @throws ArtifactFormatException if final semantic coherence fails
     * @throws IllegalArgumentException if source or encoded preflight is invalid
     * @throws IllegalStateException if source is unbound or Engine materialization fails
     * @throws ArithmeticException if checked capture arithmetic overflows
     * @throws NullPointerException if a required argument is null
     */
    public static ArtifactEnvelope captureWholeModel(Module module, Engine engine, InputArtifactFingerprint input,
                                                     ArtifactLimits limits) throws ArtifactFormatException {
        Objects.requireNonNull(input, "input");
        return WholeModelComposition.capture(module, engine, input, limits);
    }

    /**
     * Independently validates outer bounds, definition, complete value schema and alias bytes before
     * reconstructing a private NN skeleton and installing exact heap values. Requires absent input
     * identity; stored presence directs callers to supply a verified typed expected identity.
     * No Engine, initialization, RNG, autograd or native resource is used. Failure publishes no model.
     * @param envelope non-null detached whole-model image, never modified
     * @param limits non-null complete encoded ceilings, not defensive-copy or peak-heap budgets
     * @return non-null independent heap-backed evaluation model with gradient-eligible parameters
     * @throws ArtifactFormatException if stored kind, definition, values, identity or bounds fail
     * @throws IllegalStateException if Tensor identifiers are exhausted
     * @throws NullPointerException if a required argument is null
     */
    public static DeclarativeModel loadWholeModel(ArtifactEnvelope envelope, ArtifactLimits limits)
            throws ArtifactFormatException {
        return WholeModelComposition.load(envelope, null, limits);
    }

    /**
     * Loads a whole model only after independently checking complete definition/value/alias coherence
     * and exact expected input presence and bytes. Uses public target-load on a private NN skeleton;
     * failure exposes no partial candidate. No Engine, initialization, native lifetime or RNG is used.
     * Loading performs no autograd or compilation and needs no caller Arena.
     * @param envelope non-null detached whole-model image, never modified
     * @param input non-null owner-verified expected input identity; stored presence is required
     * @param limits non-null complete encoded ceilings, not peak-heap bounds
     * @return non-null independent heap-backed evaluation model with gradient-eligible parameters
     *     and exact declared identity aliases
     * @throws ArtifactFormatException if stored data, limits or expected input identity fail
     * @throws IllegalStateException if Tensor identifiers are exhausted
     * @throws NullPointerException if a required argument is null
     */
    public static DeclarativeModel loadWholeModel(ArtifactEnvelope envelope, InputArtifactFingerprint input,
                                                 ArtifactLimits limits) throws ArtifactFormatException {
        Objects.requireNonNull(input, "input");
        return WholeModelComposition.load(envelope, input, limits);
    }

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
