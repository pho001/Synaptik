package io.github.pho001.synaptik.checkpoint.model;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.nn.module.DeclarativeModel;
import io.github.pho001.synaptik.nn.module.ModelSpecification;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.Sequential;
import java.nio.ByteBuffer;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Whole-model framing and independent cross-owner coherence on the cold persistence path.
 * Values cross only the public ModelCheckpoint boundary; NN retains reconstruction/installation.
 * Encoded limits precede projected defensive copies and do not promise a peak-heap budget.
 */
final class WholeModelComposition {
    static final ArtifactLimits DEFAULT_LIMITS = new ArtifactLimits(
            268435456L, 4194304, 16384, 64, 67108864, 268435456L);
    private static final String KIND = "whole-model.v1";
    private static final String DEFINITION = "definition/schema";
    private static final String PREFIX = "model/";

    private WholeModelComposition() { }

    /**
     * Describes under caller quiescence, bounds metadata, then delegates actual value preflight.
     * @param source non-null supported, fully bound quiescent root
     * @param engine non-null Engine; unused for empty state
     * @param input optional owner-verified input identity, null for required absence
     * @param limits non-null outer encoded ceilings
     * @return detached coherent whole-model envelope, without live state
     * @throws ArtifactFormatException if final captured coherence fails
     * @throws IllegalArgumentException if source or preflight bounds are unsupported
     * @throws IllegalStateException if source is unbound or Engine violates its contract
     * @throws ArithmeticException if checked capture arithmetic overflows
     * @throws NullPointerException if a required argument is null
     */
    static ArtifactEnvelope capture(Module source, Engine engine, InputArtifactFingerprint input,
                                    ArtifactLimits limits) throws ArtifactFormatException {
        Objects.requireNonNull(source, "module");
        Objects.requireNonNull(engine, "engine");
        Objects.requireNonNull(limits, "limits");
        ModelSpecification specification = switch (source) {
            case Sequential sequence -> ModelSpecification.describe(sequence);
            case DeclarativeModel model -> model.describe();
            default -> throw new IllegalArgumentException("Unsupported whole-model root; expected Sequential or DeclarativeModel");
        };
        int size = WholeModelDefinitionCodec.size(specification);
        int innerCount = Math.addExact(specification.stateDescriptors().size(), 1);
        ArtifactLimits projected = project(limits, size, innerCount);
        byte[] definition = WholeModelDefinitionCodec.encode(specification);
        byte[] digest = hash(ByteBuffer.wrap(definition));
        byte[] inputBytes = input == null ? null : input.bytes();
        ArtifactEnvelope values = ModelCheckpoint.capture(source, engine, digest, inputBytes, projected);
        checkOuterWithValues(values, size, limits);
        coherent(specification, values, digest, inputBytes, projected);
        List<ArtifactEnvelope.Entry> entries = new ArrayList<>(innerCount + 1);
        entries.add(new ArtifactEnvelope.Entry(DEFINITION, definition));
        for (var entry : values.entries()) entries.add(new ArtifactEnvelope.Entry(PREFIX + entry.key(), copy(entry)));
        ArtifactEnvelope whole = new ArtifactEnvelope(KIND, entries);
        // Validate final correspondence anew, without constructing or publishing a model.
        validate(whole, inputBytes, limits);
        return whole;
    }

    /**
     * Validates everything before creating a private descriptor skeleton and installing values.
     * @param envelope non-null detached image; never modified
     * @param input optional verified expected input identity, null for required absence
     * @param limits non-null outer encoded ceilings
     * @return independent heap-backed evaluation model, only after complete successful installation
     * @throws ArtifactFormatException if stored metadata, values, identity or bounds are invalid
     * @throws IllegalStateException if Tensor identifier space is exhausted
     * @throws NullPointerException if envelope or limits is null
     */
    static DeclarativeModel load(ArtifactEnvelope envelope, InputArtifactFingerprint input,
                                 ArtifactLimits limits) throws ArtifactFormatException {
        Validated validated = validate(envelope, input == null ? null : input.bytes(), limits);
        try {
            DeclarativeModel model = DeclarativeModel.reconstruct(validated.specification());
            ModelCheckpoint.load(validated.values(), model, validated.digest(),
                    input == null ? null : input.bytes(), validated.projected());
            return model;
        } catch (IllegalArgumentException | ArithmeticException failure) {
            throw new ArtifactFormatException("Invalid whole-model reconstruction/state", failure);
        }
    }

    /**
     * Preflights the outer image before copies, then independently validates both semantic owners.
     * @param envelope non-null immutable outer image
     * @param expectedInput optional expected 32-byte digest, null for required absence
     * @param limits non-null outer encoded ceilings
     * @return private validated composition, never a credential for later public component calls
     * @throws ArtifactFormatException if any framing, metadata, value or expected identity fails
     * @throws NullPointerException if envelope or limits is null
     */
    private static Validated validate(ArtifactEnvelope envelope, byte[] expectedInput, ArtifactLimits limits)
            throws ArtifactFormatException {
        Objects.requireNonNull(envelope, "envelope");
        Objects.requireNonNull(limits, "limits");
        try {
            preflight(envelope, limits);
            if (!KIND.equals(envelope.kind())) {
                if ("model.v1".equals(envelope.kind()))
                    throw new ArtifactFormatException("Missing whole-model definition; use existing target-load for values-only model.v1");
                throw new ArtifactFormatException("Unsupported whole-model kind");
            }
            if (envelope.entries().isEmpty() || !DEFINITION.equals(envelope.entries().getFirst().key()))
                throw new ArtifactFormatException("Missing whole-model definition/schema");
            var definition = envelope.entries().getFirst();
            int size = definition.size();
            if (size > WholeModelDefinitionCodec.MAX_BYTES)
                throw new ArtifactFormatException("Definition exceeds 1 MiB");
            for (int i = 1; i < envelope.entries().size(); i++)
                if (!envelope.entries().get(i).key().startsWith(PREFIX))
                    throw new ArtifactFormatException("Unexpected whole-model key");
            ArtifactLimits projected = project(limits, size, envelope.entries().size() - 1);
            ModelSpecification specification = WholeModelDefinitionCodec.decode(definition.payload());
            byte[] digest = hash(definition.payload());
            List<ArtifactEnvelope.Entry> entries = new ArrayList<>(envelope.entries().size() - 1);
            for (int i = 1; i < envelope.entries().size(); i++) {
                var entry = envelope.entries().get(i);
                entries.add(new ArtifactEnvelope.Entry(entry.key().substring(PREFIX.length()), copy(entry)));
            }
            ArtifactEnvelope values = new ArtifactEnvelope("model.v1", entries);
            coherent(specification, values, digest, expectedInput, projected);
            return new Validated(specification, values, digest, projected);
        } catch (IllegalArgumentException | ArithmeticException failure) {
            throw new ArtifactFormatException("Invalid whole-model framing/limits", failure);
        }
    }

    /**
     * Compares complete ordered schemas and every declared identity group's exact raw bytes.
     * @param specification non-null NN-validated semantic definition
     * @param values non-null detached values-only component, independently validated here
     * @param digest non-null exact canonical-definition SHA-256
     * @param expectedInput optional expected digest, null for required absence
     * @param projected non-null file-equivalent component ceilings
     * @throws ArtifactFormatException if component validation or cross-owner correspondence fails
     * @throws ArithmeticException if checked derived byte length overflows
     */
    private static void coherent(ModelSpecification specification, ArtifactEnvelope values,
                                 byte[] digest, byte[] expectedInput, ArtifactLimits projected)
            throws ArtifactFormatException {
        var inspection = ModelCheckpoint.inspect(values, projected);
        if (!Arrays.equals(digest, inspection.configDigest()))
            throw new ArtifactFormatException("Whole-model configuration SHA-256 mismatch");
        if (!Arrays.equals(expectedInput, inspection.inputDigest()))
            throw new ArtifactFormatException("Input fingerprint mismatch: explicit expected identity required for stored presence; presence and bytes must match");
        var descriptors = specification.stateDescriptors();
        if (descriptors.size() != inspection.states().size())
            throw new ArtifactFormatException("Whole-model state count mismatch");
        for (int i = 0; i < descriptors.size(); i++) {
            var expected = descriptors.get(i);
            var stored = inspection.states().get(i);
            long length = Math.multiplyExact(expected.shape().knownElementCount().orElseThrow(),
                    expected.dataType().byteWidth());
            if (!expected.path().equals(stored.path()) || expected.kind() != stored.kind()
                    || expected.dataType() != stored.dataType() || !expected.shape().equals(stored.shape())
                    || !expected.requiresGrad() || length != stored.byteLength())
                throw new ArtifactFormatException("Whole-model state descriptor mismatch at " + expected.path());
        }
        var bytes = ModelCheckpoint.canonicalStateBytes(values, projected);
        for (var alias : specification.aliases()) {
            ByteBuffer first = bytes.get(alias.paths().getFirst());
            for (String path : alias.paths())
                if (!first.equals(bytes.get(path)))
                    throw new ArtifactFormatException("Whole-model alias bytes mismatch at " + path);
        }
    }

    /**
     * Computes exact framing residuals before projected-limit construction or value materialization.
     * @param limits non-null outer ceilings
     * @param definitionBytes dry-sized metadata bytes, bounded independently of values
     * @param innerCount predicted or observed number of component entries including schema
     * @return component ceilings equivalent to the outer framing bounds
     * @throws IllegalArgumentException if definition/key bounds or constructor minima fail
     * @throws ArithmeticException if checked residual arithmetic overflows
     */
    private static ArtifactLimits project(ArtifactLimits limits, int definitionBytes, int innerCount) {
        if (limits.maxKeyBytes() < DEFINITION.length() || definitionBytes > WholeModelDefinitionCodec.MAX_BYTES
                || definitionBytes > limits.maxPayloadBytes())
            throw new IllegalArgumentException("Whole-model definition/key exceeds limits");
        long delta = Math.addExact(67L, Math.multiplyExact(6L, innerCount));
        long file = Math.subtractExact(limits.maxFileBytes(), Math.addExact(definitionBytes, delta));
        long manifest = Math.subtractExact(limits.maxManifestBytes(), delta);
        int entries = Math.subtractExact(limits.maxEntries(), 1);
        int key = Math.subtractExact(limits.maxKeyBytes(), 6);
        long aggregate = Math.subtractExact(limits.maxAggregatePayloadBytes(), definitionBytes);
        if (file < 48 || manifest < 8 || entries < 0 || key < 1 || aggregate < 0)
            throw new IllegalArgumentException("Whole-model residual limits below usable minima");
        return new ArtifactLimits(file, Math.toIntExact(manifest), entries, key,
                limits.maxPayloadBytes(), aggregate);
    }

    /**
     * Checks all six outer encoded bounds using lengths before any projected payload copy.
     * @param envelope non-null immutable outer image
     * @param limits non-null outer encoded ceilings
     * @throws IllegalArgumentException if any complete encoded bound is exceeded
     * @throws ArithmeticException if checked framing arithmetic overflows
     */
    private static void preflight(ArtifactEnvelope envelope, ArtifactLimits limits) {
        if (envelope.kind().length() > limits.maxKeyBytes() || envelope.entries().size() > limits.maxEntries())
            throw new IllegalArgumentException("Whole-model kind/entry limit");
        long manifest = Math.addExact(8L, envelope.kind().length());
        long aggregate = 0;
        for (var entry : envelope.entries()) {
            if (entry.key().length() > limits.maxKeyBytes() || entry.size() > limits.maxPayloadBytes())
                throw new IllegalArgumentException("Whole-model key/payload limit");
            manifest = Math.addExact(manifest, Math.addExact(44L, entry.key().length()));
            aggregate = Math.addExact(aggregate, entry.size());
        }
        checkTotals(manifest, aggregate, limits);
    }

    /**
     * Rechecks the captured component's complete outer size before adding prefixes and copying.
     * @param values non-null captured values component
     * @param definitionBytes bounded canonical definition byte count
     * @param limits non-null outer ceilings
     * @throws IllegalArgumentException if an outer encoded bound is exceeded
     * @throws ArithmeticException if checked framing arithmetic overflows
     */
    private static void checkOuterWithValues(ArtifactEnvelope values, int definitionBytes, ArtifactLimits limits) {
        if (values.entries().size() + 1L > limits.maxEntries())
            throw new IllegalArgumentException("Whole-model entry limit");
        long manifest = 8L + KIND.length() + 44L + DEFINITION.length();
        long aggregate = definitionBytes;
        for (var entry : values.entries()) {
            if (entry.key().length() + PREFIX.length() > limits.maxKeyBytes()
                    || entry.size() > limits.maxPayloadBytes())
                throw new IllegalArgumentException("Whole-model key/payload limit");
            manifest = Math.addExact(manifest, 44L + PREFIX.length() + entry.key().length());
            aggregate = Math.addExact(aggregate, entry.size());
        }
        checkTotals(manifest, aggregate, limits);
    }

    /**
     * Includes the fixed artifact header when checking total encoded size.
     * @param manifest complete manifest bytes
     * @param aggregate sum of all canonical payload lengths in bytes
     * @param limits non-null complete encoded ceilings
     * @throws IllegalArgumentException if manifest, aggregate or file exceeds its bound
     * @throws ArithmeticException if checked total arithmetic overflows
     */
    private static void checkTotals(long manifest, long aggregate, ArtifactLimits limits) {
        long file = Math.addExact(48L, Math.addExact(manifest, aggregate));
        if (manifest > limits.maxManifestBytes() || aggregate > limits.maxAggregatePayloadBytes()
                || file > limits.maxFileBytes())
            throw new IllegalArgumentException("Whole-model manifest/aggregate/file limit");
    }

    /**
     * Copies through the public defensive entry boundary without retaining its mutable cursor.
     * @param entry non-null immutable payload entry already admitted by outer preflight
     * @return fresh exact payload bytes, never null
     */
    private static byte[] copy(ArtifactEnvelope.Entry entry) {
        byte[] bytes = new byte[entry.size()]; entry.payload().get(bytes); return bytes;
    }

    /**
     * Hashes exact definition bytes without changing the caller's buffer cursor.
     * @param bytes non-null canonical definition view at its intended position and limit
     * @return fresh 32-byte SHA-256 digest, not authentication
     */
    private static byte[] hash(ByteBuffer bytes) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            digest.update(bytes.duplicate());
            return digest.digest();
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError("Required SHA-256 unavailable", impossible);
        }
    }

    /**
     * Private composition facts; public target-load still independently validates the component.
     * @param specification validated closed semantic configuration
     * @param values detached values-only component
     * @param digest computed canonical-definition SHA-256
     * @param projected equivalent component limits
     */
    private record Validated(ModelSpecification specification, ArtifactEnvelope values,
                             byte[] digest, ArtifactLimits projected) { }
}
