package io.github.pho001.synaptik.checkpoint.model;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.checkpoint.model.WholeModelDefinitionCodecTest.*;

import io.github.pho001.synaptik.checkpoint.artifact.*;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.nn.initialization.ParameterInitialization;
import io.github.pho001.synaptik.nn.layers.*;
import io.github.pho001.synaptik.nn.module.*;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.ModelSpecification.*;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Public composition, ordering, detached lifetime and malformed-whole/valid-values coverage. */
class WholeModelCheckpointTest {
    private static final ArtifactLimits LIMITS = WholeModelComposition.DEFAULT_LIMITS;
    @TempDir Path directory;

    @Test
    void fingerprintIsExactlyOneImmutableValueWithoutTrustSemantics() {
        byte[] bytes = new byte[32]; bytes[31] = 7;
        var fingerprint = new InputArtifactFingerprint(bytes);
        bytes[31] = 8;
        byte[] expected = fingerprint.bytes(); expected[0] = 1;
        assertEquals(0, fingerprint.bytes()[0]);
        assertEquals(7, fingerprint.bytes()[31]);
        assertEquals(fingerprint, new InputArtifactFingerprint(fingerprint.bytes()));
        assertEquals(fingerprint.hashCode(), new InputArtifactFingerprint(fingerprint.bytes()).hashCode());
        assertNotEquals(fingerprint, new InputArtifactFingerprint(bytes));
        assertNotEquals(fingerprint, null);
        assertNotEquals(fingerprint, "identity");
        assertThrows(NullPointerException.class, () -> new InputArtifactFingerprint(null));
        for (int size : new int[] {0, 31, 33})
            assertThrows(IllegalArgumentException.class, () -> new InputArtifactFingerprint(new byte[size]));
    }

    @Test
    void wholeRoundTripPreservesAllFloatingRawBitsAndIndependentHeapLifetimes() throws Exception {
        long[][] patterns = {{0, Long.MIN_VALUE, 1, 0x7ff8000000001234L, 0xfff0000000002345L},
                {0, 0x80000000L, 1, 0x7fc01234, 0xff801234L},
                {0, 0x8000, 1, 0x7fc1, 0xff81}, {0, 0x8000, 1, 0x7e11, 0xfc11}};
        for (int index = 0; index < FLOATS.size(); index++) {
            DataType type = FLOATS.get(index);
            Tensor weight = tensor(type, Shape.of(1, 5), patterns[index]);
            Sequential source = new Sequential(List.of(new Linear(weight)));
            ArtifactEnvelope whole;
            try (Engine engine = Engine.standard()) {
                whole = ModelCheckpoint.captureWholeModel(source, engine, LIMITS);
            }
            var loaded = ModelCheckpoint.loadWholeModel(whole, LIMITS);
            var second = ModelCheckpoint.loadWholeModel(whole, LIMITS);
            assertEquals(ForwardMode.EVALUATION, loaded.mode());
            Tensor leaf = loaded.stateDictionary().entries().getFirst().value();
            assertTrue(leaf.descriptor().requiresGrad());
            assertTrue(leaf.provenance().isEmpty());
            assertFalse(leaf.hostStorage().orElseThrow().segment().isNative());
            assertNotSame(weight, leaf);
            assertNotSame(leaf, second.stateDictionary().entries().getFirst().value());
            leaf.hostStorage().orElseThrow().segment().set(ValueLayout.JAVA_BYTE, 0, (byte) 42);
            try (Engine engine = Engine.standard()) {
                ArtifactEnvelope resaved = ModelCheckpoint.captureWholeModel(second, engine, LIMITS);
                assertEnvelopeEquals(whole, resaved);
            }
        }
    }

    @Test
    void allTenOverloadsAndFingerprintPresenceAreEnforced() throws Exception {
        var source = source();
        var input = new InputArtifactFingerprint(new byte[32]);
        byte[] wrongBytes = new byte[32]; wrongBytes[1] = 1;
        var wrong = new InputArtifactFingerprint(wrongBytes);
        Path file = directory.resolve("whole.chk");
        ArtifactEnvelope absent;
        ArtifactEnvelope present;
        try (Engine engine = Engine.standard()) {
            ModelCheckpoint.save(file, source, engine);
            assertNotNull(ModelCheckpoint.load(file));
            ModelCheckpoint.save(file, source, engine, LIMITS);
            assertNotNull(ModelCheckpoint.load(file, LIMITS));
            absent = ModelCheckpoint.captureWholeModel(source, engine, LIMITS);
            present = ModelCheckpoint.captureWholeModel(source, engine, input, LIMITS);
            ModelCheckpoint.save(file, source, engine, input, LIMITS);
        }
        assertNotNull(ModelCheckpoint.load(file, input, LIMITS));
        assertNotNull(ModelCheckpoint.loadWholeModel(present, input, LIMITS));
        assertNotNull(ModelCheckpoint.loadWholeModel(absent, LIMITS));
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.load(file));
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.load(file, wrong, LIMITS));
        assertTrue(assertThrows(ArtifactFormatException.class,
                () -> ModelCheckpoint.loadWholeModel(present, LIMITS)).getMessage().contains("explicit expected identity"));
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.loadWholeModel(present, wrong, LIMITS));
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.loadWholeModel(absent, input, LIMITS));
        Files.delete(file);
        assertFalse(ModelCheckpoint.loadWholeModel(present, input, LIMITS).stateDictionary().entries().isEmpty());
    }

    @Test
    void emptyAndActivationOnlyTreesUseNoClosedEngineButDoNotPromiseBackendAdmission() throws Exception {
        Engine closed = Engine.standard(); closed.close();
        for (Sequential source : List.of(new Sequential(List.of()), new Sequential(List.of(
                new Activation(ActivationKind.RELU), new Sequential(List.of(new Activation(ActivationKind.TANH))))))) {
            var whole = ModelCheckpoint.captureWholeModel(source, closed, LIMITS);
            var loaded = ModelCheckpoint.loadWholeModel(whole, LIMITS);
            assertEquals(ModelSpecification.describe(source).root(), loaded.describe().root());
            assertTrue(loaded.stateDictionary().entries().isEmpty());
        }
    }

    @Test
    void unsupportedRootChildAndUnboundStateFailBeforeEngineOrFilesystem() {
        Engine closed = Engine.standard(); closed.close();
        Path impossible = directory.resolve("missing-parent").resolve("model.chk");
        assertThrows(IllegalArgumentException.class, () -> ModelCheckpoint.save(impossible, new Module() { }, closed));
        var customChild = new Sequential(List.of(new UnaryTensorModule() {
            @Override public Tensor forward(Tensor input) { throw new AssertionError("No forward during capture"); }
        }));
        assertThrows(IllegalArgumentException.class, () -> ModelCheckpoint.save(impossible, customChild, closed));
        var unbound = new Sequential(List.of(new Linear(2, false, DataType.FLOAT32,
                ParameterInitialization.zeros(), java.util.random.RandomGeneratorFactory.of("L64X128MixRandom"), 1)));
        assertThrows(IllegalStateException.class, () -> ModelCheckpoint.save(impossible, unbound, closed));
        assertFalse(Files.exists(impossible.getParent()));
    }

    @Test
    void eachOuterOnlyBoundRejectsBeforeClosedEngineAndEnvelopeCopies() throws Exception {
        var source = new Sequential(List.of(new Linear(tensor(DataType.FLOAT32, Shape.of(1, 64), new long[64]))));
        ArtifactEnvelope whole;
        try (Engine engine = Engine.standard()) { whole = ModelCheckpoint.captureWholeModel(source, engine, LIMITS); }
        int manifest = manifest(whole);
        long aggregate = whole.entries().stream().mapToLong(ArtifactEnvelope.Entry::size).sum();
        long file = 48L + manifest + aggregate;
        int maxPayload = whole.entries().stream().mapToInt(ArtifactEnvelope.Entry::size).max().orElseThrow();
        var exact = new ArtifactLimits(file, manifest, 3, 20, maxPayload, aggregate);
        assertNotNull(ModelCheckpoint.loadWholeModel(whole, exact));
        List<ArtifactLimits> below = List.of(
                new ArtifactLimits(file - 1, manifest, 3, 20, maxPayload, aggregate),
                new ArtifactLimits(file, manifest - 1, 3, 20, maxPayload, aggregate),
                new ArtifactLimits(file, manifest, 2, 20, maxPayload, aggregate),
                new ArtifactLimits(file, manifest, 3, 19, maxPayload, aggregate),
                new ArtifactLimits(file, manifest, 3, 20, maxPayload - 1, aggregate),
                new ArtifactLimits(file, manifest, 3, 20, maxPayload, aggregate - 1),
                new ArtifactLimits(48, manifest, 3, 20, maxPayload, aggregate),
                new ArtifactLimits(file, 8, 3, 20, maxPayload, aggregate),
                new ArtifactLimits(file, manifest, 0, 20, maxPayload, aggregate),
                new ArtifactLimits(file, manifest, 3, 1, maxPayload, aggregate),
                new ArtifactLimits(file, manifest, 3, 20, maxPayload, 0));
        Engine closed = Engine.standard(); closed.close();
        for (var bounds : below) {
            assertThrows(IllegalArgumentException.class, () -> ModelCheckpoint.captureWholeModel(source, closed, bounds));
            assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.loadWholeModel(whole, bounds));
        }
        try (Engine engine = Engine.standard()) {
            assertEnvelopeEquals(whole, ModelCheckpoint.captureWholeModel(source, engine, exact));
        }
        // The unprefixed component fits these four limits; only the outer delta rejects it.
        ArtifactEnvelope inner = component(whole);
        for (var bounds : below.subList(0, 4)) assertNotNull(ModelCheckpoint.inspect(inner, bounds));
    }

    @Test
    void definitionPayloadCeilingAndFixedOneMiBBoundApplyBeforeEngine() throws Exception {
        Engine closed = Engine.standard(); closed.close();
        var activations = new ArrayList<UnaryTensorModule>();
        for (int i = 0; i < 500; i++) activations.add(new Activation(ActivationKind.RELU));
        var source = new Sequential(activations);
        var whole = ModelCheckpoint.captureWholeModel(source, closed, LIMITS);
        var small = new ArtifactLimits(10000, 5000, 8, 64, 1020, 10000);
        assertThrows(IllegalArgumentException.class, () -> ModelCheckpoint.captureWholeModel(source, closed, small));
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.loadWholeModel(whole, small));
        long[] extents = new long[131072]; Arrays.fill(extents, 1);
        var giant = DeclarativeModel.reconstruct(spec(List.of(new LayerNormStep(Shape.of(extents), DataType.FLOAT32,
                epsilon(DataType.FLOAT32)))));
        assertThrows(IllegalArgumentException.class, () -> ModelCheckpoint.captureWholeModel(giant, closed, LIMITS));
    }

    @Test
    void aliasesAreExactCurrentIdentitiesAndResaveObservesSplitsAndJoins() throws Exception {
        Tensor shared = tensor(DataType.FLOAT32, Shape.of(2, 2), 1, 2, 3, 4);
        var source = new Sequential(List.of(new Linear(shared), new Linear(shared)));
        try (Engine engine = Engine.standard()) {
            var loaded = ModelCheckpoint.loadWholeModel(ModelCheckpoint.captureWholeModel(source, engine, LIMITS), LIMITS);
            var first = loaded.parametersRecursively().get("0.weight");
            var second = loaded.parametersRecursively().get("1.weight");
            assertSame(first.value(), second.value());
            second.replace(tensor(DataType.FLOAT32, Shape.of(2, 2), 1, 2, 3, 4));
            var split = ModelCheckpoint.loadWholeModel(ModelCheckpoint.captureWholeModel(loaded, engine, LIMITS), LIMITS);
            assertTrue(split.describe().aliases().isEmpty());
            assertNotSame(split.parametersRecursively().get("0.weight").value(),
                    split.parametersRecursively().get("1.weight").value());
            second.replace(first.value());
            var joined = ModelCheckpoint.loadWholeModel(ModelCheckpoint.captureWholeModel(loaded, engine, LIMITS), LIMITS);
            assertEquals(List.of(new AliasGroup(List.of("0.weight", "1.weight"))), joined.describe().aliases());
        }
    }

    @Test
    void everyLayerFamilyReconstructsCompleteValuesAndAttributesAcrossFloatingTypes() throws Exception {
        for (DataType type : FLOATS) {
            var specification = spec(List.of(new LinearStep(2, 3, true, type, true),
                    new LayerNormStep(Shape.of(3), type, epsilon(type)),
                    new EmbeddingStep(3, 2, type),
                    new Conv1dStep(4, 6, 3, true, type, new Conv1dAttrs(2, 1, 2, 2)),
                    new Conv2dStep(4, 6, 2, 3, true, type, new Conv2dAttrs(2, 3, 1, 2, 2, 1, 2)),
                    new Conv3dStep(4, 6, 2, 3, 2, true, type,
                            new Conv3dAttrs(2, 1, 3, 1, 2, 3, 2, 1, 2, 2))));
            var source = DeclarativeModel.reconstruct(specification);
            var entries = new ArrayList<StateEntry>();
            for (var state : specification.stateDescriptors()) {
                long[] raw = new long[Math.toIntExact(state.shape().knownElementCount().orElseThrow())];
                for (int i = 0; i < raw.length; i++) raw[i] = i + 1;
                entries.add(new StateEntry(state.path(), state.kind(), tensor(type, state.shape(), raw)));
            }
            source.loadStateDictionary(new StateDictionary(entries));
            try (Engine engine = Engine.standard()) {
                var whole = ModelCheckpoint.captureWholeModel(source, engine, LIMITS);
                var loaded = ModelCheckpoint.loadWholeModel(whole, LIMITS);
                assertEquals(specification.root(), loaded.describe().root());
                assertEquals(specification.stateDescriptors(), loaded.describe().stateDescriptors());
                assertEnvelopeEquals(whole, ModelCheckpoint.captureWholeModel(loaded, engine, LIMITS));
            }
        }
    }

    @Test
    void validValuesDoNotExcuseMalformedDefinitionHashSchemaOrAliasBytes() throws Exception {
        var source = source();
        ArtifactEnvelope whole;
        try (Engine engine = Engine.standard()) { whole = ModelCheckpoint.captureWholeModel(source, engine, LIMITS); }
        byte[] definition = bytes(whole.entries().getFirst());
        byte[] bad = definition.clone(); bad[11] = 2;
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.loadWholeModel(replace(whole, "definition/schema", bad), LIMITS));
        byte[] changed = definition.clone(); changed[36] = 1; // Valid config change, now inconsistent with saved digest.
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.loadWholeModel(replace(whole, "definition/schema", changed), LIMITS));
        try (Engine engine = Engine.standard()) {
            // Valid values with the right config digest, but wrong descriptors.
            var values = ModelCheckpoint.capture(new Sequential(List.of(new Linear(tensor(DataType.FLOAT32,
                    Shape.of(2, 1), 1, 2)))), engine, sha(definition), null, LIMITS);
            assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.loadWholeModel(combine(definition, values), LIMITS));
            var aliasSpec = new ModelSpecification(new Sequence(List.of(
                    new LinearStep(1, 1, false, DataType.FLOAT32, false),
                    new LinearStep(1, 1, false, DataType.FLOAT32, false))),
                    List.of(new AliasGroup(List.of("0.weight", "1.weight"))));
            byte[] aliasDefinition = WholeModelDefinitionCodec.encode(aliasSpec);
            var distinct = new Sequential(List.of(new Linear(tensor(DataType.FLOAT32, Shape.of(1, 1), 1)),
                    new Linear(tensor(DataType.FLOAT32, Shape.of(1, 1), 2))));
            var conflict = ModelCheckpoint.capture(distinct, engine, sha(aliasDefinition), null, LIMITS);
            assertTrue(assertThrows(ArtifactFormatException.class,
                    () -> ModelCheckpoint.loadWholeModel(combine(aliasDefinition, conflict), LIMITS))
                    .getMessage().contains("alias bytes"));
        }
    }

    @Test
    void missingExtraKeysWrongKindsAndValuesOnlyFileGiveClosedDiagnostics() throws Exception {
        ArtifactEnvelope whole;
        try (Engine engine = Engine.standard()) { whole = ModelCheckpoint.captureWholeModel(source(), engine, LIMITS); }
        var values = component(whole);
        assertTrue(assertThrows(ArtifactFormatException.class,
                () -> ModelCheckpoint.loadWholeModel(values, LIMITS)).getMessage().contains("target-load"));
        Path file = directory.resolve("values.chk"); ArtifactFile.save(file, values, LIMITS);
        assertTrue(assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.load(file)).getMessage().contains("definition"));
        var entries = new ArrayList<>(whole.entries()); entries.add(new ArtifactEnvelope.Entry("extra", new byte[0]));
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.loadWholeModel(new ArtifactEnvelope("whole-model.v1", entries), LIMITS));
        var missing = new ArrayList<>(whole.entries()); missing.removeLast();
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.loadWholeModel(new ArtifactEnvelope("whole-model.v1", missing), LIMITS));
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.loadWholeModel(new ArtifactEnvelope("whole-model.v2", whole.entries()), LIMITS));
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.loadWholeModel(new ArtifactEnvelope("whole-model.v1", List.of()), LIMITS));
    }

    @Test
    void requiredNullArgumentsFailWithoutPublication() {
        Engine closed = Engine.standard(); closed.close();
        var source = source();
        assertThrows(NullPointerException.class, () -> ModelCheckpoint.save(null, source, closed));
        assertThrows(NullPointerException.class, () -> ModelCheckpoint.captureWholeModel(null, closed, LIMITS));
        assertThrows(NullPointerException.class, () -> ModelCheckpoint.captureWholeModel(source, null, LIMITS));
        assertThrows(NullPointerException.class, () -> ModelCheckpoint.captureWholeModel(source, closed, null));
        assertThrows(NullPointerException.class, () -> ModelCheckpoint.captureWholeModel(source, closed, null, LIMITS));
        assertThrows(NullPointerException.class, () -> ModelCheckpoint.loadWholeModel(null, LIMITS));
        assertThrows(NullPointerException.class, () -> ModelCheckpoint.loadWholeModel(new ArtifactEnvelope("whole-model.v1", List.of()), null, LIMITS));
        assertThrows(NullPointerException.class, () -> ModelCheckpoint.load((Path) null));
        assertThrows(NullPointerException.class, () -> ModelCheckpoint.load(directory.resolve("missing"), null, LIMITS));
    }

    private static Sequential source() { return new Sequential(List.of(new Linear(tensor(DataType.FLOAT32, Shape.of(1, 2), 1, 2)))); }
    static Tensor tensor(DataType type, Shape shape, long... bits) {
        Tensor tensor = TensorFactory.allocate(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), true));
        var segment = tensor.hostStorage().orElseThrow().segment();
        for (int i = 0; i < bits.length; i++) switch (type.byteWidth()) {
            case 8 -> segment.set(ValueLayout.JAVA_LONG, i * 8L, bits[i]);
            case 4 -> segment.set(ValueLayout.JAVA_INT, i * 4L, (int) bits[i]);
            case 2 -> segment.set(ValueLayout.JAVA_SHORT, i * 2L, (short) bits[i]);
            default -> throw new IllegalArgumentException();
        }
        return tensor;
    }
    static byte[] bytes(ArtifactEnvelope.Entry entry) { byte[] b = new byte[entry.size()]; entry.payload().get(b); return b; }
    static ArtifactEnvelope component(ArtifactEnvelope whole) {
        return new ArtifactEnvelope("model.v1", whole.entries().stream().filter(e -> e.key().startsWith("model/"))
                .map(e -> new ArtifactEnvelope.Entry(e.key().substring(6), bytes(e))).toList());
    }
    static ArtifactEnvelope combine(byte[] definition, ArtifactEnvelope values) {
        var entries = new ArrayList<ArtifactEnvelope.Entry>();
        entries.add(new ArtifactEnvelope.Entry("definition/schema", definition));
        for (var e : values.entries()) entries.add(new ArtifactEnvelope.Entry("model/" + e.key(), bytes(e)));
        return new ArtifactEnvelope("whole-model.v1", entries);
    }
    private static ArtifactEnvelope replace(ArtifactEnvelope envelope, String key, byte[] bytes) {
        return new ArtifactEnvelope(envelope.kind(), envelope.entries().stream()
                .map(e -> e.key().equals(key) ? new ArtifactEnvelope.Entry(key, bytes) : e).toList());
    }
    private static int manifest(ArtifactEnvelope e) { return 8 + e.kind().length() + e.entries().stream().mapToInt(x -> 44 + x.key().length()).sum(); }
    private static byte[] sha(byte[] bytes) throws Exception { return MessageDigest.getInstance("SHA-256").digest(bytes); }
    private static void assertEnvelopeEquals(ArtifactEnvelope expected, ArtifactEnvelope actual) {
        assertEquals(expected.kind(), actual.kind()); assertEquals(expected.entries().size(), actual.entries().size());
        for (int i = 0; i < expected.entries().size(); i++) {
            assertEquals(expected.entries().get(i).key(), actual.entries().get(i).key());
            assertArrayEquals(bytes(expected.entries().get(i)), bytes(actual.entries().get(i)));
        }
    }
}
