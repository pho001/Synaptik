package io.github.pho001.synaptik.nn.checkpoint;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactEnvelope;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFile;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.module.Module;
import io.github.pho001.synaptik.nn.module.StateKind;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ReadOnlyBufferException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class ModelCheckpointTest {
    private static final ArtifactLimits LIMITS = new ArtifactLimits(16384, 8192, 32, 32, 8192, 8192);
    private static final byte[] CONFIG = new byte[32];
    @TempDir Path directory;

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void directLeavesSaveLoadAndResaveSevenTypesWithExactBits(boolean nativeSource) throws Exception {
        var source = new Fixture();
        var target = new Fixture();
        var cases = List.of(
                new Case("f64", DataType.FLOAT64, Shape.of(8), words(8,
                        0, 0x8000000000000000L, 1, 0x8000000000000001L,
                        0x7ff0000000000000L, 0xfff0000000000000L,
                        0x7ff8000000000123L, 0xfff0000000000456L), double[].class),
                new Case("f32", DataType.FLOAT32, Shape.of(8), words(4,
                        0, 0x80000000L, 1, 0x80000001L, 0x7f800000L, 0xff800000L,
                        0x7fc01234L, 0xff801234L), float[].class),
                new Case("bf16", DataType.BFLOAT16, Shape.of(8), words(2,
                        0, 0x8000, 1, 0x8001, 0x7f80, 0xff80, 0x7fc1, 0xff81), short[].class),
                new Case("f16", DataType.FLOAT16, Shape.of(8), words(2,
                        0, 0x8000, 1, 0x8001, 0x7c00, 0xfc00, 0x7e11, 0xfc11), short[].class),
                new Case("i32", DataType.INT32, Shape.of(4), words(4,
                        0x80000000L, 0x7fffffffL, 0x87654321L, 0xffffffffL), int[].class),
                new Case("i64", DataType.INT64, Shape.of(4), words(8,
                        Long.MIN_VALUE, Long.MAX_VALUE, 0xfedcba9876543210L, -1), long[].class),
                new Case("bool", DataType.BOOL, Shape.of(2), new byte[] {1, 0}, byte[].class),
                new Case("scalar", DataType.FLOAT32, Shape.scalar(), words(4, 1), float[].class),
                new Case("empty", DataType.FLOAT32, Shape.of(0, 2), new byte[0], float[].class));
        for (Case value : cases)
            target.addBuffer(value.name, leaf(value.type, value.shape, new byte[value.bytes.length], false));
        Path file = directory.resolve("model.chk");
        byte[] input = new byte[32]; input[0] = 9;
        ArtifactEnvelope captured;
        Map<String, ByteBuffer> capturedBytes;
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            for (Case value : cases)
                source.addBuffer(value.name, nativeSource
                        ? nativeLeaf(arena, value.type, value.shape, value.bytes, false)
                        : leaf(value.type, value.shape, value.bytes, false));
            var before = source.stateDictionary().entries();
            captured = ModelCheckpoint.capture(source, engine, CONFIG, input, LIMITS);
            capturedBytes = ModelCheckpoint.canonicalStateBytes(captured, LIMITS);
            ModelCheckpoint.save(file, source, engine, CONFIG, input, LIMITS);
            var after = source.stateDictionary().entries();
            for (int i = 0; i < before.size(); i++) assertSame(before.get(i).value(), after.get(i).value());
        }
        var inspected = ModelCheckpoint.inspect(file, LIMITS);
        var envelopeInspection = ModelCheckpoint.inspect(captured, LIMITS);
        assertEquals(inspected.states(), envelopeInspection.states());
        assertArrayEquals(inspected.configDigest(), envelopeInspection.configDigest());
        assertArrayEquals(inspected.inputDigest(), envelopeInspection.inputDigest());
        assertArrayEquals(inspected.schemaFingerprint(), envelopeInspection.schemaFingerprint());
        Path composed = directory.resolve("composed.chk");
        ArtifactFile.save(composed, captured, LIMITS);
        assertArrayEquals(Files.readAllBytes(file), Files.readAllBytes(composed));
        assertEquals(cases.stream().map(Case::name).toList(),
                inspected.states().stream().map(ModelCheckpointInspection.State::path).toList());
        assertEquals(StateKind.BUFFER, inspected.states().getFirst().kind());
        assertEquals(DataType.FLOAT64, inspected.states().getFirst().dataType());
        assertEquals(64, inspected.states().getFirst().byteLength());
        for (int i = 0; i < cases.size(); i++) {
            assertEquals(cases.get(i).type, inspected.states().get(i).dataType());
            assertEquals(cases.get(i).shape, inspected.states().get(i).shape());
            assertArrayEquals(cases.get(i).bytes, payload(ArtifactFile.read(file, LIMITS), i + 1));
            ByteBuffer view = capturedBytes.get(cases.get(i).name);
            assertTrue(view.isReadOnly());
            assertEquals(ByteOrder.BIG_ENDIAN, view.order());
            assertEquals(0, view.position());
            assertEquals(cases.get(i).bytes.length, view.limit());
            assertEquals(cases.get(i).bytes.length, view.capacity());
            byte[] bytes = new byte[view.remaining()]; view.get(bytes);
            assertArrayEquals(cases.get(i).bytes, bytes);
        }
        input[0] = 0;
        assertEquals(9, inspected.inputDigest()[0]);
        inspected.configDigest()[0] = 5;
        inspected.schemaFingerprint()[0] = 5;
        assertArrayEquals(CONFIG, inspected.configDigest());
        assertNotEquals(5, inspected.schemaFingerprint()[0]);
        byte[] expectedInput = new byte[32]; expectedInput[0] = 9;
        byte[] wrongInput = expectedInput.clone(); wrongInput[31] = 1;
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(captured, target, CONFIG, wrongInput, LIMITS));
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(captured, target, CONFIG, null, LIMITS));
        assertThrows(NullPointerException.class, () -> ModelCheckpoint.inspect(captured, null));
        assertThrows(NullPointerException.class, () -> ModelCheckpoint.canonicalStateBytes(captured, null));
        assertThrows(NullPointerException.class,
                () -> ModelCheckpoint.load(captured, target, CONFIG, expectedInput, null));
        ModelCheckpoint.load(file, target, CONFIG, expectedInput, LIMITS);
        ModelCheckpoint.load(captured, target, CONFIG, expectedInput, LIMITS);
        try (Engine engine = Engine.standard()) {
            Path resaved = directory.resolve("resaved.chk");
            ModelCheckpoint.save(resaved, target, engine, CONFIG, expectedInput, LIMITS);
            assertArrayEquals(inspected.schemaFingerprint(),
                    ModelCheckpoint.inspect(resaved, LIMITS).schemaFingerprint());
            assertArrayEquals(Files.readAllBytes(file), Files.readAllBytes(resaved));
            ModelCheckpoint.load(resaved, target, CONFIG, expectedInput, LIMITS);
            var restored = target.stateDictionary().entries();
            for (int i = 0; i < cases.size(); i++) {
                Case value = cases.get(i);
                Tensor tensor = restored.get(i).value();
                assertEquals(value.type, tensor.descriptor().dataType());
                assertEquals(value.shape, tensor.descriptor().shape());
                assertEquals(LayoutDescriptor.contiguous(value.shape), tensor.descriptor().layout().orElseThrow());
                assertFalse(tensor.descriptor().requiresGrad());
                assertInstanceOf(value.carrier, tensor.hostStorage().orElseThrow()
                        .segment().heapBase().orElseThrow());
                var materialized = engine.compute(tensor.contiguous(), 1024);
                byte[] actual = new byte[(int) materialized.byteSize()];
                materialized.bytes().get(actual);
                assertArrayEquals(value.bytes, actual, value.name);
            }
        }
    }

    @Test
    void targetAliasesIncludingMixedRolesShareOneRestoredTensorAndConflictFailsClosed() throws Exception {
        var source = new Fixture();
        Path file = directory.resolve("alias.chk");
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            source.addParameter("weight", nativeLeaf(arena, DataType.FLOAT32, Shape.scalar(),
                    words(4, 0x3f800000), true).contiguous());
            source.addBuffer("shadow", nativeLeaf(arena, DataType.FLOAT32, Shape.scalar(),
                    words(4, 0x3f800000), false).contiguous());
            ModelCheckpoint.save(file, source, engine, CONFIG, null, LIMITS);
        }
        Tensor shared = leaf(DataType.FLOAT32, Shape.scalar(), words(4, 0x40000000), true);
        var target = new Fixture();
        target.addParameter("weight", shared);
        target.addBuffer("shadow", shared);
        ArtifactEnvelope envelope = ArtifactFile.read(file, LIMITS);
        ModelCheckpoint.load(envelope, target, CONFIG, null, LIMITS);
        var loaded = target.stateDictionary().entries();
        assertSame(loaded.get(0).value(), loaded.get(1).value());
        assertNotSame(shared, loaded.get(0).value());
        assertTrue(loaded.get(0).value().descriptor().requiresGrad());
        try (Engine engine = Engine.standard()) {
            Path resaved = directory.resolve("alias-resaved.chk");
            ModelCheckpoint.save(resaved, target, engine, CONFIG, null, LIMITS);
            assertArrayEquals(Files.readAllBytes(file), Files.readAllBytes(resaved));
            ModelCheckpoint.load(resaved, target, CONFIG, null, LIMITS);
            assertSame(target.stateDictionary().entries().get(0).value(),
                    target.stateDictionary().entries().get(1).value());
        }
        loaded = target.stateDictionary().entries();

        byte[] schema = payload(ArtifactFile.read(file, LIMITS), 0);
        var conflict = new ArtifactEnvelope("model.v1", List.of(
                new ArtifactEnvelope.Entry("schema", schema),
                new ArtifactEnvelope.Entry("state/00000000", words(4, 0x3f800000)),
                new ArtifactEnvelope.Entry("state/00000001", words(4, 0x40000000))));
        ArtifactFile.save(file, conflict, LIMITS);
        Tensor before = loaded.getFirst().value();
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(file, target, CONFIG, null, LIMITS));
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(conflict, target, CONFIG, null, LIMITS));
        assertSame(before, target.stateDictionary().entries().getFirst().value());
        assertSame(before, target.stateDictionary().entries().get(1).value());
    }

    @Test
    void rejectsInvalidBoolDigestsSchemaAndUnboundTargetWithoutInstallation() throws Exception {
        var source = new Fixture();
        Path file = directory.resolve("flag.chk");
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            source.addBuffer("flag", nativeLeaf(arena, DataType.BOOL, Shape.scalar(),
                    new byte[] {1}, false).contiguous());
            ModelCheckpoint.save(file, source, engine, CONFIG, null, LIMITS);
        }
        var target = new Fixture();
        Tensor old = leaf(DataType.BOOL, Shape.scalar(), new byte[] {0}, false);
        target.addBuffer("flag", old);
        ArtifactEnvelope valid = ArtifactFile.read(file, LIMITS);
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(valid, target, CONFIG, new byte[32], LIMITS));
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(file, target, CONFIG, new byte[32], LIMITS));
        byte[] wrong = CONFIG.clone(); wrong[0] = 1;
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(valid, target, wrong, null, LIMITS));
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(file, target, wrong, null, LIMITS));
        assertSame(old, target.stateDictionary().entries().getFirst().value());
        assertThrows(IllegalStateException.class,
                () -> ModelCheckpoint.load(file, new LazyFixture(), CONFIG, null, LIMITS));
        assertThrows(IllegalStateException.class,
                () -> ModelCheckpoint.load(valid, new LazyFixture(), CONFIG, null, LIMITS));
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(valid, new Fixture(), CONFIG, null, LIMITS));
        var wrongPath = new Fixture(); wrongPath.addBuffer("other", old);
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(valid, wrongPath, CONFIG, null, LIMITS));
        assertSame(old, wrongPath.stateDictionary().entries().getFirst().value());
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(file, new Fixture(), CONFIG, null, LIMITS));

        ModelCheckpointInspection prior = ModelCheckpoint.inspect(file, LIMITS);
        ModelCheckpointInspection priorEnvelope = ModelCheckpoint.inspect(valid, LIMITS);
        ArtifactEnvelope invalid = new ArtifactEnvelope("model.v1", List.of(
                new ArtifactEnvelope.Entry("schema", payload(valid, 0)),
                new ArtifactEnvelope.Entry("state/00000000", new byte[] {2})));
        assertEnvelopeRejected(invalid, LIMITS, target);
        assertEquals(prior.states(), priorEnvelope.states());
        ArtifactFile.save(file, invalid, LIMITS);
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.inspect(file, LIMITS));
        assertThrows(ArtifactFormatException.class,
                () -> ModelCheckpoint.load(file, target, CONFIG, null, LIMITS));
        assertSame(old, target.stateDictionary().entries().getFirst().value());
        assertEquals("flag", prior.states().getFirst().path());
        assertArrayEquals(CONFIG, prior.configDigest());
    }

    @Test
    void laterDescriptorMismatchLeavesEarlierBindingUntouched() throws Exception {
        var source = new Fixture();
        Path file = directory.resolve("mismatch.chk");
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            source.addBuffer("first", nativeLeaf(arena, DataType.FLOAT32, Shape.scalar(),
                    words(4, 0x3f800000), false).contiguous());
            source.addBuffer("second", nativeLeaf(arena, DataType.INT32, Shape.scalar(),
                    words(4, 5), false).contiguous());
            ModelCheckpoint.save(file, source, engine, CONFIG, null, LIMITS);
        }
        var target = new Fixture();
        Tensor first = leaf(DataType.FLOAT32, Shape.scalar(), words(4, 0x40000000), false);
        Tensor second = leaf(DataType.INT32, Shape.of(1), words(4, 7), false);
        target.addBuffer("first", first);
        target.addBuffer("second", second);
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(file, target, CONFIG, null, LIMITS));
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(ArtifactFile.read(file, LIMITS), target, CONFIG, null, LIMITS));
        var state = target.stateDictionary().entries();
        assertSame(first, state.getFirst().value());
        assertSame(second, state.get(1).value());
        var wrongRole = new Fixture();
        Tensor parameter = leaf(DataType.FLOAT32, Shape.scalar(), words(4, 0x40000000), true);
        wrongRole.addParameter("first", parameter);
        wrongRole.addBuffer("second", second);
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(ArtifactFile.read(file, LIMITS), wrongRole, CONFIG, null, LIMITS));
        assertSame(parameter, wrongRole.stateDictionary().entries().getFirst().value());
    }

    @Test
    void emptyStateNeedsNoEngineAndReadLimitsFailBeforeLoad() throws Exception {
        Engine closed = Engine.standard();
        closed.close();
        Path file = directory.resolve("empty.chk");
        ModelCheckpoint.save(file, new Fixture(), closed, CONFIG, null, LIMITS);
        ArtifactEnvelope empty = ModelCheckpoint.capture(new Fixture(), closed, CONFIG, null, LIMITS);
        assertTrue(ModelCheckpoint.inspect(empty, LIMITS).states().isEmpty());
        assertTrue(ModelCheckpoint.canonicalStateBytes(empty, LIMITS).isEmpty());
        assertDoesNotThrow(() -> ModelCheckpoint.load(empty, new Fixture(), CONFIG, null, LIMITS));
        assertTrue(ModelCheckpoint.inspect(file, LIMITS).states().isEmpty());
        assertDoesNotThrow(() -> ModelCheckpoint.load(file, new Fixture(), CONFIG, null, LIMITS));
        ArtifactLimits tinyFile = new ArtifactLimits(48, 8192, 32, 32, 8192, 8192);
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.inspect(file, tinyFile));
        assertThrows(ArtifactFormatException.class,
                () -> ModelCheckpoint.load(file, new Fixture(), CONFIG, null, tinyFile));
        assertEnvelopeRejected(empty, tinyFile, new Fixture());
    }

    @Test
    void canonicalViewsHaveImmutableStructureAndIndependentAliasCursors() throws Exception {
        var source = new Fixture();
        Tensor shared = leaf(DataType.FLOAT32, Shape.of(2), words(4, 0x80000000L, 0x7fc01234L), true);
        source.addParameter("z重み/🦉 space", shared);
        source.addBuffer("a", shared);
        ArtifactEnvelope envelope;
        try (Engine engine = Engine.standard()) {
            envelope = ModelCheckpoint.capture(source, engine, CONFIG, null, LIMITS);
        }
        var first = ModelCheckpoint.canonicalStateBytes(envelope, LIMITS);
        var second = ModelCheckpoint.canonicalStateBytes(envelope, LIMITS);
        assertEquals(List.of("z重み/🦉 space", "a"), new ArrayList<>(first.keySet()));
        assertThrows(UnsupportedOperationException.class, first::clear);
        assertThrows(UnsupportedOperationException.class, () -> first.put("new", ByteBuffer.allocate(0)));
        assertThrows(UnsupportedOperationException.class,
                () -> first.entrySet().iterator().next().setValue(ByteBuffer.allocate(0)));
        ByteBuffer moved = first.get("z重み/🦉 space");
        assertNotSame(moved, first.get("a"));
        assertNotSame(moved, second.get("z重み/🦉 space"));
        assertThrows(ReadOnlyBufferException.class, () -> moved.putInt(0, 7));
        moved.position(4).limit(4).order(ByteOrder.LITTLE_ENDIAN);
        for (ByteBuffer unaffected : List.of(first.get("a"), second.get("z重み/🦉 space"),
                ModelCheckpoint.canonicalStateBytes(envelope, LIMITS).get("z重み/🦉 space"),
                envelope.entries().get(1).payload())) {
            assertEquals(0, unaffected.position());
            assertEquals(8, unaffected.limit());
            assertEquals(ByteOrder.BIG_ENDIAN, unaffected.order());
            assertEquals(0x80000000, unaffected.getInt(0));
            assertEquals(0x7fc01234, unaffected.getInt(4));
        }
    }

    @Test
    void standaloneEnvelopeOperationsApplyEveryExactLimitIncludingAliases() throws Exception {
        var source = new Fixture();
        Tensor shared = leaf(DataType.INT32, Shape.of(64), new byte[256], false);
        source.addBuffer("first", shared); source.addBuffer("second", shared);
        ArtifactEnvelope envelope;
        try (Engine engine = Engine.standard()) {
            envelope = ModelCheckpoint.capture(source, engine, CONFIG, null, LIMITS);
        }
        int schema = envelope.entries().getFirst().size();
        int manifest = 66 + 58 * 2; // v1 kind/schema plus two ordinal descriptors
        long aggregate = schema + 512L;
        long file = 48L + manifest + aggregate;
        var exact = new ArtifactLimits(file, manifest, 3, 14, 256, aggregate);
        assertEquals(2, ModelCheckpoint.inspect(envelope, exact).states().size());
        assertEquals(2, ModelCheckpoint.canonicalStateBytes(envelope, exact).size());
        assertDoesNotThrow(() -> ModelCheckpoint.load(envelope, source, CONFIG, null, exact));
        try (Engine engine = Engine.standard()) {
            assertDoesNotThrow(() -> ModelCheckpoint.capture(source, engine, CONFIG, null, exact));
        }
        List<ArtifactLimits> below = List.of(
                new ArtifactLimits(file - 1, manifest, 3, 14, 256, aggregate),
                new ArtifactLimits(file, manifest - 1, 3, 14, 256, aggregate),
                new ArtifactLimits(file, manifest, 2, 14, 256, aggregate),
                new ArtifactLimits(file, manifest, 3, 13, 256, aggregate),
                new ArtifactLimits(file, manifest, 3, 14, 255, aggregate),
                new ArtifactLimits(file, manifest, 3, 14, 256, aggregate - 1));
        Engine closed = Engine.standard(); closed.close();
        var before = source.stateDictionary().entries();
        for (ArtifactLimits limits : below) {
            assertEnvelopeRejected(envelope, limits, source);
            assertThrows(IllegalArgumentException.class,
                    () -> ModelCheckpoint.capture(source, closed, CONFIG, null, limits));
        }
        var empty = ModelCheckpoint.capture(new Fixture(), closed, CONFIG, null, LIMITS);
        var exactEmpty = new ArtifactLimits(163, 66, 1, 8, 49, 49);
        assertEquals(0, ModelCheckpoint.inspect(empty, exactEmpty).states().size());
        assertEquals(0, ModelCheckpoint.canonicalStateBytes(empty, exactEmpty).size());
        assertDoesNotThrow(() -> ModelCheckpoint.load(empty, new Fixture(), CONFIG, null, exactEmpty));
        assertDoesNotThrow(() -> ModelCheckpoint.capture(new Fixture(), closed, CONFIG, null, exactEmpty));
        assertEnvelopeRejected(empty, new ArtifactLimits(163, 66, 1, 7, 49, 49), new Fixture());
        var belowSchema = new ArtifactLimits(163, 66, 1, 8, 48, 49);
        assertEnvelopeRejected(empty, belowSchema, new Fixture());
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.capture(new Fixture(), closed, CONFIG, null, belowSchema));
        for (int i = 0; i < before.size(); i++)
            assertSame(before.get(i).value(), source.stateDictionary().entries().get(i).value());
    }

    @Test
    void preservesFileArgumentFailureOrderingAndEnvelopeNullContracts() {
        Path missing = directory.resolve("missing.chk");
        var target = new Fixture();
        assertThrows(NullPointerException.class,
                () -> ModelCheckpoint.save(null, null, null, null, null, null));
        assertThrows(NullPointerException.class,
                () -> ModelCheckpoint.load(missing, null, new byte[31], null, null));
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load((Path) null, target, new byte[31], null, null));
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load(missing, target, CONFIG, new byte[31], null));
        assertThrows(NullPointerException.class, () -> ModelCheckpoint.inspect((Path) null, null));
        assertThrows(NullPointerException.class,
                () -> ModelCheckpoint.inspect((ArtifactEnvelope) null, LIMITS));
        assertThrows(NullPointerException.class,
                () -> ModelCheckpoint.canonicalStateBytes(null, LIMITS));
        assertThrows(NullPointerException.class,
                () -> ModelCheckpoint.load((ArtifactEnvelope) null, target, CONFIG, null, LIMITS));
        assertThrows(IllegalArgumentException.class,
                () -> ModelCheckpoint.load((ArtifactEnvelope) null, target, new byte[31], null, null));
        assertThrows(NullPointerException.class,
                () -> ModelCheckpoint.capture(target, null, CONFIG, null, LIMITS));
    }

    /**
     * Checks independent facade rejection and complete binding preservation for invalid bytes/bounds.
     *
     * @param envelope detached image expected to fail semantic decode
     * @param limits bounds under which the image is invalid
     * @param target fully bound tree whose exact bindings must remain unchanged
     */
    private static void assertEnvelopeRejected(ArtifactEnvelope envelope, ArtifactLimits limits,
                                                Module target) {
        var before = target.stateDictionary().entries();
        assertThrows(ArtifactFormatException.class, () -> ModelCheckpoint.inspect(envelope, limits));
        assertThrows(ArtifactFormatException.class,
                () -> ModelCheckpoint.canonicalStateBytes(envelope, limits));
        assertThrows(ArtifactFormatException.class,
                () -> ModelCheckpoint.load(envelope, target, CONFIG, null, limits));
        var after = target.stateDictionary().entries();
        for (int i = 0; i < before.size(); i++) assertSame(before.get(i).value(), after.get(i).value());
    }

    private static byte[] payload(ArtifactEnvelope envelope, int index) {
        var entry = envelope.entries().get(index);
        byte[] result = new byte[entry.size()]; entry.payload().get(result); return result;
    }

    private static byte[] words(int width, long... bits) {
        ByteBuffer out = ByteBuffer.allocate(width * bits.length).order(ByteOrder.BIG_ENDIAN);
        for (long bit : bits) switch (width) {
            case 8 -> out.putLong(bit);
            case 4 -> out.putInt((int) bit);
            case 2 -> out.putShort((short) bit);
            default -> throw new IllegalArgumentException();
        }
        return out.array();
    }

    private static Tensor leaf(DataType type, Shape shape, byte[] canonical, boolean grad) {
        Tensor tensor = TensorFactory.allocate(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad));
        ByteBuffer source = ByteBuffer.wrap(canonical).order(ByteOrder.BIG_ENDIAN);
        MemorySegment destination = tensor.hostStorage().orElseThrow().segment();
        long offset = 0;
        switch (type) {
            case FLOAT64, INT64 -> {
                while (source.hasRemaining()) {
                    destination.set(ValueLayout.JAVA_LONG.withOrder(ByteOrder.nativeOrder()), offset, source.getLong());
                    offset += 8;
                }
            }
            case FLOAT32, INT32 -> {
                while (source.hasRemaining()) {
                    destination.set(ValueLayout.JAVA_INT.withOrder(ByteOrder.nativeOrder()), offset, source.getInt());
                    offset += 4;
                }
            }
            case BFLOAT16, FLOAT16 -> {
                while (source.hasRemaining()) {
                    destination.set(ValueLayout.JAVA_SHORT.withOrder(ByteOrder.nativeOrder()), offset, source.getShort());
                    offset += 2;
                }
            }
            case BOOL -> { while (source.hasRemaining()) destination.set(ValueLayout.JAVA_BYTE, offset++, source.get()); }
        }
        return tensor;
    }

    private static Tensor nativeLeaf(Arena arena, DataType type, Shape shape,
                                     byte[] canonical, boolean grad) {
        TensorDescriptor descriptor = new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), grad);
        MemorySegment segment = arena.allocate(canonical.length, Math.max(1, type.byteWidth()));
        Tensor tensor = TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(type, shape.knownElementCount().orElseThrow(), segment)));
        ByteBuffer source = ByteBuffer.wrap(canonical).order(ByteOrder.BIG_ENDIAN);
        long offset = 0;
        switch (type) {
            case FLOAT64, INT64 -> {
                while (source.hasRemaining()) { segment.set(ValueLayout.JAVA_LONG, offset, source.getLong()); offset += 8; }
            }
            case FLOAT32, INT32 -> {
                while (source.hasRemaining()) { segment.set(ValueLayout.JAVA_INT, offset, source.getInt()); offset += 4; }
            }
            case BFLOAT16, FLOAT16 -> {
                while (source.hasRemaining()) { segment.set(ValueLayout.JAVA_SHORT, offset, source.getShort()); offset += 2; }
            }
            case BOOL -> { while (source.hasRemaining()) segment.set(ValueLayout.JAVA_BYTE, offset++, source.get()); }
        }
        return tensor;
    }

    private record Case(String name, DataType type, Shape shape, byte[] bytes, Class<?> carrier) { }

    private static final class Fixture extends Module {
        void addBuffer(String name, Tensor value) { super.buffer(name, value); }
        void addParameter(String name, Tensor value) { super.parameter(name, value); }
    }

    private static final class LazyFixture extends Module {
        LazyFixture() { reserveParameter("future", value -> { }); }
    }
}
