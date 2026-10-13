package io.github.pho001.synaptik.checkpoint.model;

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
import java.nio.ReadOnlyBufferException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelArtifactTest {
    private static final ArtifactLimits LIMITS = new ArtifactLimits(8192, 4096, 20, 32, 4096, 4096);
    private static final byte[] CONFIG = new byte[32];

    @TempDir Path directory;

    @Test
    void capturesExactTypedBytesAndSurvivesEngineClosure() throws Exception {
        var module = new Fixture();
        ModelArtifactCandidate candidate;
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            module.add("f32", tensor(arena, DataType.FLOAT32, new float[] {-0.0f, 1.0f}));
            module.add("f64", tensor(arena, DataType.FLOAT64, new double[] {-0.0d, 1.0d}));
            module.add("bf16", tensor(arena, DataType.BFLOAT16, new short[] {(short) 0x3f80, 0}));
            module.add("f16", tensor(arena, DataType.FLOAT16, new short[] {(short) 0x3c00, 0}));
            module.add("i32", tensor(arena, DataType.INT32, new int[] {-7, 1}));
            module.add("i64", tensor(arena, DataType.INT64, new long[] {-9, 1}));
            module.add("flag", tensor(arena, DataType.BOOL, new byte[] {1, 0}));
            candidate = ModelArtifactCapture.capture(module, engine, CONFIG, null, LIMITS);
        }
        assertEquals(List.of("schema", "state/00000000", "state/00000001", "state/00000002",
                "state/00000003", "state/00000004", "state/00000005", "state/00000006"),
                candidate.envelope().entries().stream().map(ArtifactEnvelope.Entry::key).toList());
        assertArrayEquals(ByteBuffer.allocate(8).putFloat(-0.0f).putFloat(1.0f).array(), payload(candidate, 0));
        assertArrayEquals(ByteBuffer.allocate(16).putDouble(-0.0d).putDouble(1.0d).array(), payload(candidate, 1));
        assertArrayEquals(new byte[] {0x3f, (byte) 0x80, 0, 0}, payload(candidate, 2));
        assertArrayEquals(new byte[] {0x3c, 0, 0, 0}, payload(candidate, 3));
        assertArrayEquals(ByteBuffer.allocate(8).putInt(-7).putInt(1).array(), payload(candidate, 4));
        assertArrayEquals(ByteBuffer.allocate(16).putLong(-9L).putLong(1L).array(), payload(candidate, 5));
        assertArrayEquals(new byte[] {1, 0}, payload(candidate, 6));
        assertThrows(ReadOnlyBufferException.class,
                () -> candidate.envelope().entries().get(1).payload().put((byte) 3));
        assertArrayEquals(CONFIG, candidate.configDigest());
        byte[] mutable = candidate.schemaFingerprint();
        mutable[0] ^= 1;
        assertNotEquals(mutable[0], candidate.schemaFingerprint()[0]);
        assertEquals(7, ModelArtifactCodec.decode(candidate.envelope(), LIMITS).states().size());
    }

    @Test
    void emptyStateUsesNoEngineAndAliasesRemainDistinctPayloads() throws Exception {
        Engine closed = Engine.standard();
        closed.close();
        ModelArtifactCandidate empty = ModelArtifactCapture.capture(new Fixture(), closed, CONFIG, null, LIMITS);
        assertEquals(List.of("schema"), empty.envelope().entries().stream()
                .map(ArtifactEnvelope.Entry::key).toList());
        assertEquals(0, ModelArtifactCodec.decode(empty.envelope(), LIMITS).states().size());

        var module = new Fixture();
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor shared = tensor(arena, DataType.FLOAT32, new float[] {3.0f, 4.0f});
            module.add("first", shared);
            module.add("second", shared);
            ModelArtifactCandidate candidate = ModelArtifactCapture.capture(module, engine, CONFIG, null, LIMITS);
            assertEquals(2, candidate.states().size());
            assertArrayEquals(payload(candidate, 0), payload(candidate, 1));
            assertEquals("first", candidate.states().get(0).path());
            assertEquals("second", candidate.states().get(1).path());
            assertEquals(8, candidate.states().get(1).length());
        }
        ArtifactLimits aliasLimit = new ArtifactLimits(8192, 4096, 20, 32, 4096, 124);
        assertDoesNotThrow(() -> ModelArtifactCodec.preflight(2, 112, 8, aliasLimit));
        assertThrows(IllegalArgumentException.class,
                () -> ModelArtifactCodec.preflight(2, 112, 16, aliasLimit));
        assertThrows(IllegalArgumentException.class, () -> ModelArtifactCapture.capture(
                module, closed, CONFIG, null, aliasLimit));
    }

    @Test
    void unicodePathsAndFingerprintsAreCanonicalAndFileBytesDeterministic() throws Exception {
        var module = new Fixture();
        byte[] config = new byte[32]; config[0] = 7;
        byte[] input = new byte[32]; input[31] = 9;
        ModelArtifactCandidate first;
        ModelArtifactCandidate second;
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            module.add("重み/🦉 space", tensor(arena, DataType.FLOAT32, new float[] {1.0f, 2.0f}));
            first = ModelArtifactCapture.capture(module, engine, config, input, LIMITS);
            second = ModelArtifactCapture.capture(module, engine, config, input, LIMITS);
        }
        config[0] = 0;
        input[31] = 0;
        assertEquals("重み/🦉 space", ModelArtifactCodec.decode(first.envelope(), LIMITS).states().getFirst().path());
        assertEquals("state/00000000", first.envelope().entries().get(1).key());
        assertArrayEquals(first.schemaFingerprint(), second.schemaFingerprint());
        assertEquals(7, first.configDigest()[0]);
        assertEquals(9, first.inputDigest()[31]);
        Path a = directory.resolve("a.art"); Path b = directory.resolve("b.art");
        ArtifactFile.save(a, first.envelope(), LIMITS);
        ArtifactFile.save(b, second.envelope(), LIMITS);
        assertArrayEquals(Files.readAllBytes(a), Files.readAllBytes(b));

        var state = first.states().getFirst();
        var changed = new ModelArtifactCodec.State("different", state.kind(), state.type(), state.shape(), state.length());
        assertFalse(Arrays.equals(first.schemaFingerprint(), ModelArtifactCodec.fingerprint(
                ModelArtifactCodec.encode(List.of(changed), first.configDigest(), first.inputDigest(), LIMITS))));
        assertFalse(Arrays.equals(first.schemaFingerprint(), ModelArtifactCodec.fingerprint(
                ModelArtifactCodec.encode(first.states(), CONFIG, first.inputDigest(), LIMITS))));
        assertFalse(Arrays.equals(first.schemaFingerprint(), ModelArtifactCodec.fingerprint(
                ModelArtifactCodec.encode(first.states(), first.configDigest(), null, LIMITS))));
    }

    @Test
    void fixedIdsAndAllSchemaFieldsAffectFingerprint() throws Exception {
        var kinds = List.of(StateKind.PARAMETER, StateKind.BUFFER);
        var types = List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64, DataType.BOOL, DataType.FLOAT16);
        var states = new ArrayList<ModelArtifactCodec.State>();
        for (int i = 0; i < types.size(); i++)
            states.add(new ModelArtifactCodec.State("p" + i, kinds.get(i % 2), types.get(i),
                    Shape.scalar(), types.get(i).byteWidth()));
        byte[] bytes = ModelArtifactCodec.encode(states, CONFIG, null, LIMITS).bytes();
        ByteBuffer wire = ByteBuffer.wrap(bytes);
        assertEquals("SYNMOD01", new String(bytes, 0, 8, java.nio.charset.StandardCharsets.US_ASCII));
        assertEquals(1, wire.getInt(8));
        int at = 49;
        for (int i = 0; i < types.size(); i++) {
            assertEquals(2, wire.getInt(at)); // two ASCII bytes in p0 ... p6
            at += 4 + 2;
            assertEquals(i % 2 == 0 ? 1 : 2, Byte.toUnsignedInt(bytes[at++]));
            assertEquals(i + 1, Byte.toUnsignedInt(bytes[at++]));
            assertEquals(0, wire.getInt(at)); at += 4;
            assertEquals(types.get(i).byteWidth(), wire.getLong(at)); at += 8;
        }
        byte[] baseline = ModelArtifactCodec.fingerprint(
                ModelArtifactCodec.encode(List.of(states.getFirst()), CONFIG, null, LIMITS));
        for (var alternative : List.of(
                new ModelArtifactCodec.State("p1", StateKind.PARAMETER, DataType.FLOAT64, Shape.scalar(), 8),
                new ModelArtifactCodec.State("p0", StateKind.BUFFER, DataType.FLOAT64, Shape.scalar(), 8),
                new ModelArtifactCodec.State("p0", StateKind.PARAMETER, DataType.FLOAT32, Shape.scalar(), 4),
                new ModelArtifactCodec.State("p0", StateKind.PARAMETER, DataType.FLOAT64, Shape.of(1), 8),
                new ModelArtifactCodec.State("p0", StateKind.PARAMETER, DataType.FLOAT64, Shape.of(2), 16))) {
            assertFalse(Arrays.equals(baseline, ModelArtifactCodec.fingerprint(
                    ModelArtifactCodec.encode(List.of(alternative), CONFIG, null, LIMITS))));
        }
        assertFalse(Arrays.equals(baseline, ModelArtifactCodec.fingerprint(
                ModelArtifactCodec.encode(List.of(states.getFirst(), states.get(1)), CONFIG, null, LIMITS))));
        assertFalse(Arrays.equals(ModelArtifactCodec.fingerprint(
                ModelArtifactCodec.encode(List.of(states.get(0), states.get(1)), CONFIG, null, LIMITS)),
                ModelArtifactCodec.fingerprint(
                        ModelArtifactCodec.encode(List.of(states.get(1), states.get(0)), CONFIG, null, LIMITS))));
        assertEquals("state/99999999", ModelArtifactCodec.ordinal(99_999_999));
        assertThrows(IllegalArgumentException.class, () -> ModelArtifactCodec.ordinal(100_000_000));
        assertThrows(IllegalArgumentException.class, () -> ModelArtifactCodec.encode(
                new java.util.AbstractList<>() {
                    @Override public ModelArtifactCodec.State get(int index) { throw new AssertionError(); }
                    @Override public int size() { return 100_000_001; }
                }, CONFIG, null, LIMITS));
    }

    @Test
    void rejectsCorruptSchemaAndOrdinalCorrespondence() throws Exception {
        var states = List.of(new ModelArtifactCodec.State("aa", StateKind.BUFFER, DataType.FLOAT32,
                Shape.of(1), 4));
        byte[] valid = ModelArtifactCodec.encode(states, CONFIG, null, LIMITS).bytes();
        var payload = new ArtifactEnvelope.Entry("state/00000000", new byte[4]);
        var invalid = new ArrayList<byte[]>();
        invalid.add(flip(valid, 0)); // magic
        invalid.add(setInt(valid, 8, 2)); // version
        invalid.add(flip(valid, 44)); // presence flag
        invalid.add(setInt(valid, 45, 2)); // count
        invalid.add(setInt(valid, 49, -1)); // path length
        invalid.add(setInt(valid, 49, 999));
        invalid.add(setByte(valid, 53, (byte) 0xc0)); // malformed UTF-8
        invalid.add(setByte(valid, 55, (byte) 99)); // kind
        invalid.add(setByte(valid, 56, (byte) 99)); // dtype
        invalid.add(setInt(valid, 57, -1)); // rank
        invalid.add(setLong(valid, 61, -1)); // extent
        invalid.add(setLong(valid, 69, 5)); // expected length
        invalid.add(Arrays.copyOf(valid, valid.length - 1));
        invalid.add(Arrays.copyOf(valid, valid.length + 1));
        for (byte[] metadata : invalid) assertThrows(ArtifactFormatException.class,
                () -> ModelArtifactCodec.decode(envelope(metadata, payload), LIMITS));
        assertThrows(ArtifactFormatException.class, () -> ModelArtifactCodec.decode(
                new ArtifactEnvelope("model.v1", List.of(new ArtifactEnvelope.Entry("schema", valid))), LIMITS));
        assertThrows(ArtifactFormatException.class, () -> ModelArtifactCodec.decode(
                envelope(valid, new ArtifactEnvelope.Entry("state/00000001", new byte[4])), LIMITS));
        assertThrows(ArtifactFormatException.class, () -> ModelArtifactCodec.decode(
                envelope(valid, new ArtifactEnvelope.Entry("state/00000000", new byte[3])), LIMITS));
        assertThrows(ArtifactFormatException.class, () -> ModelArtifactCodec.decode(
                envelope(valid, payload, new ArtifactEnvelope.Entry("state/00000001", new byte[0])), LIMITS));
        assertThrows(ArtifactFormatException.class, () -> ModelArtifactCodec.decode(
                new ArtifactEnvelope("wrong", List.of(new ArtifactEnvelope.Entry("schema", valid), payload)), LIMITS));
        assertThrows(ArtifactFormatException.class, () -> ModelArtifactCodec.decode(
                envelope(valid, payload), new ArtifactLimits(8192, 4096, 1, 32, 4096, 4096)));
        var pair = List.of(states.getFirst(), new ModelArtifactCodec.State("bb", StateKind.BUFFER,
                DataType.FLOAT32, Shape.of(1), 4));
        byte[] duplicatePath = ModelArtifactCodec.encode(pair, CONFIG, null, LIMITS).bytes();
        duplicatePath[81] = 'a'; duplicatePath[82] = 'a';
        assertThrows(ArtifactFormatException.class, () -> ModelArtifactCodec.decode(envelope(
                duplicatePath, payload, new ArtifactEnvelope.Entry("state/00000001", new byte[4])), LIMITS));
    }

    @Test
    void preflightRejectsLimitsAndEngineFailureLeavesNoCandidateOrBindingChange() {
        var module = new Fixture();
        Tensor value = TensorFactory.scalar(2.0f, Optional.empty(), false).contiguous();
        module.add("value", value);
        Engine closed = Engine.standard(); closed.close();
        assertThrows(IllegalArgumentException.class, () -> ModelArtifactCapture.capture(
                module, closed, new byte[31], null, LIMITS));
        assertThrows(IllegalArgumentException.class, () -> ModelArtifactCapture.capture(
                module, closed, CONFIG, null, new ArtifactLimits(8192, 4096, 20, 32, 4, 4096)));
        assertThrows(IllegalArgumentException.class, () -> ModelArtifactCapture.capture(
                module, closed, CONFIG, null, new ArtifactLimits(8192, 4096, 1, 32, 4096, 4096)));
        assertThrows(IllegalArgumentException.class, () -> ModelArtifactCapture.capture(
                module, closed, CONFIG, null, new ArtifactLimits(8192, 60, 20, 32, 4096, 4096)));
        assertThrows(IllegalArgumentException.class, () -> ModelArtifactCapture.capture(
                module, closed, CONFIG, null, new ArtifactLimits(80, 4096, 20, 32, 4096, 4096)));
        assertThrows(IllegalStateException.class, () -> ModelArtifactCapture.capture(
                module, closed, CONFIG, null, LIMITS));
        assertSame(value, module.stateDictionary().entries().getFirst().value());
        assertThrows(IllegalArgumentException.class, () -> ModelArtifactCodec.encode(List.of(
                new ModelArtifactCodec.State("bad\ud800", StateKind.BUFFER, DataType.FLOAT32,
                        Shape.scalar(), 4)), CONFIG, null, LIMITS));
        assertThrows(IllegalArgumentException.class, () -> ModelArtifactCodec.encode(List.of(
                new ModelArtifactCodec.State("a", StateKind.BUFFER, DataType.FLOAT32,
                        Shape.of(2), 4)), CONFIG, null, LIMITS));
    }

    @Test
    void virtualLargeStateListStopsAtFirstMonotonicResourceBound() {
        AtomicInteger reads = new AtomicInteger();
        List<ModelArtifactCodec.State> states = new AbstractList<>() {
            @Override public ModelArtifactCodec.State get(int index) {
                reads.incrementAndGet();
                return new ModelArtifactCodec.State("p" + index, StateKind.BUFFER,
                        DataType.FLOAT32, Shape.scalar(), 4);
            }

            @Override public int size() { return 1_000_000; }
        };
        assertThrows(IllegalArgumentException.class, () -> ModelArtifactCodec.encode(
                states, CONFIG, null,
                new ArtifactLimits(100_000_000, 58_000_100, 1_000_001, 32, 4096, 100)));
        assertEquals(3, reads.get(), "aggregate rejection must not traverse the virtual list");

        reads.set(0);
        assertThrows(IllegalArgumentException.class, () -> ModelArtifactCodec.encode(
                states, CONFIG, null,
                new ArtifactLimits(58_000_180, 58_000_100, 1_000_001, 32, 4096, 1_000)));
        assertEquals(1, reads.get(), "file rejection must not traverse the virtual list");

        reads.set(0);
        assertThrows(IllegalArgumentException.class, () -> ModelArtifactCodec.encode(
                states, CONFIG, null,
                new ArtifactLimits(100_000_000, 58_000_000, 1_000_001, 32, 4096, 1_000)));
        assertEquals(0, reads.get(), "manifest rejection must precede descriptor access");
    }

    private static ArtifactEnvelope envelope(byte[] schema, ArtifactEnvelope.Entry... states) {
        var entries = new ArrayList<ArtifactEnvelope.Entry>();
        entries.add(new ArtifactEnvelope.Entry("schema", schema));
        entries.addAll(List.of(states));
        return new ArtifactEnvelope("model.v1", entries);
    }

    private static byte[] payload(ModelArtifactCandidate candidate, int ordinal) {
        var entry = candidate.envelope().entries().get(ordinal + 1);
        byte[] bytes = new byte[entry.size()]; entry.payload().get(bytes); return bytes;
    }

    private static byte[] flip(byte[] bytes, int at) {
        byte[] copy = bytes.clone(); copy[at] ^= 1; return copy;
    }

    private static byte[] setByte(byte[] bytes, int at, byte value) {
        byte[] copy = bytes.clone(); copy[at] = value; return copy;
    }

    private static byte[] setInt(byte[] bytes, int at, int value) {
        byte[] copy = bytes.clone(); ByteBuffer.wrap(copy).putInt(at, value); return copy;
    }

    private static byte[] setLong(byte[] bytes, int at, long value) {
        byte[] copy = bytes.clone(); ByteBuffer.wrap(copy).putLong(at, value); return copy;
    }

    private static Tensor tensor(Arena arena, DataType type, Object values) {
        MemorySegment source = switch (type) {
            case FLOAT64 -> MemorySegment.ofArray((double[]) values);
            case FLOAT32 -> MemorySegment.ofArray((float[]) values);
            case BFLOAT16, FLOAT16 -> MemorySegment.ofArray((short[]) values);
            case INT32 -> MemorySegment.ofArray((int[]) values);
            case INT64 -> MemorySegment.ofArray((long[]) values);
            case BOOL -> MemorySegment.ofArray((byte[]) values);
        };
        Shape shape = Shape.of(2);
        var descriptor = new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        MemorySegment segment = arena.allocate(source.byteSize(), type.byteWidth());
        MemorySegment.copy(source, 0, segment, 0, source.byteSize());
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(type, 2, segment))).contiguous();
    }

    private static final class Fixture extends Module {
        void add(String name, Tensor tensor) { buffer(name, tensor); }
    }
}
