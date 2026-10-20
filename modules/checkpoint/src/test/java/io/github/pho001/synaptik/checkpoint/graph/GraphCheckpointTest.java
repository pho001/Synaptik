package io.github.pho001.synaptik.checkpoint.graph;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactLimits;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Modifier;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Generic source ownership, signature and public-surface tests, without execution dependencies. */
final class GraphCheckpointTest {
    static final ArtifactLimits LIMITS = new ArtifactLimits(1_000_000, 100_000, 1000, 64, 100_000, 900_000);
    @TempDir Path directory;

    @Test
    void publicSurfaceContainsOnlyTheNamedBoundary() {
        assertTrue(Modifier.isFinal(GraphCheckpoint.class.getModifiers()));
        var methods = Arrays.stream(GraphCheckpoint.class.getDeclaredMethods())
                .filter(m -> Modifier.isPublic(m.getModifiers())).toList();
        assertEquals(4, methods.size());
        assertEquals(List.of("copy", "load", "rebind", "save"),
                methods.stream().map(m -> m.getName()).sorted().toList());
        for (var method : methods) {
            assertFalse(method.toGenericString().contains("compiler"));
            assertTrue(Modifier.isStatic(method.getModifiers()));
        }
        assertEquals(3, GraphCheckpoint.class.getDeclaredClasses().length);
        for (Class<?> type : List.of(GraphSource.class, GraphValueStorage.class, GraphArtifactCodec.class))
            assertFalse(Modifier.isPublic(type.getModifiers()));
    }

    @Test
    void containersSnapshotInDeclarationOrderWithoutAcquiringTensors() {
        Tensor value = prototype(DataType.FLOAT32, Shape.of(1), true, false);
        var inputs = new ArrayList<>(List.of(value));
        var outputs = new ArrayList<>(List.of(value));
        var leaves = new ArrayList<GraphCheckpoint.Leaf>();
        var source = new GraphCheckpoint.Source(inputs, outputs, leaves);
        inputs.clear(); outputs.clear(); leaves.add(leaf("value", GraphCheckpoint.Role.CONSTANT, value));
        assertEquals(List.of(value), source.inputs());
        assertEquals(List.of(value), source.outputs());
        assertTrue(source.leaves().isEmpty());
        assertThrows(UnsupportedOperationException.class, () -> source.inputs().clear());
        assertEquals("inputs", assertThrows(NullPointerException.class,
                () -> new GraphCheckpoint.Source(null, null, null)).getMessage());
        assertThrows(NullPointerException.class,
                () -> new GraphCheckpoint.Source(Arrays.asList(value, null), null, null));
        assertEquals("outputs", assertThrows(NullPointerException.class,
                () -> new GraphCheckpoint.Source(List.of(), null, null)).getMessage());
        assertEquals("name", assertThrows(NullPointerException.class,
                () -> new GraphCheckpoint.Leaf(null, null, null)).getMessage());
        assertEquals("role", assertThrows(NullPointerException.class,
                () -> new GraphCheckpoint.Leaf("a", null, null)).getMessage());
    }

    @Test
    void rawSevenTypesScalarEmptyAndLayoutPresenceRoundtripAndResaveExactly() throws Exception {
        for (DataType type : DataType.values()) {
            for (Shape shape : List.of(Shape.of(), Shape.of(0, 2), Shape.of(3))) {
                for (boolean resolved : List.of(false, true)) {
                    int count = Math.toIntExact(shape.knownElementCount().orElseThrow());
                    byte[] bits = raw(type, count);
                    Tensor state = tensor(type, shape, type.isFloating(), resolved, bits);
                    var source = new GraphCheckpoint.Source(List.of(), List.of(state),
                            List.of(leaf("重み.value", GraphCheckpoint.Role.BUFFER, state),
                                    leaf("alias", GraphCheckpoint.Role.BUFFER, state)));
                    GraphCheckpoint.Source copied = GraphCheckpoint.copy(source);
                    assertRawState(copied, state, bits);
                    Path first = directory.resolve("first"), second = directory.resolve("second");
                    GraphCheckpoint.save(first, source, LIMITS);
                    var loaded = GraphCheckpoint.load(first, LIMITS);
                    assertRawState(loaded, state, bits);
                    GraphCheckpoint.save(second, loaded, LIMITS);
                    assertArrayEquals(Files.readAllBytes(first), Files.readAllBytes(second));
                }
            }
        }
    }

    @Test
    void exactInputAssociationHandlesStateFirstDifferentSignatureOrderAndUnusedStorage() throws Exception {
        Tensor first = prototype(DataType.FLOAT32, Shape.of(2), true, true);
        Tensor second = prototype(DataType.FLOAT32, Shape.of(2), true, true);
        Tensor unused = prototype(DataType.FLOAT32, Shape.of(2), true, true);
        Tensor state = tensor(DataType.FLOAT32, Shape.of(2), true, true, raw(DataType.FLOAT32, 2));
        Tensor shared = state.mul(second);
        Tensor output = shared.add(first);
        var source = new GraphCheckpoint.Source(List.of(first, unused, second), List.of(output, shared),
                List.of(leaf("weight", GraphCheckpoint.Role.PARAMETER, state)));
        var graph = GraphSource.capture(source).graph;
        assertEquals(List.of(new io.github.pho001.synaptik.model.graph.ValueId(3),
                new io.github.pho001.synaptik.model.graph.ValueId(2)), graph.nodes().getFirst().inputs());
        var copy = GraphCheckpoint.copy(source);
        assertEquals(graph, GraphSource.capture(copy).graph);
        for (Tensor input : copy.inputs()) assertTrue(input.hostStorage().isEmpty());
        assertNotSame(first, copy.inputs().getFirst());
        for (Tensor reachable : GraphCompilationPort.captureSource(copy.outputs()).leaves())
            assertFalse(List.of(first, second, state).contains(reachable));
        Path file = directory.resolve("ordering");
        GraphCheckpoint.save(file, source, LIMITS);
        var loaded = GraphCheckpoint.load(file, LIMITS);
        assertEquals(graph, GraphSource.capture(loaded).graph);
        assertEquals(3, loaded.inputs().size());
        assertSame(loaded.outputs().get(1), loaded.outputs().getFirst().provenance().orElseThrow().producer().inputs().getFirst());
    }

    @Test
    void allRolesAliasesAndDistinctEqualValuesRetainIdentityAndGradients() {
        Tensor parameter = tensor(DataType.FLOAT32, Shape.of(1), true, false, raw(DataType.FLOAT32, 1));
        Tensor equal = tensor(DataType.FLOAT32, Shape.of(1), true, false, raw(DataType.FLOAT32, 1));
        Tensor constant = tensor(DataType.FLOAT32, Shape.of(1), true, false, raw(DataType.FLOAT32, 1));
        var source = new GraphCheckpoint.Source(List.of(), List.of(parameter.add(equal).mul(constant)), List.of(
                leaf("parameter", GraphCheckpoint.Role.PARAMETER, parameter),
                leaf("buffer", GraphCheckpoint.Role.BUFFER, equal),
                leaf("tied", GraphCheckpoint.Role.PARAMETER, parameter),
                leaf("constant", GraphCheckpoint.Role.CONSTANT, constant)));
        var copy = GraphCheckpoint.copy(source);
        assertSame(copy.leaves().get(0).tensor(), copy.leaves().get(2).tensor());
        assertNotSame(copy.leaves().get(0).tensor(), copy.leaves().get(1).tensor());
        assertTrue(copy.leaves().stream().allMatch(l -> l.tensor().descriptor().requiresGrad()));
        assertEquals(source.leaves().stream().map(GraphCheckpoint.Leaf::role).toList(),
                copy.leaves().stream().map(GraphCheckpoint.Leaf::role).toList());
    }

    @Test
    void activationsCopyLoadResaveAndRebindExactTopologyStateAndDeclaredSlots() throws Exception {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16, DataType.FLOAT16)) {
            Tensor first = prototype(type, Shape.of(3), true, true);
            Tensor second = prototype(type, Shape.of(3), true, true);
            Tensor unused = prototype(type, Shape.of(3), true, true);
            Tensor state = tensor(type, Shape.of(3), true, true, raw(type, 3));
            var store = state.hostStorage().orElseThrow();
            for (int activation = 0; activation < 3; activation++) {
                Tensor base = state.mul(second).add(first);
                Tensor active = activate(base, activation), separate = activate(base, activation);
                var source = new GraphCheckpoint.Source(List.of(first, unused, second),
                        List.of(active.mul(active), separate, active, first),
                        List.of(leaf("weight", GraphCheckpoint.Role.PARAMETER, state),
                                leaf("tied", GraphCheckpoint.Role.PARAMETER, state)));
                var graph = GraphSource.capture(source).graph;
                Path file = directory.resolve("activation"), resaved = directory.resolve("resaved");
                GraphCheckpoint.save(file, source, LIMITS);
                for (var acquired : List.of(GraphCheckpoint.copy(source), GraphCheckpoint.load(file, LIMITS))) {
                    assertEquals(graph, GraphSource.capture(acquired).graph);
                    Tensor owned = acquired.leaves().getFirst().tensor();
                    assertSame(owned, acquired.leaves().getLast().tensor());
                    assertNotSame(state, owned);
                    assertArrayEquals(raw(type, 3), GraphValueStorage.encode(owned.descriptor(), owned.hostStorage().orElseThrow().segment()));
                    assertTrue(acquired.inputs().stream().allMatch(t -> t.hostStorage().isEmpty()));
                    Tensor shared = acquired.outputs().get(2);
                    assertSame(shared, acquired.outputs().getFirst().provenance().orElseThrow().inputs().getFirst());
                    assertSame(shared, acquired.outputs().getFirst().provenance().orElseThrow().inputs().get(1));
                    assertNotSame(shared, acquired.outputs().get(1));
                    assertSame(acquired.inputs().getFirst(), acquired.outputs().get(3));
                    GraphCheckpoint.save(resaved, acquired, LIMITS);
                    assertArrayEquals(Files.readAllBytes(file), Files.readAllBytes(resaved));
                    Tensor replacement = first.abs().contiguous(); // Match the full resolved descriptor.
                    var rebound = GraphCheckpoint.rebind(acquired, List.of(replacement, unused, replacement));
                    assertSame(replacement, rebound.get(3));
                    assertEquals(shared.provenance().orElseThrow().operation(), rebound.get(2).provenance().orElseThrow().operation());
                    Tensor sum = rebound.get(2).provenance().orElseThrow().inputs().getFirst();
                    assertSame(replacement, sum.provenance().orElseThrow().inputs().get(1));
                    Tensor product = sum.provenance().orElseThrow().inputs().getFirst();
                    assertSame(owned, product.provenance().orElseThrow().inputs().getFirst());
                    assertSame(replacement, product.provenance().orElseThrow().inputs().get(1));
                }
                assertSame(store, state.hostStorage().orElseThrow());
                assertArrayEquals(raw(type, 3), GraphValueStorage.encode(state.descriptor(), store.segment()));
            }
        }
    }

    /**
     * Builds an ordinary activation without cold copying or wire reconstruction.
     * @param input floating expression to activate
     * @param activation test index 0 RELU, 1 TANH, 2 SIGMOID
     * @return ordinary expression retaining the exact input
     */
    private static Tensor activate(Tensor input, int activation) {
        return switch (activation) {
            case 0 -> input.relu(); case 1 -> input.tanh(); case 2 -> input.sigmoid();
            default -> throw new AssertionError(activation);
        };
    }

    @Test
    void rebindRetainsProducedAndRepeatedSubstitutionsWithoutStorageAdmission() {
        Tensor first = prototype(DataType.FLOAT32, Shape.of(2), true, false);
        Tensor second = prototype(DataType.FLOAT32, Shape.of(2), true, false);
        Tensor state = prototype(DataType.FLOAT32, Shape.of(2), true, false);
        var source = new GraphCheckpoint.Source(List.of(first, second),
                List.of(first, second, first.add(state)), List.of(leaf("weight", GraphCheckpoint.Role.PARAMETER, state)));
        Tensor upstream = first.neg();
        var rebound = GraphCheckpoint.rebind(source, List.of(upstream, upstream));
        assertSame(upstream, rebound.get(0)); assertSame(upstream, rebound.get(1));
        assertSame(state, rebound.get(2).provenance().orElseThrow().producer().inputs().get(1));
        assertSame(first, upstream.provenance().orElseThrow().producer().inputs().getFirst());
        assertThrows(UnsupportedOperationException.class, () -> rebound.clear());
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(source, List.of(first)));
        Tensor layoutMismatch = prototype(DataType.FLOAT32, Shape.of(2), true, true);
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(source, List.of(layoutMismatch, second)));
        Tensor gradMismatch = prototype(DataType.FLOAT32, Shape.of(2), false, false);
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(source, List.of(gradMismatch, second)));
    }

    @Test
    void metadataRejectsBeforeReplacementsStorageOrPublication() throws Exception {
        Tensor state = prototype(DataType.FLOAT32, Shape.of(1), true, false);
        var unnamed = new GraphCheckpoint.Source(List.of(), List.of(state.neg()), List.of());
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(unnamed, null));
        var valid = new GraphCheckpoint.Source(List.of(), List.of(state), List.of(leaf("state", GraphCheckpoint.Role.PARAMETER, state)));
        Path target = directory.resolve("prior"); Files.write(target, new byte[] {7});
        var impossible = new ArtifactLimits(48, 8, 0, 1, 0, 0);
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.save(target, valid, impossible));
        assertArrayEquals(new byte[] {7}, Files.readAllBytes(target));
        assertThrows(IllegalStateException.class, () -> GraphCheckpoint.copy(valid));
        assertArrayEquals(new byte[] {7}, Files.readAllBytes(target));
        assertEquals(1, Files.list(directory).count());
    }

    @Test
    void rejectsInvalidNamesRolesReachabilityAndUnsupportedFamilies() {
        Tensor a = prototype(DataType.FLOAT32, Shape.of(1), true, false);
        Tensor b = prototype(DataType.FLOAT32, Shape.of(1), true, false);
        for (String name : List.of("", " ", ".a", "a.", "a..b", "a. ", "\uD800"))
            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(
                    new GraphCheckpoint.Source(List.of(), List.of(a), List.of(leaf(name, GraphCheckpoint.Role.PARAMETER, a))), List.of()));
        for (List<GraphCheckpoint.Leaf> leaves : List.of(
                List.of(leaf("a", GraphCheckpoint.Role.PARAMETER, a), leaf("a", GraphCheckpoint.Role.PARAMETER, a)),
                List.of(leaf("a", GraphCheckpoint.Role.PARAMETER, a), leaf("a.b", GraphCheckpoint.Role.PARAMETER, a)),
                List.of(leaf("a", GraphCheckpoint.Role.PARAMETER, a), leaf("b", GraphCheckpoint.Role.BUFFER, a)),
                List.of(leaf("a", GraphCheckpoint.Role.PARAMETER, a), leaf("b", GraphCheckpoint.Role.PARAMETER, b)),
                List.of(leaf("a", GraphCheckpoint.Role.PARAMETER, a.neg()))))
            assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(
                    new GraphCheckpoint.Source(List.of(), List.of(a), leaves), List.of()));
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(
                new GraphCheckpoint.Source(List.of(a), List.of(a), List.of(leaf("a", GraphCheckpoint.Role.PARAMETER, a))), List.of(a)));
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(
                new GraphCheckpoint.Source(List.of(a, a), List.of(a), List.of()), List.of(a, a)));
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(
                new GraphCheckpoint.Source(List.of(a.neg()), List.of(a), List.of()), List.of(a)));
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(
                new GraphCheckpoint.Source(List.of(a), List.of(a, a), List.of()), List.of(a)));
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.rebind(
                new GraphCheckpoint.Source(List.of(a), List.of(a.abs()), List.of()), List.of(a)));
        Tensor integer = prototype(DataType.INT32, Shape.of(1), false, false);
        assertThrows(IllegalArgumentException.class, () -> GraphCheckpoint.copy(new GraphCheckpoint.Source(
                List.of(), List.of(integer), List.of(leaf("p", GraphCheckpoint.Role.PARAMETER, integer)))));
    }

    /**
     * Creates a descriptor-only test prototype without implicit layout closure.
     * @param type logical type
     * @param shape static logical shape
     * @param grad gradient eligibility
     * @param resolved whether to include exact canonical layout
     * @return fresh storage-free leaf
     */
    static Tensor prototype(DataType type, Shape shape, boolean grad, boolean resolved) {
        return TensorFactory.create(new TensorDescriptor(type, shape,
                resolved ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), grad));
    }

    /**
     * Imports canonical test bits to matching carriers without numeric conversion.
     * @param type logical type
     * @param shape static shape
     * @param grad gradient eligibility
     * @param resolved layout-presence flag
     * @param bits canonical big-endian logical bytes
     * @return test leaf with mutable private heap state
     */
    static Tensor tensor(DataType type, Shape shape, boolean grad, boolean resolved, byte[] bits) {
        Tensor tensor = prototype(type, shape, grad, resolved);
        tensor.replaceHostStorage(GraphValueStorage.decode(tensor.descriptor(), ByteBuffer.wrap(bits)));
        return tensor;
    }

    /**
     * Supplies non-normalized raw test lanes including NaN payload/sign and signed zero.
     * @param type logical type
     * @param count logical lane count
     * @return exact canonical bytes; BOOL uses only zero/one
     */
    static byte[] raw(DataType type, int count) {
        ByteBuffer out = ByteBuffer.allocate(count * type.byteWidth()).order(ByteOrder.BIG_ENDIAN);
        for (int i = 0; i < count; i++) switch (type) {
            case FLOAT64, INT64 -> out.putLong(i % 3 == 0 ? 0xfff8000000000123L : i % 3 == 1 ? Long.MIN_VALUE : 42);
            case FLOAT32, INT32 -> out.putInt(i % 3 == 0 ? 0xffc00123 : i % 3 == 1 ? Integer.MIN_VALUE : 42);
            case BFLOAT16, FLOAT16 -> out.putShort((short) (i % 3 == 0 ? 0xffc3 : i % 3 == 1 ? 0x8000 : 42));
            case BOOL -> out.put((byte) (i % 2));
        }
        return out.array();
    }

    /**
     * Builds an exact-role registration for test sources.
     * @param name registration path
     * @param role explicit semantic role
     * @param tensor borrowed leaf
     * @return immutable test registration
     */
    static GraphCheckpoint.Leaf leaf(String name, GraphCheckpoint.Role role, Tensor tensor) {
        return new GraphCheckpoint.Leaf(name, role, tensor);
    }

    /**
     * Verifies detached identity, descriptors, aliasing and raw bits for a two-alias state source.
     * @param source acquired or loaded source
     * @param original caller's original state
     * @param bits expected exact big-endian logical bits
     */
    private static void assertRawState(GraphCheckpoint.Source source, Tensor original, byte[] bits) {
        Tensor state = source.leaves().getFirst().tensor();
        assertNotSame(original, state);
        assertSame(state, source.outputs().getFirst());
        assertSame(state, source.leaves().get(1).tensor());
        assertEquals(original.descriptor(), state.descriptor());
        var storage = state.hostStorage().orElseThrow();
        assertFalse(storage.segment().isNative()); assertFalse(storage.isReadOnly());
        assertArrayEquals(bits, GraphValueStorage.encode(state.descriptor(), storage.segment()));
        assertNotSame(original.hostStorage().orElseThrow(), storage);
    }
}
