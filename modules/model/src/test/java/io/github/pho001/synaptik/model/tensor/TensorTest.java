package io.github.pho001.synaptik.model.tensor;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.storage.HostTensorStorage;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TensorTest {
    @Test
    void hasExactlyTheRequiredClassStateConstructorAndPublicApi() {
        assertAll(
                () -> assertTrue(Modifier.isPublic(Tensor.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(Tensor.class.getModifiers())),
                () -> assertFalse(Tensor.class.isRecord()),
                () -> assertEquals(Set.of(), Set.of(Tensor.class.getInterfaces())));

        var fields = Arrays.stream(Tensor.class.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();
        assertEquals(List.of("id", "descriptor", "label", "hostStorage"),
                fields.stream().map(field -> field.getName()).toList());
        assertAll(
                () -> assertEquals(TensorId.class, fields.get(0).getType()),
                () -> assertEquals(TensorDescriptor.class, fields.get(1).getType()),
                () -> assertEquals(Optional.class, fields.get(2).getType()),
                () -> assertEquals(HostTensorStorage.class, fields.get(3).getType()),
                () -> assertTrue(fields.stream().allMatch(
                        field -> Modifier.isPrivate(field.getModifiers()))),
                () -> assertTrue(fields.subList(0, 3).stream().allMatch(
                        field -> Modifier.isFinal(field.getModifiers()))),
                () -> assertFalse(Modifier.isFinal(fields.get(3).getModifiers())));

        var constructors = Tensor.class.getDeclaredConstructors();
        assertEquals(1, constructors.length);
        assertAll(
                () -> assertFalse(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertFalse(Modifier.isProtected(constructors[0].getModifiers())),
                () -> assertFalse(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(
                        List.of(TensorId.class, TensorDescriptor.class, Optional.class, Optional.class),
                        Arrays.asList(constructors[0].getParameterTypes())));

        Set<String> publicMethods = Arrays.stream(Tensor.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(method -> method.getName())
                .collect(Collectors.toSet());
        assertEquals(
                Set.of("id", "descriptor", "label", "hostStorage", "replaceHostStorage",
                        "clearHostStorage", "toString"),
                publicMethods);
        assertAll(
                () -> assertTrue(Modifier.isSynchronized(
                        Tensor.class.getDeclaredMethod("hostStorage").getModifiers())),
                () -> assertTrue(Modifier.isSynchronized(
                        Tensor.class.getDeclaredMethod(
                                        "replaceHostStorage", HostTensorStorage.class)
                                .getModifiers())),
                () -> assertTrue(Modifier.isSynchronized(
                        Tensor.class.getDeclaredMethod("clearHostStorage").getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(
                        Tensor.class.getDeclaredMethod("id").getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(
                        Tensor.class.getDeclaredMethod("descriptor").getModifiers())),
                () -> assertFalse(Modifier.isSynchronized(
                        Tensor.class.getDeclaredMethod("label").getModifiers())));
    }

    @Test
    void validatesConstructorReferencesAndBlankLabelInDeterministicOrder() {
        TensorDescriptor descriptor = unresolved(DataType.FLOAT32, Shape.of(2, 3));
        HostTensorStorage wrongType = storage(DataType.INT32, 0);

        NullPointerException nullId = assertThrows(
                NullPointerException.class,
                () -> new Tensor(null, null, null, null));
        NullPointerException nullDescriptor = assertThrows(
                NullPointerException.class,
                () -> new Tensor(new TensorId(1), null, null, null));
        NullPointerException nullLabel = assertThrows(
                NullPointerException.class,
                () -> new Tensor(new TensorId(1), descriptor, null, null));
        NullPointerException nullStorageOptional = assertThrows(
                NullPointerException.class,
                () -> new Tensor(new TensorId(1), descriptor, Optional.empty(), null));
        IllegalArgumentException blankWins = assertThrows(
                IllegalArgumentException.class,
                () -> new Tensor(
                        new TensorId(1), descriptor, Optional.of(" \t\n "), Optional.of(wrongType)));

        assertAll(
                () -> assertEquals("id", nullId.getMessage()),
                () -> assertEquals("descriptor", nullDescriptor.getMessage()),
                () -> assertEquals("label", nullLabel.getMessage()),
                () -> assertEquals("hostStorage", nullStorageOptional.getMessage()),
                () -> assertEquals("label must not be blank", blankWins.getMessage()));
    }

    @Test
    void retainsStableReferencesAndNormalizesLabelValue() {
        TensorId id = new TensorId(7);
        TensorDescriptor descriptor = unresolved(DataType.FLOAT32, Shape.of(2, 3));
        Tensor labeled = new Tensor(id, descriptor, Optional.of("  weights\n"), Optional.empty());
        Tensor unlabeled = new Tensor(new TensorId(8), descriptor, Optional.empty(), Optional.empty());

        assertAll(
                () -> assertSame(id, labeled.id()),
                () -> assertSame(descriptor, labeled.descriptor()),
                () -> assertEquals(Optional.of("weights"), labeled.label()),
                () -> assertEquals(Optional.empty(), unlabeled.label()),
                () -> assertEquals(Optional.empty(), labeled.hostStorage()));
    }

    @Test
    void validatesStorageTypeCapacityAndLivenessInExactOrder() {
        TensorDescriptor resolved = resolved(
                DataType.FLOAT32,
                Shape.of(2, 3),
                LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 2, true));
        Arena arena = Arena.ofConfined();
        MemorySegment deadEightBytes = arena.allocate(8, 1);
        MemorySegmentStorage wrongDead =
                new MemorySegmentStorage(DataType.INT64, 1, deadEightBytes);
        MemorySegmentStorage smallDead = new MemorySegmentStorage(
                DataType.FLOAT32, 2, deadEightBytes);
        arena.close();

        IllegalArgumentException typeWins = assertThrows(
                IllegalArgumentException.class,
                () -> new Tensor(
                        new TensorId(1), resolved, Optional.empty(), Optional.of(wrongDead)));
        IllegalArgumentException capacityWins = assertThrows(
                IllegalArgumentException.class,
                () -> new Tensor(
                        new TensorId(1), resolved, Optional.empty(), Optional.of(smallDead)));
        TensorDescriptor unresolved = unresolved(DataType.FLOAT32, Shape.of(2, 3));
        IllegalStateException liveness = assertThrows(
                IllegalStateException.class,
                () -> new Tensor(
                        new TensorId(1), unresolved, Optional.empty(), Optional.of(smallDead)));

        assertAll(
                () -> assertEquals(
                        "hostStorage data type must match descriptor data type: expected=FLOAT32, actual=INT64",
                        typeWins.getMessage()),
                () -> assertEquals(
                        "hostStorage element capacity is smaller than resolved layout span: required=8, actual=2",
                        capacityWins.getMessage()),
                () -> assertEquals("hostStorage must be alive when attached", liveness.getMessage()));
    }

    @Test
    void replacementIsAtomicAndReturnsExactPreviousReferences() {
        HostTensorStorage first = storage(DataType.FLOAT32, 6);
        HostTensorStorage second = storage(DataType.FLOAT32, 8);
        HostTensorStorage invalidType = storage(DataType.INT32, 6);
        Tensor initiallyEmpty = new Tensor(
                new TensorId(0), unresolved(DataType.FLOAT32, Shape.of(2, 3)), Optional.empty(),
                Optional.empty());
        Tensor tensor = new Tensor(
                new TensorId(1),
                resolved(DataType.FLOAT32, Shape.of(2, 3), LayoutDescriptor.contiguous(Shape.of(2, 3))),
                Optional.empty(),
                Optional.of(first));
        Optional<HostTensorStorage> initialSnapshot = tensor.hostStorage();

        Optional<HostTensorStorage> absentPrevious = initiallyEmpty.replaceHostStorage(first);
        Optional<HostTensorStorage> replaced = tensor.replaceHostStorage(second);
        IllegalArgumentException failed = assertThrows(
                IllegalArgumentException.class,
                () -> tensor.replaceHostStorage(invalidType));
        Optional<HostTensorStorage> cleared = tensor.clearHostStorage();
        Optional<HostTensorStorage> clearedAgain = tensor.clearHostStorage();

        assertAll(
                () -> assertEquals(Optional.empty(), absentPrevious),
                () -> assertSame(first, initiallyEmpty.hostStorage().orElseThrow()),
                () -> assertSame(first, initialSnapshot.orElseThrow()),
                () -> assertSame(first, replaced.orElseThrow()),
                () -> assertSame(second, cleared.orElseThrow()),
                () -> assertEquals(Optional.empty(), clearedAgain),
                () -> assertEquals(Optional.empty(), tensor.hostStorage()),
                () -> assertSame(first, initialSnapshot.orElseThrow()),
                () -> assertEquals(
                        "hostStorage data type must match descriptor data type: expected=FLOAT32, actual=INT32",
                        failed.getMessage()));
    }

    @Test
    void replacementRejectsNullWithoutChangingAssociation() {
        HostTensorStorage first = storage(DataType.FLOAT32, 0);
        Tensor tensor = new Tensor(
                new TensorId(1), unresolved(DataType.FLOAT32, Shape.of(2, 3)), Optional.empty(),
                Optional.of(first));

        NullPointerException failure = assertThrows(
                NullPointerException.class, () -> tensor.replaceHostStorage(null));

        assertAll(
                () -> assertEquals("hostStorage", failure.getMessage()),
                () -> assertSame(first, tensor.hostStorage().orElseThrow()));
    }

    @Test
    void resolvedLayoutsUseReferencedSpanAcrossAllRequiredGeometries() {
        assertCapacityBoundary(Shape.of(2, 3), LayoutDescriptor.contiguous(Shape.of(2, 3)), 6);
        assertCapacityBoundary(
                Shape.of(2, 3),
                LayoutDescriptor.of(Shape.of(2, 3), new long[] {3, 1}, 5, true),
                11);
        assertCapacityBoundary(
                Shape.of(2, 3),
                LayoutDescriptor.of(Shape.of(2, 3), new long[] {1, 2}, 0, true),
                6);
        assertCapacityBoundary(
                Shape.of(2, 3),
                LayoutDescriptor.of(Shape.of(2, 3), new long[] {0, 1}, 0, true),
                3);
        assertCapacityBoundary(Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar()), 1);

        Shape empty = Shape.of(2, 0, 4);
        LayoutDescriptor emptyWithOffset =
                LayoutDescriptor.of(empty, new long[] {0, 4, 1}, 9, true);
        Tensor zeroCapacity = new Tensor(
                new TensorId(99),
                resolved(DataType.FLOAT32, empty, emptyWithOffset),
                Optional.empty(),
                Optional.of(storage(DataType.FLOAT32, 0)));
        assertEquals(0, zeroCapacity.hostStorage().orElseThrow().elementCapacity());
    }

    @Test
    void unresolvedStaticAndDynamicLayoutsDoNotInventCapacityRequirements() {
        Tensor staticTensor = new Tensor(
                new TensorId(1),
                unresolved(DataType.FLOAT32, Shape.of(100, 100)),
                Optional.empty(),
                Optional.of(storage(DataType.FLOAT32, 0)));
        Shape dynamicShape = Shape.ofDimensions(
                new DynamicDimension("batch"), new StaticDimension(3));
        Tensor dynamicTensor = new Tensor(
                new TensorId(2),
                unresolved(DataType.FLOAT32, dynamicShape),
                Optional.empty(),
                Optional.of(storage(DataType.FLOAT32, 0)));

        assertAll(
                () -> assertEquals(0, staticTensor.hostStorage().orElseThrow().elementCapacity()),
                () -> assertEquals(0, dynamicTensor.hostStorage().orElseThrow().elementCapacity()),
                () -> assertTrue(staticTensor.descriptor().layout().isEmpty()),
                () -> assertTrue(dynamicTensor.descriptor().layout().isEmpty()));
    }

    @Test
    void acceptsReadOnlyStorageWithoutWritingIt() {
        MemorySegment readOnly = MemorySegment.ofArray(new byte[4]).asReadOnly();
        HostTensorStorage storage = new MemorySegmentStorage(DataType.FLOAT32, 1, readOnly);
        Tensor tensor = new Tensor(
                new TensorId(1),
                resolved(DataType.FLOAT32, Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar())),
                Optional.empty(),
                Optional.of(storage));

        assertAll(
                () -> assertSame(storage, tensor.hostStorage().orElseThrow()),
                () -> assertTrue(tensor.hostStorage().orElseThrow().isReadOnly()));
    }

    @Test
    void observesLateStorageDeathAndCanClearWithoutOwningTheScope() {
        Arena arena = Arena.ofConfined();
        MemorySegment segment = arena.allocate(4, 1);
        HostTensorStorage storage = new MemorySegmentStorage(DataType.FLOAT32, 1, segment);
        Tensor tensor = new Tensor(
                new TensorId(1),
                resolved(DataType.FLOAT32, Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar())),
                Optional.empty(),
                Optional.of(storage));
        arena.close();

        assertAll(
                () -> assertSame(storage, tensor.hostStorage().orElseThrow()),
                () -> assertFalse(tensor.hostStorage().orElseThrow().isAlive()),
                () -> assertThrows(IllegalStateException.class, () -> segment.get(JAVA_BYTE, 0)));
        assertSame(storage, tensor.clearHostStorage().orElseThrow());
        assertTrue(tensor.hostStorage().isEmpty());
    }

    @Test
    void sharedStorageAliasesRemainIndependentTensorAssociations() {
        MemorySegment segment = MemorySegment.ofArray(new byte[] {1, 2});
        HostTensorStorage shared = new MemorySegmentStorage(DataType.BOOL, 2, segment);
        Tensor first = new Tensor(
                new TensorId(1), unresolved(DataType.BOOL, Shape.of(2)), Optional.empty(),
                Optional.of(shared));
        Tensor second = new Tensor(
                new TensorId(2), unresolved(DataType.BOOL, Shape.of(2)), Optional.empty(),
                Optional.of(shared));

        segment.set(JAVA_BYTE, 1, (byte) 9);
        first.clearHostStorage();

        assertAll(
                () -> assertTrue(first.hostStorage().isEmpty()),
                () -> assertSame(shared, second.hostStorage().orElseThrow()),
                () -> assertEquals(9, second.hostStorage().orElseThrow().segment().get(JAVA_BYTE, 1)));
    }

    @Test
    void usesObjectIdentityEvenWithEqualStableMetadata() throws NoSuchMethodException {
        TensorId firstId = new TensorId(3);
        TensorId equalId = new TensorId(3);
        TensorDescriptor descriptor = unresolved(DataType.FLOAT32, Shape.of(2));
        HostTensorStorage storage = storage(DataType.FLOAT32, 0);
        Tensor first = new Tensor(firstId, descriptor, Optional.of("x"), Optional.of(storage));
        Tensor second = new Tensor(equalId, descriptor, Optional.of("x"), Optional.of(storage));

        assertAll(
                () -> assertNotSame(firstId, equalId),
                () -> assertEquals(firstId, equalId),
                () -> assertEquals(first, first),
                () -> assertNotEquals(first, second),
                () -> assertEquals(System.identityHashCode(first), first.hashCode()),
                () -> assertEquals(Object.class,
                        Tensor.class.getMethod("equals", Object.class).getDeclaringClass()),
                () -> assertEquals(Object.class,
                        Tensor.class.getMethod("hashCode").getDeclaringClass()));
    }

    @Test
    void diagnosticTextIsStableMetadataOnlyAcrossStorageTransitionsAndDeath() {
        Arena arena = Arena.ofConfined();
        HostTensorStorage scoped = new MemorySegmentStorage(
                DataType.FLOAT32, 1, arena.allocate(4, 1));
        Tensor tensor = new Tensor(
                new TensorId(42),
                resolved(DataType.FLOAT32, Shape.scalar(), LayoutDescriptor.contiguous(Shape.scalar())),
                Optional.of("  result  "),
                Optional.empty());
        String absent = tensor.toString();
        tensor.replaceHostStorage(scoped);
        String present = tensor.toString();
        arena.close();
        String dead = tensor.toString();
        tensor.clearHostStorage();
        String cleared = tensor.toString();

        assertAll(
                () -> assertEquals(absent, present),
                () -> assertEquals(absent, dead),
                () -> assertEquals(absent, cleared),
                () -> assertTrue(absent.contains("Tensor[")),
                () -> assertTrue(absent.contains("id=TensorId[value=42]")),
                () -> assertTrue(absent.contains("descriptor=TensorDescriptor[")),
                () -> assertTrue(absent.contains("label=Optional[result]")),
                () -> assertFalse(absent.contains("hostStorage")),
                () -> assertFalse(absent.contains("MemorySegment")),
                () -> assertFalse(absent.contains("alive")),
                () -> assertFalse(absent.contains("graph")),
                () -> assertFalse(absent.contains("runtime")));
    }

    private static void assertCapacityBoundary(
            Shape shape, LayoutDescriptor layout, long requiredCapacity) {
        TensorDescriptor descriptor = resolved(DataType.FLOAT32, shape, layout);
        HostTensorStorage exact = storage(DataType.FLOAT32, requiredCapacity);
        Tensor accepted = new Tensor(
                new TensorId(requiredCapacity), descriptor, Optional.empty(), Optional.of(exact));
        assertSame(exact, accepted.hostStorage().orElseThrow());

        if (requiredCapacity > 0) {
            HostTensorStorage tooSmall = storage(DataType.FLOAT32, requiredCapacity - 1);
            IllegalArgumentException failure = assertThrows(
                    IllegalArgumentException.class,
                    () -> new Tensor(
                            new TensorId(requiredCapacity + 100),
                            descriptor,
                            Optional.empty(),
                            Optional.of(tooSmall)));
            assertEquals(
                    "hostStorage element capacity is smaller than resolved layout span: required="
                            + requiredCapacity
                            + ", actual="
                            + (requiredCapacity - 1),
                    failure.getMessage());
        }

        HostTensorStorage larger = storage(DataType.FLOAT32, requiredCapacity + 1);
        Tensor largerAccepted = new Tensor(
                new TensorId(requiredCapacity + 200),
                descriptor,
                Optional.empty(),
                Optional.of(larger));
        assertSame(larger, largerAccepted.hostStorage().orElseThrow());
    }

    private static TensorDescriptor unresolved(DataType dataType, Shape shape) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), false);
    }

    private static TensorDescriptor resolved(
            DataType dataType, Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(dataType, shape, Optional.of(layout), false);
    }

    private static HostTensorStorage storage(DataType dataType, long capacity) {
        long byteSize = Math.multiplyExact(capacity, dataType.byteWidth());
        MemorySegment segment = MemorySegment.ofArray(new byte[Math.toIntExact(byteSize)]);
        return new MemorySegmentStorage(dataType, capacity, segment);
    }
}
