package io.github.pho001.synaptik.model.tensor;

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
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.random.GraphRngKind;
import io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

public final class GraphRngStateTest {
    @Test
    void createsTheExactStorageFreeStateTensorAndProvenance() throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        long before = next.get();

        GraphRngState state = GraphRngState.initial(0x1234L, -1L);
        Tensor tensor = state.tensor();
        TensorDescriptor descriptor = tensor.descriptor();
        TensorProvenance provenance = tensor.provenance().orElseThrow();
        GraphRngStateAttrs attrs = (GraphRngStateAttrs) provenance.operation().attrs();

        assertAll(
                () -> assertEquals(before, tensor.id().value()),
                () -> assertEquals(before + 1, next.get()),
                () -> assertEquals(DataType.INT64, descriptor.dataType()),
                () -> assertEquals(Shape.of(2), descriptor.shape()),
                () -> assertTrue(descriptor.layout().isEmpty()),
                () -> assertFalse(descriptor.requiresGrad()),
                () -> assertTrue(tensor.label().isEmpty()),
                () -> assertTrue(tensor.hostStorage().isEmpty()),
                () -> assertEquals(0, provenance.outputIndex()),
                () -> assertSame(descriptor, provenance.outputDescriptor()),
                () -> assertSame(descriptor,
                        provenance.producer().outputDescriptors().getFirst()),
                () -> assertSame(GraphRngKind.INITIAL_STATE, provenance.operation().kind()),
                () -> assertEquals(0x1234L, attrs.key()),
                () -> assertEquals(-1L, attrs.counter()),
                () -> assertTrue(provenance.inputs().isEmpty()),
                () -> assertEquals(1, provenance.producer().outputCount()));
    }

    @Test
    void equalWordsCreateDistinctExpressionOccurrencesWithIdentityEquality() {
        GraphRngState first = GraphRngState.initial(-1L, Long.MIN_VALUE);
        GraphRngState replay = GraphRngState.initial(-1L, Long.MIN_VALUE);
        Tensor firstTensor = first.tensor();
        Tensor replayTensor = replay.tensor();

        assertAll(
                () -> assertNotSame(first, replay),
                () -> assertNotEquals(first, replay),
                () -> assertNotSame(firstTensor, replayTensor),
                () -> assertNotEquals(firstTensor.id(), replayTensor.id()),
                () -> assertNotSame(
                        firstTensor.provenance().orElseThrow().producer(),
                        replayTensor.provenance().orElseThrow().producer()),
                () -> assertEquals(
                        firstTensor.provenance().orElseThrow().operation(),
                        replayTensor.provenance().orElseThrow().operation()));
    }

    @Test
    void exposesOnlyTheSelectedPublicAndPackagePrivateSurface() throws ReflectiveOperationException {
        var fields = Arrays.stream(GraphRngState.class.getDeclaredFields())
                .filter(field -> !field.isSynthetic())
                .toList();
        var constructors = GraphRngState.class.getDeclaredConstructors();
        var methods = Arrays.stream(GraphRngState.class.getDeclaredMethods())
                .filter(method -> !method.isSynthetic())
                .toList();
        var initial = GraphRngState.class.getDeclaredMethod("initial", long.class, long.class);
        var tensor = GraphRngState.class.getDeclaredMethod("tensor");

        assertAll(
                () -> assertTrue(Modifier.isPublic(GraphRngState.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(GraphRngState.class.getModifiers())),
                () -> assertEquals(1, fields.size()),
                () -> assertEquals("tensor", fields.getFirst().getName()),
                () -> assertEquals(Tensor.class, fields.getFirst().getType()),
                () -> assertTrue(Modifier.isPrivate(fields.getFirst().getModifiers())),
                () -> assertTrue(Modifier.isFinal(fields.getFirst().getModifiers())),
                () -> assertEquals(1, constructors.length),
                () -> assertEquals(List.of(Tensor.class),
                        Arrays.asList(constructors[0].getParameterTypes())),
                () -> assertFalse(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertFalse(Modifier.isProtected(constructors[0].getModifiers())),
                () -> assertFalse(Modifier.isPrivate(constructors[0].getModifiers())),
                () -> assertEquals(2, methods.size()),
                () -> assertTrue(Modifier.isPublic(initial.getModifiers())),
                () -> assertTrue(Modifier.isStatic(initial.getModifiers())),
                () -> assertEquals(GraphRngState.class, initial.getReturnType()),
                () -> assertFalse(Modifier.isPublic(tensor.getModifiers())),
                () -> assertFalse(Modifier.isProtected(tensor.getModifiers())),
                () -> assertFalse(Modifier.isPrivate(tensor.getModifiers())),
                () -> assertFalse(Modifier.isStatic(tensor.getModifiers())),
                () -> assertEquals(Tensor.class, tensor.getReturnType()),
                () -> assertEquals(0, GraphRngState.class.getDeclaredClasses().length));
    }

    @Test
    void validatesWrapperInvariantsInContractOrderWithoutAllocatingAnId()
            throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        TensorDescriptor floatWrongShape = unresolved(DataType.FLOAT32, Shape.scalar());
        TensorDescriptor intWrongShape = unresolved(DataType.INT64, Shape.scalar());
        Shape stateShape = Shape.of(2);
        TensorDescriptor intResolved = new TensorDescriptor(
                DataType.INT64,
                stateShape,
                Optional.of(LayoutDescriptor.contiguous(stateShape)),
                false);
        TensorDescriptor validDescriptor = unresolved(DataType.INT64, stateShape);
        Operation operation = new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE);

        Tensor wrongType = TensorFactory.create(floatWrongShape);
        Tensor wrongShape = TensorFactory.create(intWrongShape);
        Tensor resolved = TensorFactory.create(intResolved);
        Tensor labeled = TensorFactory.createDerived(
                validDescriptor, Optional.of("state"), operation, List.of());
        Tensor stored = TensorFactory.create(
                validDescriptor,
                Optional.empty(),
                Optional.of(new MemorySegmentStorage(
                        DataType.INT64, 2, MemorySegment.ofArray(new long[2]))));
        Tensor provenanceFree = TensorFactory.create(validDescriptor);
        long beforeFailures = next.get();

        NullPointerException nullFailure =
                assertThrows(NullPointerException.class, () -> new GraphRngState(null));
        IllegalArgumentException typeFailure =
                assertThrows(IllegalArgumentException.class, () -> new GraphRngState(wrongType));
        IllegalArgumentException shapeFailure =
                assertThrows(IllegalArgumentException.class, () -> new GraphRngState(wrongShape));
        IllegalArgumentException layoutFailure =
                assertThrows(IllegalArgumentException.class, () -> new GraphRngState(resolved));
        IllegalArgumentException labelFailure =
                assertThrows(IllegalArgumentException.class, () -> new GraphRngState(labeled));
        IllegalArgumentException storageFailure =
                assertThrows(IllegalArgumentException.class, () -> new GraphRngState(stored));
        IllegalArgumentException provenanceFailure = assertThrows(
                IllegalArgumentException.class, () -> new GraphRngState(provenanceFree));

        assertAll(
                () -> assertEquals("tensor", nullFailure.getMessage()),
                () -> assertEquals("state tensor data type must be INT64", typeFailure.getMessage()),
                () -> assertEquals("state tensor shape must be Shape[2]", shapeFailure.getMessage()),
                () -> assertEquals("state tensor layout must be unresolved", layoutFailure.getMessage()),
                () -> assertEquals("state tensor label must be absent", labelFailure.getMessage()),
                () -> assertEquals("state tensor host storage must be absent", storageFailure.getMessage()),
                () -> assertEquals("state tensor provenance must be present", provenanceFailure.getMessage()),
                () -> assertEquals(beforeFailures, next.get()));
    }

    @Test
    void packagePrivateWrappingRetainsTheExactValidatedTensorReference() {
        GraphRngState original = GraphRngState.initial(7L, 9L);
        GraphRngState secondWrapper = new GraphRngState(original.tensor());

        assertAll(
                () -> assertNotSame(original, secondWrapper),
                () -> assertSame(original.tensor(), secondWrapper.tensor()));
    }

    @Test
    void propagatesIdentifierExhaustionWithoutCreatingState() throws ReflectiveOperationException {
        AtomicLong next = nextTensorIdState();
        AtomicBoolean maximumClaimed = maximumTensorIdClaimedState();
        long savedNext = next.get();
        boolean savedClaimed = maximumClaimed.get();
        try {
            next.set(Long.MAX_VALUE);
            maximumClaimed.set(true);

            IllegalStateException failure = assertThrows(
                    IllegalStateException.class, () -> GraphRngState.initial(1L, 2L));

            assertAll(
                    () -> assertEquals("tensor identifier space exhausted", failure.getMessage()),
                    () -> assertEquals(Long.MAX_VALUE, next.get()),
                    () -> assertTrue(maximumClaimed.get()));
        } finally {
            next.set(savedNext);
            maximumClaimed.set(savedClaimed);
        }
    }

    private static TensorDescriptor unresolved(DataType dataType, Shape shape) {
        return new TensorDescriptor(dataType, shape, Optional.empty(), false);
    }

    private static AtomicLong nextTensorIdState() throws ReflectiveOperationException {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return (AtomicLong) field.get(null);
    }

    private static AtomicBoolean maximumTensorIdClaimedState() throws ReflectiveOperationException {
        var field = TensorFactory.class.getDeclaredField("MAXIMUM_TENSOR_ID_CLAIMED");
        field.setAccessible(true);
        return (AtomicBoolean) field.get(null);
    }

    private enum SampleKind implements OperationKind {
        SAMPLE;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 0, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
