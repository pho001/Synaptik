package io.github.pho001.synaptik.planning.capability;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.RecordComponent;
import java.lang.reflect.Type;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class BackendCapabilityContractTest {
    private static final TensorDescriptor FLOAT_VECTOR =
            new TensorDescriptor(DataType.FLOAT32, Shape.of(2), Optional.empty(), false);
    private static final TensorDescriptor FLOAT_MATRIX =
            new TensorDescriptor(DataType.FLOAT64, Shape.of(2, 3), Optional.empty(), true);

    @Test
    void queryHasTheExactGenericPublicRecordShape() throws ReflectiveOperationException {
        Class<OperationCapabilityQuery> type = OperationCapabilityQuery.class;
        RecordComponent[] components = type.getRecordComponents();
        Constructor<?>[] constructors = type.getDeclaredConstructors();
        ParameterizedType inputsType = (ParameterizedType) components[1].getGenericType();
        ParameterizedType outputsType = (ParameterizedType) components[2].getGenericType();

        assertAll(
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.planning.capability",
                                type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () ->
                        assertArrayEquals(
                                new String[] {"operation", "inputs", "outputs"},
                                Arrays.stream(components)
                                        .map(RecordComponent::getName)
                                        .toArray(String[]::new)),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {Operation.class, List.class, List.class},
                                Arrays.stream(components)
                                        .map(RecordComponent::getType)
                                        .toArray(Class<?>[]::new)),
                () -> assertEquals(List.class, inputsType.getRawType()),
                () ->
                        assertArrayEquals(
                                new Type[] {TensorDescriptor.class},
                                inputsType.getActualTypeArguments()),
                () -> assertEquals(List.class, outputsType.getRawType()),
                () ->
                        assertArrayEquals(
                                new Type[] {TensorDescriptor.class},
                                outputsType.getActualTypeArguments()),
                () -> assertEquals(3, type.getDeclaredFields().length),
                () ->
                        assertArrayEquals(
                                new String[] {"operation", "inputs", "outputs"},
                                Arrays.stream(type.getDeclaredFields())
                                        .map(field -> field.getName())
                                        .toArray(String[]::new)),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {Operation.class, List.class, List.class},
                                constructors[0].getParameterTypes()),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(6, type.getDeclaredMethods().length),
                () ->
                        assertEquals(
                                Set.of(
                                        "operation",
                                        "inputs",
                                        "outputs",
                                        "equals",
                                        "hashCode",
                                        "toString"),
                                Arrays.stream(type.getDeclaredMethods())
                                        .filter(method -> Modifier.isPublic(method.getModifiers()))
                                        .map(Method::getName)
                                        .collect(Collectors.toSet())),
                () ->
                        assertEquals(
                                Operation.class,
                                type.getDeclaredMethod("operation").getReturnType()),
                () -> assertEquals(List.class, type.getDeclaredMethod("inputs").getReturnType()),
                () -> assertEquals(List.class, type.getDeclaredMethod("outputs").getReturnType()),
                () ->
                        assertEquals(
                                inputsType,
                                type.getDeclaredMethod("inputs").getGenericReturnType()),
                () ->
                        assertEquals(
                                outputsType,
                                type.getDeclaredMethod("outputs").getGenericReturnType()));
    }

    @Test
    void validatesTopLevelReferencesInExactOrderWithExactMessages() {
        Operation unary = unaryOperation();

        NullPointerException operationFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> new OperationCapabilityQuery(null, null, null));
        NullPointerException inputsFailure =
                assertThrows(
                        NullPointerException.class,
                        () -> new OperationCapabilityQuery(unary, null, null));
        NullPointerException outputsFailure =
                assertThrows(
                        NullPointerException.class,
                        () ->
                                new OperationCapabilityQuery(
                                        unary, Arrays.asList((TensorDescriptor) null), null));

        assertAll(
                () -> assertEquals("operation", operationFailure.getMessage()),
                () -> assertEquals("inputs", inputsFailure.getMessage()),
                () -> assertEquals("outputs", outputsFailure.getMessage()));
    }

    @Test
    void scansInputAndOutputElementsInEncounterOrderWithExactMessages() {
        Operation unary = unaryOperation();

        NullPointerException inputFailure =
                assertThrows(
                        NullPointerException.class,
                        () ->
                                new OperationCapabilityQuery(
                                        unary,
                                        Arrays.asList(FLOAT_VECTOR, null, null),
                                        List.of(FLOAT_VECTOR)));
        NullPointerException outputFailure =
                assertThrows(
                        NullPointerException.class,
                        () ->
                                new OperationCapabilityQuery(
                                        unary,
                                        List.of(FLOAT_VECTOR),
                                        Arrays.asList(FLOAT_VECTOR, null, null)));

        assertAll(
                () -> assertEquals("inputs[1]", inputFailure.getMessage()),
                () -> assertEquals("outputs[1]", outputFailure.getMessage()));
    }

    @Test
    void inputFailurePrecedesEveryOutputElementReadAndOccurrenceValidation() {
        Operation unary = unaryOperation();
        List<TensorDescriptor> unreadableOutputs =
                new AbstractList<>() {
                    @Override
                    public TensorDescriptor get(int index) {
                        throw new AssertionError("outputs must not be scanned");
                    }

                    @Override
                    public int size() {
                        return 1;
                    }
                };

        NullPointerException failure =
                assertThrows(
                        NullPointerException.class,
                        () ->
                                new OperationCapabilityQuery(
                                        unary,
                                        Arrays.asList(FLOAT_VECTOR, null),
                                        unreadableOutputs));

        assertEquals("inputs[1]", failure.getMessage());
    }

    @Test
    void snapshotsOrderedMembershipAndRetainsExactReferences() {
        Operation binary =
                new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE);
        List<TensorDescriptor> inputs = new ArrayList<>(List.of(FLOAT_VECTOR, FLOAT_MATRIX));
        List<TensorDescriptor> outputs = new ArrayList<>(List.of(FLOAT_MATRIX));

        OperationCapabilityQuery query =
                new OperationCapabilityQuery(binary, inputs, outputs);
        inputs.clear();
        inputs.add(FLOAT_MATRIX);
        outputs.clear();
        outputs.add(FLOAT_VECTOR);

        assertAll(
                () -> assertSame(binary, query.operation()),
                () -> assertNotSame(inputs, query.inputs()),
                () -> assertNotSame(outputs, query.outputs()),
                () -> assertEquals(2, query.inputs().size()),
                () -> assertSame(FLOAT_VECTOR, query.inputs().get(0)),
                () -> assertSame(FLOAT_MATRIX, query.inputs().get(1)),
                () -> assertEquals(1, query.outputs().size()),
                () -> assertSame(FLOAT_MATRIX, query.outputs().get(0)),
                () ->
                        assertThrows(
                                UnsupportedOperationException.class,
                                () -> query.inputs().add(FLOAT_VECTOR)),
                () ->
                        assertThrows(
                                UnsupportedOperationException.class,
                                () -> query.outputs().clear()));
    }

    @Test
    void acceptsSignaturePermittedEmptyRepeatedAndMultiOutputOccurrences() {
        Operation zeroInput =
                new Operation(
                        new TestKind(
                                "ZERO_INPUT",
                                List.of(
                                        OperationSignature.fixed(
                                                NoOperationAttrs.class, 0, 1))),
                        NoOperationAttrs.INSTANCE);
        Operation binary =
                new Operation(BinaryArithmeticKind.MUL, NoOperationAttrs.INSTANCE);
        Operation twoOutput =
                new Operation(
                        new TestKind(
                                "TWO_OUTPUT",
                                List.of(
                                        OperationSignature.fixed(
                                                NoOperationAttrs.class, 1, 2))),
                        NoOperationAttrs.INSTANCE);

        OperationCapabilityQuery empty =
                new OperationCapabilityQuery(zeroInput, List.of(), List.of(FLOAT_VECTOR));
        OperationCapabilityQuery repeatedInputs =
                new OperationCapabilityQuery(
                        binary,
                        List.of(FLOAT_VECTOR, FLOAT_VECTOR),
                        List.of(FLOAT_MATRIX));
        OperationCapabilityQuery repeatedOutputs =
                new OperationCapabilityQuery(
                        twoOutput,
                        List.of(FLOAT_MATRIX),
                        List.of(FLOAT_VECTOR, FLOAT_VECTOR));

        assertAll(
                () -> assertTrue(empty.inputs().isEmpty()),
                () -> assertSame(repeatedInputs.inputs().get(0), repeatedInputs.inputs().get(1)),
                () -> assertSame(repeatedOutputs.outputs().get(0), repeatedOutputs.outputs().get(1)),
                () -> assertEquals(2, repeatedOutputs.outputs().size()));
    }

    @Test
    void propagatesSignatureCountValidationAfterBothSnapshots() {
        Operation unary = unaryOperation();

        IllegalArgumentException inputFailure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                new OperationCapabilityQuery(
                                        unary,
                                        List.of(),
                                        List.of(FLOAT_VECTOR, FLOAT_MATRIX)));
        IllegalArgumentException outputFailure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                new OperationCapabilityQuery(
                                        unary,
                                        List.of(FLOAT_VECTOR),
                                        List.of()));

        assertAll(
                () ->
                        assertEquals(
                                "input count 0 is outside accepted range [1, 1]",
                                inputFailure.getMessage()),
                () ->
                        assertEquals(
                                "output count 0 is outside accepted range [1, 1]",
                                outputFailure.getMessage()));
    }

    @Test
    void preservesOrdinaryRecordValueBehavior() {
        Operation operation = unaryOperation();
        OperationCapabilityQuery query =
                new OperationCapabilityQuery(
                        operation, List.of(FLOAT_VECTOR), List.of(FLOAT_MATRIX));
        OperationCapabilityQuery equal =
                new OperationCapabilityQuery(
                        operation,
                        new ArrayList<>(List.of(FLOAT_VECTOR)),
                        new ArrayList<>(List.of(FLOAT_MATRIX)));
        OperationCapabilityQuery different =
                new OperationCapabilityQuery(
                        operation, List.of(FLOAT_MATRIX), List.of(FLOAT_VECTOR));

        assertAll(
                () -> assertEquals(query, equal),
                () -> assertEquals(query.hashCode(), equal.hashCode()),
                () -> assertNotEquals(query, different),
                () -> assertTrue(query.toString().startsWith("OperationCapabilityQuery[")),
                () -> assertTrue(query.toString().contains("operation=")),
                () -> assertTrue(query.toString().contains("inputs=[")),
                () -> assertTrue(query.toString().contains("outputs=[")));
    }

    @Test
    void providerHasExactlyTheTwoPublicAbstractMethods() throws ReflectiveOperationException {
        Class<BackendCapabilityProvider> type = BackendCapabilityProvider.class;
        Method backendId = type.getDeclaredMethod("backendId");
        Method supports = type.getDeclaredMethod("supports", OperationCapabilityQuery.class);

        assertAll(
                () ->
                        assertEquals(
                                "io.github.pho001.synaptik.planning.capability",
                                type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isInterface(type.getModifiers())),
                () -> assertFalse(type.isSealed()),
                () -> assertEquals(0, type.getDeclaredFields().length),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(2, type.getDeclaredMethods().length),
                () ->
                        assertEquals(
                                Set.of("backendId", "supports"),
                                Arrays.stream(type.getDeclaredMethods())
                                        .map(Method::getName)
                                        .collect(Collectors.toSet())),
                () -> assertTrue(Modifier.isPublic(backendId.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(backendId.getModifiers())),
                () -> assertFalse(backendId.isDefault()),
                () -> assertEquals(BackendId.class, backendId.getReturnType()),
                () -> assertEquals(0, backendId.getParameterCount()),
                () -> assertTrue(Modifier.isPublic(supports.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(supports.getModifiers())),
                () -> assertFalse(supports.isDefault()),
                () -> assertEquals(boolean.class, supports.getReturnType()),
                () ->
                        assertArrayEquals(
                                new Class<?>[] {OperationCapabilityQuery.class},
                                supports.getParameterTypes()));
    }

    @Test
    void providerIdentityAndCapabilityAnswerAreStableNarrowAndNullRejecting() {
        BackendId cpu = new BackendId("cpu");
        TestProvider supported = new TestProvider(cpu, true);
        TestProvider unsupported = new TestProvider(new BackendId("metal"), false);
        OperationCapabilityQuery query =
                new OperationCapabilityQuery(
                        unaryOperation(), List.of(FLOAT_VECTOR), List.of(FLOAT_VECTOR));

        NullPointerException failure =
                assertThrows(NullPointerException.class, () -> supported.supports(null));

        assertAll(
                () -> assertSame(cpu, supported.backendId()),
                () -> assertSame(cpu, supported.backendId()),
                () -> assertTrue(supported.supports(query)),
                () -> assertTrue(supported.supports(query)),
                () -> assertFalse(unsupported.supports(query)),
                () -> assertEquals("query", failure.getMessage()));
    }

    private static Operation unaryOperation() {
        return new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE);
    }

    private record TestKind(String name, List<OperationSignature> signatures)
            implements OperationKind {
        private TestKind {
            name = Objects.requireNonNull(name, "name");
            signatures = List.copyOf(signatures);
        }
    }

    private record TestProvider(BackendId backendId, boolean answer)
            implements BackendCapabilityProvider {
        private TestProvider {
            Objects.requireNonNull(backendId, "backendId");
        }

        @Override
        public boolean supports(OperationCapabilityQuery query) {
            Objects.requireNonNull(query, "query");
            return answer;
        }
    }
}
