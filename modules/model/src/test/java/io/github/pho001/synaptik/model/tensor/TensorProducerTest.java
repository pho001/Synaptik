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
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

class TensorProducerTest {
    @Test
    void isFinalIdentityClassWithExactlyRequiredFieldsAndAccessors() {
        var fields = Arrays.stream(TensorProducer.class.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();
        Set<String> publicMethods = Arrays.stream(TensorProducer.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .map(method -> method.getName())
                .collect(Collectors.toSet());

        assertAll(
                () -> assertTrue(Modifier.isPublic(TensorProducer.class.getModifiers())),
                () -> assertTrue(Modifier.isFinal(TensorProducer.class.getModifiers())),
                () -> assertFalse(TensorProducer.class.isRecord()),
                () -> assertEquals(
                        List.of("operation", "inputs", "outputDescriptors"),
                        fields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(fields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(
                        Set.of("operation", "inputs", "outputDescriptors", "outputCount"),
                        publicMethods));
    }

    @Test
    void validatesContainersAndElementsInRequiredOrder() {
        Operation operation = operation(MultiKind.MULTI);
        Tensor input = tensor(1);
        TensorDescriptor descriptor = descriptor();

        NullPointerException nullOperation = assertThrows(
                NullPointerException.class, () -> new TensorProducer(null, null, null));
        NullPointerException nullInputs = assertThrows(
                NullPointerException.class, () -> new TensorProducer(operation, null, null));
        NullPointerException nullOutputs = assertThrows(
                NullPointerException.class,
                () -> new TensorProducer(operation, List.of(input), null));
        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> new TensorProducer(
                        operation, Arrays.asList(input, null), List.of(descriptor, descriptor)));
        IllegalArgumentException emptyOutputs = assertThrows(
                IllegalArgumentException.class,
                () -> new TensorProducer(operation, List.of(input, input), List.of()));
        NullPointerException nullOutput = assertThrows(
                NullPointerException.class,
                () -> new TensorProducer(
                        operation,
                        List.of(input, input),
                        Arrays.asList(descriptor, null)));

        assertAll(
                () -> assertEquals("operation", nullOperation.getMessage()),
                () -> assertEquals("inputs", nullInputs.getMessage()),
                () -> assertEquals("outputDescriptors", nullOutputs.getMessage()),
                () -> assertEquals("inputs[1]", nullInput.getMessage()),
                () -> assertEquals("outputDescriptors must not be empty", emptyOutputs.getMessage()),
                () -> assertEquals("outputDescriptors[1]", nullOutput.getMessage()));
    }

    @Test
    void snapshotsOrderedExactReferencesAndDerivesOutputCount() {
        Operation operation = operation(MultiKind.MULTI);
        Tensor input = tensor(1);
        TensorDescriptor descriptor = descriptor();
        List<Tensor> inputs = new ArrayList<>(List.of(input, input));
        List<TensorDescriptor> outputs = new ArrayList<>(List.of(descriptor, descriptor));

        TensorProducer producer = new TensorProducer(operation, inputs, outputs);
        inputs.clear();
        outputs.clear();

        assertAll(
                () -> assertSame(operation, producer.operation()),
                () -> assertEquals(2, producer.inputs().size()),
                () -> assertSame(input, producer.inputs().get(0)),
                () -> assertSame(input, producer.inputs().get(1)),
                () -> assertEquals(2, producer.outputCount()),
                () -> assertSame(descriptor, producer.outputDescriptors().get(0)),
                () -> assertSame(descriptor, producer.outputDescriptors().get(1)),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> producer.inputs().add(input)),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> producer.outputDescriptors().add(descriptor)));
    }

    @Test
    void validatesFinalCountsAndRetainsOrdinaryIdentity() {
        Tensor input = tensor(1);
        TensorDescriptor descriptor = descriptor();
        Operation operation = operation(MultiKind.MULTI);

        IllegalArgumentException wrongInputs = assertThrows(
                IllegalArgumentException.class,
                () -> new TensorProducer(operation, List.of(input), List.of(descriptor, descriptor)));
        IllegalArgumentException wrongOutputs = assertThrows(
                IllegalArgumentException.class,
                () -> new TensorProducer(operation, List.of(input, input), List.of(descriptor)));
        TensorProducer first = new TensorProducer(
                operation, List.of(input, input), List.of(descriptor, descriptor));
        TensorProducer second = new TensorProducer(
                operation, List.of(input, input), List.of(descriptor, descriptor));

        assertAll(
                () -> assertTrue(wrongInputs.getMessage().contains("input count 1")),
                () -> assertTrue(wrongOutputs.getMessage().contains("output count 1")),
                () -> assertNotSame(first, second),
                () -> assertNotEquals(first, second));
    }

    private static Operation operation(OperationKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }

    private static Tensor tensor(long id) {
        return new Tensor(
                new TensorId(id),
                descriptor(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    private static TensorDescriptor descriptor() {
        return new TensorDescriptor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), false);
    }

    private enum MultiKind implements OperationKind {
        MULTI;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 2, 2));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
