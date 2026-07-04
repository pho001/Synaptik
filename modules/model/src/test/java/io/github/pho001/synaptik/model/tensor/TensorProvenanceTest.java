package io.github.pho001.synaptik.model.tensor;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class TensorProvenanceTest {
    @Test
    void hasExactlyTheRequiredRecordComponentsAndState() {
        var components = TensorProvenance.class.getRecordComponents();
        var fields = Arrays.stream(TensorProvenance.class.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();

        assertAll(
                () -> assertTrue(Modifier.isPublic(TensorProvenance.class.getModifiers())),
                () -> assertTrue(TensorProvenance.class.isRecord()),
                () -> assertEquals(2, components.length),
                () -> assertEquals("operation", components[0].getName()),
                () -> assertEquals(Operation.class, components[0].getType()),
                () -> assertEquals("inputs", components[1].getName()),
                () -> assertEquals(List.class, components[1].getType()),
                () -> assertEquals(
                        List.of("operation", "inputs"),
                        fields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(fields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))));
    }

    @Test
    void validatesReferencesAndElementsInExactOrder() {
        Operation operation = operation();
        Tensor input = tensor(1);

        NullPointerException nullOperation = assertThrows(
                NullPointerException.class, () -> new TensorProvenance(null, null));
        NullPointerException nullInputs = assertThrows(
                NullPointerException.class, () -> new TensorProvenance(operation, null));
        NullPointerException firstNull = assertThrows(
                NullPointerException.class,
                () -> new TensorProvenance(operation, Arrays.asList(input, null, null)));

        assertAll(
                () -> assertEquals("operation", nullOperation.getMessage()),
                () -> assertEquals("inputs", nullInputs.getMessage()),
                () -> assertEquals("inputs[1]", firstNull.getMessage()));
    }

    @Test
    void snapshotsTheOrderedListAndRetainsExactReferences() {
        Operation operation = operation();
        Tensor first = tensor(1);
        Tensor second = tensor(2);
        List<Tensor> source = new ArrayList<>(List.of(second, first));

        TensorProvenance provenance = new TensorProvenance(operation, source);
        source.clear();

        assertAll(
                () -> assertSame(operation, provenance.operation()),
                () -> assertEquals(2, provenance.inputs().size()),
                () -> assertSame(second, provenance.inputs().get(0)),
                () -> assertSame(first, provenance.inputs().get(1)),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> provenance.inputs().add(first)));
    }

    @Test
    void acceptsEmptyAndRepeatedInputsWithoutSemanticValidation() {
        Operation operation = operation();
        Tensor input = tensor(1);

        TensorProvenance empty = new TensorProvenance(operation, List.of());
        TensorProvenance repeated = new TensorProvenance(operation, List.of(input, input));

        assertAll(
                () -> assertTrue(empty.inputs().isEmpty()),
                () -> assertEquals(2, repeated.inputs().size()),
                () -> assertSame(input, repeated.inputs().get(0)),
                () -> assertSame(input, repeated.inputs().get(1)));
    }

    @Test
    void usesRecordValueSemanticsWithoutConflatingTensorIdentifiers() {
        Operation firstOperation = operation();
        Operation equalOperation = operation();
        Tensor firstInput = tensor(7);
        Tensor equalIdDifferentInput = tensor(7);

        TensorProvenance first = new TensorProvenance(firstOperation, List.of(firstInput));
        TensorProvenance equal = new TensorProvenance(equalOperation, List.of(firstInput));
        TensorProvenance differentInput =
                new TensorProvenance(equalOperation, List.of(equalIdDifferentInput));

        assertAll(
                () -> assertNotSame(firstOperation, equalOperation),
                () -> assertEquals(firstOperation, equalOperation),
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertEquals(firstInput.id(), equalIdDifferentInput.id()),
                () -> assertNotEquals(firstInput, equalIdDifferentInput),
                () -> assertNotEquals(first, differentInput));
    }

    @Test
    void generatedDiagnosticTextIncludesRecordComponents() {
        Tensor input = tensor(4);

        String text = new TensorProvenance(operation(), List.of(input, input)).toString();

        assertAll(
                () -> assertTrue(text.contains("TensorProvenance")),
                () -> assertTrue(text.contains("operation=")),
                () -> assertTrue(text.contains("inputs=")),
                () -> assertTrue(text.contains("SAMPLE")),
                () -> assertTrue(text.contains("TensorId[value=4]")));
    }

    private static Operation operation() {
        return new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE);
    }

    private static Tensor tensor(long id) {
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), false);
        return new Tensor(
                new TensorId(id),
                descriptor,
                Optional.empty(),
                Optional.empty(),
                Optional.empty());
    }

    private enum SampleKind implements OperationKind {
        SAMPLE
    }
}
