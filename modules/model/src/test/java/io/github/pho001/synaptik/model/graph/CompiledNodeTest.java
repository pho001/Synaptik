package io.github.pho001.synaptik.model.graph;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class CompiledNodeTest {
    @Test
    void hasExactlyTheRequiredFourComponentRecordState() {
        var components = CompiledNode.class.getRecordComponents();
        var instanceFields = Arrays.stream(CompiledNode.class.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();

        assertAll(
                () -> assertTrue(CompiledNode.class.isRecord()),
                () -> assertEquals(4, components.length),
                () -> assertEquals("id", components[0].getName()),
                () -> assertEquals(NodeId.class, components[0].getType()),
                () -> assertEquals("operation", components[1].getName()),
                () -> assertEquals(Operation.class, components[1].getType()),
                () -> assertEquals("inputs", components[2].getName()),
                () -> assertEquals(List.class, components[2].getType()),
                () -> assertEquals("outputs", components[3].getName()),
                () -> assertEquals(List.class, components[3].getType()),
                () -> assertEquals(
                        List.of("id", "operation", "inputs", "outputs"),
                        instanceFields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(instanceFields.stream().allMatch(
                        field -> Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(1, CompiledNode.class.getDeclaredConstructors().length));
    }

    @Test
    void rejectsEachNullComponentReferenceWithItsExactName() {
        NodeId id = new NodeId(1);
        Operation operation = operation(SampleKind.SAMPLE, 2);
        List<ValueId> inputs = List.of(new ValueId(2));
        List<ValueId> outputs = List.of(new ValueId(3));

        NullPointerException nullId = assertThrows(
                NullPointerException.class,
                () -> new CompiledNode(null, operation, inputs, outputs));
        NullPointerException nullOperation = assertThrows(
                NullPointerException.class,
                () -> new CompiledNode(id, null, inputs, outputs));
        NullPointerException nullInputs = assertThrows(
                NullPointerException.class,
                () -> new CompiledNode(id, operation, null, outputs));
        NullPointerException nullOutputs = assertThrows(
                NullPointerException.class,
                () -> new CompiledNode(id, operation, inputs, null));

        assertAll(
                () -> assertEquals("id", nullId.getMessage()),
                () -> assertEquals("operation", nullOperation.getMessage()),
                () -> assertEquals("inputs", nullInputs.getMessage()),
                () -> assertEquals("outputs", nullOutputs.getMessage()));
    }

    @Test
    void rejectsNullElementsInEncounterOrderWithIndexedMessages() {
        List<ValueId> inputs = new ArrayList<>(
                Arrays.asList(new ValueId(1), null, null));
        List<ValueId> outputs = new ArrayList<>(
                Arrays.asList(new ValueId(2), new ValueId(3), null));

        NullPointerException nullInput = assertThrows(
                NullPointerException.class,
                () -> new CompiledNode(
                        new NodeId(1),
                        operation(SampleKind.SAMPLE, 2),
                        inputs,
                        List.of(new ValueId(4))));
        NullPointerException nullOutput = assertThrows(
                NullPointerException.class,
                () -> new CompiledNode(
                        new NodeId(1), operation(SampleKind.SAMPLE, 2), List.of(), outputs));

        assertAll(
                () -> assertEquals("inputs[1]", nullInput.getMessage()),
                () -> assertEquals("outputs[2]", nullOutput.getMessage()));
    }

    @Test
    void snapshotsOrderedListsAndPreventsCallerOrAccessorMutation() {
        NodeId id = new NodeId(5);
        Operation operation = operation(SampleKind.SAMPLE, 2);
        ValueId firstInput = new ValueId(10);
        ValueId secondInput = new ValueId(11);
        ValueId firstOutput = new ValueId(20);
        ValueId secondOutput = new ValueId(21);
        List<ValueId> inputs = new ArrayList<>(List.of(firstInput, secondInput));
        List<ValueId> outputs = new ArrayList<>(List.of(firstOutput, secondOutput));

        CompiledNode node = new CompiledNode(id, operation, inputs, outputs);
        inputs.clear();
        outputs.set(0, new ValueId(99));

        assertAll(
                () -> assertSame(id, node.id()),
                () -> assertSame(operation, node.operation()),
                () -> assertEquals(List.of(firstInput, secondInput), node.inputs()),
                () -> assertEquals(List.of(firstOutput, secondOutput), node.outputs()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> node.inputs().add(new ValueId(12))),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> node.outputs().set(0, new ValueId(22))));
    }

    @Test
    void permitsEmptyAndRepeatedOrderedInputs() {
        CompiledNode emptyInputs = new CompiledNode(
                new NodeId(1),
                operation(SampleKind.SOURCE, 0),
                List.of(),
                List.of(new ValueId(3)));
        ValueId repeated = new ValueId(4);
        CompiledNode repeatedInputs = new CompiledNode(
                new NodeId(2),
                operation(SampleKind.SAMPLE, 2),
                List.of(repeated, new ValueId(5), repeated),
                List.of(new ValueId(6)));

        assertAll(
                () -> assertTrue(emptyInputs.inputs().isEmpty()),
                () -> assertEquals(
                        List.of(repeated, new ValueId(5), repeated), repeatedInputs.inputs()),
                () -> assertSame(repeated, repeatedInputs.inputs().get(0)),
                () -> assertSame(repeated, repeatedInputs.inputs().get(2)));
    }

    @Test
    void acceptsSingleAndMultipleDistinctOutputsInOrder() {
        CompiledNode singleOutput = new CompiledNode(
                new NodeId(1),
                operation(SampleKind.SAMPLE, 1),
                List.of(new ValueId(2)),
                List.of(new ValueId(3)));
        CompiledNode multipleOutputs = new CompiledNode(
                new NodeId(2),
                operation(SampleKind.SAMPLE, 2),
                List.of(new ValueId(4)),
                List.of(new ValueId(7), new ValueId(6), new ValueId(5)));

        assertAll(
                () -> assertEquals(List.of(new ValueId(3)), singleOutput.outputs()),
                () -> assertEquals(
                        List.of(new ValueId(7), new ValueId(6), new ValueId(5)),
                        multipleOutputs.outputs()));
    }

    @Test
    void rejectsEmptyAndFirstLaterDuplicateOutputWithExactMessages() {
        IllegalArgumentException empty = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledNode(
                        new NodeId(1), operation(SampleKind.SAMPLE, 0), List.of(), List.of()));
        ValueId first = new ValueId(7);
        ValueId second = new ValueId(8);
        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledNode(
                        new NodeId(1),
                        operation(SampleKind.SAMPLE, 0),
                        List.of(),
                        List.of(first, second, first, second)));

        assertAll(
                () -> assertEquals("outputs must not be empty", empty.getMessage()),
                () -> assertEquals(
                        "outputs[2] duplicates ValueId[value=7]", duplicate.getMessage()));
    }

    @Test
    void performsNoCrossListOrGraphWideValidation() {
        ValueId sharedInputAndOutput = new ValueId(40);
        ValueId unresolvedElsewhere = new ValueId(999);
        CompiledNode first = new CompiledNode(
                new NodeId(1),
                operation(SampleKind.SAMPLE, -100),
                List.of(unresolvedElsewhere, sharedInputAndOutput),
                List.of(sharedInputAndOutput));
        CompiledNode second = new CompiledNode(
                new NodeId(1),
                operation(OtherSampleKind.SAMPLE, 100),
                List.of(sharedInputAndOutput),
                List.of(sharedInputAndOutput));

        assertAll(
                () -> assertEquals(sharedInputAndOutput, first.inputs().get(1)),
                () -> assertEquals(sharedInputAndOutput, first.outputs().get(0)),
                () -> assertEquals(first.id(), second.id()),
                () -> assertEquals(first.outputs(), second.outputs()),
                () -> assertNotEquals(first.operation(), second.operation()));
    }

    @Test
    void equalityHashingAndOrderSensitivityCoverCompleteState() {
        CompiledNode first = new CompiledNode(
                new NodeId(1),
                operation(SampleKind.SAMPLE, 2),
                List.of(new ValueId(2), new ValueId(3), new ValueId(2)),
                List.of(new ValueId(4), new ValueId(5)));
        CompiledNode equal = new CompiledNode(
                new NodeId(1),
                operation(SampleKind.SAMPLE, 2),
                List.of(new ValueId(2), new ValueId(3), new ValueId(2)),
                List.of(new ValueId(4), new ValueId(5)));
        CompiledNode differentId = new CompiledNode(
                new NodeId(9), first.operation(), first.inputs(), first.outputs());
        CompiledNode differentOperation = new CompiledNode(
                first.id(), operation(SampleKind.OTHER, 2), first.inputs(), first.outputs());
        CompiledNode reorderedInputs = new CompiledNode(
                first.id(),
                first.operation(),
                List.of(new ValueId(2), new ValueId(2), new ValueId(3)),
                first.outputs());
        CompiledNode reorderedOutputs = new CompiledNode(
                first.id(),
                first.operation(),
                first.inputs(),
                List.of(new ValueId(5), new ValueId(4)));

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, differentId),
                () -> assertNotEquals(first, differentOperation),
                () -> assertNotEquals(first, reorderedInputs),
                () -> assertNotEquals(first, reorderedOutputs));
    }

    @Test
    void diagnosticTextNamesAllComponentsAndRepresentativeValues() {
        String text = new CompiledNode(
                        new NodeId(1),
                        operation(SampleKind.SAMPLE, 2),
                        List.of(new ValueId(2), new ValueId(2)),
                        List.of(new ValueId(3), new ValueId(4)))
                .toString();

        assertAll(
                () -> assertTrue(text.contains("CompiledNode")),
                () -> assertTrue(text.contains("id=")),
                () -> assertTrue(text.contains("NodeId")),
                () -> assertTrue(text.contains("operation=")),
                () -> assertTrue(text.contains("Operation")),
                () -> assertTrue(text.contains("inputs=")),
                () -> assertTrue(text.contains("ValueId[value=2]")),
                () -> assertTrue(text.contains("outputs=")),
                () -> assertTrue(text.contains("ValueId[value=4]")));
    }

    @Test
    void validatesInputAndOutputCountsAgainstTheOperationSignature() {
        Operation binary = new Operation(CardinalityKind.BINARY, NoOperationAttrs.INSTANCE);
        Operation source = new Operation(CardinalityKind.SOURCE, NoOperationAttrs.INSTANCE);
        Operation multiOutput =
                new Operation(CardinalityKind.MULTI_OUTPUT, NoOperationAttrs.INSTANCE);

        CompiledNode validBinary = new CompiledNode(
                new NodeId(30),
                binary,
                List.of(new ValueId(1), new ValueId(2)),
                List.of(new ValueId(3)));
        CompiledNode validSource = new CompiledNode(
                new NodeId(31), source, List.of(), List.of(new ValueId(4)));
        CompiledNode validMultiOutput = new CompiledNode(
                new NodeId(32),
                multiOutput,
                List.of(new ValueId(4)),
                List.of(new ValueId(5), new ValueId(6)));

        IllegalArgumentException inputFailure = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledNode(
                        new NodeId(33),
                        binary,
                        List.of(new ValueId(1)),
                        List.of(new ValueId(2))));
        IllegalArgumentException outputFailure = assertThrows(
                IllegalArgumentException.class,
                () -> new CompiledNode(
                        new NodeId(34),
                        binary,
                        List.of(new ValueId(1), new ValueId(2)),
                        List.of(new ValueId(3), new ValueId(4))));

        assertAll(
                () -> assertEquals(2, validBinary.inputs().size()),
                () -> assertTrue(validSource.inputs().isEmpty()),
                () -> assertEquals(2, validMultiOutput.outputs().size()),
                () -> assertTrue(inputFailure.getMessage().contains("input count")),
                () -> assertTrue(outputFailure.getMessage().contains("output count")));
    }

    private static Operation operation(OperationKind kind, int arityHint) {
        return new Operation(kind, new SampleAttrs(arityHint));
    }

    private enum SampleKind implements OperationKind {
        SOURCE,
        SAMPLE,
        OTHER;

        private static final List<OperationSignature> SIGNATURES = List.of(new OperationSignature(
                SampleAttrs.class, 0, Integer.MAX_VALUE, 1, Integer.MAX_VALUE));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }

    private enum OtherSampleKind implements OperationKind {
        SAMPLE;

        private static final List<OperationSignature> SIGNATURES = List.of(new OperationSignature(
                SampleAttrs.class, 0, Integer.MAX_VALUE, 1, Integer.MAX_VALUE));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }

    private enum CardinalityKind implements OperationKind {
        BINARY,
        SOURCE,
        MULTI_OUTPUT;

        private static final List<OperationSignature> BINARY_SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 2, 1));
        private static final List<OperationSignature> SOURCE_SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 0, 1));
        private static final List<OperationSignature> MULTI_OUTPUT_SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 2));

        @Override
        public List<OperationSignature> signatures() {
            return switch (this) {
                case BINARY -> BINARY_SIGNATURES;
                case SOURCE -> SOURCE_SIGNATURES;
                case MULTI_OUTPUT -> MULTI_OUTPUT_SIGNATURES;
            };
        }
    }

    private record SampleAttrs(int arityHint) implements OperationAttrs {}
}
