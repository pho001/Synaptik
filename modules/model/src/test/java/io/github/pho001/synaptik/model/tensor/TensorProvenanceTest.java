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
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.reflect.Modifier;
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
                () -> assertEquals("producer", components[0].getName()),
                () -> assertEquals(TensorProducer.class, components[0].getType()),
                () -> assertEquals("outputIndex", components[1].getName()),
                () -> assertEquals(int.class, components[1].getType()),
                () -> assertEquals(
                        List.of("producer", "outputIndex"),
                        fields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(fields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))));
    }

    @Test
    void validatesProducerAndEveryOutputIndexBound() {
        TensorProducer producer = producer();

        NullPointerException nullProducer = assertThrows(
                NullPointerException.class, () -> new TensorProvenance(null, 0));
        IllegalArgumentException negative = assertThrows(
                IllegalArgumentException.class, () -> new TensorProvenance(producer, -1));
        IllegalArgumentException tooLarge = assertThrows(
                IllegalArgumentException.class, () -> new TensorProvenance(producer, 2));

        assertAll(
                () -> assertEquals("producer", nullProducer.getMessage()),
                () -> assertEquals("outputIndex must be non-negative: -1", negative.getMessage()),
                () -> assertTrue(tooLarge.getMessage().contains("outputIndex 2")),
                () -> assertTrue(tooLarge.getMessage().contains("output count 2")));
    }

    @Test
    void derivesExactOperationInputsAndSelectedDescriptor() {
        TensorProducer producer = producer();
        TensorProvenance provenance = new TensorProvenance(producer, 1);

        assertAll(
                () -> assertSame(producer, provenance.producer()),
                () -> assertEquals(1, provenance.outputIndex()),
                () -> assertSame(producer.operation(), provenance.operation()),
                () -> assertSame(producer.inputs(), provenance.inputs()),
                () -> assertSame(producer.outputDescriptors().get(1), provenance.outputDescriptor()),
                () -> assertSame(
                        producer,
                        producer.output(1).provenance().orElseThrow().producer()),
                () -> assertEquals(
                        provenance, producer.output(1).provenance().orElseThrow()),
                () -> assertSame(
                        provenance.outputDescriptor(), producer.output(1).descriptor()));
    }

    @Test
    void recordValueSemanticsUseProducerIdentityAndOutputIndex() {
        TensorProducer producer = producer();
        TensorProducer separate = producer();
        TensorProvenance first = new TensorProvenance(producer, 0);
        TensorProvenance equal = new TensorProvenance(producer, 0);
        TensorProvenance otherPosition = new TensorProvenance(producer, 1);
        TensorProvenance otherOccurrence = new TensorProvenance(separate, 0);

        assertAll(
                () -> assertNotSame(producer, separate),
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, otherPosition),
                () -> assertNotEquals(first, otherOccurrence));
    }

    private static TensorProducer producer() {
        Operation operation = new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE);
        Tensor input = tensor(1);
        return new TensorProducer(
                operation,
                List.of(input),
                List.of(descriptor(), descriptor()));
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

    private enum SampleKind implements OperationKind {
        SAMPLE;

        private static final List<OperationSignature> SIGNATURES =
                List.of(OperationSignature.fixed(NoOperationAttrs.class, 1, 2));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
