package io.github.pho001.synaptik.planning.memory;

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
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class LogicalMemoryRequirementTest {
    @Test
    void hasExactlyTheRequiredPublicFiveComponentRecordShape() throws ReflectiveOperationException {
        Class<LogicalMemoryRequirement> type = LogicalMemoryRequirement.class;
        var components = type.getRecordComponents();
        var instanceFields = Arrays.stream(type.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();
        Constructor<?>[] constructors = type.getDeclaredConstructors();
        Method valueId = type.getDeclaredMethod("valueId");
        Method descriptor = type.getDeclaredMethod("descriptor");
        Method producerPartition = type.getDeclaredMethod("producerPartition");
        Method consumerPartitions = type.getDeclaredMethod("consumerPartitions");
        Method graphOutput = type.getDeclaredMethod("graphOutput");
        ParameterizedType producerType = (ParameterizedType) components[2].getGenericType();
        ParameterizedType consumersType = (ParameterizedType) components[3].getGenericType();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.planning.memory", type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () -> assertFalse(type.isEnum()),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(5, components.length),
                () -> assertEquals(
                        List.of(
                                "valueId",
                                "descriptor",
                                "producerPartition",
                                "consumerPartitions",
                                "graphOutput"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertArrayEquals(
                        new Class<?>[] {
                            ValueId.class,
                            TensorDescriptor.class,
                            Optional.class,
                            List.class,
                            boolean.class
                        },
                        Arrays.stream(components).map(component -> component.getType())
                                .toArray(Class<?>[]::new)),
                () -> assertEquals(Optional.class, producerType.getRawType()),
                () -> assertArrayEquals(
                        new Type[] {PlannedPartition.class}, producerType.getActualTypeArguments()),
                () -> assertEquals(List.class, consumersType.getRawType()),
                () -> assertArrayEquals(
                        new Type[] {PlannedPartition.class}, consumersType.getActualTypeArguments()),
                () -> assertEquals(
                        List.of(
                                "valueId",
                                "descriptor",
                                "producerPartition",
                                "consumerPartitions",
                                "graphOutput"),
                        instanceFields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(instanceFields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertArrayEquals(
                        new Class<?>[] {
                            ValueId.class,
                            TensorDescriptor.class,
                            Optional.class,
                            List.class,
                            boolean.class
                        },
                        constructors[0].getParameterTypes()),
                () -> assertEquals(8, type.getDeclaredMethods().length),
                () -> assertEquals(ValueId.class, valueId.getReturnType()),
                () -> assertEquals(TensorDescriptor.class, descriptor.getReturnType()),
                () -> assertEquals(Optional.class, producerPartition.getReturnType()),
                () -> assertEquals(List.class, consumerPartitions.getReturnType()),
                () -> assertEquals(boolean.class, graphOutput.getReturnType()),
                () -> assertTrue(List.of(
                                valueId,
                                descriptor,
                                producerPartition,
                                consumerPartitions,
                                graphOutput)
                        .stream()
                        .allMatch(method -> Modifier.isPublic(method.getModifiers()))));
    }

    @Test
    void validatesComponentsConsumersAndDuplicatesInExactOrder() {
        ValueId valueId = new ValueId(7);
        TensorDescriptor descriptor = descriptor();
        PlannedPartition first = partition("cpu", 1);
        PlannedPartition second = partition("metal", 2);
        List<PlannedPartition> withNull = new ArrayList<>(Arrays.asList(first, null, null));

        NullPointerException nullValueId = assertThrows(
                NullPointerException.class,
                () -> new LogicalMemoryRequirement(null, null, null, null, false));
        NullPointerException nullDescriptor = assertThrows(
                NullPointerException.class,
                () -> new LogicalMemoryRequirement(valueId, null, null, null, false));
        NullPointerException nullProducer = assertThrows(
                NullPointerException.class,
                () -> new LogicalMemoryRequirement(valueId, descriptor, null, null, false));
        NullPointerException nullConsumers = assertThrows(
                NullPointerException.class,
                () -> new LogicalMemoryRequirement(
                        valueId, descriptor, Optional.empty(), null, false));
        NullPointerException nullConsumer = assertThrows(
                NullPointerException.class,
                () -> new LogicalMemoryRequirement(
                        valueId, descriptor, Optional.empty(), withNull, false));
        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class,
                () -> new LogicalMemoryRequirement(
                        valueId,
                        descriptor,
                        Optional.of(first),
                        List.of(first, second, partition("cpu", 1), second),
                        true));

        assertAll(
                () -> assertEquals("valueId", nullValueId.getMessage()),
                () -> assertEquals("descriptor", nullDescriptor.getMessage()),
                () -> assertEquals("producerPartition", nullProducer.getMessage()),
                () -> assertEquals("consumerPartitions", nullConsumers.getMessage()),
                () -> assertEquals("consumerPartitions[1]", nullConsumer.getMessage()),
                () -> assertEquals(
                        "consumerPartitions[2] duplicates " + first, duplicate.getMessage()));
    }

    @Test
    void snapshotsConsumersAndRetainsExactGraphAndPartitionReferences() {
        ValueId valueId = new ValueId(4);
        TensorDescriptor descriptor = descriptor();
        PlannedPartition producer = partition("cpu", 1);
        PlannedPartition consumer = partition("metal", 2);
        List<PlannedPartition> supplied = new ArrayList<>(List.of(producer, consumer));

        LogicalMemoryRequirement requirement = new LogicalMemoryRequirement(
                valueId, descriptor, Optional.of(producer), supplied, true);
        supplied.clear();

        assertAll(
                () -> assertSame(valueId, requirement.valueId()),
                () -> assertSame(descriptor, requirement.descriptor()),
                () -> assertSame(producer, requirement.producerPartition().orElseThrow()),
                () -> assertEquals(List.of(producer, consumer), requirement.consumerPartitions()),
                () -> assertSame(producer, requirement.consumerPartitions().get(0)),
                () -> assertSame(consumer, requirement.consumerPartitions().get(1)),
                () -> assertNotSame(supplied, requirement.consumerPartitions()),
                () -> assertTrue(requirement.graphOutput()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> requirement.consumerPartitions().add(partition("cpu", 3))));
    }

    @Test
    void permitsEmptyRelationshipsAndPreservesOrdinaryRecordValueBehavior() {
        LogicalMemoryRequirement first = new LogicalMemoryRequirement(
                new ValueId(1), descriptor(), Optional.empty(), List.of(), false);
        LogicalMemoryRequirement equal = new LogicalMemoryRequirement(
                new ValueId(1), descriptor(), Optional.empty(), List.of(), false);
        LogicalMemoryRequirement output = new LogicalMemoryRequirement(
                new ValueId(1), descriptor(), Optional.empty(), List.of(), true);
        String text = first.toString();

        assertAll(
                () -> assertTrue(first.producerPartition().isEmpty()),
                () -> assertTrue(first.consumerPartitions().isEmpty()),
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, output),
                () -> assertTrue(text.contains("LogicalMemoryRequirement")),
                () -> assertTrue(text.contains("valueId=")),
                () -> assertTrue(text.contains("descriptor=")),
                () -> assertTrue(text.contains("producerPartition=")),
                () -> assertTrue(text.contains("consumerPartitions=")),
                () -> assertTrue(text.contains("graphOutput=")));
    }

    private static PlannedPartition partition(String owner, long nodeId) {
        return new PlannedPartition(new BackendId(owner), List.of(new NodeId(nodeId)));
    }

    private static TensorDescriptor descriptor() {
        return new TensorDescriptor(DataType.FLOAT32, Shape.scalar(), Optional.empty(), false);
    }
}
