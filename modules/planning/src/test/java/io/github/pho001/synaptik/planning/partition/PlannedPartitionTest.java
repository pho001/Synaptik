package io.github.pho001.synaptik.planning.partition;

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
import io.github.pho001.synaptik.model.graph.NodeId;
import java.io.Serializable;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class PlannedPartitionTest {
    @Test
    void hasExactlyTheRequiredPublicTwoComponentRecordShape() throws ReflectiveOperationException {
        Class<PlannedPartition> type = PlannedPartition.class;
        var components = type.getRecordComponents();
        var instanceFields = Arrays.stream(type.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();
        Constructor<?>[] constructors = type.getDeclaredConstructors();
        Method owner = type.getDeclaredMethod("owner");
        Method nodeIds = type.getDeclaredMethod("nodeIds");
        ParameterizedType componentListType =
                (ParameterizedType) components[1].getGenericType();

        assertAll(
                () -> assertEquals(
                        "io.github.pho001.synaptik.planning.partition", type.getPackageName()),
                () -> assertTrue(Modifier.isPublic(type.getModifiers())),
                () -> assertTrue(Modifier.isFinal(type.getModifiers())),
                () -> assertTrue(type.isRecord()),
                () -> assertFalse(type.isEnum()),
                () -> assertEquals(0, type.getDeclaredClasses().length),
                () -> assertEquals(0, type.getInterfaces().length),
                () -> assertFalse(Serializable.class.isAssignableFrom(type)),
                () -> assertEquals(2, components.length),
                () -> assertEquals("owner", components[0].getName()),
                () -> assertEquals(BackendId.class, components[0].getType()),
                () -> assertEquals("nodeIds", components[1].getName()),
                () -> assertEquals(List.class, components[1].getType()),
                () -> assertEquals(List.class, componentListType.getRawType()),
                () -> assertArrayEquals(
                        new Type[] {NodeId.class}, componentListType.getActualTypeArguments()),
                () -> assertEquals(List.of("owner", "nodeIds"),
                        instanceFields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(instanceFields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertArrayEquals(
                        new Class<?>[] {BackendId.class, List.class},
                        constructors[0].getParameterTypes()),
                () -> assertEquals(5, type.getDeclaredMethods().length),
                () -> assertTrue(Modifier.isPublic(owner.getModifiers())),
                () -> assertEquals(BackendId.class, owner.getReturnType()),
                () -> assertTrue(Modifier.isPublic(nodeIds.getModifiers())),
                () -> assertEquals(List.class, nodeIds.getReturnType()));
    }

    @Test
    void validatesComponentsElementsAndDuplicatesInExactOrder() {
        BackendId cpu = new BackendId("cpu");
        List<NodeId> withNull = new ArrayList<>(Arrays.asList(new NodeId(1), null, null));

        NullPointerException nullOwner = assertThrows(
                NullPointerException.class, () -> new PlannedPartition(null, null));
        NullPointerException nullList = assertThrows(
                NullPointerException.class, () -> new PlannedPartition(cpu, null));
        IllegalArgumentException empty = assertThrows(
                IllegalArgumentException.class, () -> new PlannedPartition(cpu, List.of()));
        NullPointerException nullElement = assertThrows(
                NullPointerException.class, () -> new PlannedPartition(cpu, withNull));
        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class,
                () -> new PlannedPartition(
                        cpu,
                        List.of(
                                new NodeId(7),
                                new NodeId(8),
                                new NodeId(7),
                                new NodeId(8))));

        assertAll(
                () -> assertEquals("owner", nullOwner.getMessage()),
                () -> assertEquals("nodeIds", nullList.getMessage()),
                () -> assertEquals("nodeIds must not be empty", empty.getMessage()),
                () -> assertEquals("nodeIds[1]", nullElement.getMessage()),
                () -> assertEquals(
                        "nodeIds[2] duplicates NodeId[value=7]", duplicate.getMessage()));
    }

    @Test
    void snapshotsMembershipRetainsExactReferencesAndIsImmutable() {
        BackendId owner = new BackendId(new String("cpu"));
        NodeId first = new NodeId(4);
        NodeId second = new NodeId(5);
        List<NodeId> supplied = new ArrayList<>(List.of(first, second));

        PlannedPartition partition = new PlannedPartition(owner, supplied);
        supplied.clear();

        assertAll(
                () -> assertSame(owner, partition.owner()),
                () -> assertEquals(List.of(first, second), partition.nodeIds()),
                () -> assertSame(first, partition.nodeIds().get(0)),
                () -> assertSame(second, partition.nodeIds().get(1)),
                () -> assertNotSame(supplied, partition.nodeIds()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> partition.nodeIds().add(new NodeId(6))));
    }

    @Test
    void preservesOrdinaryRecordEqualityHashingAndDiagnosticText() {
        PlannedPartition first = new PlannedPartition(
                new BackendId("cpu"), List.of(new NodeId(1), new NodeId(2)));
        PlannedPartition equal = new PlannedPartition(
                new BackendId("cpu"), List.of(new NodeId(1), new NodeId(2)));
        PlannedPartition reordered = new PlannedPartition(
                new BackendId("cpu"), List.of(new NodeId(2), new NodeId(1)));
        String text = first.toString();

        assertAll(
                () -> assertEquals(first, equal),
                () -> assertEquals(first.hashCode(), equal.hashCode()),
                () -> assertNotEquals(first, reordered),
                () -> assertTrue(text.contains("PlannedPartition")),
                () -> assertTrue(text.contains("owner=")),
                () -> assertTrue(text.contains("nodeIds=")));
    }
}
