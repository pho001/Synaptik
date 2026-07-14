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

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
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

class LogicalMemoryPlanTest {
    @Test
    void hasExactlyTheRequiredPublicOneComponentRecordShape() throws ReflectiveOperationException {
        Class<LogicalMemoryPlan> type = LogicalMemoryPlan.class;
        var components = type.getRecordComponents();
        var instanceFields = Arrays.stream(type.getDeclaredFields())
                .filter(field -> !Modifier.isStatic(field.getModifiers()))
                .toList();
        Constructor<?>[] constructors = type.getDeclaredConstructors();
        Method requirements = type.getDeclaredMethod("requirements");
        ParameterizedType listType = (ParameterizedType) components[0].getGenericType();

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
                () -> assertEquals(1, components.length),
                () -> assertEquals("requirements", components[0].getName()),
                () -> assertEquals(List.class, components[0].getType()),
                () -> assertEquals(List.class, listType.getRawType()),
                () -> assertArrayEquals(
                        new Type[] {LogicalMemoryRequirement.class},
                        listType.getActualTypeArguments()),
                () -> assertEquals(List.of("requirements"),
                        instanceFields.stream().map(field -> field.getName()).toList()),
                () -> assertTrue(instanceFields.stream().allMatch(field ->
                        Modifier.isPrivate(field.getModifiers())
                                && Modifier.isFinal(field.getModifiers()))),
                () -> assertEquals(1, constructors.length),
                () -> assertTrue(Modifier.isPublic(constructors[0].getModifiers())),
                () -> assertArrayEquals(
                        new Class<?>[] {List.class}, constructors[0].getParameterTypes()),
                () -> assertEquals(4, type.getDeclaredMethods().length),
                () -> assertTrue(Modifier.isPublic(requirements.getModifiers())),
                () -> assertEquals(List.class, requirements.getReturnType()));
    }

    @Test
    void validatesListElementsAndDuplicateValueIdentitiesInExactOrder() {
        LogicalMemoryRequirement first = requirement(new ValueId(7));
        List<LogicalMemoryRequirement> withNull =
                new ArrayList<>(Arrays.asList(first, null, null));

        NullPointerException nullList = assertThrows(
                NullPointerException.class, () -> new LogicalMemoryPlan(null));
        NullPointerException nullElement = assertThrows(
                NullPointerException.class, () -> new LogicalMemoryPlan(withNull));
        IllegalArgumentException duplicate = assertThrows(
                IllegalArgumentException.class,
                () -> new LogicalMemoryPlan(List.of(
                        first, requirement(new ValueId(8)), requirement(new ValueId(7)))));

        assertAll(
                () -> assertEquals("requirements", nullList.getMessage()),
                () -> assertEquals("requirements[1]", nullElement.getMessage()),
                () -> assertEquals(
                        "requirements[2] duplicates ValueId[value=7]",
                        duplicate.getMessage()));
    }

    @Test
    void snapshotsMembershipRetainsExactReferencesAndAllowsEmptyStandalonePlan() {
        LogicalMemoryRequirement first = requirement(new ValueId(1));
        LogicalMemoryRequirement second = requirement(new ValueId(2));
        List<LogicalMemoryRequirement> supplied = new ArrayList<>(List.of(first, second));

        LogicalMemoryPlan plan = new LogicalMemoryPlan(supplied);
        supplied.clear();
        LogicalMemoryPlan empty = new LogicalMemoryPlan(List.of());

        assertAll(
                () -> assertEquals(List.of(first, second), plan.requirements()),
                () -> assertSame(first, plan.requirements().get(0)),
                () -> assertSame(second, plan.requirements().get(1)),
                () -> assertNotSame(supplied, plan.requirements()),
                () -> assertTrue(empty.requirements().isEmpty()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> plan.requirements().add(requirement(new ValueId(3)))));
    }

    @Test
    void preservesOrdinaryOrderedRecordValueBehaviorAndDiagnosticText() {
        LogicalMemoryRequirement first = requirement(new ValueId(1));
        LogicalMemoryRequirement second = requirement(new ValueId(2));
        LogicalMemoryPlan plan = new LogicalMemoryPlan(List.of(first, second));
        LogicalMemoryPlan equal = new LogicalMemoryPlan(List.of(
                requirement(new ValueId(1)), requirement(new ValueId(2))));
        LogicalMemoryPlan reordered = new LogicalMemoryPlan(List.of(second, first));
        String text = plan.toString();

        assertAll(
                () -> assertEquals(plan, equal),
                () -> assertEquals(plan.hashCode(), equal.hashCode()),
                () -> assertNotEquals(plan, reordered),
                () -> assertTrue(text.contains("LogicalMemoryPlan")),
                () -> assertTrue(text.contains("requirements=")));
    }

    private static LogicalMemoryRequirement requirement(ValueId valueId) {
        return new LogicalMemoryRequirement(
                valueId, descriptor(), Optional.empty(), List.of(), false);
    }

    private static TensorDescriptor descriptor() {
        return new TensorDescriptor(DataType.FLOAT32, Shape.scalar(), Optional.empty(), false);
    }
}
