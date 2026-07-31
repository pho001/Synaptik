package io.github.pho001.synaptik.prepare;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class FinalizationPublicShapeTest {
    @Test
    void exposesOnlyTheFourPlannedPublicRootTypes() {
        assertAll(
                () -> assertPublicSealedInterface(PreparationResourceAssignment.class),
                () -> assertPublicRecord(BackendPartitionFinalization.class),
                () -> assertPublicInterface(BackendPartitionFinalizer.class),
                () -> assertPublicRecord(PreparedPartition.class),
                () -> assertFalse(Modifier.isPublic(
                        BackendPartitionFinalizationHandoff.class.getModifiers())),
                () -> assertEquals(
                        BackendPreparationPlan.class,
                        BackendPartitionFinalization.class
                                .getTypeParameters()[0]
                                .getBounds()[0]),
                () -> assertEquals(
                        BackendPreparationPlan.class,
                        BackendPartitionFinalizer.class
                                .getTypeParameters()[0]
                                .getBounds()[0]));
    }

    @Test
    void locksAssignmentAndInternalHandoffRecordShapes() {
        assertAll(
                () -> assertEquals(
                        java.util.Set.of(
                                PreparationResourceAssignment.Buffer.class,
                                PreparationResourceAssignment.Workspace.class),
                        java.util.Set.of(PreparationResourceAssignment.class.getPermittedSubclasses())),
                () -> assertRecordComponents(
                        PreparationResourceAssignment.Buffer.class,
                        "requirement", "slot", "planIndex"),
                () -> assertRecordComponents(
                        PreparationResourceAssignment.Workspace.class,
                        "requirement", "slot", "planIndex"),
                () -> assertRecordComponents(
                        BackendPartitionFinalization.class,
                        "analysis", "memoryPlan", "assignments"),
                () -> assertRecordComponents(
                        PreparedPartition.class, "partition", "executable"),
                () -> assertRecordComponents(
                        BackendPartitionFinalizationHandoff.Entry.class,
                        "context", "analysis", "finalizer"),
                () -> assertRecordComponents(
                        BackendPartitionFinalizationHandoff.Result.class,
                        "memoryPlan", "partitions"),
                () -> assertTrue(Arrays.stream(
                                BackendPartitionFinalizationHandoff.class.getDeclaredMethods())
                        .anyMatch(method -> method.getName().equals("finalizePartitions")
                                && Modifier.isStatic(method.getModifiers())
                                && !Modifier.isPublic(method.getModifiers()))),
                () -> assertTrue(BackendPartitionFinalization.class
                                .getRecordComponents()[2]
                                .getGenericType()
                        instanceof ParameterizedType));
    }

    private static void assertPublicRecord(Class<?> type) {
        assertTrue(Modifier.isPublic(type.getModifiers()));
        assertTrue(type.isRecord());
    }

    private static void assertPublicInterface(Class<?> type) {
        assertTrue(Modifier.isPublic(type.getModifiers()));
        assertTrue(type.isInterface());
    }

    private static void assertPublicSealedInterface(Class<?> type) {
        assertPublicInterface(type);
        assertTrue(type.isSealed());
    }

    private static void assertRecordComponents(Class<?> type, String... names) {
        assertTrue(type.isRecord());
        assertEquals(
                java.util.List.of(names),
                Arrays.stream(type.getRecordComponents()).map(component -> component.getName()).toList());
    }
}
