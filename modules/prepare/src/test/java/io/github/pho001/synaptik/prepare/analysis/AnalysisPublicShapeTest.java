package io.github.pho001.synaptik.prepare.analysis;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class AnalysisPublicShapeTest {
    @Test
    void exposesTheExactMarkerRolesPartitionDagAndFiveComponentContextRecord() {
        Class<BackendAnalysisInputs> inputs = BackendAnalysisInputs.class;
        Class<BackendPreparationPlan> plan = BackendPreparationPlan.class;
        Class<PrepareContext> context = PrepareContext.class;
        Class<PartitionDag> dag = PartitionDag.class;
        var components = context.getRecordComponents();
        Type contextBound = context.getTypeParameters()[0].getBounds()[0];

        assertAll(
                () -> assertTrue(inputs.isInterface()),
                () -> assertTrue(Modifier.isPublic(inputs.getModifiers())),
                () -> assertFalse(inputs.isSealed()),
                () -> assertEquals(0, inputs.getDeclaredMethods().length),
                () -> assertEquals(0, inputs.getInterfaces().length),
                () -> assertTrue(plan.isInterface()),
                () -> assertTrue(Modifier.isPublic(plan.getModifiers())),
                () -> assertFalse(plan.isSealed()),
                () -> assertEquals(0, plan.getDeclaredMethods().length),
                () -> assertEquals(0, plan.getInterfaces().length),
                () -> assertEquals(
                        "io.github.pho001.synaptik.prepare.analysis", context.getPackageName()),
                () -> assertTrue(Modifier.isPublic(dag.getModifiers())),
                () -> assertTrue(Modifier.isFinal(dag.getModifiers())),
                () -> assertFalse(dag.isRecord()),
                () -> assertEquals(
                        List.of(
                                "consumers",
                                "edges",
                                "equals",
                                "externalInputs",
                                "hashCode",
                                "localSinks",
                                "node",
                                "nodes",
                                "partition",
                                "producer",
                                "toString"),
                        Arrays.stream(dag.getDeclaredMethods())
                                .filter(method -> Modifier.isPublic(method.getModifiers()))
                                .map(method -> method.getName())
                                .sorted()
                                .toList()),
                () -> assertTrue(PartitionDag.ProducerOccurrence.class.isRecord()),
                () -> assertTrue(Modifier.isPublic(
                        PartitionDag.ProducerOccurrence.class.getModifiers())),
                () -> assertTrue(PartitionDag.ConsumerOccurrence.class.isRecord()),
                () -> assertTrue(Modifier.isPublic(
                        PartitionDag.ConsumerOccurrence.class.getModifiers())),
                () -> assertTrue(PartitionDag.Edge.class.isRecord()),
                () -> assertTrue(Modifier.isPublic(PartitionDag.Edge.class.getModifiers())),
                () -> assertEquals(0, dag.getDeclaredFields().length
                        - Arrays.stream(dag.getDeclaredFields())
                                .filter(field -> Modifier.isPrivate(field.getModifiers())
                                        && Modifier.isFinal(field.getModifiers()))
                                .count()),
                () -> assertTrue(Modifier.isPublic(context.getModifiers())),
                () -> assertTrue(Modifier.isFinal(context.getModifiers())),
                () -> assertTrue(context.isRecord()),
                () -> assertEquals(2, context.getDeclaredConstructors().length),
                () -> assertTrue(Arrays.stream(context.getDeclaredConstructors()).anyMatch(
                        constructor -> Arrays.equals(
                                constructor.getParameterTypes(),
                                new Class<?>[] {
                                    PartitionDag.class,
                                    List.class,
                                    List.class,
                                    Map.class,
                                    BackendAnalysisInputs.class
                                }))),
                () -> assertTrue(Arrays.stream(context.getDeclaredConstructors()).anyMatch(
                        constructor -> Arrays.equals(
                                constructor.getParameterTypes(),
                                new Class<?>[] {
                                    PlannedPartition.class,
                                    List.class,
                                    List.class,
                                    List.class,
                                    Map.class,
                                    BackendAnalysisInputs.class
                                }))),
                () -> assertEquals(1, context.getTypeParameters().length),
                () -> assertEquals(BackendAnalysisInputs.class, contextBound),
                () -> assertEquals(
                        List.of(
                                "partitionDag",
                                "values",
                                "memoryRequirements",
                                "constants",
                                "backendInputs"),
                        Arrays.stream(components).map(component -> component.getName()).toList()),
                () -> assertArrayEquals(
                        new Class<?>[] {
                            PartitionDag.class,
                            List.class,
                            List.class,
                            Map.class,
                            BackendAnalysisInputs.class
                        },
                        Arrays.stream(components).map(component -> component.getType())
                                .toArray(Class<?>[]::new)),
                () -> assertListArgument(components[1].getGenericType(), GraphValue.class),
                () -> assertListArgument(
                        components[2].getGenericType(), LogicalMemoryRequirement.class),
                () -> assertMapArguments(
                        components[3].getGenericType(), ValueId.class, ScalarValue.class),
                () -> assertEquals(
                        context.getTypeParameters()[0], components[4].getGenericType()));
    }

    @Test
    void exposesTheExactSealedResourcesAnalysisAndTypedPreparerRelationship()
            throws ReflectiveOperationException {
        Class<PreparationResourceRequirement> requirement =
                PreparationResourceRequirement.class;
        Class<BackendPartitionAnalysis> analysis = BackendPartitionAnalysis.class;
        Class<BackendPartitionPreparer> preparer = BackendPartitionPreparer.class;
        var analysisComponents = analysis.getRecordComponents();
        var method = preparer.getDeclaredMethod("analyze", PrepareContext.class);
        ParameterizedType returnType = (ParameterizedType) method.getGenericReturnType();
        ParameterizedType parameterType =
                (ParameterizedType) method.getGenericParameterTypes()[0];

        assertAll(
                () -> assertTrue(requirement.isInterface()),
                () -> assertTrue(requirement.isSealed()),
                () -> assertEquals(
                        List.of(
                                PreparationResourceRequirement.Buffer.class,
                                PreparationResourceRequirement.Workspace.class),
                        Arrays.asList(requirement.getPermittedSubclasses())),
                () -> assertTrue(PreparationResourceRequirement.Buffer.class.isRecord()),
                () -> assertTrue(PreparationResourceRequirement.Workspace.class.isRecord()),
                () -> assertTrue(analysis.isRecord()),
                () -> assertTrue(Modifier.isPublic(analysis.getModifiers())),
                () -> assertEquals(1, analysis.getTypeParameters().length),
                () -> assertEquals(
                        BackendPreparationPlan.class,
                        analysis.getTypeParameters()[0].getBounds()[0]),
                () -> assertEquals(
                        List.of("partition", "plan", "requirements"),
                        Arrays.stream(analysisComponents)
                                .map(component -> component.getName())
                                .toList()),
                () -> assertEquals(PlannedPartition.class, analysisComponents[0].getType()),
                () -> assertEquals(
                        analysis.getTypeParameters()[0],
                        analysisComponents[1].getGenericType()),
                () -> assertListArgument(
                        analysisComponents[2].getGenericType(),
                        PreparationResourceRequirement.class),
                () -> assertTrue(preparer.isInterface()),
                () -> assertTrue(Modifier.isPublic(preparer.getModifiers())),
                () -> assertEquals(2, preparer.getTypeParameters().length),
                () -> assertEquals(
                        BackendAnalysisInputs.class,
                        preparer.getTypeParameters()[0].getBounds()[0]),
                () -> assertEquals(
                        BackendPreparationPlan.class,
                        preparer.getTypeParameters()[1].getBounds()[0]),
                () -> assertEquals(1, preparer.getDeclaredMethods().length),
                () -> assertTrue(Modifier.isPublic(method.getModifiers())),
                () -> assertTrue(Modifier.isAbstract(method.getModifiers())),
                () -> assertEquals(BackendPartitionAnalysis.class, returnType.getRawType()),
                () -> assertArrayEquals(
                        new Type[] {preparer.getTypeParameters()[1]},
                        returnType.getActualTypeArguments()),
                () -> assertEquals(PrepareContext.class, parameterType.getRawType()),
                () -> assertArrayEquals(
                        new Type[] {preparer.getTypeParameters()[0]},
                        parameterType.getActualTypeArguments()));
    }

    private static void assertListArgument(Type supplied, Type argument) {
        ParameterizedType type = (ParameterizedType) supplied;
        assertEquals(List.class, type.getRawType());
        assertArrayEquals(new Type[] {argument}, type.getActualTypeArguments());
    }

    private static void assertMapArguments(Type supplied, Type key, Type value) {
        ParameterizedType type = (ParameterizedType) supplied;
        assertEquals(Map.class, type.getRawType());
        assertArrayEquals(new Type[] {key, value}, type.getActualTypeArguments());
    }
}
