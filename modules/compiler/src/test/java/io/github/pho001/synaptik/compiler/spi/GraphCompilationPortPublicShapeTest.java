package io.github.pho001.synaptik.compiler.spi;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.CompileConstantPlan;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class GraphCompilationPortPublicShapeTest {
    @Test
    void invokesThePortFromAnotherPackageUsingOnlyPublicTypes() {
        Tensor output = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2), Optional.empty(), false));

        CompileArtifacts artifacts = GraphCompilationPort.compile(
                CompileMode.FORWARD_ONLY,
                List.of(output),
                Optional.empty(),
                GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                List.of(),
                List.of());

        assertTrue(artifacts.graph().nodes().isEmpty());
        assertEquals(List.of(artifacts.graph().inputs().getFirst()),
                artifacts.constants().bindableInputs());
        assertEquals(
                List.of(new CompileConstantPlan.BindableInput(
                        output.id(), artifacts.graph().inputs().getFirst())),
                artifacts.constants().bindableInputBindings());
    }

    @Test
    void preservesPublicAggregatePortAndTypedBindingShapes() throws Exception {
        assertTrue(CompileConstantPlan.BindableInput.class.isRecord());
        assertTrue(Modifier.isPublic(CompileConstantPlan.BindableInput.class.getModifiers()));
        assertEquals(List.of("tensorId", "valueId"), Arrays.stream(
                        CompileConstantPlan.BindableInput.class.getRecordComponents())
                .map(component -> component.getName()).toList());
        assertEquals(List.of(TensorId.class, ValueId.class), Arrays.stream(
                        CompileConstantPlan.BindableInput.class.getRecordComponents())
                .map(component -> component.getType()).toList());
        assertEquals(8, CompileArtifacts.class.getRecordComponents().length);
        assertEquals(1, Arrays.stream(GraphCompilationPort.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .count());
        assertEquals(List.class,
                CompileConstantPlan.class.getMethod("bindableInputs").getReturnType());
        assertEquals(List.class,
                CompileConstantPlan.class.getMethod("bindableInputBindings").getReturnType());
    }
}
