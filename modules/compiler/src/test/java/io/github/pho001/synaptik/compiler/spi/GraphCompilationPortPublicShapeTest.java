package io.github.pho001.synaptik.compiler.spi;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
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
    }
}
