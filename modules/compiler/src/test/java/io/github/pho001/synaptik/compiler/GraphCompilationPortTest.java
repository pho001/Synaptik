package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class GraphCompilationPortTest {
    @Test
    void exposesOnlyTheSpecifiedPublicCompileShapeAndPreservesCompilerPrivacy()
            throws Exception {
        assertTrue(Modifier.isPublic(GraphCompilationPort.class.getModifiers()));
        assertTrue(Modifier.isFinal(GraphCompilationPort.class.getModifiers()));
        assertEquals(0, GraphCompilationPort.class.getDeclaredConstructors()[0].getParameterCount());
        assertTrue(Modifier.isPrivate(
                GraphCompilationPort.class.getDeclaredConstructors()[0].getModifiers()));

        var publicMethods = Arrays.stream(GraphCompilationPort.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers()))
                .toList();
        assertEquals(1, publicMethods.size());
        var compile = publicMethods.getFirst();
        assertEquals("compile", compile.getName());
        assertTrue(Modifier.isStatic(compile.getModifiers()));
        assertSame(CompileArtifacts.class, compile.getReturnType());
        assertEquals(
                List.of(
                        CompileMode.class,
                        List.class,
                        Optional.class,
                        GraphOptimizationConfig.class,
                        BackendIntent.class,
                        PartitionScoringConfig.class,
                        List.class,
                        List.class),
                List.of(compile.getParameterTypes()));

        assertFalse(Modifier.isPublic(GraphCompiler.class.getModifiers()));
        var graphEntry = GraphCompiler.class.getDeclaredMethod(
                "compile",
                CompileMode.class,
                List.class,
                Optional.class,
                CompileTimeConstantGraph.Ingress.class,
                GraphOptimizationConfig.class);
        var completeEntry = GraphCompiler.class.getDeclaredMethod(
                "compile",
                CompileMode.class,
                List.class,
                Optional.class,
                CompileTimeConstantGraph.Ingress.class,
                GraphOptimizationConfig.class,
                BackendIntent.class,
                PartitionScoringConfig.class,
                List.class,
                List.class);
        assertFalse(Modifier.isPublic(graphEntry.getModifiers()));
        assertFalse(Modifier.isPublic(completeEntry.getModifiers()));
        assertEquals(2, Arrays.stream(GraphCompiler.class.getDeclaredMethods())
                .filter(method -> method.getName().equals("compile"))
                .count());
    }

    @Test
    void delegatesForwardCompilationWithEmptyConstantIngressWithoutChangingArtifacts() {
        Tensor input = tensor(true);
        Tensor output = input.neg();
        BackendId backendId = new BackendId("cpu");
        BackendCapabilityProvider provider = provider(backendId);
        BackendAvailabilitySnapshot snapshot = snapshot(backendId);

        CompileArtifacts direct = GraphCompiler.compile(
                CompileMode.FORWARD_ONLY,
                List.of(output),
                Optional.empty(),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.standard(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                List.of(provider),
                List.of(snapshot));
        CompileArtifacts throughPort = GraphCompilationPort.compile(
                CompileMode.FORWARD_ONLY,
                List.of(output),
                Optional.empty(),
                GraphOptimizationConfig.standard(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                List.of(provider),
                List.of(snapshot));

        assertArtifactEquivalent(direct, throughPort);
        assertEquals(List.of(throughPort.graph().inputs().getFirst()),
                throughPort.constants().bindableInputs());
        assertTrue(throughPort.constants().constantSources().isEmpty());
    }

    @Test
    void delegatesBackwardRequestAndPreservesProviderFailureIdentity() {
        Tensor target = tensor(true);
        Tensor objective = target.mul(target).sum();
        BackendId backendId = new BackendId("cpu");
        FunctionalGradientRequest request = FunctionalGradientTestSupport.request(
                objective, List.of(target));

        CompileArtifacts direct = GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(request),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                List.of(provider(backendId)),
                List.of(snapshot(backendId)));
        CompileArtifacts throughPort = GraphCompilationPort.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(request),
                GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                List.of(provider(backendId)),
                List.of(snapshot(backendId)));
        assertArtifactEquivalent(direct, throughPort);

        RuntimeException providerFailure = new RuntimeException("provider failure");
        BackendCapabilityProvider failingProvider = new BackendCapabilityProvider() {
            @Override
            public BackendId backendId() {
                return backendId;
            }

            @Override
            public boolean supports(
                    io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery query) {
                throw providerFailure;
            }
        };
        RuntimeException propagated = assertThrows(
                RuntimeException.class,
                () -> GraphCompilationPort.compile(
                        CompileMode.FORWARD_ONLY,
                        List.of(tensor(true).neg()),
                        Optional.empty(),
                        GraphOptimizationConfig.disabled(),
                        BackendIntent.unconstrained(),
                        PartitionScoringConfig.neutral(),
                        List.of(failingProvider),
                        List.of(snapshot(backendId))));
        assertSame(providerFailure, propagated);
    }

    @Test
    void preservesDeclarationOrderValidationFromTheCompleteEntry() {
        NullPointerException failure = assertThrows(
                NullPointerException.class,
                () -> GraphCompilationPort.compile(
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null,
                        null));

        assertEquals("mode", failure.getMessage());
    }

    private static Tensor tensor(boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2), Optional.empty(), requiresGrad));
    }

    private static void assertArtifactEquivalent(
            CompileArtifacts expected, CompileArtifacts actual) {
        assertEquals(expected.mode(), actual.mode());
        assertEquals(expected.graph(), actual.graph());
        assertEquals(expected.partitions(), actual.partitions());
        assertEquals(expected.memory(), actual.memory());
        assertEquals(
                expected.publication().forwardBindings(),
                actual.publication().forwardBindings());
        assertEquals(
                expected.publication().gradientBindings(),
                actual.publication().gradientBindings());
        assertEquals(
                expected.constants().bindableInputs(),
                actual.constants().bindableInputs());
        assertEquals(
                expected.constants().constantSources(),
                actual.constants().constantSources());
        assertEquals(
                expected.diagnostics().deferredConstraints(),
                actual.diagnostics().deferredConstraints());
        assertEquals(
                expected.derivatives().derivativeOrderByNode(),
                actual.derivatives().derivativeOrderByNode());
    }

    private static BackendCapabilityProvider provider(BackendId backendId) {
        return new BackendCapabilityProvider() {
            @Override
            public BackendId backendId() {
                return backendId;
            }

            @Override
            public boolean supports(
                    io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery query) {
                return true;
            }
        };
    }

    private static BackendAvailabilitySnapshot snapshot(BackendId backendId) {
        BackendDeviceId deviceId = new BackendDeviceId(backendId, "0");
        return new BackendAvailabilitySnapshot(
                backendId, Map.of(deviceId, DeviceClass.CPU));
    }
}
