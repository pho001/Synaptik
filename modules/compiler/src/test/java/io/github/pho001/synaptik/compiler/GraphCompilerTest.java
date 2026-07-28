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
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

final class GraphCompilerTest {
    @Test
    void exposesTheExactPackagePrivateDirectCompileAndResultShapes() throws Exception {
        var compile = GraphCompiler.class.getDeclaredMethod(
                "compile",
                CompileMode.class,
                List.class,
                Optional.class,
                CompileTimeConstantGraph.Ingress.class,
                GraphOptimizationConfig.class);

        assertTrue(Modifier.isStatic(compile.getModifiers()));
        assertFalse(Modifier.isPublic(compile.getModifiers()));
        assertSame(GraphCompilation.class, compile.getReturnType());
        assertEquals(
                List.of("mode", "validatedGraph", "forwardOutputs", "gradientResults"),
                java.util.Arrays.stream(GraphCompilation.class.getRecordComponents())
                        .map(component -> component.getName())
                        .toList());
        assertTrue(GraphCompilation.class.isRecord());
        assertTrue(GraphCompilation.GradientResultRole.class.isRecord());
        assertFalse(Modifier.isPublic(GraphCompilation.class.getModifiers()));

        var completeCompile = GraphCompiler.class.getDeclaredMethod(
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
        assertTrue(Modifier.isStatic(completeCompile.getModifiers()));
        assertFalse(Modifier.isPublic(completeCompile.getModifiers()));
        assertSame(CompileArtifacts.class, completeCompile.getReturnType());
        assertEquals(2, Arrays.stream(GraphCompiler.class.getDeclaredMethods())
                .filter(method -> method.getName().equals("compile"))
                .count());
    }

    @Test
    void completeCompilePlansEachFinalNodeAndReturnsCrossValidatedArtifacts() {
        Tensor input = tensor();
        Tensor output = input.neg();
        BackendId backendId = new BackendId("cpu");
        List<OperationCapabilityQuery> queries = new ArrayList<>();
        BackendCapabilityProvider provider = provider(backendId, queries, true);

        CompileArtifacts artifacts = GraphCompiler.compile(
                CompileMode.FORWARD_ONLY,
                List.of(output),
                Optional.empty(),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                List.of(provider),
                List.of(snapshot(backendId)));

        assertEquals(artifacts.graph().nodes().size(), queries.size());
        assertEquals(1, artifacts.partitions().size());
        assertSame(backendId, artifacts.partitions().getFirst().owner());
        assertSame(artifacts.graph(), artifacts.publication().graph());
        assertEquals(output.id(), artifacts.publication()
                .forwardOutputs().getFirst().tensorId());
        assertTrue(artifacts.publication().gradientResults().isEmpty());
        assertEquals(artifacts.graph().values().size(), artifacts.memory().requirements().size());
        assertTrue(artifacts.diagnostics().deferredConstraints().isEmpty());

        Map<ValueId, GraphValue> values = artifacts.graph().values().stream()
                .collect(java.util.stream.Collectors.toMap(GraphValue::id, value -> value));
        for (int nodeIndex = 0; nodeIndex < artifacts.graph().nodes().size(); nodeIndex++) {
            var node = artifacts.graph().nodes().get(nodeIndex);
            OperationCapabilityQuery query = queries.get(nodeIndex);
            assertSame(node.operation(), query.operation());
            for (int inputIndex = 0; inputIndex < node.inputs().size(); inputIndex++) {
                assertSame(
                        values.get(node.inputs().get(inputIndex)).descriptor(),
                        query.inputs().get(inputIndex));
            }
            for (int outputIndex = 0; outputIndex < node.outputs().size(); outputIndex++) {
                assertSame(
                        values.get(node.outputs().get(outputIndex)).descriptor(),
                        query.outputs().get(outputIndex));
            }
        }
    }

    @Test
    void completeCompileSkipsUnusedProviderElementsForZeroNodePassThroughGraph() {
        Tensor output = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2), Optional.empty(), false));
        List<BackendCapabilityProvider> providers = new ArrayList<>();
        providers.add(null);
        List<BackendAvailabilitySnapshot> snapshots = new ArrayList<>();
        snapshots.add(null);

        CompileArtifacts artifacts = GraphCompiler.compile(
                CompileMode.FORWARD_ONLY,
                List.of(output),
                Optional.empty(),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                providers,
                snapshots);

        assertTrue(artifacts.graph().nodes().isEmpty());
        assertTrue(artifacts.partitions().isEmpty());
        assertEquals(1, artifacts.memory().requirements().size());
        assertEquals(1, artifacts.constants().bindableInputs().size());
    }

    @Test
    void completeCompileValidatesPlanningInputsBeforeGraphConstruction() {
        Tensor output = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2), Optional.empty(), false));
        List<BackendCapabilityProvider> providers = new ArrayList<>();
        providers.add(null);
        List<BackendAvailabilitySnapshot> snapshots = new ArrayList<>();
        snapshots.add(null);

        assertEquals(
                "backendIntent",
                assertThrows(
                        NullPointerException.class,
                        () -> GraphCompiler.compile(
                                CompileMode.FORWARD_ONLY,
                                List.of(output),
                                Optional.empty(),
                                CompileTimeConstantGraph.Ingress.empty(),
                                GraphOptimizationConfig.disabled(),
                                null,
                                null,
                                providers,
                                snapshots))
                        .getMessage());
        assertEquals(
                "partitionScoringConfig",
                assertThrows(
                        NullPointerException.class,
                        () -> GraphCompiler.compile(
                                CompileMode.FORWARD_ONLY,
                                List.of(output),
                                Optional.empty(),
                                CompileTimeConstantGraph.Ingress.empty(),
                                GraphOptimizationConfig.disabled(),
                                BackendIntent.unconstrained(),
                                null,
                                providers,
                                snapshots))
                        .getMessage());
        assertEquals(
                "capabilityProviders",
                assertThrows(
                        NullPointerException.class,
                        () -> GraphCompiler.compile(
                                CompileMode.FORWARD_ONLY,
                                List.of(output),
                                Optional.empty(),
                                CompileTimeConstantGraph.Ingress.empty(),
                                GraphOptimizationConfig.disabled(),
                                BackendIntent.unconstrained(),
                                PartitionScoringConfig.neutral(),
                                null,
                                snapshots))
                        .getMessage());
        assertEquals(
                "availabilitySnapshots",
                assertThrows(
                        NullPointerException.class,
                        () -> GraphCompiler.compile(
                                CompileMode.FORWARD_ONLY,
                                List.of(output),
                                Optional.empty(),
                                CompileTimeConstantGraph.Ingress.empty(),
                                GraphOptimizationConfig.disabled(),
                                BackendIntent.unconstrained(),
                                PartitionScoringConfig.neutral(),
                                providers,
                                null))
                        .getMessage());
    }

    @Test
    void completeCompileTransportsExactLogicalConstantAndBackwardPublicationRoles() {
        Tensor constant = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2), Optional.empty(), false));
        ScalarValue value = ScalarValue.float32(-0.0f);
        CompileArtifacts constantArtifacts = GraphCompiler.compile(
                CompileMode.FORWARD_ONLY,
                List.of(constant),
                Optional.empty(),
                new CompileTimeConstantGraph.Ingress(List.of(
                        new CompileTimeConstantGraph.Binding(
                                constant, new CompileTimeConstantGraph.Splat(value)))),
                GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                List.of(),
                List.of());

        assertTrue(constantArtifacts.constants().bindableInputs().isEmpty());
        assertSame(
                value,
                constantArtifacts.constants().constantSources().getFirst().value());
        assertSame(
                constantArtifacts.graph().inputs().getFirst(),
                constantArtifacts.constants().constantSources().getFirst().valueId());

        Tensor target = tensor();
        Tensor objective = target.mul(target).sum();
        BackendId backendId = new BackendId("cpu");
        CompileArtifacts backwardArtifacts = GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(new AutogradPreflight.FirstOrderRequest(
                        objective, List.of(target))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.standard(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                List.of(provider(backendId, new ArrayList<>(), true)),
                List.of(snapshot(backendId)));

        assertEquals(1, backwardArtifacts.publication().gradientResults().size());
        assertEquals(
                target.id(),
                backwardArtifacts.publication().gradientResults().getFirst().tensorId());
        assertTrue(backwardArtifacts.graph().nodePhases().containsValue(GraphPhase.BACKWARD));
    }

    @Test
    void completeCompileAddsNodeContextOnlyToTerminalNoEligibleFailure() {
        Tensor output = tensor().neg();
        BackendId backendId = new BackendId("cpu");

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> GraphCompiler.compile(
                        CompileMode.FORWARD_ONLY,
                        List.of(output),
                        Optional.empty(),
                        CompileTimeConstantGraph.Ingress.empty(),
                        GraphOptimizationConfig.disabled(),
                        BackendIntent.unconstrained(),
                        PartitionScoringConfig.neutral(),
                        List.of(provider(backendId, new ArrayList<>(), false)),
                        List.of(snapshot(backendId))));

        var node = GraphCompiler.compile(
                        CompileMode.FORWARD_ONLY,
                        List.of(output),
                        Optional.empty(),
                        CompileTimeConstantGraph.Ingress.empty(),
                        GraphOptimizationConfig.disabled())
                .validatedGraph().graph().nodes().getFirst();
        assertEquals(
                "nodes[0] " + node.id() + " "
                        + node.operation().kind().getClass().getName() + "."
                        + node.operation().kind().name()
                        + ": no hard-eligible backend is available for ownership selection",
                failure.getMessage());
        assertEquals(
                "no hard-eligible backend is available for ownership selection",
                failure.getCause().getMessage());
    }

    @Test
    void compilesForwardOnlyWithoutBackwardState() {
        Tensor input = tensor();
        Tensor output = input.neg();

        GraphCompilation result = GraphCompiler.compile(
                CompileMode.FORWARD_ONLY,
                List.of(output),
                Optional.empty(),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled());

        assertEquals(result.validatedGraph().graph().outputs(), result.forwardOutputs());
        assertTrue(result.gradientResults().isEmpty());
        assertFalse(result.validatedGraph().graph().nodePhases()
                .containsValue(GraphPhase.BACKWARD));
    }

    @Test
    void compilesBackwardAndTrainingModesAsOnePhaseAwareGraph() {
        for (CompileMode mode :
                List.of(CompileMode.FORWARD_AND_BACKWARD, CompileMode.TRAINING_STEP)) {
            Tensor target = tensor();
            Tensor objective = target.mul(target).sum();
            GraphCompilation result = GraphCompiler.compile(
                    mode,
                    List.of(objective),
                    Optional.of(new AutogradPreflight.FirstOrderRequest(
                            objective, List.of(target))),
                    CompileTimeConstantGraph.Ingress.empty(),
                    GraphOptimizationConfig.standard());

            assertEquals(1, result.forwardOutputs().size());
            assertEquals(1, result.gradientResults().size());
            assertEquals(target.id(), result.gradientResults().getFirst().target());
            assertTrue(result.validatedGraph().graph().nodePhases()
                    .containsValue(GraphPhase.BACKWARD));
            assertTrue(result.validatedGraph().constants().size() >= 1);
        }
    }

    @Test
    void enforcesModeRequestPresenceBeforeExpansion() {
        Tensor output = tensor().sum();
        var request = new AutogradPreflight.FirstOrderRequest(output, List.of(output));
        assertThrows(IllegalArgumentException.class, () -> GraphCompiler.compile(
                CompileMode.FORWARD_ONLY,
                List.of(output),
                Optional.of(request),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled()));
        assertThrows(IllegalArgumentException.class, () -> GraphCompiler.compile(
                CompileMode.TRAINING_STEP,
                List.of(output),
                Optional.empty(),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled()));
    }

    @Test
    void stableDeduplicatesEqualGradientBoundaryValuesButPreservesEveryTargetRole() {
        Tensor target = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), true));
        Tensor objective = target.add(ScalarValue.float32(2.0f));

        GraphCompilation result = GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(new AutogradPreflight.FirstOrderRequest(
                        objective, List.of(target, objective))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled());

        assertEquals(2, result.gradientResults().size());
        assertEquals(
                result.gradientResults().get(0).gradient(),
                result.gradientResults().get(1).gradient());
        assertEquals(2, result.validatedGraph().graph().outputs().size());
    }

    @Test
    void knownUnsupportedPreflightFailureConsumesNoTensorIdentity() throws Exception {
        Tensor target = tensor();
        Tensor other = tensor();
        Tensor objective = target.softmax(0).mul(other).sum();
        var request = new AutogradPreflight.FirstOrderRequest(objective, List.of(target));
        long before = nextTensorId();

        assertThrows(IllegalArgumentException.class, () -> GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(request),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled()));

        assertEquals(before, nextTensorId());
    }

    @Test
    void compilesSharedAlgebraDivisionMeanAndDirectZeroInBothOptimizationModes() {
        for (GraphOptimizationConfig optimization :
                List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
            Tensor narrow = TensorFactory.create(new TensorDescriptor(
                    DataType.BFLOAT16, Shape.of(2, 1), Optional.empty(), true));
            Tensor wide = TensorFactory.create(new TensorDescriptor(
                    DataType.FLOAT64, Shape.of(2, 3), Optional.empty(), true));
            Tensor objective = narrow.div(wide).floor().mean();

            GraphCompilation result = GraphCompiler.compile(
                    CompileMode.FORWARD_AND_BACKWARD,
                    List.of(objective),
                    Optional.of(new AutogradPreflight.FirstOrderRequest(
                            objective, List.of(narrow))),
                    CompileTimeConstantGraph.Ingress.empty(),
                    optimization);

            assertEquals(narrow.id(), result.gradientResults().getFirst().target());
            assertTrue(result.validatedGraph().graph().nodePhases()
                    .containsValue(GraphPhase.BACKWARD));
        }
    }

    @Test
    void preflightOnly0004AFailuresConsumeNoTensorIdentity() throws Exception {
        DynamicDimension dynamic = new DynamicDimension("N");
        Tensor source = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(dynamic),
                Optional.empty(),
                true));
        Tensor bindingDependent = source
                .sumToShape(Shape.ofDimensions(new DynamicDimension("M")))
                .sum();
        long beforeShapeFailure = nextTensorId();
        assertThrows(IllegalArgumentException.class, () -> GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(bindingDependent),
                Optional.of(new AutogradPreflight.FirstOrderRequest(
                        bindingDependent, List.of(source))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled()));
        assertEquals(beforeShapeFailure, nextTensorId());

        Tensor base = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(dynamic),
                Optional.empty(),
                true));
        Tensor update = TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(2)),
                Optional.empty(),
                true));
        Tensor replacement = base.sliceUpdate(
                        update,
                        new long[] {0},
                        new int[] {0},
                        new long[] {1})
                .sum();
        long beforeRoleFailure = nextTensorId();
        assertThrows(IllegalArgumentException.class, () -> GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(replacement),
                Optional.of(new AutogradPreflight.FirstOrderRequest(
                        replacement, List.of(update))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled()));
        assertEquals(beforeRoleFailure, nextTensorId());
    }

    @Test
    void capturesRepresentative0004AFormulasOnceWithPhasesRolesAndConstants() {
        for (GraphOptimizationConfig optimization :
                List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
            Tensor target = TensorFactory.create(new TensorDescriptor(
                    DataType.FLOAT32, Shape.of(2, 2), Optional.empty(), true));
            Tensor objective = target.erf()
                    .matmul(target)
                    .slice(
                            new long[] {0},
                            new long[] {2},
                            new int[] {0},
                            new long[] {1})
                    .pad(
                            new long[] {0, 1},
                            new long[] {0, 0},
                            ScalarValue.float32(0.0f))
                    .sum();

            GraphCompilation result = GraphCompiler.compile(
                    CompileMode.FORWARD_AND_BACKWARD,
                    List.of(objective),
                    Optional.of(new AutogradPreflight.FirstOrderRequest(
                            objective, List.of(target))),
                    CompileTimeConstantGraph.Ingress.empty(),
                    optimization);

            assertEquals(1, result.forwardOutputs().size());
            assertEquals(target.id(), result.gradientResults().getFirst().target());
            assertTrue(result.validatedGraph().graph().nodePhases()
                    .containsValue(GraphPhase.FORWARD));
            assertTrue(result.validatedGraph().graph().nodePhases()
                    .containsValue(GraphPhase.BACKWARD));
            assertTrue(result.validatedGraph().constants().size() >= 2);
        }
    }

    @Test
    void capturesAndOptimizesRepresentative0005AFormulasThroughTheSharedPipeline() {
        for (GraphOptimizationConfig optimization :
                List.of(GraphOptimizationConfig.disabled(), GraphOptimizationConfig.standard())) {
            Tensor target = tensor();
            Tensor other = tensor();
            Tensor objective = target.minimum(other)
                    .pow(ScalarValue.float32(2.0f))
                    .geluTanhApproximation()
                    .silu()
                    .sum();

            GraphCompilation result = GraphCompiler.compile(
                    CompileMode.FORWARD_AND_BACKWARD,
                    List.of(objective),
                    Optional.of(new AutogradPreflight.FirstOrderRequest(
                            objective, List.of(target))),
                    CompileTimeConstantGraph.Ingress.empty(),
                    optimization);

            assertEquals(target.id(), result.gradientResults().getFirst().target());
            assertTrue(result.validatedGraph().graph().nodePhases()
                    .containsValue(GraphPhase.FORWARD));
            assertTrue(result.validatedGraph().graph().nodePhases()
                    .containsValue(GraphPhase.BACKWARD));
            assertTrue(result.validatedGraph().constants().size() >= 2);
        }
    }

    private static Tensor tensor() {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, Shape.of(2), Optional.empty(), true));
    }

    private static BackendCapabilityProvider provider(
            BackendId backendId,
            List<OperationCapabilityQuery> queries,
            boolean supported) {
        return new BackendCapabilityProvider() {
            @Override
            public BackendId backendId() {
                return backendId;
            }

            @Override
            public boolean supports(OperationCapabilityQuery query) {
                queries.add(query);
                return supported;
            }
        };
    }

    private static BackendAvailabilitySnapshot snapshot(BackendId backendId) {
        BackendDeviceId deviceId = new BackendDeviceId(backendId, "0");
        return new BackendAvailabilitySnapshot(
                backendId, Map.of(deviceId, DeviceClass.CPU));
    }

    private static long nextTensorId() throws Exception {
        var field = TensorFactory.class.getDeclaredField("NEXT_TENSOR_ID");
        field.setAccessible(true);
        return ((AtomicLong) field.get(null)).get();
    }
}
