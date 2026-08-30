package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.operation.layout.Window3dAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.pooling.AveragePool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.MaxPool3dAttrs;
import io.github.pho001.synaptik.model.operation.pooling.Pool3dKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class Pool3dAndWindow3dCompilerTest {
    private static final Window3dAttrs WINDOW = new Window3dAttrs(
            2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);

    @Test
    void adoptsAllFiveSignaturesAsOrdinaryForwardNodes() {
        Tensor input = tensor(Shape.of(0, 0, 4, 4, 4), true);
        Tensor max = input.maxPool3d(new MaxPool3dAttrs(
                2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false));
        Tensor average = input.averagePool3d(new AveragePool3dAttrs(
                2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, true));
        Tensor direct = input.unfold3d(WINDOW);
        Tensor typed = input.unfold3d(WINDOW, ScalarValue.float32(-1));
        Tensor folded = direct.fold3d(input.descriptor().shape(), WINDOW);

        ValidatedGraph validated = compile(List.of(max, average, direct, typed, folded));
        CompiledGraphModel graph = validated.graph();

        assertEquals(5, graph.outputs().size());
        assertEquals(5, graph.nodes().size());
        assertEquals(List.of(
                        Pool3dKind.MAX_POOL3D,
                        Pool3dKind.AVERAGE_POOL3D,
                        WindowTransformKind.UNFOLD3D,
                        WindowTransformKind.UNFOLD3D,
                        WindowTransformKind.FOLD3D),
                graph.nodes().stream().map(node -> node.operation().kind()).toList());
        assertEquals(Shape.of(0, 0, 3, 3, 3), max.descriptor().shape());
        assertEquals(Shape.of(0, 0, 27), direct.descriptor().shape());
        assertSame(input.descriptor().shape(), folded.descriptor().shape());
        assertTrue(validated.constraints().isEmpty());
    }

    @Test
    void retainsCanonicalSymbolicGeometryAndExactConstraintOrder() {
        Dimension n = new DynamicDimension("N");
        Dimension c = new DynamicDimension("C");
        Dimension d = new DynamicDimension("D");
        Dimension h = new DynamicDimension("H");
        Dimension w = new DynamicDimension("W");
        Tensor input = tensor(Shape.ofDimensions(n, c, d, h, w), false);
        MaxPool3dAttrs pooling = new MaxPool3dAttrs(
                3, 3, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, true);
        Tensor pool = input.maxPool3d(pooling);
        Tensor columns = input.unfold3d(WINDOW);

        ValidatedGraph validated = compile(List.of(pool, columns));
        assertEquals(List.of(
                        "pool3d depth numerator non-negative",
                        "pool3d height numerator non-negative",
                        "pool3d width numerator non-negative",
                        "unfold3d depth domain",
                        "unfold3d height domain",
                        "unfold3d width domain"),
                validated.constraints().stream()
                        .map(DeferredGraphConstraint::subject)
                        .toList());
        assertSame(n, pool.descriptor().shape().dimension(0));
        assertSame(c, pool.descriptor().shape().dimension(1));
        assertEquals(DimensionExpressions.multiply(
                        DimensionExpressions.multiply(
                                DimensionExpressions.multiply(c, 2), 2), 2),
                columns.descriptor().shape().dimension(1));
        assertEquals(
                DimensionExpressions.multiply(
                        DimensionExpressions.multiply(
                                DimensionExpressions.addConstant(d, -1),
                                DimensionExpressions.addConstant(h, -1)),
                        DimensionExpressions.addConstant(w, -1)),
                columns.descriptor().shape().dimension(2));
    }

    @Test
    void preservesExactCseAndPublicRootOccurrenceIdentity() {
        Tensor input = tensor(Shape.of(1, 2, 4, 4, 4), false);
        MaxPool3dAttrs attrs = new MaxPool3dAttrs(
                2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        Tensor first = input.maxPool3d(attrs);
        Tensor second = input.maxPool3d(attrs);

        CompiledGraphModel internal = compile(List.of(first.neg(), second.neg())).graph();
        assertEquals(1, internal.nodes().stream()
                .filter(node -> node.operation().kind() == Pool3dKind.MAX_POOL3D)
                .count());

        CompiledGraphModel publicRoots = compile(List.of(first, second)).graph();
        assertEquals(2, publicRoots.nodes().size());
        assertTrue(!publicRoots.nodes().get(0).id().equals(publicRoots.nodes().get(1).id()));

        Tensor differentAttrs = input.maxPool3d(new MaxPool3dAttrs(
                1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, false));
        Tensor differentInput = tensor(Shape.of(1, 2, 4, 4, 4), false).maxPool3d(attrs);
        CompiledGraphModel distinct = compile(List.of(
                first.neg(), differentAttrs.neg(), differentInput.neg())).graph();
        assertEquals(3, distinct.nodes().stream()
                .filter(node -> node.operation().kind() instanceof Pool3dKind)
                .count());

        Tensor unrelated = tensor(Shape.of(1), false);
        CompiledGraphModel captured = GraphCapture.capture(List.of(first, unrelated));
        CompiledGraphModel deadRoot = new CompiledGraphModel(
                captured.values(), captured.nodes(), captured.inputs(),
                List.of(captured.outputs().get(1)), captured.nodePhases());
        assertTrue(ForwardDeadCodeElimination.eliminate(deadRoot).nodes().isEmpty());
    }

    @Test
    void preservesPublicationDiagnosticsAndPlanningHandoff() {
        Dimension depth = new DynamicDimension("D");
        Tensor input = tensor(Shape.ofDimensions(
                new io.github.pho001.synaptik.model.shape.StaticDimension(1),
                new io.github.pho001.synaptik.model.shape.StaticDimension(2),
                depth,
                new io.github.pho001.synaptik.model.shape.StaticDimension(4),
                new io.github.pho001.synaptik.model.shape.StaticDimension(4)), true);
        Tensor output = input.unfold3d(WINDOW, ScalarValue.float32(0));
        BackendId backendId = new BackendId("recording-window3d-test-backend");
        List<OperationCapabilityQuery> queries = new ArrayList<>();

        CompileArtifacts artifacts = GraphCompiler.compile(
                CompileMode.FORWARD_ONLY,
                List.of(output),
                Optional.empty(),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.standard(),
                BackendIntent.unconstrained(),
                PartitionScoringConfig.neutral(),
                List.of(recordingProvider(backendId, queries)),
                List.of(new BackendAvailabilitySnapshot(
                        backendId,
                        Map.of(new BackendDeviceId(backendId, "0"), DeviceClass.CPU))));

        assertEquals(1, queries.size());
        assertSame(output.provenance().orElseThrow().operation(), queries.getFirst().operation());
        assertEquals(output.provenance().orElseThrow().producer().inputs().stream()
                .map(Tensor::descriptor).toList(), queries.getFirst().inputs());
        assertEquals(output.provenance().orElseThrow().producer().outputDescriptors(),
                queries.getFirst().outputs());
        assertEquals(1, artifacts.publication().forwardBindings().size());
        assertEquals(List.of("unfold3d depth domain"),
                artifacts.diagnostics().deferredConstraints().stream()
                        .map(CompileDiagnostics.DeferredConstraintDiagnostic::subject)
                        .toList());
        assertEquals(1, artifacts.partitions().size());
    }

    @Test
    void twoStageClosureUsesDisconnectedPolicyAndKeepsDifferentiableIncomingBranches() {
        Tensor linearTarget = tensor(Shape.of(1, 1, 3, 3, 3), true);
        Tensor linearObjective = linearTarget.unfold3d(WINDOW).sum();
        assertThrows(
                IllegalArgumentException.class,
                () -> compileTwoStage(
                        linearObjective,
                        linearTarget,
                        FunctionalGradientRequest.DisconnectedPolicy.ERROR));
        GraphCompilation linearZero = compileTwoStage(
                linearObjective,
                linearTarget,
                FunctionalGradientRequest.DisconnectedPolicy.ZERO);
        assertEquals(2, linearZero.gradientResults().size());
        assertEquals(2, linearZero.gradientResults().get(1).derivativeOrder());

        Tensor maximumTarget = tensor(Shape.of(1, 1, 3, 3, 3), true);
        Tensor maximum = maximumTarget.maxPool3d(new MaxPool3dAttrs(
                1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, false));
        Tensor connectedObjective = maximum.mul(maximumTarget).sum();
        GraphCompilation connected = compileTwoStage(
                connectedObjective,
                maximumTarget,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR);
        assertEquals(List.of(1, 2), connected.gradientResults().stream()
                .map(GradientPublicationBinding::derivativeOrder)
                .toList());
    }

    private static GraphCompilation compileTwoStage(
            Tensor objective,
            Tensor target,
            FunctionalGradientRequest.DisconnectedPolicy secondPolicy) {
        FunctionalGradientRequest.Stage first = new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.ForwardTensorReference(objective)),
                List.of(Optional.empty()),
                List.of(target),
                true,
                FunctionalGradientRequest.DisconnectedPolicy.ERROR);
        Tensor secondSeed = tensor(target.descriptor().shape(), false);
        FunctionalGradientRequest.Stage second = new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.FirstStageGradientReference(0)),
                List.of(Optional.of(secondSeed)),
                List.of(target),
                false,
                secondPolicy);
        return GraphCompiler.compile(
                CompileMode.FORWARD_AND_BACKWARD,
                List.of(objective),
                Optional.of(new FunctionalGradientRequest(List.of(first, second))),
                CompileTimeConstantGraph.Ingress.empty(),
                GraphOptimizationConfig.disabled());
    }

    private static ValidatedGraph compile(List<Tensor> outputs) {
        return GraphCompiler.compile(
                        CompileMode.FORWARD_ONLY,
                        outputs,
                        Optional.empty(),
                        CompileTimeConstantGraph.Ingress.empty(),
                        GraphOptimizationConfig.standard())
                .validatedGraph();
    }

    private static Tensor tensor(Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                DataType.FLOAT32, shape, Optional.empty(), requiresGrad));
    }

    private static BackendCapabilityProvider recordingProvider(
            BackendId backendId, List<OperationCapabilityQuery> queries) {
        return new BackendCapabilityProvider() {
            @Override
            public BackendId backendId() {
                return backendId;
            }

            @Override
            public boolean supports(OperationCapabilityQuery query) {
                queries.add(query);
                return true;
            }
        };
    }
}
