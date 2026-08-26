package io.github.pho001.synaptik.compiler;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
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
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.TensorProducer;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

final class Conv3dCompilerTest {
    @Test
    void capturesBiasedAndUnbiasedOccurrencesAsOrdinaryFlatNodes() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(0, 4, 7, 9, 11), true);
        Tensor weight = tensor(DataType.BFLOAT16, Shape.of(6, 2, 3, 3, 3), false);
        Tensor bias = tensor(DataType.FLOAT64, Shape.of(6), true);
        Conv3dAttrs attrs = new Conv3dAttrs(2, 2, 2, 1, 1, 1, 1, 1, 1, 2);
        Tensor unbiased = input.conv3d(weight, attrs);
        Tensor biased = input.conv3d(weight, bias, attrs);
        Tensor zeroInput = tensor(DataType.FLOAT32, Shape.of(2, 0, 1, 1, 1), false);
        Tensor zeroWeight = tensor(DataType.FLOAT32, Shape.of(0, 0, 1, 1, 1), false);
        Tensor zeroChannels = zeroInput.conv3d(zeroWeight, Conv3dAttrs.defaults());

        GraphCompilation compilation = compileForward(
                List.of(unbiased, biased, zeroChannels), GraphOptimizationConfig.standard());
        CompiledGraphModel graph = compilation.validatedGraph().graph();
        List<TensorProducer> producers =
                List.of(producer(unbiased), producer(biased), producer(zeroChannels));

        assertAll(
                () -> assertEquals(3, graph.nodes().size()),
                () -> assertEquals(3, graph.outputs().size()),
                () -> assertEquals(List.of(2, 3, 2),
                        graph.nodes().stream().map(node -> node.inputs().size()).toList()),
                () -> assertTrue(graph.nodes().stream()
                        .allMatch(node -> node.operation().kind() == Conv3dKind.CONV3D)),
                () -> assertSame(producers.get(0).operation(), graph.nodes().get(0).operation()),
                () -> assertSame(producers.get(1).operation(), graph.nodes().get(1).operation()),
                () -> assertSame(producers.get(2).operation(), graph.nodes().get(2).operation()),
                () -> assertSame(attrs, graph.nodes().get(0).operation().attrs()),
                () -> assertSame(attrs, graph.nodes().get(1).operation().attrs()),
                () -> assertEquals(DataType.FLOAT32,
                        descriptor(graph, graph.nodes().get(0).outputs().getFirst()).dataType()),
                () -> assertEquals(DataType.FLOAT64,
                        descriptor(graph, graph.nodes().get(1).outputs().getFirst()).dataType()),
                () -> assertEquals(Shape.of(0, 6, 4, 5, 6),
                        descriptor(graph, graph.nodes().get(0).outputs().getFirst()).shape()),
                () -> assertEquals(Shape.of(2, 0, 1, 1, 1),
                        descriptor(graph, graph.nodes().get(2).outputs().getFirst()).shape()),
                () -> assertTrue(compilation.validatedGraph().constraints().isEmpty()));
    }

    @Test
    void retainsExactOrderedSymbolicConstraintsAndCanonicalGeometry() {
        Dimension batch = new DynamicDimension("N");
        Dimension inputChannels = new DynamicDimension("Cin");
        Dimension outputChannels = new DynamicDimension("Cout");
        Dimension channelsPerGroup = new DynamicDimension("Cpg");
        Dimension biasChannels = new DynamicDimension("B");
        Dimension depth = new DynamicDimension("D");
        Dimension height = new DynamicDimension("H");
        Dimension width = new DynamicDimension("W");
        Tensor input = tensor(DataType.BFLOAT16,
                Shape.ofDimensions(batch, inputChannels, depth, height, width), false);
        Tensor weight = tensor(DataType.FLOAT32,
                Shape.ofDimensions(outputChannels, channelsPerGroup,
                        new StaticDimension(3), new StaticDimension(5),
                        new StaticDimension(3)), true);
        Tensor bias = tensor(DataType.FLOAT64,
                Shape.ofDimensions(biasChannels), false);
        Conv3dAttrs attrs = new Conv3dAttrs(2, 3, 4, 1, 0, 2, 2, 1, 3, 4);
        Tensor output = input.conv3d(weight, bias, attrs);

        ValidatedGraph validated = compileForward(
                List.of(output), GraphOptimizationConfig.standard()).validatedGraph();
        TensorDescriptor result = descriptor(
                validated.graph(), validated.graph().outputs().getFirst());
        List<DeferredGraphConstraint> constraints = validated.constraints();

        Dimension depthNumerator = DimensionExpressions.addConstant(depth, -3);
        Dimension heightNumerator = DimensionExpressions.addConstant(height, -5);
        Dimension widthNumerator = DimensionExpressions.addConstant(width, -3);
        assertAll(
                () -> assertEquals(DataType.FLOAT64, result.dataType()),
                () -> assertSame(batch, result.shape().dimension(0)),
                () -> assertSame(outputChannels, result.shape().dimension(1)),
                () -> assertEquals(DimensionExpressions.addConstant(
                                DimensionExpressions.floorDivide(depthNumerator, 2), 1),
                        result.shape().dimension(2)),
                () -> assertEquals(DimensionExpressions.addConstant(
                                DimensionExpressions.floorDivide(heightNumerator, 3), 1),
                        result.shape().dimension(3)),
                () -> assertEquals(DimensionExpressions.addConstant(
                                DimensionExpressions.floorDivide(widthNumerator, 4), 1),
                        result.shape().dimension(4)),
                () -> assertTrue(result.requiresGrad()),
                () -> assertEquals(List.of(
                                "conv3d input channels divisible by groups",
                                "conv3d output channels divisible by groups",
                                "conv3d weight channels per group",
                                "conv3d bias channels",
                                "conv3d depth numerator non-negative",
                                "conv3d height numerator non-negative",
                                "conv3d width numerator non-negative"),
                        constraints.stream().map(DeferredGraphConstraint::subject).toList()),
                () -> assertEquals(new DimensionDivisible(inputChannels, 4),
                        constraints.get(0).predicate()),
                () -> assertEquals(new DimensionDivisible(outputChannels, 4),
                        constraints.get(1).predicate()),
                () -> assertEquals(new DimensionEqual(
                                DimensionExpressions.multiply(channelsPerGroup, 4), inputChannels),
                        constraints.get(2).predicate()),
                () -> assertEquals(new DimensionEqual(biasChannels, outputChannels),
                        constraints.get(3).predicate()),
                () -> assertEquals(new DimensionAtLeast(depthNumerator, 0),
                        constraints.get(4).predicate()),
                () -> assertEquals(new DimensionAtLeast(heightNumerator, 0),
                        constraints.get(5).predicate()),
                () -> assertEquals(new DimensionAtLeast(widthNumerator, 0),
                        constraints.get(6).predicate()));
    }

    @Test
    void preservesOrdinaryExactCseAndGraphOutputExclusion() {
        Tensor input = tensor(DataType.FLOAT32, Shape.of(1, 2, 5, 5, 5), false);
        Tensor weight = tensor(DataType.FLOAT32, Shape.of(4, 2, 1, 1, 1), false);
        Tensor equalFirst = input.conv3d(weight, Conv3dAttrs.defaults());
        Tensor equalSecond = input.conv3d(weight, Conv3dAttrs.defaults());

        CompiledGraphModel merged = compileForward(
                List.of(equalFirst.neg(), equalSecond.neg()),
                GraphOptimizationConfig.standard()).validatedGraph().graph();
        assertEquals(1, conv3dNodes(merged).size());

        Conv3dAttrs alternate = new Conv3dAttrs(2, 2, 2, 2, 2, 2, 1, 1, 1, 1);
        Tensor differentAttrs = input.conv3d(weight, alternate);
        Tensor differentInput = tensor(DataType.FLOAT32, Shape.of(1, 2, 5, 5, 5), false)
                .conv3d(weight, Conv3dAttrs.defaults());
        CompiledGraphModel distinct = compileForward(
                List.of(equalFirst.neg(), differentAttrs.neg(), differentInput.neg()),
                GraphOptimizationConfig.standard()).validatedGraph().graph();
        assertEquals(3, conv3dNodes(distinct).size());

        CompiledGraphModel publicRoots = compileForward(
                List.of(equalFirst, equalSecond), GraphOptimizationConfig.standard())
                .validatedGraph().graph();
        assertEquals(2, conv3dNodes(publicRoots).size());
        assertNotEquals(
                conv3dNodes(publicRoots).get(0).id(), conv3dNodes(publicRoots).get(1).id());

        Tensor separateRoot = tensor(DataType.FLOAT32, Shape.of(1), false);
        CompiledGraphModel captured = GraphCapture.capture(List.of(equalFirst, separateRoot));
        CompiledGraphModel withDeadConv3d = new CompiledGraphModel(
                captured.values(),
                captured.nodes(),
                captured.inputs(),
                List.of(captured.outputs().get(1)),
                captured.nodePhases());
        assertTrue(conv3dNodes(ForwardDeadCodeElimination.eliminate(withDeadConv3d)).isEmpty());
    }

    @Test
    void passesExactPublicationDiagnosticsAndPlanningQueryWithoutProviderClaim() {
        Dimension channels = new DynamicDimension("Cin");
        Tensor input = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(2), channels,
                        new StaticDimension(5), new StaticDimension(5),
                        new StaticDimension(5)), true);
        Tensor weight = tensor(DataType.FLOAT32,
                Shape.ofDimensions(new StaticDimension(6), new DynamicDimension("Cpg"),
                        new StaticDimension(3), new StaticDimension(3),
                        new StaticDimension(3)), false);
        Conv3dAttrs attrs = new Conv3dAttrs(1, 1, 1, 0, 0, 0, 1, 1, 1, 2);
        Tensor output = input.conv3d(weight, attrs);
        TensorProducer producer = producer(output);
        BackendId backendId = new BackendId("recording-conv3d-test-backend");
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
                List.of(snapshot(backendId)));

        OperationCapabilityQuery query = queries.getFirst();
        CompiledNode node = artifacts.graph().nodes().getFirst();
        assertAll(
                () -> assertEquals(1, queries.size()),
                () -> assertSame(producer.operation(), query.operation()),
                () -> assertSame(attrs, query.operation().attrs()),
                () -> assertEquals(
                        producer.inputs().stream().map(Tensor::descriptor).toList(),
                        query.inputs()),
                () -> assertEquals(producer.outputDescriptors(), query.outputs()),
                () -> assertEquals(1, artifacts.publication().forwardBindings().size()),
                () -> assertEquals(node.outputs().getFirst(), artifacts.graph().outputs().getFirst()),
                () -> assertEquals(1, artifacts.partitions().size()),
                () -> assertEquals(List.of(
                                "conv3d input channels divisible by groups",
                                "conv3d weight channels per group"),
                        artifacts.diagnostics().deferredConstraints().stream()
                                .map(CompileDiagnostics.DeferredConstraintDiagnostic::subject)
                                .toList()));
    }

    private static GraphCompilation compileForward(
            List<Tensor> outputs, GraphOptimizationConfig optimization) {
        return GraphCompiler.compile(
                CompileMode.FORWARD_ONLY,
                outputs,
                Optional.empty(),
                CompileTimeConstantGraph.Ingress.empty(),
                optimization);
    }

    private static List<CompiledNode> conv3dNodes(CompiledGraphModel graph) {
        return graph.nodes().stream()
                .filter(node -> node.operation().kind() == Conv3dKind.CONV3D)
                .toList();
    }

    private static TensorProducer producer(Tensor tensor) {
        return tensor.provenance().orElseThrow().producer();
    }

    private static TensorDescriptor descriptor(
            CompiledGraphModel graph,
            io.github.pho001.synaptik.model.graph.ValueId id) {
        return graph.values().stream()
                .filter(value -> value.id().equals(id))
                .findFirst()
                .orElseThrow()
                .descriptor();
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

    private static BackendAvailabilitySnapshot snapshot(BackendId backendId) {
        BackendDeviceId deviceId = new BackendDeviceId(backendId, "0");
        return new BackendAvailabilitySnapshot(
                backendId, Map.of(deviceId, DeviceClass.CPU));
    }

    private static Tensor tensor(DataType type, Shape shape, boolean requiresGrad) {
        return TensorFactory.create(new TensorDescriptor(
                type, shape, Optional.empty(), requiresGrad));
    }
}
