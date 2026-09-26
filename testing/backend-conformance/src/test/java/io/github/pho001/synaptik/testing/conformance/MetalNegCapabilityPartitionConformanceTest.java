package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.classification.FloatingClassificationKind;
import io.github.pho001.synaptik.model.operation.elementwise.logical.BooleanLogicalKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.index.SelectKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.operation.layout.PermutationAttrs;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.operation.reduction.AxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.MultiAxisReductionAttrs;
import io.github.pho001.synaptik.model.operation.reduction.SumToShapeAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.partition.MaximalSameOwnerPartitioning;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Conformance checks for public Metal capability truth and Planning maximal closure. */
final class MetalNegCapabilityPartitionConformanceTest {
    /** Proves the strict matrix is a subset of the complete accelerator matrix. */
    @Test
    void advertisesExactProfileQualifiedDomain() {
        var provider = new MetalCapabilityProvider();
        TensorDescriptor matrix = descriptor(Shape.of(2, 3));
        TensorDescriptor row = descriptor(Shape.of(3));
        for (UnaryElementwiseKind kind : UnaryElementwiseKind.values()) {
            assertEquals(
                    kind == UnaryElementwiseKind.NEG || kind == UnaryElementwiseKind.ABS,
                    provider.supports(query(
                            NumericalProfile.STRICT_IEEE,
                            operation(kind),
                            List.of(matrix),
                            List.of(matrix))),
                    "strict " + kind);
            assertEquals(
                    kind == UnaryElementwiseKind.NEG || kind == UnaryElementwiseKind.ABS,
                    provider.supports(query(
                            NumericalProfile.ACCELERATOR,
                            operation(kind),
                            List.of(matrix),
                            List.of(matrix))),
                    "accelerator " + kind);
        }
        for (BinaryArithmeticKind kind : BinaryArithmeticKind.values()) {
            boolean supported = kind == BinaryArithmeticKind.ADD
                    || kind == BinaryArithmeticKind.SUB
                    || kind == BinaryArithmeticKind.MUL
                    || kind == BinaryArithmeticKind.DIV
                    || kind == BinaryArithmeticKind.MIN
                    || kind == BinaryArithmeticKind.MAX;
            assertFalse(provider.supports(query(
                    NumericalProfile.STRICT_IEEE,
                    operation(kind),
                    List.of(matrix, row),
                    List.of(matrix))));
            assertEquals(supported, provider.supports(query(
                    NumericalProfile.ACCELERATOR,
                    operation(kind),
                    List.of(matrix, row),
                    List.of(matrix))));
        }
        TensorDescriptor cube = descriptor(Shape.of(2, 3, 4));
        TensorDescriptor scalar = descriptor(Shape.scalar());
        for (AggregateReductionKind kind : List.of(
                AggregateReductionKind.SUM,
                AggregateReductionKind.MEAN,
                AggregateReductionKind.PROD,
                AggregateReductionKind.MIN,
                AggregateReductionKind.MAX,
                AggregateReductionKind.ALL,
                AggregateReductionKind.ANY)) {
            Operation full = new Operation(kind, NoOperationAttrs.INSTANCE);
            assertFalse(provider.supports(query(
                    NumericalProfile.STRICT_IEEE,
                    full,
                    List.of(cube),
                    List.of(scalar))),
                    "strict " + kind);
            assertEquals(
                    kind == AggregateReductionKind.SUM
                            || kind == AggregateReductionKind.MEAN
                            || kind == AggregateReductionKind.MIN
                            || kind == AggregateReductionKind.MAX,
                    provider.supports(query(
                            NumericalProfile.ACCELERATOR,
                            full,
                            List.of(cube),
                            List.of(scalar))),
                    "accelerator " + kind);
        }
        assertTrue(provider.supports(query(
                NumericalProfile.ACCELERATOR,
                new Operation(
                        AggregateReductionKind.MEAN,
                        new AxisReductionAttrs(1, false)),
                List.of(cube),
                List.of(descriptor(Shape.of(2, 4))))));
        assertTrue(provider.supports(query(
                NumericalProfile.ACCELERATOR,
                new Operation(
                        AggregateReductionKind.SUM,
                        new AxisReductionAttrs(1, true)),
                List.of(cube),
                List.of(descriptor(Shape.of(2, 1, 4))))));
        assertTrue(provider.supports(query(
                NumericalProfile.ACCELERATOR,
                new Operation(
                        AggregateReductionKind.SUM,
                        new MultiAxisReductionAttrs(List.of(), false)),
                List.of(cube),
                List.of(cube))));
        assertTrue(provider.supports(query(
                NumericalProfile.ACCELERATOR,
                new Operation(
                        AggregateReductionKind.SUM,
                        new SumToShapeAttrs(Shape.of(1, 4))),
                List.of(cube),
                List.of(descriptor(Shape.of(1, 4))))));
        TensorDescriptor right = descriptor(Shape.of(3, 4));
        TensorDescriptor product = descriptor(Shape.of(2, 4));
        Operation matmul = new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE);
        assertTrue(provider.supports(query(
                NumericalProfile.ACCELERATOR,
                matmul,
                List.of(matrix, right),
                List.of(product))));
        assertFalse(provider.supports(query(
                NumericalProfile.STRICT_IEEE,
                matmul,
                List.of(matrix, right),
                List.of(product))));
        TensorDescriptor transposedRight = view(Shape.of(3, 4), 1, 3);
        assertTrue(provider.supports(query(
                NumericalProfile.ACCELERATOR,
                new Operation(
                        AxisTransformKind.PERMUTE,
                        new PermutationAttrs(List.of(1, 0))),
                List.of(descriptor(Shape.of(4, 3))),
                List.of(transposedRight))));
        assertTrue(provider.supports(query(
                NumericalProfile.ACCELERATOR,
                matmul,
                List.of(matrix, transposedRight),
                List.of(product))));

        Operation contiguous = new Operation(
                ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE);
        assertTrue(provider.supports(query(
                NumericalProfile.STRICT_IEEE,
                contiguous,
                List.of(matrix),
                List.of(matrix))));
        assertTrue(provider.supports(query(
                NumericalProfile.ACCELERATOR,
                contiguous,
                List.of(matrix),
                List.of(matrix))));
    }

    /** Proves an eligible NEG chain becomes one whole maximal Metal partition. */
    @Test void eligibleNegOccurrencesBecomeOneMaximalPartition() {
        TensorDescriptor descriptor = descriptor(Shape.of(4));
        ValueId input = new ValueId(0);
        ValueId negated = new ValueId(1);
        ValueId restored = new ValueId(2);
        ValueId output = new ValueId(3);
        CompiledNode first = new CompiledNode(
                new NodeId(0),
                operation(UnaryElementwiseKind.NEG),
                List.of(input),
                List.of(negated));
        CompiledNode second = new CompiledNode(
                new NodeId(1),
                operation(UnaryElementwiseKind.NEG),
                List.of(negated),
                List.of(restored));
        CompiledNode third = new CompiledNode(
                new NodeId(2),
                operation(UnaryElementwiseKind.NEG),
                List.of(restored),
                List.of(output));
        List<CompiledNode> nodes = List.of(first, second, third);
        var graph = new CompiledGraphModel(
                List.of(
                        new GraphValue(input, descriptor),
                        new GraphValue(negated, descriptor),
                        new GraphValue(restored, descriptor),
                        new GraphValue(output, descriptor)),
                nodes,
                List.of(input),
                List.of(restored, output),
                Map.of(
                        first.id(), GraphPhase.FORWARD,
                        second.id(), GraphPhase.FORWARD,
                        third.id(), GraphPhase.FORWARD));

        var partitions = MaximalSameOwnerPartitioning.partition(
                graph,
                Map.of(
                        first.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        second.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        third.id(), MetalCapabilityProvider.METAL_BACKEND_ID));

        assertEquals(1, partitions.size());
        assertSame(MetalCapabilityProvider.METAL_BACKEND_ID, partitions.getFirst().owner());
        assertEquals(nodes.stream().map(CompiledNode::id).toList(),
                partitions.getFirst().nodeIds());
    }

    /** Proves all four accelerator binaries form one ordered maximal partition. */
    @Test
    void eligibleAcceleratorBinaryOccurrencesBecomeOneMaximalPartition() {
        TensorDescriptor matrix = descriptor(Shape.of(2, 3));
        TensorDescriptor row = descriptor(Shape.of(3));
        ValueId matrixInput = new ValueId(20);
        ValueId rowInput = new ValueId(21);
        ValueId added = new ValueId(22);
        ValueId subtracted = new ValueId(23);
        ValueId multiplied = new ValueId(24);
        ValueId divided = new ValueId(25);
        List<CompiledNode> nodes = List.of(
                new CompiledNode(
                        new NodeId(20),
                        operation(BinaryArithmeticKind.ADD),
                        List.of(matrixInput, rowInput),
                        List.of(added)),
                new CompiledNode(
                        new NodeId(21),
                        operation(BinaryArithmeticKind.SUB),
                        List.of(rowInput, matrixInput),
                        List.of(subtracted)),
                new CompiledNode(
                        new NodeId(22),
                        operation(BinaryArithmeticKind.MUL),
                        List.of(matrixInput, matrixInput),
                        List.of(multiplied)),
                new CompiledNode(
                        new NodeId(23),
                        operation(BinaryArithmeticKind.DIV),
                        List.of(added, rowInput),
                        List.of(divided)));
        List<GraphValue> values = List.of(
                new GraphValue(matrixInput, matrix),
                new GraphValue(rowInput, row),
                new GraphValue(added, matrix),
                new GraphValue(subtracted, matrix),
                new GraphValue(multiplied, matrix),
                new GraphValue(divided, matrix));
        Map<NodeId, GraphPhase> phases = nodes.stream().collect(
                java.util.stream.Collectors.toMap(
                        CompiledNode::id, ignored -> GraphPhase.FORWARD));
        var graph = new CompiledGraphModel(
                values,
                nodes,
                List.of(matrixInput, rowInput),
                List.of(added, subtracted, multiplied, divided),
                phases);
        Map<NodeId, io.github.pho001.synaptik.backend.contract.BackendId> owners =
                nodes.stream().collect(java.util.stream.Collectors.toMap(
                        CompiledNode::id,
                        ignored -> MetalCapabilityProvider.METAL_BACKEND_ID));

        var partitions = MaximalSameOwnerPartitioning.partition(graph, owners);
        assertEquals(1, partitions.size());
        assertSame(MetalCapabilityProvider.METAL_BACKEND_ID, partitions.getFirst().owner());
        assertEquals(
                nodes.stream().map(CompiledNode::id).toList(),
                partitions.getFirst().nodeIds());
    }

    /** Proves the visible local transpose and MATMUL become one accelerator partition. */
    @Test
    void acceleratorLinearTopologyBecomesOneMaximalPartition() {
        ValueId left = new ValueId(40);
        ValueId weight = new ValueId(41);
        ValueId transposed = new ValueId(42);
        ValueId output = new ValueId(43);
        CompiledNode transpose = new CompiledNode(
                new NodeId(40),
                new Operation(
                        AxisTransformKind.PERMUTE,
                        new PermutationAttrs(List.of(1, 0))),
                List.of(weight),
                List.of(transposed));
        CompiledNode matmul = new CompiledNode(
                new NodeId(41),
                new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(left, transposed),
                List.of(output));
        List<CompiledNode> nodes = List.of(transpose, matmul);
        var graph = new CompiledGraphModel(
                List.of(
                        new GraphValue(left, descriptor(Shape.of(2, 3))),
                        new GraphValue(weight, descriptor(Shape.of(4, 3))),
                        new GraphValue(transposed, view(Shape.of(3, 4), 1, 3)),
                        new GraphValue(output, descriptor(Shape.of(2, 4)))),
                nodes,
                List.of(left, weight),
                List.of(output),
                Map.of(
                        transpose.id(), GraphPhase.FORWARD,
                        matmul.id(), GraphPhase.FORWARD));
        var partitions = MaximalSameOwnerPartitioning.partition(
                graph,
                Map.of(
                        transpose.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        matmul.id(), MetalCapabilityProvider.METAL_BACKEND_ID));
        assertEquals(1, partitions.size());
        assertEquals(
                nodes.stream().map(CompiledNode::id).toList(),
                partitions.getFirst().nodeIds());
    }

    /** Proves chained affine views, CONTIGUOUS, and NEG remain one maximal Metal partition. */
    @Test
    void eligibleAffineCompositionBecomesOneMaximalPartition() {
        ValueId input = new ValueId(10);
        ValueId reshaped = new ValueId(11);
        ValueId expanded = new ValueId(12);
        ValueId contiguous = new ValueId(13);
        ValueId negated = new ValueId(14);
        ValueId output = new ValueId(15);
        List<CompiledNode> nodes = List.of(
                new CompiledNode(
                        new NodeId(10),
                        new Operation(
                                ShapeTransformKind.RESHAPE,
                                new TargetShapeAttrs(Shape.of(2, 1, 3))),
                        List.of(input),
                        List.of(reshaped)),
                new CompiledNode(
                        new NodeId(11),
                        new Operation(
                                ShapeTransformKind.EXPAND,
                                new TargetShapeAttrs(Shape.of(2, 4, 3))),
                        List.of(reshaped),
                        List.of(expanded)),
                new CompiledNode(
                        new NodeId(12),
                        new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE),
                        List.of(expanded),
                        List.of(contiguous)),
                new CompiledNode(
                        new NodeId(13),
                        operation(UnaryElementwiseKind.NEG),
                        List.of(contiguous),
                        List.of(negated)),
                new CompiledNode(
                        new NodeId(14),
                        new Operation(
                                ShapeTransformKind.RESHAPE,
                                new TargetShapeAttrs(Shape.of(4, 6))),
                        List.of(negated),
                        List.of(output)));
        List<GraphValue> values = List.of(
                new GraphValue(input, descriptor(Shape.of(6))),
                new GraphValue(reshaped, view(Shape.of(2, 1, 3), 3, 3, 1)),
                new GraphValue(expanded, view(Shape.of(2, 4, 3), 3, 0, 1)),
                new GraphValue(contiguous, descriptor(Shape.of(2, 4, 3))),
                new GraphValue(negated, descriptor(Shape.of(2, 4, 3))),
                new GraphValue(output, view(Shape.of(4, 6), 6, 1)));
        Map<NodeId, GraphPhase> phases = nodes.stream().collect(
                java.util.stream.Collectors.toMap(
                        CompiledNode::id, ignored -> GraphPhase.FORWARD));
        var graph = new CompiledGraphModel(
                values, nodes, List.of(input), List.of(expanded, output), phases);
        Map<NodeId, io.github.pho001.synaptik.backend.contract.BackendId> owners =
                nodes.stream().collect(java.util.stream.Collectors.toMap(
                        CompiledNode::id,
                        ignored -> MetalCapabilityProvider.METAL_BACKEND_ID));

        var partitions = MaximalSameOwnerPartitioning.partition(graph, owners);
        assertEquals(1, partitions.size());
        assertSame(MetalCapabilityProvider.METAL_BACKEND_ID, partitions.getFirst().owner());
        assertEquals(nodes.stream().map(CompiledNode::id).toList(),
                partitions.getFirst().nodeIds());
    }

    /** Proves exact indexing operations, including replacement scatter, compose maximally. */
    @Test
    void exactInt32IndexingOccurrencesProduceOneMaximalMetalRegion() {
        TensorDescriptor dataDescriptor = descriptor(Shape.of(2, 3));
        TensorDescriptor indexDescriptor = typed(DataType.INT32, Shape.of(2));
        TensorDescriptor gatheredDescriptor = descriptor(Shape.of(2, 2));
        TensorDescriptor oneHotDescriptor = typed(DataType.BOOL, Shape.of(2, 4));
        TensorDescriptor scatterIndexDescriptor = typed(DataType.INT32, Shape.of(2, 2));
        TensorDescriptor updateDescriptor = descriptor(Shape.of(2, 2));
        ValueId data = new ValueId(60);
        ValueId indices = new ValueId(61);
        ValueId gathered = new ValueId(62);
        ValueId negated = new ValueId(63);
        ValueId oneHot = new ValueId(64);
        ValueId scatterIndices = new ValueId(65);
        ValueId updates = new ValueId(66);
        ValueId scattered = new ValueId(67);
        ValueId scatterNegated = new ValueId(68);
        CompiledNode gather = new CompiledNode(
                new NodeId(60),
                new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)),
                List.of(data, indices),
                List.of(gathered));
        CompiledNode neg = new CompiledNode(
                new NodeId(61),
                operation(UnaryElementwiseKind.NEG),
                List.of(gathered),
                List.of(negated));
        CompiledNode encode = new CompiledNode(
                new NodeId(62),
                new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(4)),
                List.of(indices),
                List.of(oneHot));
        CompiledNode scatter = new CompiledNode(
                new NodeId(63),
                new Operation(
                        AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(1, ScatterReduction.NONE)),
                List.of(data, scatterIndices, updates),
                List.of(scattered));
        CompiledNode scatterNeg = new CompiledNode(
                new NodeId(64),
                operation(UnaryElementwiseKind.NEG),
                List.of(scattered),
                List.of(scatterNegated));
        List<CompiledNode> nodes = List.of(gather, neg, encode, scatter, scatterNeg);
        var provider = new MetalCapabilityProvider();
        for (NumericalProfile profile : NumericalProfile.values()) {
            assertTrue(provider.supports(query(
                    profile, gather.operation(),
                    List.of(dataDescriptor, indexDescriptor),
                    List.of(gatheredDescriptor))));
            assertTrue(provider.supports(query(
                    profile, encode.operation(),
                    List.of(indexDescriptor),
                    List.of(oneHotDescriptor))));
            assertTrue(provider.supports(query(
                    profile,
                    scatter.operation(),
                    List.of(dataDescriptor, scatterIndexDescriptor, updateDescriptor),
                    List.of(dataDescriptor))));
        }
        var graph = new CompiledGraphModel(
                List.of(
                        new GraphValue(data, dataDescriptor),
                        new GraphValue(indices, indexDescriptor),
                        new GraphValue(gathered, gatheredDescriptor),
                        new GraphValue(negated, gatheredDescriptor),
                        new GraphValue(oneHot, oneHotDescriptor),
                        new GraphValue(scatterIndices, scatterIndexDescriptor),
                        new GraphValue(updates, updateDescriptor),
                        new GraphValue(scattered, dataDescriptor),
                        new GraphValue(scatterNegated, dataDescriptor)),
                nodes,
                List.of(data, indices, scatterIndices, updates),
                List.of(negated, oneHot, scatterNegated),
                Map.of(
                        gather.id(), GraphPhase.FORWARD,
                        neg.id(), GraphPhase.FORWARD,
                        encode.id(), GraphPhase.FORWARD,
                        scatter.id(), GraphPhase.FORWARD,
                        scatterNeg.id(), GraphPhase.FORWARD));
        var partitions = MaximalSameOwnerPartitioning.partition(
                graph,
                Map.of(
                        gather.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        neg.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        encode.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        scatter.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        scatterNeg.id(), MetalCapabilityProvider.METAL_BACKEND_ID));
        assertEquals(1, partitions.size());
        assertEquals(
                List.of(gather.id(), neg.id(), encode.id(), scatter.id(), scatterNeg.id()),
                partitions.getFirst().nodeIds());
        assertSame(
                MetalCapabilityProvider.METAL_BACKEND_ID,
                partitions.getFirst().owner());
    }

    /** Proves the public exact BOOL matrix and explicit scalar/select exclusions. */
    @Test
    void advertisesExactBoolClassificationLogicWhereAndExcludesScalarSelect() {
        var provider = new MetalCapabilityProvider();
        TensorDescriptor floatMatrix = descriptor(Shape.of(2, 3));
        TensorDescriptor boolMatrix = typed(DataType.BOOL, Shape.of(2, 3));
        TensorDescriptor boolColumn = typed(DataType.BOOL, Shape.of(2, 1));
        TensorDescriptor boolRow = typed(DataType.BOOL, Shape.of(3));
        TensorDescriptor trueBranch = descriptor(Shape.of(2, 1));
        TensorDescriptor falseBranch = descriptor(Shape.of(3));

        for (NumericalProfile profile : NumericalProfile.values()) {
            for (FloatingClassificationKind kind : FloatingClassificationKind.values()) {
                assertTrue(provider.supports(query(
                        profile,
                        new Operation(kind, NoOperationAttrs.INSTANCE),
                        List.of(floatMatrix),
                        List.of(boolMatrix))),
                        profile + " " + kind);
            }
            for (BooleanLogicalKind kind :
                    List.of(BooleanLogicalKind.AND, BooleanLogicalKind.OR)) {
                assertTrue(provider.supports(query(
                        profile,
                        new Operation(kind, NoOperationAttrs.INSTANCE),
                        List.of(boolColumn, boolRow),
                        List.of(boolMatrix))),
                        profile + " " + kind);
            }
            assertTrue(provider.supports(query(
                    profile,
                    new Operation(BooleanLogicalKind.NOT, NoOperationAttrs.INSTANCE),
                    List.of(boolMatrix),
                    List.of(boolMatrix))));
            assertTrue(provider.supports(query(
                    profile,
                    new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                    List.of(boolRow, trueBranch, falseBranch),
                    List.of(floatMatrix))));
            assertFalse(provider.supports(query(
                    profile,
                    new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                    List.of(
                            typed(DataType.BOOL, Shape.scalar()),
                            trueBranch,
                            falseBranch),
                    List.of(floatMatrix))),
                    "rank-zero BOOL condition remains outside Metal capability");
            assertFalse(provider.supports(query(
                    profile,
                    new Operation(SelectKind.SELECT, new SelectAttrs(1, 0)),
                    List.of(floatMatrix),
                    List.of(descriptor(Shape.of(2))))),
                    "wire 73 scalar-index SELECT remains outside Metal capability");
        }
    }

    /** Proves materialized general-axis windows compose with exact Metal elementwise work. */
    @Test
    void unfoldAxisAndNegProduceOneMaximalMetalRegion() {
        ValueId input = new ValueId(80);
        ValueId unfolded = new ValueId(81);
        ValueId negated = new ValueId(82);
        CompiledNode unfold = new CompiledNode(
                new NodeId(80),
                new Operation(
                        WindowTransformKind.UNFOLD_AXIS,
                        new UnfoldAxisAttrs(1, 3, 2)),
                List.of(input),
                List.of(unfolded));
        CompiledNode neg = new CompiledNode(
                new NodeId(81),
                operation(UnaryElementwiseKind.NEG),
                List.of(unfolded),
                List.of(negated));
        TensorDescriptor inputDescriptor = descriptor(Shape.of(2, 6));
        TensorDescriptor outputDescriptor = descriptor(Shape.of(2, 2, 3));
        var provider = new MetalCapabilityProvider();
        for (NumericalProfile profile : NumericalProfile.values()) {
            assertTrue(provider.supports(query(
                    profile, unfold.operation(),
                    List.of(inputDescriptor), List.of(outputDescriptor))));
        }
        var graph = new CompiledGraphModel(
                List.of(
                        new GraphValue(input, inputDescriptor),
                        new GraphValue(unfolded, outputDescriptor),
                        new GraphValue(negated, outputDescriptor)),
                List.of(unfold, neg),
                List.of(input),
                List.of(negated),
                Map.of(
                        unfold.id(), GraphPhase.FORWARD,
                        neg.id(), GraphPhase.FORWARD));
        var partitions = MaximalSameOwnerPartitioning.partition(
                graph,
                Map.of(
                        unfold.id(), MetalCapabilityProvider.METAL_BACKEND_ID,
                        neg.id(), MetalCapabilityProvider.METAL_BACKEND_ID));
        assertEquals(1, partitions.size());
        assertSame(MetalCapabilityProvider.METAL_BACKEND_ID, partitions.getFirst().owner());
        assertEquals(List.of(unfold.id(), neg.id()), partitions.getFirst().nodeIds());
    }

    private static OperationCapabilityQuery query(
            NumericalProfile profile,
            Operation operation,
            List<TensorDescriptor> inputs,
            List<TensorDescriptor> outputs) {
        return new OperationCapabilityQuery(profile, operation, inputs, outputs);
    }

    private static TensorDescriptor descriptor(Shape shape) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static TensorDescriptor typed(DataType dataType, Shape shape) {
        return new TensorDescriptor(dataType, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static TensorDescriptor view(Shape shape, long... strides) {
        return new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, strides, 0L, true)), false);
    }

    private static Operation operation(UnaryElementwiseKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }

    private static Operation operation(BinaryArithmeticKind kind) {
        return new Operation(kind, NoOperationAttrs.INSTANCE);
    }
}
