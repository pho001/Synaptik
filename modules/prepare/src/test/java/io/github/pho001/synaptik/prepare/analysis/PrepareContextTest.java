package io.github.pho001.synaptik.prepare.analysis;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledScanBody;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.ordering.TopKAttrs;
import io.github.pho001.synaptik.model.operation.ordering.TopKKind;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.scan.ScanKind;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.ScanActivityPlan;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;

class PrepareContextTest {
    @Test
    void activityCannotTurnPrimalCarryIntoSelectedEmissionSeed() {
        Fixture fixture = typedLengthFixture(DataType.FLOAT32,
                new io.github.pho001.synaptik.model.shape.StaticDimension(1), 1, -2);
        var node = fixture.nodes.getFirst();
        var plan = new io.github.pho001.synaptik.planning.capability.ScanActivityPlan(
                node.scanBody().orElseThrow().graph(), List.of(), List.of(0),
                List.of(List.of(0)), Map.of(), Map.of(), List.of());
        assertThrows(IllegalArgumentException.class, () -> new PrepareContext<>(
                fixture.partition, fixture.nodes, fixture.values, fixture.requirements,
                fixture.constants, fixture.inputs, Map.of(node.id(), plan)));
    }

    @Test
    void activityCannotReinterpretRowCarryIdentityAsSharedTemporalAddition() {
        Fixture fixture = typedLengthFixture(DataType.FLOAT32,
                new io.github.pho001.synaptik.model.shape.StaticDimension(0), 0, -2);
        var node = fixture.nodes.getFirst();
        var plan = new io.github.pho001.synaptik.planning.capability.ScanActivityPlan(
                node.scanBody().orElseThrow().graph(), List.of(), List.of(),
                List.of(List.of()), Map.of(), Map.of(),
                List.of(new io.github.pho001.synaptik.planning.capability.ScanActivityPlan.Accumulator(0, 0)));
        assertThrows(IllegalArgumentException.class, () -> new PrepareContext<>(
                fixture.partition, fixture.nodes, fixture.values, fixture.requirements,
                fixture.constants, fixture.inputs, Map.of(node.id(), plan)));
    }

    @Test
    void lengthSharedCarryCannotBeACarriedRootButAllValidCarryCan() {
        for (boolean lengthMode : List.of(false, true)) {
            Fixture fixture = sharedActivityFixture(lengthMode);
            var graph = fixture.nodes.getFirst().scanBody().orElseThrow().graph();
            var plan = new ScanActivityPlan(graph, List.of(new ScanActivityPlan.Carry(0, true, 0)),
                    List.of(), List.of(List.of(0)), Map.of(new NodeId(0), 0, new NodeId(1), 0),
                    Map.of(), List.of());
            if (lengthMode) {
                assertThrows(IllegalArgumentException.class, () -> activityContext(fixture, plan));
            } else {
                activityContext(fixture, plan);
            }
        }
    }

    @Test
    void numericCarriesRequireCompleteClassificationAndExactContributionAvailability() {
        for (boolean lengthMode : List.of(false, true)) {
            Fixture fixture = sharedActivityFixture(lengthMode);
            var graph = fixture.nodes.getFirst().scanBody().orElseThrow().graph();
            List<List<Integer>> scopes = List.of(List.of(0), List.of(1), List.of(0, 1));
            Map<NodeId, ScanActivityPlan.Merge> merges =
                    Map.of(new NodeId(0), new ScanActivityPlan.Merge(0, 1));
            var valid = new ScanActivityPlan(graph, List.of(), List.of(1, 2), scopes,
                    Map.of(new NodeId(0), 2, new NodeId(1), 2), merges,
                    List.of(new ScanActivityPlan.Accumulator(0, 2)));
            activityContext(fixture, valid);

            var missingCarry = new ScanActivityPlan(graph, List.of(), valid.seedFormals(), scopes,
                    valid.nodeScopes(), merges, List.of());
            assertThrows(IllegalArgumentException.class, () -> activityContext(fixture, missingCarry));

            // The contribution is available at either root, not just root zero. Implication
            // permits the narrower ADD owner, but does not prove the declared completed term.
            var narrowerContribution = new ScanActivityPlan(graph, List.of(), valid.seedFormals(), scopes,
                    Map.of(new NodeId(0), 2, new NodeId(1), 0), merges,
                    List.of(new ScanActivityPlan.Accumulator(0, 0)));
            assertThrows(IllegalArgumentException.class,
                    () -> activityContext(fixture, narrowerContribution));
        }
    }

    @Test
    void temporalContributionMustMatchAccumulatorWithoutBroadcastOrPromotion() {
        for (DataType seedType : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            for (Shape seedShape : List.of(Shape.of(1), Shape.of(2))) {
                Fixture fixture = directActivityFixture(seedType, seedShape, false, false);
                var graph = fixture.nodes.getFirst().scanBody().orElseThrow().graph();
                var plan = new ScanActivityPlan(graph, List.of(), List.of(1),
                        List.of(List.of(0)), Map.of(new NodeId(0), 0), Map.of(),
                        List.of(new ScanActivityPlan.Accumulator(0, 0)));
                // Ordinary ADD accepts every fixture. Only the exact completed term is valid.
                context(fixture, fixture.values, fixture.requirements, Map.of());
                if (seedType == DataType.FLOAT64 && seedShape.equals(Shape.of(2))) {
                    activityContext(fixture, plan);
                } else {
                    assertThrows(IllegalArgumentException.class, () -> activityContext(fixture, plan));
                }
            }
        }
    }

    @Test
    void optionalRawMergeRequiresExactOperandsWhileOrdinaryAddKeepsPromotion() {
        for (DataType seedType : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            for (Shape seedShape : List.of(Shape.of(1), Shape.of(2))) {
                Fixture fixture = directActivityFixture(seedType, seedShape, false, false);
                var graph = fixture.nodes.getFirst().scanBody().orElseThrow().graph();
                // Every ordinary ADD remains valid, including broadcasting and promotion.
                context(fixture, fixture.values, fixture.requirements, Map.of());
                org.junit.jupiter.api.function.Executable association = () -> activityContext(fixture,
                        new ScanActivityPlan(graph, List.of(new ScanActivityPlan.Carry(0, false, 2)),
                                List.of(1), List.of(List.of(0), List.of(1), List.of(0, 1)),
                                Map.of(new NodeId(0), 2),
                                Map.of(new NodeId(0), new ScanActivityPlan.Merge(0, 1)), List.of()));
                if (seedType == DataType.FLOAT64 && seedShape.equals(Shape.of(2))) {
                    assertDoesNotThrow(association);
                } else {
                    // Both the intrinsic carrier and the independent Prepare barrier reject
                    // malformed associations; this does not pin their order or diagnostics.
                    assertThrows(IllegalArgumentException.class, association);
                }
            }
        }
    }

    @Test
    void unconditionalNoNodeNextCarryCannotDeclareStructuralAbsence() {
        Fixture absent = directActivityFixture(DataType.FLOAT64, Shape.of(2), true, true);
        var absentGraph = absent.nodes.getFirst().scanBody().orElseThrow().graph();
        var falseAbsence = new ScanActivityPlan(absentGraph,
                List.of(new ScanActivityPlan.Carry(0, true, 0)), List.of(),
                List.of(List.of()), Map.of(), Map.of(), List.of());
        context(absent, absent.values, absent.requirements, Map.of());
        assertThrows(IllegalArgumentException.class, () -> activityContext(absent, falseAbsence));

        Fixture identity = directActivityFixture(DataType.FLOAT64, Shape.of(2), true, false);
        var identityGraph = identity.nodes.getFirst().scanBody().orElseThrow().graph();
        activityContext(identity, new ScanActivityPlan(identityGraph,
                List.of(new ScanActivityPlan.Carry(0, true, 0)), List.of(),
                List.of(List.of(0)), Map.of(), Map.of(), List.of()));
    }

    @Test
    void optionalMergeCannotTreatUnconditionalFormalAsAbsentOperand() {
        Fixture fixture = sharedActivityFixture(false);
        var graph = fixture.nodes.getFirst().scanBody().orElseThrow().graph();
        var plan = new ScanActivityPlan(graph, List.of(), List.of(2),
                List.of(List.of(), List.of(0)), Map.of(new NodeId(0), 1, new NodeId(1), 1),
                Map.of(new NodeId(0), new ScanActivityPlan.Merge(0, 1)),
                List.of(new ScanActivityPlan.Accumulator(0, 1)));
        assertThrows(IllegalArgumentException.class, () -> activityContext(fixture, plan));
    }

    /**
     * Builds a complete all-valid direct projection with either an ordinary temporal ADD or
     * a zero-node carry binding, bypassing annotation inference.
     * @param seedType ordinary sequence-step dtype, possibly promoted by ADD
     * @param seedShape ordinary sequence-step shape, possibly broadcast by ADD
     * @param identity whether the body has no ordinary nodes
     * @param invariantNext whether a zero-node next carry binds an unconditional invariant
     * @return complete descriptor-valid projection without an activity association
     */
    private static Fixture directActivityFixture(
            DataType seedType, Shape seedShape, boolean identity, boolean invariantNext) {
        var state = scanDescriptor(identity ? DataType.FLOAT32 : DataType.FLOAT64, false, Shape.of(2));
        var seed = scanDescriptor(seedType, false, seedShape);
        var local = new ArrayList<>(List.of(new GraphValue(new ValueId(0), state),
                new GraphValue(new ValueId(1), seed), new GraphValue(new ValueId(2), state)));
        var addition = new CompiledNode(new NodeId(0),
                new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(3)));
        if (!identity) local.add(new GraphValue(new ValueId(3), state));
        ValueId next = new ValueId(identity ? invariantNext ? 2 : 0 : 3);
        var graph = new CompiledGraphModel(local, identity ? List.of() : List.of(addition),
                List.of(new ValueId(0), new ValueId(1), new ValueId(2)), List.of(next),
                identity ? Map.of() : Map.of(addition.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(next), List.of(), List.of(), List.of(0));
        var node = new CompiledNode(new NodeId(0),
                new Operation(ScanKind.SCAN, new ScanAttrs(1, 1, 1, 1, 0, 0, false)),
                graph.inputs(), List.of(new ValueId(10)), Optional.of(body));
        var partition = new PlannedPartition(new BackendId("direct-activity"), List.of(node.id()));
        var values = List.of(new GraphValue(new ValueId(0), state),
                new GraphValue(new ValueId(1), scanDescriptor(seedType, false, stacked(1, seedShape))),
                new GraphValue(new ValueId(2), state), new GraphValue(new ValueId(10), state));
        var requirements = values.stream().map(v -> new LogicalMemoryRequirement(v.id(), v.descriptor(),
                v.id().equals(new ValueId(10)) ? Optional.of(partition) : Optional.empty(),
                v.id().equals(new ValueId(10)) ? List.<PlannedPartition>of() : List.of(partition),
                v.id().equals(new ValueId(10)))).toList();
        return new Fixture(partition, List.of(node), values, requirements, Map.of(),
                new FakeInputs("direct-activity"));
    }

    @Test
    void symbolicOrdinaryReshapeProvesCountsWithoutImposingLengthRowLocality() {
        var batch = new DynamicDimension("Customers");
        var two = new io.github.pho001.synaptik.model.shape.StaticDimension(2);
        var three = new io.github.pho001.synaptik.model.shape.StaticDimension(3);
        Fixture invalid = symbolicReshapeFixture(Shape.ofDimensions(batch, three));
        assertThrows(IllegalArgumentException.class,
                () -> context(invalid, invalid.values, invalid.requirements, Map.of()));
        for (Shape target : List.of(Shape.ofDimensions(two, batch),
                Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.StaticDimension(1), batch, two))) {
            Fixture valid = symbolicReshapeFixture(target);
            context(valid, valid.values, valid.requirements, Map.of());
        }
    }

    /**
     * Builds an all-valid carry-free symbolic reshape directly at the public Prepare boundary.
     * @param target declared reshape target using the actual Customers coordinate
     * @return complete projection whose ordinary symbolic count relation still requires proof
     */
    private static Fixture symbolicReshapeFixture(Shape target) {
        Shape source = Shape.ofDimensions(new DynamicDimension("Customers"),
                new io.github.pho001.synaptik.model.shape.StaticDimension(2));
        var input = new TensorDescriptor(DataType.FLOAT32, source, Optional.empty(), false);
        var output = new TensorDescriptor(DataType.FLOAT32, target, Optional.empty(), false);
        var reshape = new CompiledNode(new NodeId(0), new Operation(
                io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind.RESHAPE,
                new io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs(target)),
                List.of(new ValueId(0)), List.of(new ValueId(1)));
        var graph = new CompiledGraphModel(
                List.of(new GraphValue(new ValueId(0), input), new GraphValue(new ValueId(1), output)),
                List.of(reshape), List.of(new ValueId(0)), List.of(new ValueId(1)),
                Map.of(reshape.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(), List.of(new ValueId(1)), List.of(), List.of(0));
        var node = new CompiledNode(new NodeId(0),
                new Operation(ScanKind.SCAN, new ScanAttrs(1, 0, 1, 0, 1, 0, false)),
                List.of(new ValueId(0)), List.of(new ValueId(10)), Optional.of(body));
        var partition = new PlannedPartition(new BackendId("symbolic-reshape"), List.of(node.id()));
        var values = List.of(new GraphValue(new ValueId(0),
                        new TensorDescriptor(DataType.FLOAT32, stacked(1, source), Optional.empty(), false)),
                new GraphValue(new ValueId(10),
                        new TensorDescriptor(DataType.FLOAT32, stacked(1, target), Optional.empty(), false)));
        var requirements = values.stream().map(v -> new LogicalMemoryRequirement(v.id(), v.descriptor(),
                v.id().equals(new ValueId(10)) ? Optional.of(partition) : Optional.empty(),
                v.id().equals(new ValueId(10)) ? List.<PlannedPartition>of() : List.of(partition),
                v.id().equals(new ValueId(10)))).toList();
        return new Fixture(partition, List.of(node), values, requirements, Map.of(),
                new FakeInputs("symbolic-reshape"));
    }

    /**
     * Builds an ordinary shared temporal ADD with two independent selected contribution seeds.
     * @param lengthMode whether to append a separate lengths actual with no ROW carry
     * @return complete direct projection, valid without relying on a backend's activity checks
     */
    private static Fixture sharedActivityFixture(boolean lengthMode) {
        var shared = scanDescriptor(DataType.FLOAT32, false, Shape.of(2));
        var term = scanDescriptor(DataType.FLOAT32, true, Shape.of(2));
        var local = List.of(new GraphValue(new ValueId(0), shared),
                new GraphValue(new ValueId(1), term), new GraphValue(new ValueId(2), term),
                new GraphValue(new ValueId(3), term), new GraphValue(new ValueId(4), term));
        var contribution = new CompiledNode(new NodeId(0),
                new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(1), new ValueId(2)), List.of(new ValueId(3)));
        var accumulation = new CompiledNode(new NodeId(1),
                new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0), new ValueId(3)), List.of(new ValueId(4)));
        var residuals = List.of(new ValueId(0), new ValueId(3), new ValueId(4));
        var graph = new CompiledGraphModel(local, List.of(contribution, accumulation),
                List.of(new ValueId(0), new ValueId(1), new ValueId(2)), residuals,
                Map.of(contribution.id(), GraphPhase.FORWARD, accumulation.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(new ValueId(4)), List.of(), residuals, List.of(0));
        var values = new ArrayList<>(List.of(new GraphValue(new ValueId(0), shared),
                new GraphValue(new ValueId(1), scanDescriptor(DataType.FLOAT32, true, Shape.of(1, 2))),
                new GraphValue(new ValueId(2), scanDescriptor(DataType.FLOAT32, true, Shape.of(1, 2)))));
        var inputs = new ArrayList<>(List.of(new ValueId(0), new ValueId(1), new ValueId(2)));
        if (lengthMode) {
            inputs.add(new ValueId(3));
            values.add(new GraphValue(new ValueId(3), scanDescriptor(DataType.INT64, false, Shape.of(2))));
        }
        var outputs = List.of(new ValueId(10), new ValueId(11), new ValueId(12), new ValueId(13));
        values.add(new GraphValue(outputs.get(0), term));
        for (int port = 1; port < outputs.size(); port++) {
            values.add(new GraphValue(outputs.get(port),
                    scanDescriptor(DataType.FLOAT32, true, Shape.of(1, 2))));
        }
        ScanAttrs attrs = lengthMode
                ? new ScanAttrs(1, 1, 2, 0, 0, 3, false, true, 0,
                        List.of(OptionalInt.empty(), OptionalInt.empty()))
                : new ScanAttrs(1, 1, 2, 0, 0, 3, false);
        var node = new CompiledNode(new NodeId(0), new Operation(ScanKind.SCAN, attrs),
                inputs, outputs, Optional.of(body));
        var partition = new PlannedPartition(new BackendId("activity"), List.of(node.id()));
        var requirements = values.stream().map(v -> new LogicalMemoryRequirement(v.id(), v.descriptor(),
                outputs.contains(v.id()) ? Optional.of(partition) : Optional.empty(),
                outputs.contains(v.id()) ? List.<PlannedPartition>of() : List.of(partition),
                v.id().equals(outputs.getFirst()))).toList();
        return new Fixture(partition, List.of(node), values, requirements, Map.of(), new FakeInputs("activity"));
    }

    /**
     * Applies an exact local activity association directly at the pre-analyzer boundary.
     * @param fixture complete containing Scan projection
     * @param plan activity retaining that Scan's exact local graph
     * @return validated context; no backend analyzer is involved
     */
    private static PrepareContext<FakeInputs> activityContext(Fixture fixture, ScanActivityPlan plan) {
        return new PrepareContext<>(fixture.partition, fixture.nodes, fixture.values, fixture.requirements,
                fixture.constants, fixture.inputs, Map.of(fixture.nodes.getFirst().id(), plan));
    }

    /**
     * Builds raw direct projection facts, bypassing Compiler inference completely.
     * @param type newly admitted interior/declaration type
     * @param batch static zero or one named row Dimension
     * @param time fixed time count
     * @param unusedRole -2 for homogeneous identity, -1 for an F32/type/F32 cast chain, otherwise unused carry/step/invariant
     * @return structurally complete direct fixture with canonical residual ports and logical uses
     */
    private static Fixture typedLengthFixture(DataType type, io.github.pho001.synaptik.model.shape.Dimension batch,
            long time, int unusedRole) {
        Shape row = Shape.ofDimensions(batch, new io.github.pho001.synaptik.model.shape.StaticDimension(2));
        java.util.function.BiFunction<DataType, Shape, TensorDescriptor> descriptor = (dtype, shape) ->
                new TensorDescriptor(dtype, shape, shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(), false);
        DataType boundary = unusedRole == -2 ? type : DataType.FLOAT32;
        var rd = descriptor.apply(boundary, row);
        var typed = descriptor.apply(type, unusedRole == 2 ? Shape.of(2) : row);
        var local = new ArrayList<GraphValue>(); var nodes = new ArrayList<CompiledNode>();
        var formals = new ArrayList<ValueId>(); var actuals = new ArrayList<TensorDescriptor>();
        local.add(new GraphValue(new ValueId(0), rd)); formals.add(new ValueId(0)); actuals.add(rd);
        if (unusedRole == 0) { local.add(new GraphValue(new ValueId(1), typed)); formals.add(new ValueId(1)); actuals.add(typed); }
        int step = local.size(); local.add(new GraphValue(new ValueId(step), rd)); formals.add(new ValueId(step)); actuals.add(descriptor.apply(boundary, stacked(time, row)));
        if (unusedRole > 0) {
            int id = local.size(); local.add(new GraphValue(new ValueId(id), typed)); formals.add(new ValueId(id));
            actuals.add(descriptor.apply(type, unusedRole == 1 ? stacked(time, row) : Shape.of(2)));
        }
        ValueId root = new ValueId(0);
        var residuals = new ArrayList<ValueId>(); residuals.add(root);
        if (unusedRole == -1) {
            for (DataType target : List.of(type, DataType.FLOAT32)) {
                ValueId out = new ValueId(local.size());
                local.add(new GraphValue(out, descriptor.apply(target, row)));
                nodes.add(new CompiledNode(new NodeId(nodes.size()), new Operation(
                        io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind.CAST,
                        new io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs(target)), List.of(root), List.of(out)));
                root = out; residuals.add(out);
            }
        } else if (unusedRole == 0) residuals.add(new ValueId(1));
        var roots = new java.util.LinkedHashSet<ValueId>(); roots.add(root); roots.addAll(residuals);
        if (unusedRole == 0) roots.add(new ValueId(1));
        var localGraph = new CompiledGraphModel(local, nodes, formals, List.copyOf(roots),
                nodes.stream().collect(java.util.stream.Collectors.toMap(CompiledNode::id, n -> GraphPhase.FORWARD)));
        int carries = unusedRole == 0 ? 2 : 1, sequences = unusedRole == 1 ? 2 : 1;
        var next = unusedRole == 0 ? List.of(root, new ValueId(1)) : List.of(root);
        var body = new CompiledScanBody(localGraph, next, List.of(), residuals, List.of(0));
        actuals.add(descriptor.apply(DataType.INT64, Shape.ofDimensions(batch)));
        var values = new ArrayList<GraphValue>();
        for (var d : actuals) values.add(new GraphValue(new ValueId(values.size()), d));
        var outputs = new ArrayList<ValueId>(); var roles = new ArrayList<>(next); roles.addAll(residuals);
        for (int i = 0; i < roles.size(); i++) {
            var d = local.get((int) roles.get(i).value()).descriptor(); ValueId id = new ValueId(values.size()); outputs.add(id);
            values.add(new GraphValue(id, descriptor.apply(d.dataType(), i < carries ? d.shape() : stacked(time, d.shape()))));
        }
        var node = new CompiledNode(new NodeId(0), new Operation(ScanKind.SCAN,
                new ScanAttrs(time, carries, sequences, unusedRole == 2 ? 1 : 0, 0, residuals.size(), false, true, carries, java.util.Collections.nCopies(sequences, OptionalInt.of(0)))),
                java.util.stream.IntStream.range(0, actuals.size()).mapToObj(i -> new ValueId(i)).toList(), outputs, Optional.of(body));
        var partition = new PlannedPartition(new BackendId("typed-length"), List.of(node.id()));
        var requirements = values.stream().map(v -> new LogicalMemoryRequirement(v.id(), v.descriptor(),
                outputs.contains(v.id()) ? Optional.of(partition) : Optional.empty(), outputs.contains(v.id()) ? List.<PlannedPartition>of() : List.of(partition),
                v.id().equals(outputs.getFirst()))).toList();
        return new Fixture(partition, List.of(node), values, requirements, Map.of(), new FakeInputs("typed-length"));
    }

    @Test
    void directSymbolicLengthProjectionAcceptsCompleteTemplatesButRejectsEveryIsolatedMutation() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (long time : new long[] {0, 3}) {
            var artifacts = BackendPartitionPreparerTest.symbolicLengthArtifacts(type, time, false);
            var late = artifacts.partitions().getLast(); var node = artifacts.graph().nodes().getLast();
            var projected = new PrepareContext<>(late, List.of(node),
                    artifacts.graph().values(), artifacts.memory().requirements(), Map.of(), new FakeInputs("symbolic-length"));
            assertTrue(projected.requiresBatchSize()); assertSame(node.scanBody().orElseThrow(),projected.nodes().getFirst().scanBody().orElseThrow());
            Fixture fixture = lengthFixture(type, time, 2, 2, false);
            Shape symbol = Shape.ofDimensions(new DynamicDimension("Customers"));
            for (GraphValue value : fixture.values) {
                var d = value.descriptor();
                Fixture changed = withScanOuter(fixture, value.id(), new TensorDescriptor(d.dataType(), symbol, Optional.empty(), d.requiresGrad()));
                assertThrows(IllegalArgumentException.class, () -> context(changed, changed.values, changed.requirements, Map.of()));
            }
            for (GraphValue value : fixture.nodes.getFirst().scanBody().orElseThrow().graph().values()) {
                var d = value.descriptor();
                Fixture changed = withScanLocal(fixture, value.id(), new TensorDescriptor(d.dataType(), symbol, Optional.empty(), d.requiresGrad()));
                assertThrows(IllegalArgumentException.class, () -> context(changed, changed.values, changed.requirements, Map.of()));
            }
        }
    }

    @Test
    void lengthModeSeparatesActualsAndFormalsWithMixedRolesAndEmptyDomains() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            for (boolean reverse : List.of(false, true)) for (long time : List.of(0L, 3L)) {
                for (long batch : List.of(0L, 1L, 2L)) for (long features : List.of(0L, 2L)) {
                    Fixture fixture = lengthFixture(type, time, batch, features, reverse);
                    var context = context(fixture, fixture.values, fixture.requirements, Map.of());
                    var node = fixture.nodes.getFirst();
                    var body = node.scanBody().orElseThrow();
                    assertSame(node, context.nodes().getFirst());
                    assertSame(body, context.nodes().getFirst().scanBody().orElseThrow());
                    assertSame(body.graph(), context.nodes().getFirst().scanBody().orElseThrow().graph());
                    assertEquals(7, node.inputs().size());
                    assertEquals(6, body.graph().inputs().size());
                    assertEquals(DataType.INT64, fixture.values.get(6).descriptor().dataType());
                    assertEquals(Shape.of(batch, features), body.graph().values().get(6).descriptor().shape());
                    assertEquals(Shape.of(batch), fixture.values.get(6).descriptor().shape());
                    assertEquals(body.nextCarry().getFirst(), body.emissions().getFirst());
                    assertEquals(List.of(0), body.demandedOutputIndices());
                    assertTrue(fixture.values.subList(7, fixture.values.size()).stream()
                            .allMatch(v -> v.descriptor().requiresGrad()));
                    assertEquals(false, body.graph().values().getFirst().descriptor().requiresGrad());
                    assertEquals(Shape.of(time, features), fixture.values.get(13).descriptor().shape());
                }
            }
        }
    }

    @Test
    void lengthModePreservesActualLayoutsWithoutCpuReadOrCanonicalLengthsPredicates() {
        Fixture fixture = lengthFixture(DataType.FLOAT64, 3, 2, 2, false);
        for (ValueId id : fixture.nodes.getFirst().inputs()) {
            TensorDescriptor descriptor = fixture.values.get((int) id.value()).descriptor();
            long[] strides = new long[descriptor.shape().rank()];
            Arrays.fill(strides, 3);
            for (Optional<LayoutDescriptor> layout : List.of(Optional.<LayoutDescriptor>empty(),
                    Optional.of(LayoutDescriptor.of(descriptor.shape(), strides, 5, true)),
                    Optional.of(LayoutDescriptor.of(descriptor.shape(), new long[strides.length], 0, true)))) {
                Fixture changed = withScanOuter(fixture, id,
                        new TensorDescriptor(descriptor.dataType(), descriptor.shape(), layout, descriptor.requiresGrad()));
                var projected = context(changed, changed.values, changed.requirements, Map.of());
                assertSame(changed.values.get((int) id.value()), projected.values().get((int) id.value()));
                assertEquals(layout, projected.values().get((int) id.value()).descriptor().layout());
            }
        }
    }

    @Test
    void lengthModeRejectsBadLengthsAndStaticBoundaryRolesEvenWhenUnusedOrEmpty() {
        for (long time : List.of(0L, 3L)) for (long batch : List.of(0L, 1L, 2L)) {
            Fixture fixture = lengthFixture(DataType.FLOAT32, time, batch, 2, false);
            for (TensorDescriptor bad : List.of(
                    scanDescriptor(DataType.INT32, false, Shape.of(batch)),
                    scanDescriptor(DataType.FLOAT32, true, Shape.of(batch)),
                    scanDescriptor(DataType.INT64, false, Shape.of(batch, 1)),
                    scanDescriptor(DataType.INT64, false, Shape.of(batch + 1)),
                    scanDescriptor(DataType.INT64, false, Shape.of()))) {
                assertInvalidLengthFixture(withScanOuter(fixture, new ValueId(6), bad));
            }
            // Unused declarations still constrain the explicitly declared row extent.
            for (Shape badShape : List.of(Shape.of(2, batch + 1), Shape.of(batch))) {
                Fixture bad = withLengthFormalAndActual(fixture, 3, badShape, time);
                assertInvalidLengthFixture(bad);
            }
            Fixture badTime = withScanOuter(fixture, new ValueId(4),
                    scanDescriptor(DataType.FLOAT32, false, Shape.of(time + 1, 2, batch)));
            assertInvalidLengthFixture(badTime); // SHARED still has a time dimension.
            assertInvalidLengthFixture(withLengthFormalAndActual(fixture, 0, Shape.of(batch + 1, 2), time));
            assertInvalidLengthFixture(withScanLocal(fixture, new ValueId(6),
                    scanDescriptor(DataType.FLOAT32, true, Shape.of(batch + 1, 2))));
            assertInvalidLengthFixture(withScanLocal(fixture, new ValueId(0),
                    scanDescriptor(DataType.FLOAT32, false, Shape.of())));
        }
    }

    @Test
    void lengthModeChecksEveryLocalFormalAndOuterRoleWithoutReinferringAlgebra() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (long time : List.of(0L, 3L)) {
            Fixture fixture = lengthFixture(type, time, 2, 2, false);
            var body = fixture.nodes.getFirst().scanBody().orElseThrow();
            for (GraphValue value : body.graph().values()) {
                TensorDescriptor d = value.descriptor();
                assertInvalidLengthFixture(withScanLocal(fixture, value.id(), new TensorDescriptor(type,
                        Shape.ofDimensions(new DynamicDimension("local")), Optional.empty(), d.requiresGrad())));
                assertInvalidLengthFixture(withScanLocal(fixture, value.id(),
                        scanDescriptor(type == DataType.FLOAT32 ? DataType.FLOAT64 : DataType.FLOAT32,
                                d.requiresGrad(), d.shape())));
            }
            for (ValueId id : body.graph().inputs()) {
                TensorDescriptor d = body.graph().values().get((int) id.value()).descriptor();
                assertInvalidLengthFixture(withScanLocal(fixture, id,
                        scanDescriptor(type, !d.requiresGrad(), d.shape())));
                assertInvalidLengthFixture(withScanLocal(fixture, id,
                        new TensorDescriptor(type, d.shape(), Optional.empty(), d.requiresGrad())));
                assertInvalidLengthFixture(withScanLocal(fixture, id,
                        scanDescriptor(type, d.requiresGrad(), Shape.of(4))));
            }
            for (ValueId id : fixture.nodes.getFirst().outputs()) {
                TensorDescriptor d = fixture.values.stream().filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor();
                for (TensorDescriptor bad : List.of(scanDescriptor(type, false, d.shape()),
                        new TensorDescriptor(type, d.shape(), Optional.empty(), true),
                        scanDescriptor(type, true, Shape.of(4)),
                        scanDescriptor(type == DataType.FLOAT32 ? DataType.FLOAT64 : DataType.FLOAT32, true, d.shape()))) {
                    assertInvalidLengthFixture(withScanOuter(fixture, id, bad));
                }
            }
        }
    }

    @Test
    void lengthModeTiedPositionsUseExplicitOuterAxesNotEqualSizes() {
        Fixture fixture = lengthFixture(DataType.FLOAT32, 3, 2, 2, false);
        var node = fixture.nodes.getFirst();
        var inputs = new ArrayList<>(node.inputs());
        inputs.set(4, inputs.get(3)); // Same shape, but ROW(1) and SHARED are not compatible.
        Fixture incompatible = withLengthNode(fixture, node.operation(), inputs);
        assertThrows(IllegalArgumentException.class,
                () -> context(incompatible, incompatible.values, incompatible.requirements, Map.of()));
        var a = (ScanAttrs) node.operation().attrs();
        var compatible = withLengthNode(fixture, new Operation(ScanKind.SCAN,
                new ScanAttrs(a.timeCount(), 2, 3, 1, 2, 5, false, true, 1, List.of(OptionalInt.of(0), OptionalInt.of(1), OptionalInt.of(1)))), inputs);
        var context = context(compatible, compatible.values, compatible.requirements, Map.of());
        assertEquals(2, context.partitionDag().consumers(inputs.get(3)).size());
        assertEquals(3, context.partitionDag().consumers(inputs.get(3)).get(0).inputPosition());
        assertEquals(4, context.partitionDag().consumers(inputs.get(3)).get(1).inputPosition());
    }

    @Test
    void lengthModeRejectsElementAndCanonicalStrideOverflowIncludingUnusedAndZeroBatch() {
        Fixture fixture = lengthFixture(DataType.FLOAT64, 0, 0, 2, false);
        TensorDescriptor huge = new TensorDescriptor(DataType.FLOAT64, Shape.of(Long.MAX_VALUE, 2), Optional.empty(), false);
        Fixture invalidLocal = withScanLocal(fixture, new ValueId(4), huge);
        assertThrows(ArithmeticException.class,
                () -> context(invalidLocal, invalidLocal.values, invalidLocal.requirements, Map.of()));
        Shape hugeRow = Shape.of(0, Long.MAX_VALUE, 2);
        Fixture invalidBoundary = withScanOuter(fixture, new ValueId(0),
                new TensorDescriptor(DataType.FLOAT64, hugeRow, Optional.empty(), false));
        assertThrows(ArithmeticException.class,
                () -> context(invalidBoundary, invalidBoundary.values, invalidBoundary.requirements, Map.of()));
        assertThrows(ArithmeticException.class,
                () -> new ScanAttrs(0, Integer.MAX_VALUE, 1, 0, 0, 0, false, true, 1, List.of(OptionalInt.of(0))));
    }

    @Test
    void bothScanModesCheckEveryFixedFeatureProductWithoutARankOrEmptyDomainShortcut() {
        var b = new DynamicDimension("B");
        Shape validRow = Shape.ofDimensions(b,
                new io.github.pho001.synaptik.model.shape.StaticDimension(2),
                new io.github.pho001.synaptik.model.shape.StaticDimension(3),
                new io.github.pho001.synaptik.model.shape.StaticDimension(1));
        Shape hugeRow = Shape.ofDimensions(b,
                new io.github.pho001.synaptik.model.shape.StaticDimension(Long.MAX_VALUE),
                new io.github.pho001.synaptik.model.shape.StaticDimension(2),
                new io.github.pho001.synaptik.model.shape.StaticDimension(1));
        for (boolean lengthMode : List.of(false, true)) for (long time : List.of(0L, 3L)) {
            Fixture fixture = carryFreeIdentityFixture(lengthMode, time, validRow);
            context(fixture, fixture.values, fixture.requirements, Map.of());
            Fixture overflowingIdentity = carryFreeIdentityFixture(lengthMode, time, hugeRow);
            assertThrows(ArithmeticException.class, () -> context(overflowingIdentity,
                    overflowingIdentity.values, overflowingIdentity.requirements, Map.of()));

            // Each mutation independently overflows, with every other declaration still valid.
            // The second local/actual is unused; no ordinary node can enforce this obligation.
            for (int position : List.of(0, 1)) {
                TensorDescriptor descriptor = new TensorDescriptor(DataType.FLOAT32,
                        position == 0 ? hugeRow : Shape.of(Long.MAX_VALUE, 2, 1), Optional.empty(), false);
                Fixture invalidLocal = withScanLocal(fixture, new ValueId(position), descriptor);
                assertThrows(ArithmeticException.class, () -> context(invalidLocal,
                        invalidLocal.values, invalidLocal.requirements, Map.of()));
                TensorDescriptor actual = new TensorDescriptor(DataType.FLOAT32,
                        position == 0 ? stacked(time, hugeRow) : descriptor.shape(), Optional.empty(), false);
                Fixture invalidActual = withScanOuter(fixture, new ValueId(position), actual);
                assertThrows(ArithmeticException.class, () -> context(invalidActual,
                        invalidActual.values, invalidActual.requirements, Map.of()));
            }
            Fixture invalidResult = withScanOuter(fixture, new ValueId(10),
                    new TensorDescriptor(DataType.FLOAT32, stacked(time, hugeRow), Optional.empty(), false));
            assertThrows(ArithmeticException.class, () -> context(invalidResult,
                    invalidResult.values, invalidResult.requirements, Map.of()));
        }
    }

    /**
     * Builds a high-rank named-B zero-node identity with an unused fixed invariant and no carries.
     * @param lengthMode whether to append the separate named-B lengths actual
     * @param time fixed temporal extent, including zero
     * @param row unresolved leading-B local shape; fixed factors may deliberately overflow
     * @return raw complete fixture whose descriptor products have not been evaluated
     */
    private static Fixture carryFreeIdentityFixture(boolean lengthMode, long time, Shape row) {
        var step = new TensorDescriptor(DataType.FLOAT32, row, Optional.empty(), false);
        var invariant = scanDescriptor(DataType.FLOAT32, false, Shape.of(1));
        var graph = new CompiledGraphModel(
                List.of(new GraphValue(new ValueId(0), step), new GraphValue(new ValueId(1), invariant)),
                List.of(), List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(0)), Map.of());
        var body = new CompiledScanBody(graph, List.of(), List.of(new ValueId(0)), List.of(), List.of(0));
        var values = new ArrayList<>(List.of(
                new GraphValue(new ValueId(0), new TensorDescriptor(DataType.FLOAT32,
                        stacked(time, row), Optional.empty(), false)),
                new GraphValue(new ValueId(1), invariant)));
        var inputs = new ArrayList<>(List.of(new ValueId(0), new ValueId(1)));
        if (lengthMode) {
            inputs.add(new ValueId(2));
            values.add(new GraphValue(new ValueId(2), new TensorDescriptor(DataType.INT64,
                    Shape.ofDimensions(row.dimension(0)), Optional.empty(), false)));
        }
        values.add(new GraphValue(new ValueId(10), new TensorDescriptor(DataType.FLOAT32,
                stacked(time, row), Optional.empty(), false)));
        ScanAttrs attrs = lengthMode
                ? new ScanAttrs(time, 0, 1, 1, 1, 0, false, true, 0, List.of(OptionalInt.of(0)))
                : new ScanAttrs(time, 0, 1, 1, 1, 0, false);
        var node = new CompiledNode(new NodeId(0), new Operation(ScanKind.SCAN, attrs),
                inputs, List.of(new ValueId(10)), Optional.of(body));
        var partition = new PlannedPartition(new BackendId("high-rank-identity"), List.of(node.id()));
        var requirements = values.stream().map(v -> new LogicalMemoryRequirement(v.id(), v.descriptor(),
                v.id().value() == 10 ? Optional.of(partition) : Optional.empty(),
                v.id().value() == 10 ? List.<PlannedPartition>of() : List.of(partition),
                v.id().value() == 10)).toList();
        return new Fixture(partition, List.of(node), values, requirements, Map.of(), new FakeInputs("identity"));
    }

    @Test
    void lengthModeDirectDemandAndLengthsLogicalUseRemainChecked() {
        Fixture fixture = lengthFixture(DataType.FLOAT64, 0, 0, 0, true);
        for (List<Integer> demand : List.of(List.<Integer>of(), List.of(0, 1), List.of(4))) {
            assertInvalidLengthFixture(withScanDemand(fixture, demand));
        }
        var requirements = new ArrayList<>(fixture.requirements);
        var lengths = requirements.get(6);
        requirements.set(6, new LogicalMemoryRequirement(lengths.valueId(), lengths.descriptor(),
                lengths.producerPartition(), List.of(), false));
        assertThrows(IllegalArgumentException.class,
                () -> context(fixture, fixture.values, requirements, Map.of()));
        var external = new PlannedPartition(new BackendId("external"), List.of(new NodeId(99)));
        var residual = requirements.get(11);
        requirements.set(6, lengths);
        requirements.set(11, new LogicalMemoryRequirement(residual.valueId(), residual.descriptor(),
                residual.producerPartition(), List.of(external), false));
        Fixture demanded = withScanDemand(fixture, List.of(0, 4));
        context(demanded, demanded.values, requirements, Map.of());
    }
    @Test
    void compatibleFixedScanConstructorStillHasNoBodyAndUsesOrdinaryOuterValidation() {
        var inputs = java.util.stream.LongStream.range(0, 5).mapToObj(ValueId::new).toList();
        var outputs = List.of(new ValueId(5), new ValueId(6));
        var node = new CompiledNode(new NodeId(0),
                new Operation(RecurrentScanKind.RNN_TANH, RecurrentDirection.FORWARD), inputs, outputs);
        var partition = new PlannedPartition(new BackendId("fixed"), List.of(node.id()));
        var descriptors = List.of(
                scanDescriptor(DataType.FLOAT32, false, Shape.of(3, 2, 4)),
                scanDescriptor(DataType.INT64, false, Shape.of(2)),
                scanDescriptor(DataType.FLOAT32, false, Shape.of(2, 5)),
                scanDescriptor(DataType.FLOAT32, false, Shape.of(5, 4)),
                scanDescriptor(DataType.FLOAT32, false, Shape.of(5, 5)),
                scanDescriptor(DataType.FLOAT32, false, Shape.of(3, 2, 5)),
                scanDescriptor(DataType.FLOAT32, false, Shape.of(2, 5)));
        var values = java.util.stream.IntStream.range(0, descriptors.size())
                .mapToObj(id -> new GraphValue(new ValueId(id), descriptors.get(id))).toList();
        var requirements = values.stream().map(v -> new LogicalMemoryRequirement(v.id(), v.descriptor(),
                v.id().value() < 5 ? Optional.empty() : Optional.of(partition),
                v.id().value() < 5 ? List.of(partition) : List.<PlannedPartition>of(), false)).toList();
        var context = new PrepareContext<>(partition, List.of(node), values, requirements, Map.of(), new FakeInputs("fixed"));
        assertSame(node, context.nodes().getFirst());
        assertTrue(context.nodes().getFirst().scanBody().isEmpty());
        assertEquals(values, context.values());
    }

    @Test
    void scanKeepsTwoCarriesTiedActualsRepeatedRootsAndSeparateDescriptorScopesAtZeroTime() {
        for (long time : List.of(0L, 3L)) {
            Fixture fixture = scanFixture(time);
            var context = context(fixture, fixture.values, fixture.requirements, Map.of());
            CompiledNode node = fixture.nodes.getFirst();
            CompiledScanBody body = node.scanBody().orElseThrow();
            assertSame(node, context.nodes().getFirst());
            assertSame(body, context.nodes().getFirst().scanBody().orElseThrow());
            assertSame(body.graph(), context.nodes().getFirst().scanBody().orElseThrow().graph());
            assertEquals(node.inputs().get(0), node.inputs().get(1));
            assertEquals(body.nextCarry().getFirst(), body.emissions().get(1));
            assertEquals(4, body.graph().inputs().size()); // Includes unused invariant.
            assertEquals(13, context.values().size()); // Only three unique actuals and ten ports.
            assertEquals(1, context.partitionDag().nodes().size());
            assertTrue(context.partitionDag().edges().isEmpty());
            assertEquals(List.of(0), body.demandedOutputIndices());
            assertEquals(Shape.of(time, 2), context.values().get(1).descriptor().shape());
            assertEquals(Shape.of(2), body.graph().values().get(1).descriptor().shape());
            assertEquals(false, body.graph().values().getFirst().descriptor().requiresGrad());
            assertTrue(context.values().get(5).descriptor().requiresGrad()); // emit(previous carry)
            assertEquals(DataType.INT64, context.values().getLast().descriptor().dataType());
            assertEquals(false, context.values().getLast().descriptor().requiresGrad());
        }
    }

    @Test
    void scanRejectsEveryDynamicLocalIncludingUnusedFormalAndUnusedSiblingEvenAtZeroTime() {
        for (long time : List.of(0L, 3L)) {
            Fixture fixture = scanFixture(time);
            var body = fixture.nodes.getFirst().scanBody().orElseThrow();
            for (GraphValue value : body.graph().values()) {
                var dynamic = new TensorDescriptor(value.descriptor().dataType(),
                        Shape.ofDimensions(new DynamicDimension("local")), Optional.empty(), false);
                Fixture invalid = withScanLocal(fixture, value.id(), dynamic);
                assertThrows(IllegalArgumentException.class,
                        () -> context(invalid, invalid.values, invalid.requirements, Map.of()));
            }
        }
    }

    @Test
    void matchingNamedBatchProjectsUnresolvedLocalsAndLatentPortsWithoutBackendGeometry() {
        var b = new DynamicDimension("B");
        for (long time : List.of(0L,3L)) {
            Fixture fixture = scanFixture(time);
            for (var value : fixture.nodes.getFirst().scanBody().orElseThrow().graph().values()) {
                if (!value.descriptor().shape().equals(Shape.of(2))) continue;
                fixture = withScanLocal(fixture,value.id(),new TensorDescriptor(value.descriptor().dataType(),
                        Shape.ofDimensions(b,new io.github.pho001.synaptik.model.shape.StaticDimension(2)),Optional.empty(),value.descriptor().requiresGrad()));
            }
            fixture = withScanLocal(fixture, new ValueId(5),
                    scanDescriptor(DataType.FLOAT32, true, Shape.of(1, 2)));
            fixture = withScanLocal(fixture, new ValueId(6),
                    scanDescriptor(DataType.INT64, false, Shape.of(1, 2)));
            var body = fixture.nodes.getFirst().scanBody().orElseThrow(); var graph = body.graph();
            fixture = withScanOuter(fixture,new ValueId(0),graph.values().getFirst().descriptor());
            fixture = withScanOuter(fixture,new ValueId(1),new TensorDescriptor(DataType.FLOAT32,
                    stacked(time,graph.values().get(2).descriptor().shape()),Optional.empty(),true));
            var roles = new ArrayList<>(body.nextCarry()); roles.addAll(body.emissions()); roles.addAll(body.residuals());
            var outputs = fixture.nodes.getFirst().outputs();
            for (int p = 0; p < outputs.size(); p++) {
                var id = roles.get(p); var local = graph.values().stream().filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor();
                Shape shape = p < 2 ? local.shape() : stacked(time,local.shape());
                fixture = withScanOuter(fixture,outputs.get(p),new TensorDescriptor(local.dataType(),shape,
                        shape.isFullyStatic() ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(),local.dataType().isDifferentiable()));
            }
            var context = context(fixture,fixture.values,fixture.requirements,Map.of());
            assertTrue(context.requiresBatchSize()); assertSame(body,context.nodes().getFirst().scanBody().orElseThrow());
            assertEquals(13,context.values().size());
            assertTrue(context.nodes().getFirst().scanBody().orElseThrow().graph().values().getFirst().descriptor().layout().isEmpty());
        }
    }

    @Test
    void scanRejectsFormalTypeShapeGradientAndCanonicalLayoutMismatchesForEveryPosition() {
        for (long time : List.of(0L, 3L)) {
            Fixture fixture = scanFixture(time);
            var body = fixture.nodes.getFirst().scanBody().orElseThrow();
            for (ValueId id : body.graph().inputs()) {
                TensorDescriptor formal = body.graph().values().stream()
                        .filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor();
                var alternatives = List.of(
                        scanDescriptor(DataType.FLOAT64, formal.requiresGrad(), formal.shape()),
                        scanDescriptor(formal.dataType(), formal.requiresGrad(), Shape.of(4)),
                        scanDescriptor(formal.dataType(), !formal.requiresGrad(), formal.shape()),
                        new TensorDescriptor(formal.dataType(), formal.shape(), Optional.empty(), formal.requiresGrad()));
                for (TensorDescriptor alternative : alternatives) {
                    Fixture invalid = withScanLocal(fixture, id, alternative);
                    assertThrows(IllegalArgumentException.class,
                            () -> context(invalid, invalid.values, invalid.requirements, Map.of()));
                }
            }
        }
    }

    @Test
    void scanRejectsTimeNextCarryAndEveryOuterRoleDescriptorWithoutRewritingLocalFlags() {
        for (long time : List.of(0L, 3L)) {
            Fixture fixture = scanFixture(time);
            for (TensorDescriptor badSequence : List.of(
                    scanDescriptor(DataType.FLOAT32, true, Shape.of(time + 1, 2)),
                    scanDescriptor(DataType.FLOAT32, true, Shape.scalar()))) {
                Fixture invalid = withScanOuter(fixture, fixture.values.get(1).id(), badSequence);
                assertThrows(IllegalArgumentException.class,
                        () -> context(invalid, invalid.values, invalid.requirements, Map.of()));
            }
            for (TensorDescriptor badCarry : List.of(
                    scanDescriptor(DataType.FLOAT64, true, Shape.of(2)),
                    scanDescriptor(DataType.FLOAT32, true, Shape.of(4)))) {
                Fixture invalid = withScanLocal(fixture, new ValueId(4), badCarry);
                assertThrows(IllegalArgumentException.class,
                        () -> context(invalid, invalid.values, invalid.requirements, Map.of()));
            }
            for (GraphValue value : fixture.values.subList(3, fixture.values.size())) {
                TensorDescriptor descriptor = value.descriptor();
                var alternatives = new ArrayList<>(List.of(
                        scanDescriptor(DataType.FLOAT64, descriptor.requiresGrad(), descriptor.shape()),
                        scanDescriptor(descriptor.dataType(), descriptor.requiresGrad(), Shape.of(4)),
                        new TensorDescriptor(descriptor.dataType(), descriptor.shape(), Optional.empty(), descriptor.requiresGrad())));
                if (descriptor.dataType().isDifferentiable()) {
                    alternatives.add(scanDescriptor(descriptor.dataType(), false, descriptor.shape()));
                }
                for (TensorDescriptor alternative : alternatives) {
                    Fixture invalid = withScanOuter(fixture, value.id(), alternative);
                    assertThrows(IllegalArgumentException.class,
                            () -> context(invalid, invalid.values, invalid.requirements, Map.of()));
                }
            }
        }
    }

    @Test
    void directScanContextChecksDemandAndOnlyLocallyKnowableRequirementConsistency() {
        Fixture fixture = scanFixture(0);
        for (List<Integer> demand : List.of(List.<Integer>of(), List.of(0, 1), List.of(2))) {
            Fixture invalid = withScanDemand(fixture, demand);
            assertThrows(IllegalArgumentException.class,
                    () -> context(invalid, invalid.values, invalid.requirements, Map.of()));
        }
        // A direct context can trust an external consumer obligation but cannot prove that edge.
        var external = new PlannedPartition(new BackendId("other"), List.of(new NodeId(99)));
        var requirements = new ArrayList<>(fixture.requirements);
        var previous = requirements.get(4);
        requirements.set(4, new LogicalMemoryRequirement(previous.valueId(), previous.descriptor(),
                previous.producerPartition(), List.of(external), false));
        Fixture demanded = withScanDemand(fixture, List.of(0, 1));
        context(demanded, demanded.values, requirements, Map.of());
        requirements.set(4, new LogicalMemoryRequirement(previous.valueId(), previous.descriptor(),
                previous.producerPartition(), List.of(fixture.partition), false));
        assertThrows(IllegalArgumentException.class,
                () -> context(demanded, demanded.values, requirements, Map.of()));
        requirements.set(4, new LogicalMemoryRequirement(previous.valueId(), previous.descriptor(),
                Optional.of(external), List.of(), false));
        assertThrows(IllegalArgumentException.class,
                () -> context(fixture, fixture.values, requirements, Map.of()));
        var inputRequirements = new ArrayList<>(fixture.requirements);
        var input = inputRequirements.getFirst();
        for (LogicalMemoryRequirement invalid : List.of(
                new LogicalMemoryRequirement(input.valueId(), input.descriptor(), Optional.empty(), List.of(), false),
                new LogicalMemoryRequirement(input.valueId(), input.descriptor(), Optional.of(fixture.partition), List.of(fixture.partition), false))) {
            inputRequirements.set(0, invalid);
            assertThrows(IllegalArgumentException.class,
                    () -> context(fixture, fixture.values, inputRequirements, Map.of()));
        }
    }

    @Test
    void malformedScanStructureRejectsAtExistingModelConstructors() {
        Fixture fixture = scanFixture(0);
        CompiledNode node = fixture.nodes.getFirst();
        CompiledScanBody body = node.scanBody().orElseThrow();
        assertThrows(IllegalArgumentException.class,
                () -> new CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs()));
        assertThrows(IllegalArgumentException.class,
                () -> new CompiledNode(node.id(), new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE),
                        node.inputs(), node.outputs(), Optional.of(body)));
        assertThrows(IllegalArgumentException.class,
                () -> new CompiledNode(node.id(), new Operation(ScanKind.SCAN,
                        new ScanAttrs(0, 1, 2, 1, 3, 5, false)), node.inputs(), node.outputs(), Optional.of(body)));
        assertThrows(IllegalArgumentException.class,
                () -> new CompiledScanBody(body.graph(), List.of(), List.of(), body.residuals(), List.of()));
        assertThrows(IllegalArgumentException.class,
                () -> new CompiledScanBody(body.graph(), body.nextCarry(), List.of(), List.of(), List.of()));
        for (List<Integer> demand : List.of(List.of(-1), List.of(10), List.of(1, 0), List.of(0, 0))) {
            assertThrows(IllegalArgumentException.class,
                    () -> new CompiledScanBody(body.graph(), body.nextCarry(), body.emissions(), body.residuals(), demand));
        }
    }

    private static Fixture lengthFixture(DataType type, long time, long batch, long features, boolean reverse) {
        var row = scanDescriptor(type, false, Shape.of(batch, features));
        var shared = scanDescriptor(type, false, Shape.of(features));
        var step = scanDescriptor(type, true, Shape.of(batch, features));
        var transposed = scanDescriptor(type, false, Shape.of(features, batch));
        var local = List.of(new GraphValue(new ValueId(0), row), new GraphValue(new ValueId(1), shared),
                new GraphValue(new ValueId(2), step), new GraphValue(new ValueId(3), transposed),
                new GraphValue(new ValueId(4), transposed), new GraphValue(new ValueId(5), shared),
                new GraphValue(new ValueId(6), step), new GraphValue(new ValueId(7), shared),
                new GraphValue(new ValueId(8), step));
        var addRow = new CompiledNode(new NodeId(0), new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0), new ValueId(2)), List.of(new ValueId(6)));
        var addShared = new CompiledNode(new NodeId(1), new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(1), new ValueId(5)), List.of(new ValueId(7)));
        var neg = new CompiledNode(new NodeId(2), new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(6)), List.of(new ValueId(8)));
        var formals = java.util.stream.LongStream.range(0, 6).mapToObj(ValueId::new).toList();
        var graph = new CompiledGraphModel(local, List.of(addRow, addShared, neg), formals,
                List.of(new ValueId(0), new ValueId(1), new ValueId(6), new ValueId(7), new ValueId(8)),
                Map.of(addRow.id(), GraphPhase.FORWARD, addShared.id(), GraphPhase.FORWARD, neg.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(new ValueId(6), new ValueId(7)),
                List.of(new ValueId(6), new ValueId(0)),
                List.of(new ValueId(0), new ValueId(6), new ValueId(1), new ValueId(7), new ValueId(8)), List.of(0));
        var outputs = java.util.stream.LongStream.range(10, 19).mapToObj(ValueId::new).toList();
        var node = new CompiledNode(new NodeId(0), new Operation(ScanKind.SCAN,
                new ScanAttrs(time, 2, 3, 1, 2, 5, reverse, true, 1, List.of(OptionalInt.of(0), OptionalInt.of(1), OptionalInt.empty()))),
                java.util.stream.LongStream.range(0, 7).mapToObj(ValueId::new).toList(), outputs, Optional.of(body));
        var partition = new PlannedPartition(new BackendId("length"), List.of(node.id()));
        var values = new ArrayList<GraphValue>();
        for (int i = 0; i < formals.size(); i++) {
            TensorDescriptor source = local.get(i).descriptor();
            Shape shape = i >= 2 && i < 5 ? stacked(time, source.shape()) : source.shape();
            values.add(new GraphValue(new ValueId(i), scanDescriptor(type, source.requiresGrad(), shape)));
        }
        values.add(new GraphValue(new ValueId(6), scanDescriptor(DataType.INT64, false, Shape.of(batch))));
        var roles = new ArrayList<>(body.nextCarry()); roles.addAll(body.emissions()); roles.addAll(body.residuals());
        for (int i = 0; i < roles.size(); i++) {
            Shape shape = local.get((int) roles.get(i).value()).descriptor().shape();
            values.add(new GraphValue(outputs.get(i), scanDescriptor(type, true, i < 2 ? shape : stacked(time, shape))));
        }
        var requirements = values.stream().map(v -> new LogicalMemoryRequirement(v.id(), v.descriptor(),
                outputs.contains(v.id()) ? Optional.of(partition) : Optional.empty(),
                outputs.contains(v.id()) ? List.<PlannedPartition>of() : List.of(partition),
                v.id().equals(outputs.getFirst()))).toList();
        return new Fixture(partition, List.of(node), values, requirements, Map.of(), new FakeInputs("length"));
    }

    private static Shape stacked(long time, Shape local) {
        var dimensions = new ArrayList<io.github.pho001.synaptik.model.shape.Dimension>();
        dimensions.add(new io.github.pho001.synaptik.model.shape.StaticDimension(time));
        dimensions.addAll(local.dimensions());
        return Shape.ofDimensions(dimensions.toArray(io.github.pho001.synaptik.model.shape.Dimension[]::new));
    }

    private static Fixture withLengthFormalAndActual(Fixture fixture, int position, Shape shape, long time) {
        var descriptor = fixture.values.get(position).descriptor();
        Fixture local = withScanLocal(fixture, new ValueId(position), scanDescriptor(descriptor.dataType(), descriptor.requiresGrad(), shape));
        Shape actual = position >= 2 && position < 5 ? stacked(time, shape) : shape;
        return withScanOuter(local, new ValueId(position), scanDescriptor(descriptor.dataType(), descriptor.requiresGrad(), actual));
    }

    private static Fixture withLengthNode(Fixture fixture, Operation operation, List<ValueId> inputs) {
        var node = fixture.nodes.getFirst();
        var changed = new CompiledNode(node.id(), operation, inputs, node.outputs(), node.scanBody());
        return new Fixture(fixture.partition, List.of(changed), fixture.values, fixture.requirements, Map.of(), fixture.inputs);
    }

    private static void assertInvalidLengthFixture(Fixture fixture) {
        assertThrows(IllegalArgumentException.class,
                () -> context(fixture, fixture.values, fixture.requirements, Map.of()));
    }

    private static Fixture scanFixture(long time) {
        var carry = scanDescriptor(DataType.FLOAT32, false, Shape.of(2));
        var step = scanDescriptor(DataType.FLOAT32, true, Shape.of(2));
        var invariant = scanDescriptor(DataType.FLOAT32, false, Shape.of(3));
        var local = List.of(new GraphValue(new ValueId(0), carry), new GraphValue(new ValueId(1), carry),
                new GraphValue(new ValueId(2), step), new GraphValue(new ValueId(3), invariant),
                new GraphValue(new ValueId(4), step),
                new GraphValue(new ValueId(5), scanDescriptor(DataType.FLOAT32, true, Shape.of(1))),
                new GraphValue(new ValueId(6), scanDescriptor(DataType.INT64, false, Shape.of(1))));
        var add = new CompiledNode(new NodeId(0), new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0), new ValueId(2)), List.of(new ValueId(4)));
        var top = new CompiledNode(new NodeId(1), new Operation(TopKKind.TOP_K, new TopKAttrs(0, 1, true, true)),
                List.of(new ValueId(2)), List.of(new ValueId(5), new ValueId(6)));
        var graph = new CompiledGraphModel(local, List.of(add, top),
                List.of(new ValueId(0), new ValueId(1), new ValueId(2), new ValueId(3)),
                List.of(new ValueId(0), new ValueId(1), new ValueId(4), new ValueId(5), new ValueId(6)),
                Map.of(add.id(), GraphPhase.FORWARD, top.id(), GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph, List.of(new ValueId(4), new ValueId(1)),
                List.of(new ValueId(0), new ValueId(4), new ValueId(5)),
                List.of(new ValueId(0), new ValueId(4), new ValueId(1), new ValueId(5), new ValueId(6)), List.of(0));
        var outputs = java.util.stream.LongStream.range(10, 20).mapToObj(ValueId::new).toList();
        var node = new CompiledNode(new NodeId(0), new Operation(ScanKind.SCAN, new ScanAttrs(time, 2, 1, 1, 3, 5, false)),
                List.of(new ValueId(0), new ValueId(0), new ValueId(1), new ValueId(2)), outputs, Optional.of(body));
        var partition = new PlannedPartition(new BackendId("scan"), List.of(node.id()));
        var values = new ArrayList<>(List.of(new GraphValue(new ValueId(0), carry),
                new GraphValue(new ValueId(1), scanDescriptor(DataType.FLOAT32, true, Shape.of(time, 2))),
                new GraphValue(new ValueId(2), invariant)));
        var roles = new ArrayList<>(body.nextCarry()); roles.addAll(body.emissions()); roles.addAll(body.residuals());
        for (int i = 0; i < roles.size(); i++) {
            TensorDescriptor source = local.get((int) roles.get(i).value()).descriptor();
            Shape shape = i < 2 ? source.shape() : Shape.of(time, source.shape().dimension(0).staticSize().orElseThrow());
            values.add(new GraphValue(outputs.get(i), scanDescriptor(source.dataType(), source.dataType().isDifferentiable(), shape)));
        }
        var requirements = values.stream().map(v -> new LogicalMemoryRequirement(v.id(), v.descriptor(),
                outputs.contains(v.id()) ? Optional.of(partition) : Optional.empty(),
                outputs.contains(v.id()) ? List.<PlannedPartition>of() : List.of(partition), v.id().equals(outputs.getFirst()))).toList();
        return new Fixture(partition, List.of(node), values, requirements, Map.of(), new FakeInputs("scan"));
    }

    private static TensorDescriptor scanDescriptor(DataType type, boolean gradient, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), gradient);
    }

    private static Fixture withScanLocal(Fixture fixture, ValueId id, TensorDescriptor descriptor) {
        var node = fixture.nodes.getFirst();
        var body = node.scanBody().orElseThrow();
        var graph = body.graph();
        var values = graph.values().stream().map(v -> v.id().equals(id) ? new GraphValue(id, descriptor) : v).toList();
        var replacement = new CompiledScanBody(new CompiledGraphModel(values, graph.nodes(), graph.inputs(), graph.outputs(), graph.nodePhases()),
                body.nextCarry(), body.emissions(), body.residuals(), body.demandedOutputIndices());
        return withScanBody(fixture, replacement);
    }

    private static Fixture withScanDemand(Fixture fixture, List<Integer> demand) {
        var body = fixture.nodes.getFirst().scanBody().orElseThrow();
        return withScanBody(fixture, new CompiledScanBody(body.graph(), body.nextCarry(), body.emissions(), body.residuals(), demand));
    }

    private static Fixture withScanBody(Fixture fixture, CompiledScanBody body) {
        var node = fixture.nodes.getFirst();
        var replacement = new CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs(), Optional.of(body));
        return new Fixture(fixture.partition, List.of(replacement), fixture.values, fixture.requirements, Map.of(), fixture.inputs);
    }

    private static Fixture withScanOuter(Fixture fixture, ValueId id, TensorDescriptor descriptor) {
        var values = fixture.values.stream().map(v -> v.id().equals(id) ? new GraphValue(id, descriptor) : v).toList();
        var requirements = fixture.requirements.stream().map(r -> r.valueId().equals(id)
                ? new LogicalMemoryRequirement(id, descriptor, r.producerPartition(), r.consumerPartitions(), r.graphOutput()) : r).toList();
        return new Fixture(fixture.partition, fixture.nodes, values, requirements, Map.of(), fixture.inputs);
    }

    @Test
    void rejectsNullComponentsAndInvalidNodeOrder() {
        Fixture fixture = fixture();
        assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>((PartitionDag) null, fixture.values, fixture.requirements, fixture.constants, fixture.inputs));
        List<CompiledNode> nodesWithNull =
                new ArrayList<>(Arrays.asList(fixture.nodes.getFirst(), null, null));

        assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>((PlannedPartition) null, null, null, null, null, null));
        assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>(fixture.partition, null, null, null, null, null));
        assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>(fixture.partition, fixture.nodes, null, null, null, null));
        assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>(fixture.partition, fixture.nodes, fixture.values, null, null, null));
        assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>(fixture.partition, fixture.nodes, fixture.values, fixture.requirements, null, null));
        assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>(fixture.partition, fixture.nodes, fixture.values, fixture.requirements, fixture.constants, null));
        assertThrows(
                NullPointerException.class,
                () -> new PrepareContext<>(fixture.partition, nodesWithNull, fixture.values, fixture.requirements, fixture.constants, fixture.inputs));
        assertThrows(
                IllegalArgumentException.class,
                () -> new PrepareContext<>(fixture.partition, List.of(fixture.nodes.getFirst()), fixture.values, fixture.requirements, fixture.constants, fixture.inputs));
        assertThrows(
                IllegalArgumentException.class,
                () -> new PrepareContext<>(fixture.partition, List.of(fixture.nodes.get(1), fixture.nodes.get(0)), fixture.values, fixture.requirements, fixture.constants, fixture.inputs));
    }

    @Test
    void rejectsMalformedValueAndRequirementProjections() {
        Fixture fixture = fixture();
        List<GraphValue> valuesWithNull =
                new ArrayList<>(Arrays.asList(fixture.values.getFirst(), null, null));
        GraphValue duplicate = new GraphValue(
                fixture.values.getFirst().id(), fixture.values.getFirst().descriptor());
        GraphValue dynamic = new GraphValue(
                new ValueId(9),
                new TensorDescriptor(
                        DataType.FLOAT32,
                        Shape.ofDimensions(new DynamicDimension("N"),new DynamicDimension("N")),
                        Optional.empty(),
                        false));
        List<LogicalMemoryRequirement> requirementsWithNull =
                new ArrayList<>(Arrays.asList(fixture.requirements.getFirst(), null, null));
        LogicalMemoryRequirement duplicateRequirement = new LogicalMemoryRequirement(
                fixture.requirements.getFirst().valueId(),
                fixture.requirements.getFirst().descriptor(),
                Optional.empty(),
                List.of(fixture.partition),
                false);

        assertThrows(
                NullPointerException.class,
                () -> context(fixture, valuesWithNull, fixture.requirements, fixture.constants));
        assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        List.of(fixture.values.getFirst(), duplicate),
                        List.of(
                                fixture.requirements.getFirst(),
                                fixture.requirements.getFirst()),
                        Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        List.of(dynamic),
                        List.of(requirement(dynamic, Optional.empty())),
                        Map.of()));
        assertThrows(
                NullPointerException.class,
                () -> context(fixture, fixture.values, requirementsWithNull, fixture.constants));
        assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        fixture.values,
                        List.of(fixture.requirements.getFirst(), duplicateRequirement),
                        Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        fixture.values.subList(1, fixture.values.size()),
                        fixture.requirements.subList(1, fixture.requirements.size()),
                        Map.of()));
    }

    @Test
    void validatesOneMatchingRequirementAndExactTypedGraphInputConstants() {
        Fixture fixture = fixture();
        LogicalMemoryRequirement mismatched = new LogicalMemoryRequirement(
                fixture.values.getFirst().id(),
                descriptor(DataType.FLOAT64),
                Optional.empty(),
                List.of(fixture.partition),
                false);
        LogicalMemoryRequirement extra = requirement(
                new GraphValue(new ValueId(20), descriptor(DataType.FLOAT32)),
                Optional.empty());
        Map<ValueId, ScalarValue> absentConstant =
                Map.of(new ValueId(50), ScalarValue.float32(1.0f));
        Map<ValueId, ScalarValue> producedConstant =
                Map.of(fixture.values.get(1).id(), ScalarValue.float32(1.0f));
        Map<ValueId, ScalarValue> wrongType =
                Map.of(fixture.values.getFirst().id(), ScalarValue.float64(1.0));

        assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        fixture.values,
                        fixture.requirements.subList(0, fixture.requirements.size() - 1),
                        Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        fixture.values,
                        List.of(
                                mismatched,
                                fixture.requirements.get(1),
                                fixture.requirements.get(2)),
                        Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> context(
                        fixture,
                        fixture.values,
                        List.of(
                                fixture.requirements.get(0),
                                fixture.requirements.get(1),
                                fixture.requirements.get(2),
                                extra),
                        Map.of()));
        assertThrows(
                IllegalArgumentException.class,
                () -> context(fixture, fixture.values, fixture.requirements, absentConstant));
        assertThrows(
                IllegalArgumentException.class,
                () -> context(fixture, fixture.values, fixture.requirements, producedConstant));
        assertThrows(
                IllegalArgumentException.class,
                () -> context(fixture, fixture.values, fixture.requirements, wrongType));
    }

    @Test
    void snapshotsCollectionsRetainsExactReferencesAndPreservesEncounterOrder() {
        Fixture fixture = fixture();
        List<CompiledNode> suppliedNodes = new ArrayList<>(fixture.nodes);
        List<GraphValue> suppliedValues = new ArrayList<>(fixture.values);
        List<LogicalMemoryRequirement> suppliedRequirements =
                new ArrayList<>(fixture.requirements);
        Map<ValueId, ScalarValue> suppliedConstants = new LinkedHashMap<>();
        ScalarValue firstConstant = fixture.constants.values().iterator().next();
        suppliedConstants.put(fixture.values.getFirst().id(), firstConstant);

        PrepareContext<FakeInputs> context = new PrepareContext<>(fixture.partition, suppliedNodes, suppliedValues, suppliedRequirements, suppliedConstants, fixture.inputs);
        PartitionDag partitionDag = context.partitionDag();
        PrepareContext<FakeInputs> canonical = new PrepareContext<>(partitionDag, fixture.values, fixture.requirements, fixture.constants, fixture.inputs);
        suppliedNodes.clear();
        suppliedValues.clear();
        suppliedRequirements.clear();
        suppliedConstants.clear();

        assertAll(
                () -> assertSame(fixture.partition, context.partition()),
                () -> assertSame(partitionDag.nodes(), context.nodes()),
                () -> assertSame(partitionDag, canonical.partitionDag()),
                () -> assertEquals(context, canonical),
                () -> assertEquals(context.hashCode(), canonical.hashCode()),
                () -> assertEquals(fixture.nodes, context.nodes()),
                () -> assertSame(fixture.nodes.getFirst(), context.nodes().getFirst()),
                () -> assertNotSame(suppliedNodes, context.nodes()),
                () -> assertEquals(fixture.values, context.values()),
                () -> assertSame(fixture.values.getFirst(), context.values().getFirst()),
                () -> assertEquals(fixture.requirements, context.memoryRequirements()),
                () -> assertSame(
                        fixture.requirements.getFirst(),
                        context.memoryRequirements().getFirst()),
                () -> assertEquals(
                        List.of(fixture.values.getFirst().id()),
                        context.constants().keySet().stream().toList()),
                () -> assertSame(firstConstant, context.constants().values().iterator().next()),
                () -> assertSame(fixture.inputs, context.backendInputs()),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> context.nodes().clear()),
                () -> assertThrows(
                        UnsupportedOperationException.class, () -> context.values().clear()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> context.memoryRequirements().clear()),
                () -> assertThrows(
                        UnsupportedOperationException.class,
                        () -> context.constants().clear()));
    }

    private static PrepareContext<FakeInputs> context(
            Fixture fixture,
            List<GraphValue> values,
            List<LogicalMemoryRequirement> requirements,
            Map<ValueId, ScalarValue> constants) {
        return new PrepareContext<>(fixture.partition, fixture.nodes, values, requirements, constants, fixture.inputs);
    }

    private static Fixture fixture() {
        ValueId input = new ValueId(0);
        ValueId middle = new ValueId(1);
        ValueId output = new ValueId(2);
        CompiledNode first = node(10, List.of(input), List.of(middle));
        CompiledNode second = node(11, List.of(middle), List.of(output));
        PlannedPartition partition = new PlannedPartition(
                new BackendId("cpu"), List.of(first.id(), second.id()));
        List<GraphValue> values = List.of(
                new GraphValue(input, descriptor(DataType.FLOAT32)),
                new GraphValue(middle, descriptor(DataType.FLOAT32)),
                new GraphValue(output, descriptor(DataType.FLOAT32)));
        List<LogicalMemoryRequirement> requirements = List.of(
                new LogicalMemoryRequirement(
                        input,
                        values.get(0).descriptor(),
                        Optional.empty(),
                        List.of(partition),
                        false),
                new LogicalMemoryRequirement(
                        middle,
                        values.get(1).descriptor(),
                        Optional.of(partition),
                        List.of(partition),
                        false),
                new LogicalMemoryRequirement(
                        output,
                        values.get(2).descriptor(),
                        Optional.of(partition),
                        List.of(),
                        true));
        ScalarValue constant = ScalarValue.float32(3.0f);
        return new Fixture(
                partition,
                List.of(first, second),
                values,
                requirements,
                Map.of(input, constant),
                new FakeInputs("cpu-v1"));
    }

    private static LogicalMemoryRequirement requirement(
            GraphValue value, Optional<PlannedPartition> producer) {
        return new LogicalMemoryRequirement(
                value.id(), value.descriptor(), producer, List.of(), false);
    }

    private static CompiledNode node(long id, List<ValueId> inputs, List<ValueId> outputs) {
        return new CompiledNode(
                new NodeId(id),
                new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE),
                inputs,
                outputs);
    }

    private static TensorDescriptor descriptor(DataType dataType) {
        return new TensorDescriptor(dataType, Shape.of(2, 3), Optional.empty(), false);
    }

    private record FakeInputs(String target) implements BackendAnalysisInputs {}

    private record Fixture(
            PlannedPartition partition,
            List<CompiledNode> nodes,
            List<GraphValue> values,
            List<LogicalMemoryRequirement> requirements,
            Map<ValueId, ScalarValue> constants,
            FakeInputs inputs) {}

    private enum SampleKind implements OperationKind {
        SAMPLE;

        private static final List<OperationSignature> SIGNATURES = List.of(
                new OperationSignature(
                        NoOperationAttrs.class, 0, Integer.MAX_VALUE, 1, Integer.MAX_VALUE));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }
}
