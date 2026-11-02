package io.github.pho001.synaptik.prepare;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.BackendAvailabilitySnapshot;
import io.github.pho001.synaptik.backend.contract.BackendDeviceId;
import io.github.pho001.synaptik.backend.contract.DeviceClass;
import io.github.pho001.synaptik.compiler.CompileArtifacts;
import io.github.pho001.synaptik.compiler.CompileConstantPlan;
import io.github.pho001.synaptik.compiler.CompileDiagnostics;
import io.github.pho001.synaptik.compiler.DerivativeGraphMetadata;
import io.github.pho001.synaptik.compiler.GradientPublicationBinding;
import io.github.pho001.synaptik.compiler.FunctionalGradientRequest;
import io.github.pho001.synaptik.compiler.PublicationPlan;
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.config.compile.CompileMode;
import io.github.pho001.synaptik.config.compile.GraphOptimizationConfig;
import io.github.pho001.synaptik.config.compile.BackendIntent;
import io.github.pho001.synaptik.config.compile.PartitionScoringConfig;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.CompiledScanBody;
import io.github.pho001.synaptik.model.graph.ForwardPublicationBinding;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.operation.scan.ScanAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.ScanBody;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlan;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryPlanning;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.BackendPartitionAnalysis;
import io.github.pho001.synaptik.prepare.analysis.BackendPreparationPlan;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.execution.BoundInvocation;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.execution.PreparedExecution;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan;
import io.github.pho001.synaptik.runtime.resource.PreparedResource;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CallerInput;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.CreatedBuffer;
import io.github.pho001.synaptik.runtime.resource.PreparedRepresentationPlan.InitializedBuffer;
import io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation;
import io.github.pho001.synaptik.runtime.run.PreparedPublication;
import io.github.pho001.synaptik.runtime.run.RunState;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.ExecutionStep;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.PublicationStep;
import io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.RepresentationCreationStep;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.Set;
import java.util.HashSet;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class GraphPreparationTest {
    @Test
    void rejectsMissingDuplicateAndForeignCallerContributionsBeforeAnalysis() {
        Shape shape = Shape.of(1);
        var fixed = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        CompileArtifacts artifacts = callerSourceArtifacts(List.of(fixed, fixed, fixed), null);
        List<ProducerlessPublishedValueResource> complete = sourceContributions(artifacts, Map.of());
        ProducerlessPublishedValueResource caller = complete.stream()
                .filter(resource -> resource.value().id().equals(new ValueId(90))).findFirst().orElseThrow();
        var duplicate = new ArrayList<>(complete);
        duplicate.add(caller);
        var foreignValue = new ArrayList<>(complete);
        foreignValue.set(1, new ProducerlessPublishedValueResource(
                new GraphValue(caller.value().id(), caller.value().descriptor()),
                caller.logicalRequirement(), caller.owner(), caller.size(), caller.byteAlignment()));
        var logical = caller.logicalRequirement();
        var foreignRequirement = new ArrayList<>(complete);
        foreignRequirement.set(1, new ProducerlessPublishedValueResource(caller.value(),
                new LogicalMemoryRequirement(logical.valueId(), logical.descriptor(),
                        logical.producerPartition(), logical.consumerPartitions(), logical.graphOutput()),
                caller.owner(), caller.size(), caller.byteAlignment()));
        var extra = new ArrayList<>(complete);
        GraphValue foreign = new GraphValue(new ValueId(999), fixed);
        extra.add(new ProducerlessPublishedValueResource(foreign,
                new LogicalMemoryRequirement(foreign.id(), fixed, Optional.empty(), List.of(), true),
                caller.owner(), caller.size(), caller.byteAlignment()));
        for (List<ProducerlessPublishedValueResource> invalid : List.of(
                List.<ProducerlessPublishedValueResource>of(),
                complete.subList(0, complete.size() - 1), duplicate, foreignValue, foreignRequirement, extra)) {
            AtomicInteger analyses = new AtomicInteger();
            assertThrows(IllegalArgumentException.class, () -> GraphPreparation.prepare(
                    artifacts, List.of(callerSourcePreparation(artifacts, analyses, List.of())),
                    invalid, sourceAssembler(artifacts, Map.of())));
            assertEquals(0, analyses.get());
        }
        AtomicInteger analyses = new AtomicInteger();
        PreparedScheduleAssembler wrongReference = new PreparedScheduleAssembler() {
            @Override public ProducerlessPublishedValueResource producerlessPublishedValue(
                    GraphValue value, LogicalMemoryRequirement requirement, Optional<ScalarValue> scalar) {
                return value == caller.value() ? foreignValue.get(1) : complete.stream()
                        .filter(resource -> resource.value() == value).findFirst().orElseThrow();
            }
            @Override public PreparedSchedule assemble(PreparedScheduleContext context) {
                throw new AssertionError("invalid contribution reached schedule assembly");
            }
        };
        assertThrows(IllegalArgumentException.class, () -> GraphPreparation.prepare(
                artifacts, List.of(callerSourcePreparation(artifacts, analyses, List.of())), wrongReference));
        assertEquals(0, analyses.get());
    }

    @Test
    void preservesRawCallerAffineBroadcastAndEmptySpanGeometryWithoutArithmeticAdmission() {
        for (DataType type : DataType.values()) {
            Shape pair = Shape.of(2);
            Shape empty = Shape.of(0);
            var strided = new TensorDescriptor(type, pair,
                    Optional.of(LayoutDescriptor.of(pair, new long[] {3}, 5, true)), false);
            var broadcast = new TensorDescriptor(type, pair,
                    Optional.of(LayoutDescriptor.of(pair, new long[] {0}, 7, true)), false);
            var noElements = new TensorDescriptor(type, empty,
                    Optional.of(LayoutDescriptor.of(empty, new long[] {3}, 11, true)), false);
            CompileArtifacts artifacts = callerSourceArtifacts(List.of(strided, broadcast, noElements), null);
            PreparedExecution execution = GraphPreparation.prepare(
                    artifacts,
                    List.of(callerSourcePreparation(artifacts, new AtomicInteger(), List.of())),
                    sourceAssembler(artifacts, Map.of()));
            assertEquals(9L * type.byteWidth(), execution.memoryPlan().buffers().get(0).byteSize());
            assertEquals(8L * type.byteWidth(), execution.memoryPlan().buffers().get(2).byteSize());
            assertEquals(0, execution.memoryPlan().buffers().get(4).byteSize());
            assertTrue(!execution.memoryPlan().requiresBatchSize());
            execution.close();
        }
    }

    @Test
    void lengthForwardProjectsSharedCarrySuffixMixedSequencesAndReusedTemplateScopes() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (long time : List.of(0L, 3L)) {
            for (long batch : List.of(0L, 1L, 2L)) for (long features : List.of(0L, 2L)) {
                Tensor h = scanTensor(type, false, batch, features), old = scanTensor(type, false, features);
                Tensor x = scanTensor(type, true, batch, features), transposed = scanTensor(type, false, features, batch);
                Tensor shared = scanTensor(type, false, features, batch), w = scanTensor(type, false, features);
                Tensor next = h.add(x).add(transposed.permute(1, 0));
                var body = new ScanBody(List.of(h, old), List.of(x, transposed, shared), List.of(w),
                        List.of(next, old.add(shared.sum(1, false).add(w))), List.of(h, next, next));
                var carry = List.of(scanTensor(type, false, batch, features), scanTensor(type, false, features));
                var sequences = List.of(scanTensor(type, true, time, batch, features),
                        scanTensor(type, false, time, features, batch), scanTensor(type, false, time, features, batch));
                Tensor lengths = scanTensor(DataType.INT64, false, batch);
                var axes = List.of(OptionalInt.of(0), OptionalInt.of(1), OptionalInt.empty());
                var a = Scan.apply(carry, sequences, List.of(scanTensor(type, false, features)), lengths, 1, axes, body, false);
                var b = Scan.apply(carry, sequences, List.of(scanTensor(type, false, features)), lengths, 1, axes, body, true);
                CompileArtifacts source = compileScan(List.of(a.finalCarry().getFirst(), a.finalCarry().getLast(), b.outputs().getLast()));
                CompileArtifacts artifacts = splitScanPartitions(source);
                var contexts = new ArrayList<PrepareContext<FakeInputs>>();
                var finals = new ArrayList<BackendPartitionFinalization<FakePlan>>();
                var events = new ArrayList<String>();
                try (var execution = GraphPreparation.prepare(artifacts,
                        scanPreparations(artifacts, contexts, finals, events, Set.of()), GraphPreparationTest::scanSchedule)) {
                    assertEquals(List.of("analyze-0", "analyze-1", "finalize-0", "finalize-1"), events);
                    assertEquals(2, contexts.size());
                    for (int i = 0; i < contexts.size(); i++) {
                        var node = source.graph().nodes().get(i);
                        var projected = contexts.get(i).nodes().getFirst();
                        assertSame(node, projected);
                        assertSame(node.operation().attrs(), projected.operation().attrs());
                        assertSame(node.scanBody().orElseThrow(), projected.scanBody().orElseThrow());
                        assertEquals(axes, ((ScanAttrs) projected.operation().attrs()).sequenceRowAxes());
                        assertEquals(6, projected.scanBody().orElseThrow().graph().inputs().size());
                        assertEquals(7, projected.inputs().size());
                        assertLengthDeclarations(contexts.get(i), finals.get(i));
                        var carryBanks = (PreparationResourceRequirement.Workspace) finals.get(i).analysis().requirements().stream()
                                .filter(r -> r instanceof PreparationResourceRequirement.Workspace s && s.requirementId() == 0).findFirst().orElseThrow();
                        assertEquals(Math.multiplyExact(2L * type.byteWidth(), Math.addExact(Math.multiplyExact(batch, features), features)), carryBanks.byteSize());
                        assertSame(execution.memoryPlan(), finals.get(i).memoryPlan());
                    }
                    assertNotSame(contexts.get(0).nodes().getFirst().scanBody().orElseThrow().graph(),
                            contexts.get(1).nodes().getFirst().scanBody().orElseThrow().graph());
                    assertEquals(contexts.get(0).nodes().getFirst().scanBody().orElseThrow().graph().inputs(),
                            contexts.get(1).nodes().getFirst().scanBody().orElseThrow().graph().inputs());
                }
            }
        }
    }

    @Test
    void lengthReverseRejectsMissingExtraAndStaleDemandBeforeAnalysis() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (boolean reverse : List.of(false, true)) {
            for (long time : List.of(0L, 3L)) for (long batch : List.of(0L, 1L, 2L)) {
                Tensor h = scanTensor(type, false, batch, 2), x = scanTensor(type, true, batch, 2);
                Tensor unused = scanTensor(type, false, batch, 2), w = scanTensor(type, true, 2, 2);
                Tensor wt = w.permute(1, 0), ht = h.permute(1, 0);
                Tensor activation = h.matmul(wt).add(x).permute(1, 0).tanh();
                Tensor next = activation.permute(1, 0).add(ht.permute(1, 0));
                // The second emission is intentionally unpublished, independent of saved-primal choices.
                var body = new ScanBody(List.of(h), List.of(x, unused), List.of(w), List.of(next), List.of(h, x));
                Tensor seq = scanTensor(type, true, time, batch, 2), weight = scanTensor(type, true, 2, 2);
                Tensor unusedActual = scanTensor(type, false, time, batch, 2), lengths = scanTensor(DataType.INT64, false, batch);
                var source = Scan.apply(List.of(scanTensor(type, false, batch, 2)), List.of(seq, unusedActual), List.of(weight),
                        lengths, 1, List.of(OptionalInt.of(0), OptionalInt.of(0)), body, reverse);
                var roots = List.of(source.finalCarry().getFirst(), source.outputs().getFirst());
                CompileArtifacts artifacts = splitScanPartitions(compileScan(roots, List.of(seq, weight)));
                var scans = artifacts.graph().nodes().stream().filter(n -> n.scanBody().isPresent()).toList();
                var contexts = new ArrayList<PrepareContext<FakeInputs>>();
                var finals = new ArrayList<BackendPartitionFinalization<FakePlan>>();
                try (var ignored = GraphPreparation.prepare(artifacts,
                        scanPreparations(artifacts, contexts, finals, new ArrayList<>(), Set.of()), GraphPreparationTest::scanSchedule)) {
                    for (CompiledNode node : scans) {
                        var demand = node.scanBody().orElseThrow().demandedOutputIndices();
                        var missing = new ArrayList<>(demand); missing.removeFirst();
                        assertScanRejectedBeforeAnalysis(withScanDemand(artifacts, node, missing), true);
                        int dead = java.util.stream.IntStream.range(0, node.outputs().size())
                                .filter(i -> !demand.contains(i)).findFirst().orElseThrow();
                        var extra = new ArrayList<>(demand); extra.add(dead); extra.sort(Integer::compareTo);
                        assertScanRejectedBeforeAnalysis(withScanDemand(artifacts, node, extra), true);
                        var stale = new ArrayList<>(missing); stale.add(dead); stale.sort(Integer::compareTo);
                        assertScanRejectedBeforeAnalysis(withScanDemand(artifacts, node, stale), true);
                    }
                }
            }
        }
    }

    @Test
    void actualLengthReverseKeepsTiedPositionsRepeatedRootsAndTemplateOccurrences() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) {
            Tensor h = scanTensor(type, true, 1, 2), k = scanTensor(type, true, 1, 2);
            Tensor x = scanTensor(type, true, 1, 2), y = scanTensor(type, true, 1, 2);
            Tensor w = scanTensor(type, true, 2), u = scanTensor(type, true, 2);
            Tensor next = h.mul(w).add(k.mul(u)).add(x).add(y);
            var body = new ScanBody(List.of(h, k), List.of(x, y), List.of(w, u), List.of(next, next), List.of(next, next));
            Tensor initial = scanTensor(type, true, 1, 2), seq = scanTensor(type, true, 3, 1, 2), weight = scanTensor(type, true, 2);
            Tensor lengths = scanTensor(DataType.INT64, false, 1);
            var roots = new ArrayList<Tensor>();
            for (boolean reverse : List.of(false, true)) {
                var scan = Scan.apply(List.of(initial, initial), List.of(seq, seq), List.of(weight, weight), lengths,
                        2, List.of(OptionalInt.of(0), OptionalInt.of(0)), body, reverse);
                roots.addAll(scan.finalCarry()); roots.addAll(scan.outputs());
            }
            CompileArtifacts artifacts = compileScan(roots, List.of(initial, seq, weight));
            var projected = GraphPreparation.project(artifacts, artifacts.partitions().getFirst(), new FakeInputs("tied"));
            var scans = projected.nodes().stream().filter(n -> n.scanBody().isPresent()).toList();
            assertEquals(4, scans.size());
            var forwards = scans.stream().filter(n -> artifacts.graph().nodePhases().get(n.id()) == GraphPhase.FORWARD).toList();
            assertEquals(2, forwards.size());
            for (var node : forwards) {
                assertEquals(node.inputs().get(0), node.inputs().get(1));
                assertEquals(node.inputs().get(2), node.inputs().get(3));
                assertEquals(node.inputs().get(4), node.inputs().get(5));
                var local = node.scanBody().orElseThrow();
                assertEquals(local.nextCarry().get(0), local.nextCarry().get(1));
                assertEquals(local.nextCarry().getFirst(), local.emissions().getLast());
                assertSame(node, projected.partitionDag().node(node.id()).orElseThrow());
            }
            assertNotSame(forwards.get(0).scanBody().orElseThrow().graph(), forwards.get(1).scanBody().orElseThrow().graph());
            assertEquals(3, artifacts.publication().gradientBindings().size());
            for (var binding : artifacts.publication().gradientBindings()) {
                var producer = artifacts.graph().nodes().stream().filter(n -> n.outputs().contains(binding.valueId())).findFirst().orElseThrow();
                assertEquals("ADD", producer.operation().kind().name()); // Positional and cross-occurrence sums survive.
            }
            try (var ignored = GraphPreparation.prepare(artifacts,
                    scanPreparations(artifacts, new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), Set.of()),
                    GraphPreparationTest::scanSchedule)) {
                assertSame(artifacts.graph().nodes().getFirst(), projected.nodes().getFirst());
            }
        }
    }

    @Test
    void lengthMetadataHandoffKeepsExactSlotsRollbackAndOneTimeResourceTransfer() {
        for (DataType type : List.of(DataType.FLOAT32, DataType.FLOAT64)) for (long time : List.of(0L, 3L)) {
            for (long batch : List.of(0L, 2L)) {
                CompileArtifacts artifacts = lengthHandoffArtifacts(type, time, batch);
                var contexts = new ArrayList<PrepareContext<FakeInputs>>();
                var finals = new ArrayList<BackendPartitionFinalization<FakePlan>>();
                var events = new ArrayList<String>();
                var closes = new ArrayList<String>();
                var resources = scanResources(closes, artifacts.partitions().size());
                PreparedExecution execution = GraphPreparation.prepare(artifacts,
                        scanPreparations(artifacts, contexts, finals, events, Set.of(), resources), context -> {
                            assertTrue(closes.isEmpty());
                            ValueId crossing = artifacts.graph().nodes().getLast().inputs().getFirst();
                            var assigned = context.bufferAssignments().stream().filter(b -> b.valueId().equals(crossing)).findFirst().orElseThrow();
                            assertEquals(artifacts.partitions().stream().map(PlannedPartition::owner).toList(), assigned.representationOwners());
                            return scanSchedule(context);
                        });
                assertEquals(List.of("analyze-0", "analyze-1", "finalize-0", "finalize-1"), events);
                assertTrue(closes.isEmpty());
                assertLengthDeclarations(contexts.getFirst(), finals.getFirst());
                ValueId crossing = artifacts.graph().nodes().getLast().inputs().getFirst();
                var producer = bufferAssignment(finals.getFirst(), crossing);
                var consumer = bufferAssignment(finals.getLast(), crossing);
                assertSame(producer.slot(), consumer.slot());
                assertEquals(producer.planIndex(), consumer.planIndex());
                assertEquals(0, producer.representationIndex());
                assertEquals(1, consumer.representationIndex());
                assertEquals(type.byteWidth() * time * batch * 2, producer.requirement().byteSize());
                assertSame(execution.memoryPlan().buffers().get(producer.planIndex()).slot(), producer.slot());
                try (var other = GraphPreparation.prepare(artifacts,
                        scanPreparations(artifacts, new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), Set.of()),
                        GraphPreparationTest::scanSchedule)) {
                    assertNotSame(execution.memoryPlan(), other.memoryPlan());
                    assertNotSame(execution.schedule(), other.schedule());
                }
                execution.close(); execution.close();
                assertEquals(List.of("resource-1", "resource-0"), closes);

                var rollbackCloses = new ArrayList<String>();
                var failure = new IllegalStateException("length assembly rejected");
                assertSame(failure, assertThrows(IllegalStateException.class, () -> GraphPreparation.prepare(artifacts,
                        scanPreparations(artifacts, new ArrayList<>(), new ArrayList<>(), new ArrayList<>(), Set.of(),
                                scanResources(rollbackCloses, artifacts.partitions().size())), context -> { throw failure; })));
                assertEquals(List.of("resource-1", "resource-0"), rollbackCloses);
                for (int omitted : List.of(0, 1)) {
                    var rejected = new ArrayList<String>();
                    assertTrue(assertThrows(IllegalArgumentException.class, () -> GraphPreparation.prepare(artifacts,
                            scanPreparations(artifacts, new ArrayList<>(), new ArrayList<>(), rejected, Set.of(omitted)),
                            GraphPreparationTest::scanSchedule)).getMessage().contains("cross-partition value"));
                    assertEquals(List.of("analyze-0", "analyze-1"), rejected);
                }
            }
        }
    }

    @Test
    void lengthLaterPartitionRejectsUnusedDynamicDtypeAndOverflowBeforeAnyAnalyzer() {
        Tensor h = scanTensor(DataType.FLOAT64, false, 2, 2), x = scanTensor(DataType.FLOAT64, false, 2, 2);
        Tensor unused = scanTensor(DataType.FLOAT64, false, 5);
        var template = new ScanBody(List.of(h), List.of(x), List.of(unused), List.of(h.add(x).tanh()), List.of(h));
        Tensor initial = scanTensor(DataType.FLOAT64, false, 2, 2), seq = scanTensor(DataType.FLOAT64, false, 0, 2, 2);
        Tensor actual = scanTensor(DataType.FLOAT64, false, 5), lengths = scanTensor(DataType.INT64, false, 2);
        var a = Scan.apply(List.of(initial), List.of(seq), List.of(actual), lengths, 1, List.of(OptionalInt.of(0)), template, false);
        var b = Scan.apply(List.of(initial), List.of(seq), List.of(actual), lengths, 1, List.of(OptionalInt.of(0)), template, true);
        CompileArtifacts source = splitScanPartitions(compileScan(List.of(a.finalCarry().getFirst(), b.finalCarry().getFirst())));
        var node = source.graph().nodes().getLast();
        var body = node.scanBody().orElseThrow();
        var local = body.graph();
        for (ValueId id : List.of(local.inputs().getLast(), body.residuals().getLast())) {
            for (TensorDescriptor bad : List.of(
                    new TensorDescriptor(DataType.FLOAT64, Shape.ofDimensions(new DynamicDimension("unused")), Optional.empty(), false),
                    scanTensor(DataType.FLOAT32, false, 5).descriptor(),
                    new TensorDescriptor(DataType.FLOAT64, Shape.of(Long.MAX_VALUE, 2), Optional.empty(), false))) {
                var values = local.values().stream().map(v -> v.id().equals(id) ? new GraphValue(id, bad) : v).toList();
                var invalidBody = new CompiledScanBody(new CompiledGraphModel(values, local.nodes(), local.inputs(), local.outputs(), local.nodePhases()),
                        body.nextCarry(), body.emissions(), body.residuals(), body.demandedOutputIndices());
                var invalid = withScanBody(source, node, invalidBody);
                var events = new ArrayList<String>();
                Class<? extends RuntimeException> expected = bad.shape().equals(Shape.of(Long.MAX_VALUE, 2))
                        ? ArithmeticException.class : IllegalArgumentException.class;
                assertThrows(expected, () -> GraphPreparation.prepare(invalid,
                        scanPreparations(invalid, new ArrayList<>(), new ArrayList<>(), events, Set.of()), GraphPreparationTest::scanSchedule));
                assertTrue(events.isEmpty());
            }
        }
    }

    @Test
    void projectsValidBodiesOutsideInitialCpuWhitelistIncludingZeroTimeAndUnusedGradientActual() {
        for (long time : List.of(0L, 3L)) {
            for (DataType type : List.of(DataType.FLOAT16, DataType.FLOAT32)) {
                Tensor h = scanTensor(type, false, 2), x = scanTensor(type, false, 2);
                Tensor unused = scanTensor(type, true, 5);
                Tensor next = type == DataType.FLOAT16 ? h.add(x).tanh() : h.add(x).relu();
                var template = new ScanBody(List.of(h), List.of(x), List.of(unused), List.of(next), List.of(h, next));
                Tensor initial = scanTensor(type, false, 2), sequence = scanTensor(type, false, time, 2);
                Tensor actualUnused = scanTensor(type, true, 5);
                var a = Scan.apply(List.of(initial), List.of(sequence), List.of(actualUnused), template, false);
                var b = Scan.apply(List.of(initial), List.of(sequence), List.of(actualUnused), template, true);
                CompileArtifacts source = compileScan(List.of(a.finalCarry().getFirst(), b.outputs().getFirst()));
                var partitions = List.of(
                        new PlannedPartition(new BackendId("first"), List.of(source.graph().nodes().get(0).id())),
                        new PlannedPartition(new BackendId("second"), List.of(source.graph().nodes().get(1).id())));
                CompileArtifacts artifacts = scanArtifacts(source, source.graph(), partitions);
                var events = new ArrayList<String>();
                var contexts = new ArrayList<PrepareContext<FakeInputs>>();
                var finalizations = new ArrayList<BackendPartitionFinalization<FakePlan>>();
                try (PreparedExecution execution = GraphPreparation.prepare(artifacts,
                        scanPreparations(artifacts, contexts, finalizations, events, Set.of()),
                        context -> { events.add("assemble"); return scanSchedule(context); })) {
                    assertEquals(List.of("analyze-0", "analyze-1", "finalize-0", "finalize-1", "assemble"), events);
                    assertEquals(2, contexts.size());
                    for (int i = 0; i < 2; i++) {
                        var node = artifacts.graph().nodes().get(i);
                        var projected = contexts.get(i);
                        assertSame(node, projected.nodes().getFirst());
                        var body = node.scanBody().orElseThrow();
                        assertSame(body, projected.nodes().getFirst().scanBody().orElseThrow());
                        assertSame(body.graph(), projected.nodes().getFirst().scanBody().orElseThrow().graph());
                        assertEquals(1, projected.partitionDag().nodes().size());
                        assertTrue(projected.partitionDag().edges().isEmpty());
                        assertEquals(3, body.graph().inputs().size());
                        assertTrue(projected.values().stream().allMatch(v -> artifacts.graph().values().contains(v)));
                        assertTrue(body.graph().values().stream().noneMatch(v -> projected.values().stream().anyMatch(o -> o == v)));
                        assertTrue(node.outputs().stream().allMatch(id -> projected.values().stream()
                                .filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor().requiresGrad()));
                        assertTrue(body.graph().values().stream().filter(v -> !v.id().equals(body.graph().inputs().getLast()))
                                .noneMatch(v -> v.descriptor().requiresGrad()));
                        assertSame(execution.memoryPlan(), finalizations.get(i).memoryPlan());
                    }
                    assertNotSame(contexts.get(0).nodes().getFirst().scanBody().orElseThrow().graph(),
                            contexts.get(1).nodes().getFirst().scanBody().orElseThrow().graph());
                    assertEquals(List.of(0), contexts.get(0).nodes().getFirst().scanBody().orElseThrow().demandedOutputIndices());
                    assertEquals(List.of(1), contexts.get(1).nodes().getFirst().scanBody().orElseThrow().demandedOutputIndices());
                }
            }
        }
    }

    @Test
    void rejectsStaleDemandFromPublicationsExternalConsumersAndForwardReverseEdgesBeforeAnyAnalysis() {
        for (long time : List.of(0L, 3L)) {
            Tensor h = scanTensor(DataType.FLOAT32, false, 2), x = scanTensor(DataType.FLOAT32, false, 2);
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x).tanh()), List.of(h));
            Tensor initial = scanTensor(DataType.FLOAT32, false, 2), sequence = scanTensor(DataType.FLOAT32, false, time, 2);
            var forward = Scan.apply(List.of(initial), List.of(sequence), List.of(), body, false);
            var reverse = Scan.apply(List.of(initial), List.of(forward.residuals().getLast()), List.of(), body, true);
            for (List<Tensor> roots : List.of(List.of(forward.finalCarry().getFirst()),
                    List.of(forward.residuals().getLast()), List.of(forward.residuals().getLast().neg()),
                    List.of(forward.finalCarry().getFirst(), reverse.finalCarry().getFirst()))) {
                CompileArtifacts source = compileScan(roots);
                // The same complete graph also projects with local scan-to-scan consumers;
                // partition-local sinks alone do not determine demand.
                var localContext = GraphPreparation.project(source, source.partitions().getFirst(), new FakeInputs("local"));
                assertEquals(source.graph().nodes(), localContext.nodes());
                // Distinct owners force the downstream consumer out of the scan's partition.
                var partitions = new ArrayList<PlannedPartition>();
                for (int i = 0; i < source.graph().nodes().size(); i++) {
                    partitions.add(new PlannedPartition(new BackendId("owner-" + i), List.of(source.graph().nodes().get(i).id())));
                }
                CompileArtifacts artifacts = scanArtifacts(source, source.graph(), partitions);
                for (CompiledNode node : artifacts.graph().nodes()) {
                    if (node.scanBody().isEmpty()) continue;
                    var bodyDemand = node.scanBody().orElseThrow().demandedOutputIndices();
                    var missing = new ArrayList<>(bodyDemand);
                    missing.removeLast();
                    assertScanRejectedBeforeAnalysis(withScanDemand(artifacts, node, missing), true);
                    var extra = new ArrayList<>(bodyDemand);
                    for (int i = 0; i < node.outputs().size(); i++) {
                        if (!extra.contains(i)) { extra.add(i); break; }
                    }
                    extra.sort(Integer::compareTo);
                    assertScanRejectedBeforeAnalysis(withScanDemand(artifacts, node, extra), true);
                }
                var contexts = new ArrayList<PrepareContext<FakeInputs>>();
                var finalizations = new ArrayList<BackendPartitionFinalization<FakePlan>>();
                var events = new ArrayList<String>();
                try (var ignored = GraphPreparation.prepare(artifacts,
                        scanPreparations(artifacts, contexts, finalizations, events, Set.of()), GraphPreparationTest::scanSchedule)) {
                    assertEquals(partitions.size(), contexts.size());
                    if (artifacts.graph().nodes().size() == 2) {
                        var first = artifacts.graph().nodes().getFirst();
                        var last = artifacts.graph().nodes().getLast();
                        assertTrue(last.inputs().stream().anyMatch(first.outputs()::contains));
                        if (last.scanBody().isPresent()) {
                            assertEquals(List.of(0), last.scanBody().orElseThrow().demandedOutputIndices());
                        }
                    }
                }
            }
        }
    }

    @Test
    void laterPartitionDynamicUnusedFormalFailsBeforeTheFirstAnalyzer() {
        Tensor h = scanTensor(DataType.FLOAT32, false, 2), x = scanTensor(DataType.FLOAT32, false, 2);
        var template = new ScanBody(List.of(h), List.of(x), List.of(scanTensor(DataType.FLOAT32, false, 5)),
                List.of(h.add(x)), List.of());
        Tensor initial = scanTensor(DataType.FLOAT32, false, 2), sequence = scanTensor(DataType.FLOAT32, false, 0, 2);
        Tensor unused = scanTensor(DataType.FLOAT32, false, 5);
        var a = Scan.apply(List.of(initial), List.of(sequence), List.of(unused), template, false);
        var b = Scan.apply(List.of(initial), List.of(sequence), List.of(unused), template, true);
        CompileArtifacts source = compileScan(List.of(a.finalCarry().getFirst(), b.finalCarry().getFirst()));
        var partitions = List.of(new PlannedPartition(new BackendId("first"), List.of(source.graph().nodes().getFirst().id())),
                new PlannedPartition(new BackendId("second"), List.of(source.graph().nodes().getLast().id())));
        var node = source.graph().nodes().getLast();
        var body = node.scanBody().orElseThrow();
        var local = body.graph();
        ValueId id = local.inputs().getLast();
        var values = local.values().stream().map(v -> v.id().equals(id)
                ? new GraphValue(id, new TensorDescriptor(DataType.FLOAT32, Shape.ofDimensions(new DynamicDimension("unused")), Optional.empty(), false)) : v).toList();
        var invalid = new CompiledScanBody(new CompiledGraphModel(values, local.nodes(), local.inputs(), local.outputs(), local.nodePhases()),
                body.nextCarry(), body.emissions(), body.residuals(), body.demandedOutputIndices());
        var nodes = List.of(source.graph().nodes().getFirst(), new CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs(), Optional.of(invalid)));
        var graph = new CompiledGraphModel(source.graph().values(), nodes, source.graph().inputs(), source.graph().outputs(), source.graph().nodePhases());
        assertScanRejectedBeforeAnalysis(scanArtifacts(source, graph, partitions), false);
    }

    @Test
    void symbolicOccurrencesProjectRankThreePortsWithoutInventingLocalSlotsOrLayouts() {
        var batch = new DynamicDimension("B");
        for (long time : List.of(0L,3L)) {
            var row = new TensorDescriptor(DataType.FLOAT32,Shape.ofDimensions(batch,new io.github.pho001.synaptik.model.shape.StaticDimension(3)),Optional.empty(),true);
            var sequence = new TensorDescriptor(DataType.FLOAT32,Shape.ofDimensions(new io.github.pho001.synaptik.model.shape.StaticDimension(time),batch,new io.github.pho001.synaptik.model.shape.StaticDimension(3)),Optional.empty(),true);
            Tensor h = TensorFactory.create(row), x = TensorFactory.create(row), initial = TensorFactory.create(row), input = TensorFactory.create(sequence);
            var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x).sigmoid()),List.of(h));
            var first = Scan.apply(List.of(initial),List.of(input),List.of(),body,false);
            var second = Scan.apply(List.of(initial),List.of(input),List.of(),body,true);
            var source = compileScan(List.of(first.finalCarry().getFirst(),first.outputs().getFirst(),second.outputs().getFirst()));
            for (var partition : source.partitions()) {
                var context = GraphPreparation.project(source,partition,new FakeInputs("scan"));
                assertTrue(context.requiresBatchSize());
                assertTrue(context.values().stream().allMatch(v -> v.descriptor().shape().isFullyStatic() || v.descriptor().layout().isEmpty()));
                for (var node : context.nodes()) node.scanBody().ifPresent(scoped -> {
                    assertTrue(scoped.graph().values().stream().anyMatch(v -> !v.descriptor().shape().isFullyStatic()));
                    assertTrue(scoped.graph().values().stream().noneMatch(v -> v.descriptor().layout().isPresent()));
                });
                assertEquals(source.graph().values().stream().filter(v -> context.values().stream().anyMatch(c -> c.id().equals(v.id()))).count(),context.values().size());
            }
        }
    }

    @Test
    void scanResidualHandoffRetainsExactDeclarationsCheckedGeometryAndNoLocalBufferInventory() {
        for (long time : List.of(0L, 3L)) {
            Tensor h = scanTensor(DataType.FLOAT32, false, 2), x = scanTensor(DataType.FLOAT32, false, 2);
            var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x).tanh()), List.of(h));
            var scan = Scan.apply(List.of(scanTensor(DataType.FLOAT32, false, 2)),
                    List.of(scanTensor(DataType.FLOAT32, false, time, 2)), List.of(), body, false);
            CompileArtifacts source = compileScan(List.of(scan.finalCarry().getFirst(), scan.residuals().getLast().neg()));
            var partitions = List.of(new PlannedPartition(new BackendId("first"), List.of(source.graph().nodes().getFirst().id())),
                    new PlannedPartition(new BackendId("second"), List.of(source.graph().nodes().getLast().id())));
            CompileArtifacts artifacts = scanArtifacts(source, source.graph(), partitions);
            var first = artifacts.graph().nodes().getFirst();
            ValueId residual = artifacts.graph().nodes().getLast().inputs().getFirst();
            var contexts = new ArrayList<PrepareContext<FakeInputs>>();
            var finalizations = new ArrayList<BackendPartitionFinalization<FakePlan>>();
            var events = new ArrayList<String>();
            try (var execution = GraphPreparation.prepare(artifacts,
                    scanPreparations(artifacts, contexts, finalizations, events, Set.of()), context -> {
                        events.add("assemble");
                        var crossing = context.bufferAssignments().stream().filter(a -> a.valueId().equals(residual)).findFirst().orElseThrow();
                        assertEquals(List.of(partitions.get(0).owner(), partitions.get(1).owner()), crossing.representationOwners());
                        return scanSchedule(context);
                    })) {
                assertEquals(List.of("analyze-0", "analyze-1", "finalize-0", "finalize-1", "assemble"), events);
                var firstAssignment = finalizations.get(0).assignments().stream()
                        .filter(a -> a instanceof PreparationResourceAssignment.Buffer buffer && buffer.requirement().valueId().equals(residual))
                        .map(a -> (PreparationResourceAssignment.Buffer) a).findFirst().orElseThrow();
                var secondAssignment = finalizations.get(1).assignments().stream()
                        .filter(a -> a instanceof PreparationResourceAssignment.Buffer buffer && buffer.requirement().valueId().equals(residual))
                        .map(a -> (PreparationResourceAssignment.Buffer) a).findFirst().orElseThrow();
                assertSame(firstAssignment.slot(), secondAssignment.slot());
                assertEquals(firstAssignment.planIndex(), secondAssignment.planIndex());
                assertEquals(0, firstAssignment.representationIndex());
                assertEquals(1, secondAssignment.representationIndex());
                assertEquals(Math.multiplyExact(time, 8), firstAssignment.requirement().byteSize());
                assertEquals(Math.multiplyExact(time, 8) + 16, execution.memoryPlan().buffers().get(firstAssignment.planIndex()).byteSize());
                assertEquals(16, execution.memoryPlan().buffers().get(firstAssignment.planIndex()).byteAlignment());
                for (ValueId output : first.outputs()) {
                    int index = first.outputs().indexOf(output);
                    if (!first.scanBody().orElseThrow().demandedOutputIndices().contains(index)) {
                        assertTrue(finalizations.getFirst().assignments().stream().noneMatch(a -> a instanceof PreparationResourceAssignment.Buffer buffer
                                && buffer.requirement().valueId().equals(output)));
                    }
                }
            }
            // Both missing producer and missing consumer declarations fail before finalization.
            for (int omittedPartition : List.of(0, 1)) {
                var rejectedEvents = new ArrayList<String>();
                assertTrue(assertThrows(IllegalArgumentException.class, () -> GraphPreparation.prepare(artifacts,
                        scanPreparations(artifacts, new ArrayList<>(), new ArrayList<>(), rejectedEvents,
                                Set.of(omittedPartition)), GraphPreparationTest::scanSchedule))
                        .getMessage().contains("cross-partition value"));
                assertEquals(List.of("analyze-0", "analyze-1"), rejectedEvents);
            }
        }
    }

    private static Tensor scanTensor(DataType type, boolean gradient, long... dimensions) {
        Shape shape = Shape.of(dimensions);
        return TensorFactory.create(new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), gradient));
    }

    private static CompileArtifacts lengthHandoffArtifacts(DataType type, long time, long batch) {
        Tensor h = scanTensor(type, false, batch, 2), x = scanTensor(type, false, batch, 2);
        var body = new ScanBody(List.of(h), List.of(x), List.of(), List.of(h.add(x).tanh()), List.of(h));
        var scan = Scan.apply(List.of(scanTensor(type, false, batch, 2)), List.of(scanTensor(type, false, time, batch, 2)),
                List.of(), scanTensor(DataType.INT64, false, batch), 1, List.of(OptionalInt.of(0)), body, false);
        return splitScanPartitions(compileScan(List.of(scan.finalCarry().getFirst(), scan.residuals().getLast().neg())));
    }

    private static List<List<PreparedResource>> scanResources(List<String> closes, int count) {
        var resources = new ArrayList<List<PreparedResource>>();
        for (int i = 0; i < count; i++) {
            String name = "resource-" + i;
            resources.add(List.of(() -> closes.add(name)));
        }
        return resources;
    }

    private static PreparationResourceAssignment.Buffer bufferAssignment(BackendPartitionFinalization<?> finalization, ValueId id) {
        return finalization.assignments().stream()
                .filter(a -> a instanceof PreparationResourceAssignment.Buffer b && b.requirement().valueId().equals(id))
                .map(a -> (PreparationResourceAssignment.Buffer) a).findFirst().orElseThrow();
    }

    private static CompileArtifacts compileScan(List<Tensor> roots) {
        return compileScan(roots, Optional.empty());
    }

    private static CompileArtifacts compileScan(List<Tensor> roots, List<Tensor> targets) {
        var request = new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(
                roots.stream().map(t -> (FunctionalGradientRequest.OutputReference)
                        new FunctionalGradientRequest.ForwardTensorReference(t)).toList(),
                roots.stream().map(t -> Optional.of(scanTensor(t.descriptor().dataType(), false,
                        t.descriptor().shape().toLongArray()))).toList(), targets, false,
                FunctionalGradientRequest.DisconnectedPolicy.ZERO)));
        return compileScan(roots, Optional.of(request));
    }

    private static CompileArtifacts compileScan(List<Tensor> roots, Optional<FunctionalGradientRequest> request) {
        var owner = new BackendId("scan-test-only");
        BackendCapabilityProvider provider = new BackendCapabilityProvider() {
            @Override public BackendId backendId() { return owner; }
            @Override public boolean supports(OperationCapabilityQuery query) { return true; }
        };
        return GraphCompilationPort.compile(request.isPresent() ? CompileMode.FORWARD_AND_BACKWARD : CompileMode.FORWARD_ONLY, roots, request,
                GraphOptimizationConfig.disabled(), BackendIntent.unconstrained(), PartitionScoringConfig.neutral(), List.of(provider),
                List.of(new BackendAvailabilitySnapshot(owner, Map.of(new BackendDeviceId(owner, "0"), DeviceClass.CPU))));
    }

    private static CompileArtifacts splitScanPartitions(CompileArtifacts source) {
        var partitions = new ArrayList<PlannedPartition>();
        for (var node : source.graph().nodes()) {
            partitions.add(new PlannedPartition(new BackendId("scan-owner-" + partitions.size()), List.of(node.id())));
        }
        return scanArtifacts(source, source.graph(), partitions);
    }

    private static TensorDescriptor descriptor(CompileArtifacts artifacts, ValueId id) {
        return artifacts.graph().values().stream().filter(v -> v.id().equals(id)).findFirst().orElseThrow().descriptor();
    }

    private static CompileArtifacts scanArtifacts(CompileArtifacts source, CompiledGraphModel graph, List<PlannedPartition> partitions) {
        PublicationPlan publication = construct(PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class}, graph,
                source.publication().forwardBindings(), source.publication().gradientBindings());
        return new CompileArtifacts(source.mode(), graph, partitions, LogicalMemoryPlanning.plan(graph, partitions), publication,
                source.constants(), source.diagnostics(), new DerivativeGraphMetadata(graph, source.derivatives().derivativeOrderByNode()));
    }

    private static CompileArtifacts withScanDemand(CompileArtifacts source, CompiledNode node, List<Integer> demand) {
        var body = node.scanBody().orElseThrow();
        return withScanBody(source, node, new CompiledScanBody(body.graph(), body.nextCarry(), body.emissions(), body.residuals(), demand));
    }

    private static CompileArtifacts withScanBody(CompileArtifacts source, CompiledNode node, CompiledScanBody body) {
        var replacement = new CompiledNode(node.id(), node.operation(), node.inputs(), node.outputs(),
                Optional.of(body));
        var graph = source.graph();
        var nodes = graph.nodes().stream().map(n -> n == node ? replacement : n).toList();
        return scanArtifacts(source, new CompiledGraphModel(graph.values(), nodes, graph.inputs(), graph.outputs(), graph.nodePhases()), source.partitions());
    }

    private static void assertScanRejectedBeforeAnalysis(CompileArtifacts artifacts, boolean checkProjection) {
        var events = new ArrayList<String>();
        var preparations = scanPreparations(artifacts, new ArrayList<>(), new ArrayList<>(), events, Set.of());
        assertThrows(IllegalArgumentException.class,
                () -> GraphPreparation.prepare(artifacts, preparations, GraphPreparationTest::scanSchedule));
        assertTrue(events.isEmpty());
        if (checkProjection) {
            assertThrows(IllegalArgumentException.class,
                    () -> GraphPreparation.project(artifacts, artifacts.partitions().getFirst(), new FakeInputs("scan")));
        }
    }

    private static List<PartitionPreparation<?, ?>> scanPreparations(CompileArtifacts artifacts,
            List<PrepareContext<FakeInputs>> contexts, List<BackendPartitionFinalization<FakePlan>> finalizations,
            List<String> events, Set<Integer> omitCrossing) {
        return scanPreparations(artifacts, contexts, finalizations, events, omitCrossing, List.of());
    }

    private static List<PartitionPreparation<?, ?>> scanPreparations(CompileArtifacts artifacts,
            List<PrepareContext<FakeInputs>> contexts, List<BackendPartitionFinalization<FakePlan>> finalizations,
            List<String> events, Set<Integer> omitCrossing, List<List<PreparedResource>> resources) {
        var preparations = new ArrayList<PartitionPreparation<?, ?>>();
        for (int index = 0; index < artifacts.partitions().size(); index++) {
            int position = index;
            var partition = artifacts.partitions().get(index);
            var plan = new FakePlan("scan-plan-" + index);
            preparations.add(new PartitionPreparation<>(new FakeInputs("scan"), context -> {
                events.add("analyze-" + position);
                contexts.add(context);
                var selected = new HashSet<ValueId>();
                for (CompiledNode node : context.nodes()) {
                    selected.addAll(node.inputs());
                    if (node.scanBody().isPresent()) {
                        node.scanBody().orElseThrow().demandedOutputIndices().forEach(i -> selected.add(node.outputs().get(i)));
                    } else selected.addAll(node.outputs());
                }
                var declarations = new ArrayList<PreparationResourceRequirement>();
                boolean lengthMode = context.nodes().stream().anyMatch(n -> n.operation().attrs() instanceof ScanAttrs a && a.rowCarryCount() > 0);
                for (GraphValue value : context.values()) {
                    if (!selected.contains(value.id())) continue;
                    var logical = context.memoryRequirements().stream().filter(r -> r.valueId().equals(value.id())).findFirst().orElseThrow();
                    boolean crossing = logical.producerPartition().isPresent()
                            && logical.consumerPartitions().stream().anyMatch(p -> p != logical.producerPartition().orElseThrow());
                    if (crossing && omitCrossing.contains(position)) continue;
                    long bytes = descriptorBytes(value.descriptor());
                    declarations.add(new PreparationResourceRequirement.Buffer(value.id(),
                            crossing && position > 0 && !lengthMode ? Math.addExact(bytes, 16) : bytes,
                            crossing && position > 0 && !lengthMode ? 16 : value.descriptor().dataType().byteWidth()));
                }
                if (lengthMode) declarations.addAll(lengthScratch(context));
                else declarations.add(new PreparationResourceRequirement.Workspace(0, 32, 8));
                return new BackendPartitionAnalysis<>(context.partition(), plan, declarations);
            }, new BackendPartitionFinalizer<>() {
                @Override public BackendId backendId() { return partition.owner(); }
                @Override public BackendPartitionFinalizationResult finalizePartition(BackendPartitionFinalization<FakePlan> finalization) {
                    events.add("finalize-" + position);
                    finalizations.add(finalization);
                    assertSame(plan, finalization.analysis().plan());
                    for (int i = 0; i < finalization.assignments().size(); i++) {
                        var declaration = finalization.analysis().requirements().get(i);
                        switch (finalization.assignments().get(i)) {
                            case PreparationResourceAssignment.Buffer buffer -> {
                                assertSame(declaration, buffer.requirement());
                                var entry = finalization.memoryPlan().buffers().get(buffer.planIndex());
                                assertSame(entry.slot(), buffer.slot());
                                assertTrue(entry.byteSize() >= buffer.requirement().byteSize());
                                assertTrue(entry.byteAlignment() >= buffer.requirement().byteAlignment());
                            }
                            case PreparationResourceAssignment.Workspace workspace -> {
                                assertSame(declaration, workspace.requirement());
                                var entry = finalization.memoryPlan().workspaces().get(workspace.planIndex());
                                assertSame(entry.slot(), workspace.slot());
                                var requirement = (PreparationResourceRequirement.Workspace) declaration;
                                assertEquals(requirement.byteSize(), entry.byteSize());
                                assertEquals(requirement.byteAlignment(), entry.byteAlignment());
                            }
                        }
                    }
                    return new BackendPartitionFinalizationResult(new TestExecutable(finalization.memoryPlan()),
                            resources.isEmpty() ? List.of() : resources.get(position));
                }
            }));
        }
        return preparations;
    }

    /** Test-only backend geometry: two carry banks, one step staging bank and dense local cuts. */
    private static List<PreparationResourceRequirement.Workspace> lengthScratch(PrepareContext<?> context) {
        var result = new ArrayList<PreparationResourceRequirement.Workspace>();
        for (int index = 0; index < context.nodes().size(); index++) {
            var node = context.nodes().get(index);
            if (!(node.operation().attrs() instanceof ScanAttrs a) || a.rowCarryCount() == 0) continue;
            var local = node.scanBody().orElseThrow().graph();
            var descriptors = new LinkedHashMap<ValueId, TensorDescriptor>();
            local.values().forEach(v -> descriptors.put(v.id(), v.descriptor()));
            long carries = 0, staging = 0, temporaries = 0;
            for (int i = 0; i < a.carryCount(); i++) {
                carries = Math.addExact(carries, descriptorBytes(descriptors.get(local.inputs().get(i))));
            }
            for (int i = a.carryCount(); i < a.carryCount() + a.sequenceCount(); i++) {
                staging = Math.addExact(staging, descriptorBytes(descriptors.get(local.inputs().get(i))));
            }
            for (var value : local.values()) {
                if (!local.inputs().contains(value.id())) temporaries = Math.addExact(temporaries, descriptorBytes(value.descriptor()));
            }
            int width = descriptors.get(local.inputs().getFirst()).dataType().byteWidth();
            result.add(new PreparationResourceRequirement.Workspace(index * 3L, Math.multiplyExact(2, carries), width));
            result.add(new PreparationResourceRequirement.Workspace(index * 3L + 1, staging, width));
            result.add(new PreparationResourceRequirement.Workspace(index * 3L + 2, temporaries, width));
        }
        return result;
    }

    private static long descriptorBytes(TensorDescriptor descriptor) {
        return Math.multiplyExact((long) descriptor.dataType().byteWidth(), descriptor.shape().knownElementCount().orElseThrow());
    }

    private static void assertLengthDeclarations(PrepareContext<?> context, BackendPartitionFinalization<?> finalization) {
        var selected = new HashSet<ValueId>();
        for (var node : context.nodes()) {
            selected.addAll(node.inputs());
            node.scanBody().orElseThrow().demandedOutputIndices().forEach(i -> selected.add(node.outputs().get(i)));
        }
        var buffers = finalization.analysis().requirements().stream().filter(r -> r instanceof PreparationResourceRequirement.Buffer)
                .map(r -> (PreparationResourceRequirement.Buffer) r).toList();
        assertEquals(selected, buffers.stream().map(PreparationResourceRequirement.Buffer::valueId).collect(java.util.stream.Collectors.toSet()));
        for (var buffer : buffers) {
            TensorDescriptor d = context.values().stream().filter(v -> v.id().equals(buffer.valueId())).findFirst().orElseThrow().descriptor();
            assertEquals(descriptorBytes(d), buffer.byteSize());
            assertEquals(d.dataType().byteWidth(), buffer.byteAlignment());
        }
        var workspace = finalization.analysis().requirements().stream().filter(r -> r instanceof PreparationResourceRequirement.Workspace).toList();
        assertEquals(lengthScratch(context), workspace);
        for (var node : context.nodes()) {
            ScanAttrs a = (ScanAttrs) node.operation().attrs();
            var body = node.scanBody().orElseThrow();
            for (int i = a.carryCount() + a.emissionCount(); i < node.outputs().size(); i++) {
                if (!body.demandedOutputIndices().contains(i)) {
                    ValueId dead = node.outputs().get(i);
                    assertTrue(buffers.stream().noneMatch(b -> b.valueId().equals(dead)), "dead history has zero declaration bytes");
                }
            }
            var lengths = buffers.stream().filter(b -> b.valueId().equals(node.inputs().getLast())).findFirst().orElseThrow();
            long batch = context.values().stream().filter(v -> v.id().equals(node.inputs().getLast())).findFirst().orElseThrow()
                    .descriptor().shape().dimension(0).staticSize().orElseThrow();
            assertEquals(Math.multiplyExact(batch, 8L), lengths.byteSize());
        }
    }

    private static PreparedSchedule scanSchedule(PreparedScheduleContext context) {
        var buffers = new ArrayList<List<PreparedRepresentationPlan.BufferPreparation>>();
        for (PreparedBufferAssignment assignment : context.bufferAssignments()) {
            var preparations = new ArrayList<PreparedRepresentationPlan.BufferPreparation>();
            for (int i = 0; i < assignment.representationOwners().size(); i++) {
                preparations.add(context.constants().containsKey(assignment.valueId())
                        ? new InitializedBuffer((requestedBytes, requestedAlignment) -> { throw new AssertionError("must not allocate"); })
                        : i == 0 && context.bindableInputValueIds().contains(assignment.valueId())
                                ? new CallerInput() : new CreatedBuffer((requestedBytes, requestedAlignment) -> { throw new AssertionError("must not allocate"); }));
            }
            buffers.add(preparations);
        }
        var workspaceCreators = context.memoryPlan().workspaces().stream()
                .map(ignored -> (PreparedRepresentationPlan.WorkspaceCreator) (requestedBytes, requestedAlignment) -> { throw new AssertionError("must not allocate"); }).toList();
        var representations = new PreparedRepresentationPlan(context.memoryPlan(), buffers, workspaceCreators);
        var steps = new ArrayList<PreparedSchedule.Step>();
        steps.add(new RepresentationCreationStep(representations));
        context.partitions().forEach(p -> steps.add(new ExecutionStep(p.executable())));
        for (int i = 0; i < context.publicationValueIds().size(); i++) {
            ValueId id = context.publicationValueIds().get(i);
            int bufferIndex = context.bufferAssignments().stream().filter(a -> a.valueId().equals(id)).findFirst().orElseThrow().planIndex();
            steps.add(new PublicationStep(new PreparedPublication(context.memoryPlan(), bufferIndex, 0, i)));
        }
        return new PreparedSchedule(context.memoryPlan(), steps);
    }

    @Test
    void requiresEveryProducerAndExternalConsumerToDeclareCrossPartitionValues() {
        Fixture fixture = fixture();
        ValueId crossing = new ValueId(2);
        var missingProducer = List.<PartitionPreparation<?, ?>>of(
                selectivePreparation(0, fixture, java.util.Set.of(new ValueId(0))),
                selectivePreparation(1, fixture,
                        java.util.Set.of(new ValueId(1), crossing, new ValueId(3))));
        var missingConsumer = List.<PartitionPreparation<?, ?>>of(
                selectivePreparation(0, fixture, java.util.Set.of(new ValueId(0), crossing)),
                selectivePreparation(1, fixture,
                        java.util.Set.of(new ValueId(1), new ValueId(3))));

        assertAll(
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> GraphPreparation.prepare(fixture.artifacts, missingProducer,
                                GraphPreparationTest::validSchedule)).getMessage()
                        .contains("cross-partition value ValueId[value=2] must be declared")),
                () -> assertTrue(assertThrows(IllegalArgumentException.class,
                        () -> GraphPreparation.prepare(fixture.artifacts, missingConsumer,
                                GraphPreparationTest::validSchedule)).getMessage()
                        .contains("cross-partition value ValueId[value=2] must be declared")));
    }

    @Test
    void preparesAllContextsBeforeOrderedAnalysisAndFinalizationThenAssemblesOnce() {
        Fixture fixture = fixture();
        List<String> events = new ArrayList<>();
        List<PrepareContext<FakeInputs>> contexts = new ArrayList<>();
        AtomicInteger assemblerCalls = new AtomicInteger();

        List<PartitionPreparation<?, ?>> preparations = List.of(
                preparation(0, fixture, contexts, events),
                preparation(1, fixture, contexts, events));
        PreparedExecution execution = GraphPreparation.prepare(
                fixture.artifacts,
                preparations,
                context -> {
                    assemblerCalls.incrementAndGet();
                    events.add("assemble");
                    return validSchedule(context);
                });

        assertAll(
                () -> assertEquals(
                        List.of("analyze-0", "analyze-1", "finalize-0", "finalize-1", "assemble"),
                        events),
                () -> assertEquals(1, assemblerCalls.get()),
                () -> assertEquals(2, contexts.size()),
                () -> assertSame(fixture.partitions.get(0), contexts.get(0).partition()),
                () -> assertSame(fixture.partitions.get(1), contexts.get(1).partition()),
                () -> assertSame(
                        contexts.get(0).partitionDag().nodes(), contexts.get(0).nodes()),
                () -> assertSame(
                        contexts.get(1).partitionDag().nodes(), contexts.get(1).nodes()),
                () -> assertEquals(
                        List.of(fixture.nodes.get(0)), contexts.get(0).nodes()),
                () -> assertEquals(
                        List.of(fixture.nodes.get(1)), contexts.get(1).nodes()),
                () -> assertEquals(
                        List.of(fixture.values.get(0), fixture.values.get(2)),
                        contexts.get(0).values()),
                () -> assertEquals(
                        List.of(fixture.values.get(1), fixture.values.get(2), fixture.values.get(3)),
                        contexts.get(1).values()),
                () -> assertTrue(contexts.get(0).constants().isEmpty()),
                () -> assertSame(
                        fixture.constantValue,
                        contexts.get(1).constants().get(fixture.values.get(1).id())),
                () -> assertSame(execution.memoryPlan(), execution.schedule().memoryPlan()),
                () -> assertSame(
                        execution.memoryPlan(),
                        ((RepresentationCreationStep) execution.schedule().steps().getFirst())
                                .representationPlan().memoryPlan()));
    }

    @Test
    void ownsResourcesThroughAssemblyAndTransfersThemOnlyOnSuccess() {
        Fixture fixture = fixture();
        var closes = new ArrayList<String>();
        PreparedResource first = () -> closes.add("first");
        PreparedResource second = () -> closes.add("second");
        List<PartitionPreparation<?, ?>> preparations = List.of(
                preparation(0, fixture, new ArrayList<>(), new ArrayList<>(), List.of(first)),
                preparation(1, fixture, new ArrayList<>(), new ArrayList<>(), List.of(second)));
        RuntimeException assemblyFailure = new RuntimeException("assembly");

        assertSame(assemblyFailure, assertThrows(RuntimeException.class,
                () -> GraphPreparation.prepare(
                        fixture.artifacts, preparations, context -> { throw assemblyFailure; })));
        assertEquals(List.of("second", "first"), closes);

        closes.clear();
        assertThrows(IllegalArgumentException.class,
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> new PreparedSchedule(
                                new PreparedMemoryPlan(List.of(), List.of()), List.of())));
        assertEquals(List.of("second", "first"), closes);

        closes.clear();
        PreparedExecution execution = GraphPreparation.prepare(
                fixture.artifacts, preparations, GraphPreparationTest::validSchedule);
        assertTrue(closes.isEmpty());
        execution.close();
        assertEquals(List.of("second", "first"), closes);
    }

    @Test
    void validatesEarlierPhasesBeforeInvokingLaterCollaborators() {
        Fixture fixture = fixture();
        AtomicInteger analyses = new AtomicInteger();
        AtomicInteger finalizers = new AtomicInteger();
        AtomicInteger assemblers = new AtomicInteger();
        var preparation = new PartitionPreparation<>(
                new FakeInputs("target"),
                context -> {
                    analyses.incrementAndGet();
                    return new BackendPartitionAnalysis<>(
                            context.partition(),
                            new FakePlan("route"),
                            declarations(context));
                },
                new BackendPartitionFinalizer<FakePlan>() {
                    @Override
                    public BackendId backendId() {
                        return fixture.partitions.getFirst().owner();
                    }

                    @Override
                    public BackendPartitionFinalizationResult finalizePartition(
                            BackendPartitionFinalization<FakePlan> finalization) {
                        finalizers.incrementAndGet();
                        return new BackendPartitionFinalizationResult(
                                new TestExecutable(finalization.memoryPlan()));
                    }
                });

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "artifacts",
                        () -> GraphPreparation.prepare(null, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "preparations",
                        () -> GraphPreparation.prepare(fixture.artifacts, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "scheduleAssembler",
                        () -> GraphPreparation.prepare(fixture.artifacts, List.of(), null)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "preparations size must equal compile partition count 2",
                        () -> GraphPreparation.prepare(
                                fixture.artifacts, List.of(preparation), context -> null)),
                () -> assertEquals(0, analyses.get()),
                () -> assertEquals(0, finalizers.get()),
                () -> assertEquals(0, assemblers.get()));

        var nullAnalysis = new PartitionPreparation<>(
                new FakeInputs("target"), context -> null, preparation.finalizer());
        assertFailure(
                NullPointerException.class,
                "preparations[0].preparer returned null",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        List.of(nullAnalysis, preparation),
                        context -> {
                            assemblers.incrementAndGet();
                            return null;
                        }));
        assertAll(
                () -> assertEquals(0, finalizers.get()),
                () -> assertEquals(0, assemblers.get()));
    }

    @Test
    void projectsEveryContextBeforeInvokingTheFirstPreparer() {
        Fixture fixture = fixture(
                descriptor(),
                new TensorDescriptor(
                        DataType.FLOAT32,
                        Shape.ofDimensions(new DynamicDimension("N"),new DynamicDimension("N")),
                        Optional.empty(),
                        false));
        AtomicInteger analyses = new AtomicInteger();
        List<PartitionPreparation<?, ?>> preparations = List.of(
                countingPreparation(0, fixture, analyses),
                countingPreparation(1, fixture, analyses));

        assertThrows(IllegalArgumentException.class,
                () -> GraphPreparation.prepare(
                        fixture.artifacts, preparations, GraphPreparationTest::validSchedule));

        assertEquals(0, analyses.get());
    }

    @Test
    void rejectsInvalidSchedulesWithExactMessages() {
        Fixture fixture = fixture();
        List<PartitionPreparation<?, ?>> preparations = List.of(
                preparation(0, fixture, new ArrayList<>(), new ArrayList<>()),
                preparation(1, fixture, new ArrayList<>(), new ArrayList<>()));

        assertFailure(
                NullPointerException.class,
                "scheduleAssembler returned null",
                () -> GraphPreparation.prepare(fixture.artifacts, preparations, context -> null));
        assertFailure(
                IllegalArgumentException.class,
                "schedule memory plan does not match prepared memory plan",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> new PreparedSchedule(
                                new PreparedMemoryPlan(List.of(), List.of()), List.of())));
        assertFailure(
                IllegalArgumentException.class,
                "non-empty prepared memory plan requires a representation creation occurrence",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> new PreparedSchedule(context.memoryPlan(), List.of(
                                new ExecutionStep(context.partitions().get(0).executable()),
                                new ExecutionStep(context.partitions().get(1).executable()),
                                new PublicationStep(new PreparedPublication(
                                        context.memoryPlan(), 3, 0, 0))))));
        assertFailure(
                IllegalArgumentException.class,
                "caller-input occurrence count must equal bindable input count 1",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> scheduleWithPreparations(
                                context,
                                List.of(
                                        List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer())),
                                        List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer())),
                                        List.of(new InitializedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer())),
                                        List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer()))))));
        assertFailure(
                IllegalArgumentException.class,
                "constant source ValueId[value=1] requires an initialized buffer representation",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> scheduleWithPreparations(
                                context,
                                List.of(
                                        List.of(new CallerInput()),
                                        List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer())),
                                        List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer())),
                                        List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer()))))));
        assertFailure(
                IllegalArgumentException.class,
                "constant source ValueId[value=1] must not use a caller-input representation",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> scheduleWithPreparations(
                                context,
                                List.of(
                                        List.of(new CallerInput()),
                                        List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer())),
                                        List.of(
                                                new InitializedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer()),
                                                new CallerInput()),
                                        List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer()))))));
        assertFailure(
                IllegalArgumentException.class,
                "buffer ValueId[value=2] has an initialized representation but is not a constant source",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> scheduleWithPreparations(
                                context,
                                List.of(
                                        List.of(new CallerInput()),
                                        List.of(new InitializedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer())),
                                        List.of(new InitializedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer())),
                                        List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer()))))));
        assertFailure(
                IllegalArgumentException.class,
                "execution occurrence count must equal prepared partition count 2",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> new PreparedSchedule(
                                context.memoryPlan(),
                                List.of(
                                        new RepresentationCreationStep(validRepresentations(context)),
                                        new ExecutionStep(
                                                context.partitions().getFirst().executable()),
                                        publication(context, 3, 0)))));
        assertFailure(
                IllegalArgumentException.class,
                "steps[3] publication representationIndex out of creation-plan range: 1",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> new PreparedSchedule(
                                context.memoryPlan(),
                                List.of(
                                        new RepresentationCreationStep(validRepresentations(context)),
                                        new ExecutionStep(
                                                context.partitions().get(0).executable()),
                                        new ExecutionStep(
                                                context.partitions().get(1).executable()),
                                        publication(context, 3, 1)))));
        assertFailure(
                IllegalArgumentException.class,
                "publication occurrence count must equal requested result count 1",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> new PreparedSchedule(
                                context.memoryPlan(),
                                List.of(
                                        new RepresentationCreationStep(validRepresentations(context)),
                                        new ExecutionStep(
                                                context.partitions().get(0).executable()),
                                        new ExecutionStep(
                                                context.partitions().get(1).executable())))));
        assertFailure(
                IllegalArgumentException.class,
                "publication occurrence 0 bufferIndex does not match requested value ValueId[value=3]",
                () -> GraphPreparation.prepare(
                        fixture.artifacts,
                        preparations,
                        context -> new PreparedSchedule(
                                context.memoryPlan(),
                                List.of(
                                        new RepresentationCreationStep(validRepresentations(context)),
                                        new ExecutionStep(
                                                context.partitions().get(0).executable()),
                                        new ExecutionStep(
                                                context.partitions().get(1).executable()),
                                        publication(context, 0, 0)))));
    }

    @Test
    void freshPreparationCallsProduceIndependentRecipesAndNeverInvokeCreators() {
        Fixture fixture = fixture();
        AtomicInteger creators = new AtomicInteger();
        List<PartitionPreparation<?, ?>> preparations = List.of(
                preparation(0, fixture, new ArrayList<>(), new ArrayList<>()),
                preparation(1, fixture, new ArrayList<>(), new ArrayList<>()));
        PreparedScheduleAssembler assembler = context -> {
            PreparedRepresentationPlan representations = new PreparedRepresentationPlan(
                    context.memoryPlan(),
                    List.of(
                            List.of(new CallerInput()),
                            List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> {
                                creators.incrementAndGet();
                                return new FakeBuffer();
                            })),
                            List.of(new InitializedBuffer((requestedBytes, requestedAlignment) -> {
                                creators.incrementAndGet();
                                return new FakeBuffer();
                            })),
                            List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> {
                                creators.incrementAndGet();
                                return new FakeBuffer();
                            }))),
                    List.of());
            return completeSchedule(context, representations);
        };

        PreparedExecution first = GraphPreparation.prepare(fixture.artifacts, preparations, assembler);
        PreparedExecution second = GraphPreparation.prepare(fixture.artifacts, preparations, assembler);

        assertAll(
                () -> assertNotSame(first, second),
                () -> assertNotSame(first.memoryPlan(), second.memoryPlan()),
                () -> assertNotSame(first.schedule(), second.schedule()),
                () -> assertEquals(0, creators.get()));
    }

    @Test
    void rejectsAnUnresolvedSourceCallerBeforeScheduleAssembly() {
        CompileArtifacts artifacts = passThroughArtifacts();

        assertThrows(IllegalArgumentException.class,
                () -> GraphPreparation.prepare(
                        artifacts,
                        List.of(),
                        context -> new PreparedSchedule(context.memoryPlan(), List.of())));
    }

    @Test
    void preservesAliasedForwardAndRepeatedGradientPublicationOccurrences() {
        Fixture fixture = fixture();
        CompileArtifacts artifacts = withAliasedResults(fixture.artifacts());
        List<PartitionPreparation<?, ?>> preparations = List.of(
                preparation(0, fixture, new ArrayList<>(), new ArrayList<>()),
                preparation(1, fixture, new ArrayList<>(), new ArrayList<>()));

        PreparedExecution execution = GraphPreparation.prepare(
                artifacts,
                preparations,
                context -> {
                    PreparedRepresentationPlan representations = validRepresentations(context);
                    return new PreparedSchedule(
                            context.memoryPlan(),
                            List.of(
                                    new RepresentationCreationStep(representations),
                                    new ExecutionStep(
                                            context.partitions().get(0).executable()),
                                    new ExecutionStep(
                                            context.partitions().get(1).executable()),
                                    new PublicationStep(new PreparedPublication(
                                            context.memoryPlan(), 3, 0, 0)),
                                    new PublicationStep(new PreparedPublication(
                                            context.memoryPlan(), 3, 0, 1)),
                                    new PublicationStep(new PreparedPublication(
                                            context.memoryPlan(), 3, 0, 2))));
                });

        assertAll(
                () -> assertEquals(3, execution.schedule().publicationCount()),
                () -> assertEquals(
                        List.of(3, 3, 3),
                        execution.schedule().steps().stream()
                                .filter(PublicationStep.class::isInstance)
                                .map(PublicationStep.class::cast)
                                .map(step -> step.publication().bufferIndex())
                                .toList()));
    }

    @Test
    void validatesAndSnapshotsProducerlessContributionsBeforeBackendWork() {
        ProducerlessFixture fixture = producerlessFixture(1);
        AtomicInteger analyses = new AtomicInteger();
        AtomicInteger finalizers = new AtomicInteger();
        AtomicInteger assemblers = new AtomicInteger();
        PartitionPreparation<FakeInputs, FakePlan> preparation = producerlessPreparation(
                fixture, analyses, finalizers, new ArrayList<>());
        ProducerlessPublishedValueResource resource = resource(fixture, 0, 12, 8);

        assertAll(
                () -> assertFailure(
                        NullPointerException.class,
                        "artifacts",
                        () -> GraphPreparation.prepare(null, null, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "preparations",
                        () -> GraphPreparation.prepare(fixture.artifacts, null, null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "producerlessResources",
                        () -> GraphPreparation.prepare(
                                fixture.artifacts, List.of(preparation), null, null)),
                () -> assertFailure(
                        NullPointerException.class,
                        "scheduleAssembler",
                        () -> GraphPreparation.prepare(
                                fixture.artifacts, List.of(preparation), List.of(resource), null)),
                () -> {
                    var preparations = new ArrayList<PartitionPreparation<?, ?>>();
                    preparations.add(null);
                    var resources = new ArrayList<ProducerlessPublishedValueResource>();
                    resources.add(null);
                    assertFailure(
                            NullPointerException.class,
                            "preparations[0]",
                            () -> GraphPreparation.prepare(
                                    fixture.artifacts,
                                    preparations,
                                    resources,
                                    context -> null));
                },
                () -> {
                    var resources = new ArrayList<ProducerlessPublishedValueResource>();
                    resources.add(null);
                    assertFailure(
                            NullPointerException.class,
                            "producerlessResources[0]",
                            () -> GraphPreparation.prepare(
                                    fixture.artifacts,
                                    List.of(preparation),
                                    resources,
                                    context -> null));
                },
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "producerlessResources has no contribution for required ValueId[value=70]",
                        () -> GraphPreparation.prepare(
                                fixture.artifacts,
                                List.of(preparation),
                                List.of(),
                                context -> null)),
                () -> assertFailure(
                        IllegalArgumentException.class,
                        "producerlessResources[1] duplicates ValueId[value=70]",
                        () -> GraphPreparation.prepare(
                                fixture.artifacts,
                                List.of(preparation),
                                List.of(resource, resource),
                                context -> null)),
                () -> {
                    GraphValue equalValue = new GraphValue(
                            resource.value().id(), resource.value().descriptor());
                    ProducerlessPublishedValueResource foreignValue =
                            new ProducerlessPublishedValueResource(equalValue, resource.logicalRequirement(), new io.github.pho001.synaptik.backend.contract.BackendId("test"), new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.FixedByteSize(12), 8);
                    assertFailure(
                            IllegalArgumentException.class,
                            "producerlessResources[0] value reference does not match artifacts graph value: ValueId[value=70]",
                            () -> GraphPreparation.prepare(
                                    fixture.artifacts,
                                    List.of(preparation),
                                    List.of(foreignValue),
                                    context -> null));
                },
                () -> {
                    LogicalMemoryRequirement logical = resource.logicalRequirement();
                    LogicalMemoryRequirement equalRequirement = new LogicalMemoryRequirement(
                            logical.valueId(),
                            logical.descriptor(),
                            logical.producerPartition(),
                            logical.consumerPartitions(),
                            logical.graphOutput());
                    ProducerlessPublishedValueResource foreignRequirement =
                            new ProducerlessPublishedValueResource(resource.value(), equalRequirement, new io.github.pho001.synaptik.backend.contract.BackendId("test"), new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.FixedByteSize(12), 8);
                    assertFailure(
                            IllegalArgumentException.class,
                            "producerlessResources[0] logical requirement reference does not match artifacts memory requirement: ValueId[value=70]",
                            () -> GraphPreparation.prepare(
                                    fixture.artifacts,
                                    List.of(preparation),
                                    List.of(foreignRequirement),
                                    context -> null));
                },
                () -> {
                    CompileArtifacts bindableArtifacts = resolvedBindableZeroNodeArtifacts();
                    GraphValue bindableValue = bindableArtifacts.graph().values().getFirst();
                    LogicalMemoryRequirement bindableRequirement =
                            bindableArtifacts.memory().requirements().getFirst();
                    ProducerlessPublishedValueResource extra =
                            new ProducerlessPublishedValueResource(bindableValue, bindableRequirement, new io.github.pho001.synaptik.backend.contract.BackendId("test"), new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.FixedByteSize(4), 4);
                    assertThrows(IllegalArgumentException.class, () -> GraphPreparation.prepare(
                            bindableArtifacts,
                            List.of(),
                            List.of(extra),
                            context -> null));
                },
                () -> assertEquals(0, analyses.get()),
                () -> assertEquals(0, finalizers.get()),
                () -> assertEquals(0, assemblers.get()));

        var mutable = new ArrayList<>(List.of(resource));
        GraphPreparation.prepare(
                fixture.artifacts,
                List.of(producerlessPreparation(
                        fixture, new AtomicInteger(), new AtomicInteger(), new ArrayList<>())),
                mutable,
                context -> {
                    mutable.clear();
                    assemblers.incrementAndGet();
                    return producerlessSchedule(context);
                });
        assertEquals(1, assemblers.get());
    }

    @Test
    void appendsProducerlessAssignmentsInGraphValueOrderWithoutChangingPartitionInputs() {
        ProducerlessFixture fixture = producerlessFixture(2);
        AtomicInteger analyses = new AtomicInteger();
        AtomicInteger finalizers = new AtomicInteger();
        List<PrepareContext<FakeInputs>> contexts = new ArrayList<>();
        ProducerlessPublishedValueResource first = resource(fixture, 0, 13, 16);
        ProducerlessPublishedValueResource second = resource(fixture, 1, 29, 32);

        PreparedExecution execution = GraphPreparation.prepare(
                fixture.artifacts,
                List.of(producerlessPreparation(fixture, analyses, finalizers, contexts)),
                List.of(second, first),
                GraphPreparationTest::producerlessSchedule);

        assertAll(
                () -> assertEquals(1, analyses.get()),
                () -> assertEquals(1, finalizers.get()),
                () -> assertEquals(1, contexts.size()),
                () -> assertEquals(
                        List.of(fixture.ordinaryInput, fixture.ordinaryOutput),
                        contexts.getFirst().values()),
                () -> assertTrue(contexts.getFirst().constants().isEmpty()),
                () -> assertEquals(4, execution.memoryPlan().buffers().size()),
                () -> assertEquals(13, execution.memoryPlan().buffers().get(2).byteSize()),
                () -> assertEquals(16, execution.memoryPlan().buffers().get(2).byteAlignment()),
                () -> assertEquals(29, execution.memoryPlan().buffers().get(3).byteSize()),
                () -> assertEquals(32, execution.memoryPlan().buffers().get(3).byteAlignment()),
                () -> assertEquals(
                        1,
                        execution.schedule().steps().stream()
                                .filter(ExecutionStep.class::isInstance)
                                .count()),
                () -> assertEquals(3, execution.schedule().publicationCount()));
    }

    @Test
    void rejectsProducerlessResourcesForPureZeroNodeGraphsBeforeAssembly() {
        CompileArtifacts artifacts = producerlessZeroNodeArtifacts();
        GraphValue value = artifacts.graph().values().getFirst();
        LogicalMemoryRequirement requirement = artifacts.memory().requirements().getFirst();
        AtomicInteger assemblers = new AtomicInteger();

        assertThrows(IllegalArgumentException.class, () -> GraphPreparation.prepare(
                artifacts,
                List.of(),
                List.of(new ProducerlessPublishedValueResource(value, requirement, new io.github.pho001.synaptik.backend.contract.BackendId("test"), new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.FixedByteSize(4), 4)),
                context -> {
                    assemblers.incrementAndGet();
                    return new PreparedSchedule(context.memoryPlan(), List.of());
                }));
        assertEquals(0, assemblers.get());
    }

    private static PartitionPreparation<FakeInputs, FakePlan> preparation(
            int index,
            Fixture fixture,
            List<PrepareContext<FakeInputs>> contexts,
            List<String> events) {
        return preparation(index, fixture, contexts, events, List.of());
    }

    private static PartitionPreparation<FakeInputs, FakePlan> preparation(
            int index,
            Fixture fixture,
            List<PrepareContext<FakeInputs>> contexts,
            List<String> events,
            List<PreparedResource> resources) {
        BackendId owner = fixture.partitions.get(index).owner();
        return new PartitionPreparation<>(
                new FakeInputs("target-" + index),
                context -> {
                    contexts.add(context);
                    events.add("analyze-" + index);
                    return new BackendPartitionAnalysis<>(
                            context.partition(),
                            new FakePlan("route-" + index),
                            declarations(context));
                },
                new BackendPartitionFinalizer<>() {
                    @Override
                    public BackendId backendId() {
                        return owner;
                    }

                    @Override
                    public BackendPartitionFinalizationResult finalizePartition(
                            BackendPartitionFinalization<FakePlan> finalization) {
                        events.add("finalize-" + index);
                        return new BackendPartitionFinalizationResult(
                                new TestExecutable(finalization.memoryPlan()), resources);
                    }
                });
    }

    private static PartitionPreparation<FakeInputs, FakePlan> producerlessPreparation(
            ProducerlessFixture fixture,
            AtomicInteger analyses,
            AtomicInteger finalizers,
            List<PrepareContext<FakeInputs>> contexts) {
        return new PartitionPreparation<>(
                new FakeInputs("producerless"),
                context -> {
                    analyses.incrementAndGet();
                    contexts.add(context);
                    return new BackendPartitionAnalysis<>(
                            context.partition(), new FakePlan("route"), declarations(context));
                },
                new BackendPartitionFinalizer<>() {
                    @Override
                    public BackendId backendId() {
                        return fixture.partition.owner();
                    }

                    @Override
                    public BackendPartitionFinalizationResult finalizePartition(
                            BackendPartitionFinalization<FakePlan> finalization) {
                        finalizers.incrementAndGet();
                        assertEquals(2, finalization.assignments().size());
                        return new BackendPartitionFinalizationResult(
                                new TestExecutable(finalization.memoryPlan()));
                    }
                });
    }

    private static PartitionPreparation<FakeInputs, FakePlan> countingPreparation(
            int index, Fixture fixture, AtomicInteger analyses) {
        BackendId owner = fixture.partitions.get(index).owner();
        return new PartitionPreparation<>(
                new FakeInputs("target-" + index),
                context -> {
                    analyses.incrementAndGet();
                    return new BackendPartitionAnalysis<>(
                            context.partition(), new FakePlan("route"), declarations(context));
                },
                new BackendPartitionFinalizer<>() {
                    @Override
                    public BackendId backendId() {
                        return owner;
                    }

                    @Override
                    public BackendPartitionFinalizationResult finalizePartition(
                            BackendPartitionFinalization<FakePlan> finalization) {
                        return new BackendPartitionFinalizationResult(
                                new TestExecutable(finalization.memoryPlan()));
                    }
                });
    }

    private static PartitionPreparation<FakeInputs, FakePlan> selectivePreparation(
            int index, Fixture fixture, java.util.Set<ValueId> declaredValues) {
        BackendId owner = fixture.partitions.get(index).owner();
        return new PartitionPreparation<>(new FakeInputs("selective-" + index), context ->
                new BackendPartitionAnalysis<>(context.partition(), new FakePlan("route"),
                        context.values().stream().filter(value -> declaredValues.contains(value.id()))
                                .map(value -> (PreparationResourceRequirement)
                                        new PreparationResourceRequirement.Buffer(value.id(), 24, 8))
                                .toList()),
                new BackendPartitionFinalizer<>() {
                    @Override public BackendId backendId() { return owner; }
                    @Override public BackendPartitionFinalizationResult finalizePartition(
                            BackendPartitionFinalization<FakePlan> finalization) {
                        return new BackendPartitionFinalizationResult(
                                new TestExecutable(finalization.memoryPlan()));
                    }
                });
    }

    private static List<PreparationResourceRequirement> declarations(
            PrepareContext<FakeInputs> context) {
        return context.values().stream()
                .map(value -> (PreparationResourceRequirement)
                        new PreparationResourceRequirement.Buffer(value.id(), 24, 8))
                .toList();
    }

    private static PreparedSchedule validSchedule(PreparedScheduleContext context) {
        return completeSchedule(context, validRepresentations(context));
    }

    private static PreparedRepresentationPlan validRepresentations(
            PreparedScheduleContext context) {
        return new PreparedRepresentationPlan(
                context.memoryPlan(),
                List.of(
                        List.of(new CallerInput()),
                        List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer())),
                        List.of(new InitializedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer())),
                        List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer()))),
                List.of());
    }

    private static PreparedSchedule scheduleWithPreparations(
            PreparedScheduleContext context,
            List<List<PreparedRepresentationPlan.BufferPreparation>> preparations) {
        return completeSchedule(
                context,
                new PreparedRepresentationPlan(context.memoryPlan(), preparations, List.of()));
    }

    private static PreparedSchedule completeSchedule(
            PreparedScheduleContext context, PreparedRepresentationPlan representations) {
        int outputBuffer = context.bufferAssignments().stream()
                .filter(assignment -> assignment.valueId().equals(new ValueId(3)))
                .findFirst()
                .orElseThrow()
                .planIndex();
        return new PreparedSchedule(
                context.memoryPlan(),
                List.of(
                        new RepresentationCreationStep(representations),
                        new ExecutionStep(context.partitions().get(0).executable()),
                        new ExecutionStep(context.partitions().get(1).executable()),
                        new PublicationStep(new PreparedPublication(
                                context.memoryPlan(), outputBuffer, 0, 0))));
    }

    private static PreparedSchedule producerlessSchedule(PreparedScheduleContext context) {
        var preparations = new ArrayList<List<PreparedRepresentationPlan.BufferPreparation>>();
        for (PreparedBufferAssignment assignment : context.bufferAssignments()) {
            if (context.bindableInputValueIds().contains(assignment.valueId())) {
                preparations.add(List.of(new CallerInput()));
            } else if (context.constants().containsKey(assignment.valueId())) {
                preparations.add(List.of(new InitializedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer())));
            } else {
                preparations.add(List.of(new CreatedBuffer((requestedBytes, requestedAlignment) -> new FakeBuffer())));
            }
        }
        PreparedRepresentationPlan representations = new PreparedRepresentationPlan(
                context.memoryPlan(), preparations, List.of());
        var steps = new ArrayList<PreparedSchedule.Step>();
        steps.add(new RepresentationCreationStep(representations));
        steps.add(new ExecutionStep(context.partitions().getFirst().executable()));
        int resultIndex = 0;
        for (ValueId valueId : context.publicationValueIds()) {
            int bufferIndex = context.bufferAssignments().stream()
                    .filter(assignment -> assignment.valueId().equals(valueId))
                    .findFirst()
                    .orElseThrow()
                    .planIndex();
            steps.add(new PublicationStep(new PreparedPublication(
                    context.memoryPlan(), bufferIndex, 0, resultIndex++)));
        }
        return new PreparedSchedule(context.memoryPlan(), steps);
    }

    private static PublicationStep publication(
            PreparedScheduleContext context, int bufferIndex, int representationIndex) {
        return new PublicationStep(new PreparedPublication(
                context.memoryPlan(), bufferIndex, representationIndex, 0));
    }

    static Fixture fixture() {
        return fixture(descriptor(), descriptor());
    }

    private static Fixture fixture(
            TensorDescriptor firstDescriptor, TensorDescriptor secondDescriptor) {
        ValueId input = new ValueId(0);
        ValueId constant = new ValueId(1);
        ValueId intermediate = new ValueId(2);
        ValueId output = new ValueId(3);
        List<GraphValue> values = List.of(
                new GraphValue(input, firstDescriptor),
                new GraphValue(constant, secondDescriptor),
                new GraphValue(intermediate, firstDescriptor),
                new GraphValue(output, secondDescriptor));
        CompiledNode first = node(10, List.of(input), intermediate);
        CompiledNode second = node(11, List.of(intermediate, constant), output);
        List<CompiledNode> nodes = List.of(first, second);
        CompiledGraphModel graph = new CompiledGraphModel(
                values,
                nodes,
                List.of(input, constant),
                List.of(output),
                Map.of(first.id(), GraphPhase.FORWARD, second.id(), GraphPhase.FORWARD));
        PlannedPartition firstPartition =
                new PlannedPartition(new BackendId("cpu"), List.of(first.id()));
        PlannedPartition secondPartition =
                new PlannedPartition(new BackendId("gpu"), List.of(second.id()));
        List<PlannedPartition> partitions = List.of(firstPartition, secondPartition);
        LogicalMemoryPlan memory = LogicalMemoryPlanning.plan(graph, partitions);
        ScalarValue scalar = ScalarValue.float32(2.0f);
        CompileConstantPlan constants = construct(
                CompileConstantPlan.class,
                new Class<?>[] {List.class, List.class},
                List.of(new CompileConstantPlan.BindableInput(new TensorId(9001), input)),
                List.of(new CompileConstantPlan.ConstantSource(constant, scalar)));
        PublicationPlan publication = construct(
                PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class},
                graph,
                List.of(new ForwardPublicationBinding(new TensorId(100), output)),
                List.of());
        CompileDiagnostics diagnostics = construct(
                CompileDiagnostics.class,
                new Class<?>[] {List.class},
                List.of());
        var derivativeOrders = new LinkedHashMap<NodeId, Integer>();
        derivativeOrders.put(first.id(), 0);
        derivativeOrders.put(second.id(), 0);
        CompileArtifacts artifacts = new CompileArtifacts(CompileMode.FORWARD_ONLY, graph, partitions, memory, publication, constants, diagnostics, new DerivativeGraphMetadata(graph, derivativeOrders));
        return new Fixture(artifacts, partitions, nodes, values, scalar);
    }

    private static TensorDescriptor descriptor() {
        return new TensorDescriptor(DataType.FLOAT32, Shape.of(2, 3), Optional.empty(), false);
    }

    private static CompiledNode node(long id, List<ValueId> inputs, ValueId output) {
        return new CompiledNode(
                new NodeId(id),
                new Operation(SampleKind.SAMPLE, NoOperationAttrs.INSTANCE),
                inputs,
                List.of(output));
    }

    private static <T> T construct(
            Class<T> type, Class<?>[] parameterTypes, Object... arguments) {
        try {
            Constructor<T> constructor = type.getDeclaredConstructor(parameterTypes);
            constructor.setAccessible(true);
            return constructor.newInstance(arguments);
        } catch (ReflectiveOperationException failure) {
            throw new AssertionError(failure);
        }
    }

    private static CompileArtifacts passThroughArtifacts() {
        ValueId valueId = new ValueId(50);
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(new GraphValue(valueId, descriptor())),
                List.of(),
                List.of(valueId),
                List.of(valueId),
                Map.of());
        List<PlannedPartition> partitions = List.of();
        CompileConstantPlan constants = construct(
                CompileConstantPlan.class,
                new Class<?>[] {List.class, List.class},
                List.of(new CompileConstantPlan.BindableInput(new TensorId(9002), valueId)),
                List.of());
        PublicationPlan publication = construct(
                PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class},
                graph,
                List.of(new ForwardPublicationBinding(new TensorId(200), valueId)),
                List.of());
        CompileDiagnostics diagnostics = construct(
                CompileDiagnostics.class, new Class<?>[] {List.class}, List.of());
        return new CompileArtifacts(CompileMode.FORWARD_ONLY, graph, partitions, LogicalMemoryPlanning.plan(graph, partitions), publication, constants, diagnostics, new DerivativeGraphMetadata(graph, Map.of()));
    }

    private static ProducerlessFixture producerlessFixture(int constantCount) {
        ValueId inputId = new ValueId(60);
        ValueId outputId = new ValueId(61);
        GraphValue input = new GraphValue(inputId, descriptor());
        GraphValue output = new GraphValue(outputId, descriptor());
        var values = new ArrayList<GraphValue>();
        values.add(input);
        var constants = new ArrayList<GraphValue>();
        var constantSources = new ArrayList<CompileConstantPlan.ConstantSource>();
        for (int index = 0; index < constantCount; index++) {
            Shape shape = Shape.of(index + 1L);
            TensorDescriptor descriptor = new TensorDescriptor(
                    DataType.FLOAT32,
                    shape,
                    Optional.of(LayoutDescriptor.contiguous(shape)),
                    false);
            GraphValue constant = new GraphValue(new ValueId(70 + index), descriptor);
            constants.add(constant);
            values.add(constant);
            constantSources.add(new CompileConstantPlan.ConstantSource(
                    constant.id(), ScalarValue.float32(index + 1.0f)));
        }
        values.add(output);
        CompiledNode node = node(60, List.of(inputId), outputId);
        var outputs = new ArrayList<ValueId>();
        outputs.add(outputId);
        constants.forEach(value -> outputs.add(value.id()));
        var inputs = new ArrayList<ValueId>();
        inputs.add(inputId);
        constants.forEach(value -> inputs.add(value.id()));
        CompiledGraphModel graph = new CompiledGraphModel(
                values,
                List.of(node),
                inputs,
                outputs,
                Map.of(node.id(), GraphPhase.FORWARD));
        PlannedPartition partition = new PlannedPartition(
                new BackendId("cpu"), List.of(node.id()));
        List<PlannedPartition> partitions = List.of(partition);
        CompileConstantPlan constantPlan = construct(
                CompileConstantPlan.class,
                new Class<?>[] {List.class, List.class},
                List.of(new CompileConstantPlan.BindableInput(new TensorId(9060), inputId)),
                constantSources);
        var forwardBindings = new ArrayList<ForwardPublicationBinding>();
        for (int index = 0; index < outputs.size(); index++) {
            forwardBindings.add(new ForwardPublicationBinding(
                    new TensorId(9600 + index), outputs.get(index)));
        }
        PublicationPlan publication = construct(
                PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class},
                graph,
                forwardBindings,
                List.of());
        CompileArtifacts artifacts = new CompileArtifacts(CompileMode.FORWARD_ONLY, graph, partitions, LogicalMemoryPlanning.plan(graph, partitions), publication, constantPlan, construct(CompileDiagnostics.class, new Class<?>[] {List.class}, List.of()), new DerivativeGraphMetadata(graph, Map.of(node.id(), 0)));
        return new ProducerlessFixture(artifacts, partition, input, output, constants);
    }

    private static ProducerlessPublishedValueResource resource(
            ProducerlessFixture fixture, int index, long byteSize, long byteAlignment) {
        GraphValue value = fixture.constants.get(index);
        LogicalMemoryRequirement requirement = fixture.artifacts.memory().requirements().stream()
                .filter(candidate -> candidate.valueId().equals(value.id()))
                .findFirst()
                .orElseThrow();
        return new ProducerlessPublishedValueResource(value, requirement, new io.github.pho001.synaptik.backend.contract.BackendId("test"), new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.FixedByteSize(byteSize), byteAlignment);
    }

    @Test
    void sourceCallersLeadInBindableOrderAndConstantsRemainLastOnBothPreparationPaths() {
        Shape shape = Shape.of(2, 3);
        var affine = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {8, 2}, 5, true)), true);
        CompileArtifacts artifacts = callerSourceArtifacts(List.of(affine, affine, affine), null);
        for (boolean automatic : List.of(false, true)) {
            AtomicInteger analyses = new AtomicInteger();
            var assembler = sourceAssembler(artifacts, Map.of());
            PreparedExecution execution = automatic
                    ? GraphPreparation.prepare(artifacts,
                            List.of(callerSourcePreparation(artifacts, analyses, List.of())), assembler)
                    : GraphPreparation.prepare(artifacts,
                            List.of(callerSourcePreparation(artifacts, analyses, List.of())),
                            sourceContributions(artifacts, Map.of()).reversed(), assembler);
            assertEquals(1, analyses.get());
            assertEquals(8, execution.memoryPlan().buffers().size());
            for (int index : new int[] {0, 2, 4}) {
                assertEquals(72, execution.memoryPlan().buffers().get(index).byteSize());
            }
            assertEquals(List.of(5, 6, 0, 7, 2, 4),
                    execution.schedule().steps().stream()
                            .filter(PublicationStep.class::isInstance).map(PublicationStep.class::cast)
                            .map(step -> step.publication().bufferIndex()).toList());
            assertSame(affine, artifacts.graph().values().stream()
                    .filter(value -> value.id().equals(new ValueId(90))).findFirst().orElseThrow()
                    .descriptor());
            execution.close();
        }
    }

    @Test
    void sourceOnlyNamedAuthorityRetainsRunBoundRecipesEvenForPermanentZeroBytes() {
        for (long rowBytes : new long[] {0, 12}) {
            Shape shape = Shape.ofDimensions(new DynamicDimension("Customers"),
                    new io.github.pho001.synaptik.model.shape.StaticDimension(rowBytes == 0 ? 0 : 3));
            var named = new TensorDescriptor(DataType.FLOAT32, shape, Optional.empty(), true);
            CompileArtifacts artifacts = callerSourceArtifacts(List.of(named, named, named), null);
            PreparedMemoryPlan.ByteSize size = new SourceRowsByteSize(rowBytes);
            Map<ValueId, PreparedMemoryPlan.ByteSize> sizes = Map.of(
                    new ValueId(90), size, new ValueId(91), size, new ValueId(92), size);
            for (boolean automatic : List.of(false, true)) {
                PreparedExecution execution = automatic
                        ? GraphPreparation.prepare(artifacts,
                                List.of(callerSourcePreparation(artifacts, new AtomicInteger(), List.of())),
                                sourceAssembler(artifacts, sizes))
                        : GraphPreparation.prepare(artifacts,
                                List.of(callerSourcePreparation(artifacts, new AtomicInteger(), List.of())),
                                sourceContributions(artifacts, sizes).reversed(),
                                sourceAssembler(artifacts, sizes));
                assertTrue(execution.memoryPlan().requiresBatchSize());
                for (int index : new int[] {0, 2, 4}) {
                    assertSame(size, execution.memoryPlan().buffers().get(index).size());
                    assertTrue(execution.memoryPlan().buffers().get(index).size().fixedByteSize().isEmpty());
                }
                assertThrows(IllegalArgumentException.class, () ->
                        io.github.pho001.synaptik.runtime.run.RunMemoryGeometry.resolve(
                                execution.memoryPlan(), java.util.OptionalLong.empty()));
                for (long batch : new long[] {2, 0, 5}) {
                    var geometry = io.github.pho001.synaptik.runtime.run.RunMemoryGeometry.resolve(
                            execution.memoryPlan(), java.util.OptionalLong.of(batch));
                    assertSame(execution.memoryPlan(), geometry.memoryPlan());
                    for (int index : new int[] {0, 2, 4}) {
                        assertEquals(batch * rowBytes, geometry.bufferByteSize(index));
                    }
                    for (int index : new int[] {1, 3, 5}) {
                        assertEquals(24, geometry.bufferByteSize(index));
                    }
                }
                execution.close();
            }
        }
    }

    @Test
    void rejectsInvalidSourceShapesAndConstantLayoutsBeforeEitherPathCanAnalyze() {
        Shape one = Shape.of(1);
        var fixed = new TensorDescriptor(DataType.FLOAT32, one,
                Optional.of(LayoutDescriptor.contiguous(one)), false);
        var unresolved = new TensorDescriptor(DataType.FLOAT32, one, Optional.empty(), false);
        var batch = new TensorDescriptor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("Customers")), Optional.empty(), false);
        var independent = new TensorDescriptor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("Other")), Optional.empty(), false);
        var repeated = new TensorDescriptor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("Customers"), new DynamicDimension("Customers")),
                Optional.empty(), false);
        List<CompileArtifacts> invalid = List.of(
                callerSourceArtifacts(List.of(unresolved, fixed, fixed), null),
                callerSourceArtifacts(List.of(batch, independent, fixed), null),
                callerSourceArtifacts(List.of(repeated, fixed, fixed), null),
                callerSourceArtifacts(List.of(fixed, fixed, fixed), unresolved),
                callerSourceArtifacts(List.of(batch, fixed, fixed), batch),
                callerSourceArtifacts(List.of(fixed, fixed, fixed), batch));
        for (CompileArtifacts artifacts : invalid) for (boolean automatic : List.of(false, true)) {
            AtomicInteger analyses = new AtomicInteger();
            var preparation = callerSourcePreparation(artifacts, analyses, List.of());
            assertThrows(IllegalArgumentException.class, () -> {
                if (automatic) GraphPreparation.prepare(
                        artifacts, List.of(preparation), sourceAssembler(artifacts, Map.of()));
                else GraphPreparation.prepare(
                        artifacts, List.of(preparation), List.of(),
                        sourceAssembler(artifacts, Map.of()));
            });
            assertEquals(0, analyses.get());
        }
    }

    @Test
    void rejectedSourceCallerRepresentationRollsBackFinalizedResourcesExactlyOnce() {
        Shape shape = Shape.of(1);
        var fixed = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        CompileArtifacts artifacts = callerSourceArtifacts(List.of(fixed, fixed, fixed), null);
        var closes = new ArrayList<String>();
        PreparedResource first = () -> closes.add("first");
        PreparedResource second = () -> closes.add("second");
        AtomicInteger analyses = new AtomicInteger();
        assertThrows(IllegalArgumentException.class, () -> GraphPreparation.prepare(
                artifacts,
                List.of(callerSourcePreparation(artifacts, analyses, List.of(first, second))),
                sourceContributions(artifacts, Map.of()),
                context -> {
                    var preparations = new ArrayList<List<PreparedRepresentationPlan.BufferPreparation>>();
                    for (PreparedBufferAssignment assignment : context.bufferAssignments()) {
                        if (context.bindableInputValueIds().contains(assignment.valueId())
                                && !assignment.valueId().equals(new ValueId(90))) {
                            preparations.add(List.of(new CallerInput()));
                        } else {
                            preparations.add(List.of(new InitializedBuffer(
                                    (requestedBytes, requestedAlignment) -> new FakeBuffer())));
                        }
                    }
                    PreparedSchedule valid = producerlessSchedule(context);
                    var steps = new ArrayList<>(valid.steps());
                    steps.set(0, new RepresentationCreationStep(new PreparedRepresentationPlan(
                            context.memoryPlan(), preparations, List.of())));
                    return new PreparedSchedule(context.memoryPlan(), steps);
                }));
        assertEquals(1, analyses.get());
        assertEquals(List.of("second", "first"), closes);
    }

    private static CompileArtifacts callerSourceArtifacts(
            List<TensorDescriptor> callers, TensorDescriptor constantDescriptor) {
        Shape one = Shape.of(1);
        if (constantDescriptor == null) {
            constantDescriptor = new TensorDescriptor(DataType.FLOAT32, one,
                    Optional.of(LayoutDescriptor.contiguous(one)), false);
        }
        List<GraphValue> values = List.of(
                new GraphValue(new ValueId(70), constantDescriptor),
                new GraphValue(new ValueId(90), callers.get(0)),
                new GraphValue(new ValueId(60), descriptor()),
                new GraphValue(new ValueId(71), constantDescriptor),
                new GraphValue(new ValueId(91), callers.get(1)),
                new GraphValue(new ValueId(62), descriptor()),
                new GraphValue(new ValueId(92), callers.get(2)),
                new GraphValue(new ValueId(61), descriptor()));
        List<ValueId> inputs = values.subList(0, 7).stream().map(GraphValue::id).toList();
        List<ValueId> outputs = List.of(new ValueId(61), new ValueId(70), new ValueId(90),
                new ValueId(71), new ValueId(91), new ValueId(92));
        CompiledNode node = node(60, List.of(new ValueId(60), new ValueId(62)), new ValueId(61));
        CompiledGraphModel graph = new CompiledGraphModel(values, List.of(node), inputs, outputs,
                Map.of(node.id(), GraphPhase.FORWARD));
        var partition = new PlannedPartition(new BackendId("cpu"), List.of(node.id()));
        var bindables = inputs.stream().filter(id -> id.value() != 70 && id.value() != 71)
                .map(id -> new CompileConstantPlan.BindableInput(new TensorId(9000 + id.value()), id))
                .toList();
        CompileConstantPlan constants = construct(CompileConstantPlan.class,
                new Class<?>[] {List.class, List.class}, bindables,
                List.of(new CompileConstantPlan.ConstantSource(new ValueId(70), ScalarValue.float32(1)),
                        new CompileConstantPlan.ConstantSource(new ValueId(71), ScalarValue.float32(2))));
        var bindings = new ArrayList<ForwardPublicationBinding>();
        for (int index = 0; index < outputs.size(); index++) {
            bindings.add(new ForwardPublicationBinding(new TensorId(9600 + index), outputs.get(index)));
        }
        PublicationPlan publication = construct(PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class},
                graph, bindings, List.of());
        List<PlannedPartition> partitions = List.of(partition);
        return new CompileArtifacts(CompileMode.FORWARD_ONLY, graph, partitions,
                LogicalMemoryPlanning.plan(graph, partitions), publication, constants,
                construct(CompileDiagnostics.class, new Class<?>[] {List.class}, List.of()),
                new DerivativeGraphMetadata(graph, Map.of(node.id(), 0)));
    }

    private static PartitionPreparation<FakeInputs, FakePlan> callerSourcePreparation(
            CompileArtifacts artifacts, AtomicInteger analyses, List<PreparedResource> resources) {
        PlannedPartition partition = artifacts.partitions().getFirst();
        return new PartitionPreparation<>(new FakeInputs("source-callers"), context -> {
            analyses.incrementAndGet();
            assertEquals(List.of(new ValueId(60), new ValueId(62), new ValueId(61)),
                    context.values().stream().map(GraphValue::id).toList());
            assertTrue(context.constants().isEmpty());
            assertTrue(!context.requiresBatchSize());
            return new BackendPartitionAnalysis<>(partition, new FakePlan("fixed"), declarations(context));
        }, new BackendPartitionFinalizer<>() {
            @Override public BackendId backendId() { return partition.owner(); }
            @Override public BackendPartitionFinalizationResult finalizePartition(
                    BackendPartitionFinalization<FakePlan> finalization) {
                assertEquals(List.of(1, 3, 5), finalization.assignments().stream()
                        .map(PreparationResourceAssignment.Buffer.class::cast)
                        .map(PreparationResourceAssignment.Buffer::planIndex).toList());
                return new BackendPartitionFinalizationResult(
                        new TestExecutable(finalization.memoryPlan()), resources);
            }
        });
    }

    private static List<ProducerlessPublishedValueResource> sourceContributions(
            CompileArtifacts artifacts, Map<ValueId, PreparedMemoryPlan.ByteSize> sizes) {
        var resources = new ArrayList<ProducerlessPublishedValueResource>();
        for (GraphValue value : artifacts.graph().values()) {
            if (value.id().value() != 70 && value.id().value() != 71 && value.id().value() < 90) continue;
            LogicalMemoryRequirement requirement = artifacts.memory().requirements().stream()
                    .filter(candidate -> candidate.valueId().equals(value.id())).findFirst().orElseThrow();
            PreparedMemoryPlan.ByteSize size = sizes.get(value.id());
            if (size == null) size = new PreparedMemoryPlan.FixedByteSize(Math.multiplyExact(
                    value.descriptor().layout().orElseThrow().referencedElementSpan(),
                    value.descriptor().dataType().byteWidth()));
            resources.add(new ProducerlessPublishedValueResource(
                    value, requirement, new BackendId("cpu"), size, 8));
        }
        return resources;
    }

    private static PreparedScheduleAssembler sourceAssembler(
            CompileArtifacts artifacts, Map<ValueId, PreparedMemoryPlan.ByteSize> sizes) {
        return new PreparedScheduleAssembler() {
            @Override public ProducerlessPublishedValueResource producerlessPublishedValue(
                    GraphValue value, LogicalMemoryRequirement requirement, Optional<ScalarValue> scalar) {
                assertEquals(value.id().value() == 70 || value.id().value() == 71, scalar.isPresent());
                if (scalar.isPresent()) assertSame(artifacts.constants().constantSources().stream()
                        .filter(source -> source.valueId().equals(value.id())).findFirst().orElseThrow()
                        .value(), scalar.orElseThrow());
                return sourceContributions(artifacts, sizes).stream()
                        .filter(resource -> resource.value() == value).findFirst().orElseThrow();
            }
            @Override public PreparedSchedule assemble(PreparedScheduleContext context) {
                assertEquals(List.of(new ValueId(90), new ValueId(60), new ValueId(91),
                                new ValueId(62), new ValueId(92), new ValueId(61),
                                new ValueId(70), new ValueId(71)),
                        context.bufferAssignments().stream().map(PreparedBufferAssignment::valueId).toList());
                assertTrue(context.bufferAssignments().stream()
                        .allMatch(assignment -> assignment.representationOwners().equals(List.of(new BackendId("cpu")))));
                return producerlessSchedule(context);
            }
        };
    }

    private record SourceRowsByteSize(long rowBytes) implements PreparedMemoryPlan.ByteSize {
        @Override public java.util.OptionalLong fixedByteSize() { return java.util.OptionalLong.empty(); }
        @Override public long byteSize(long batchSize) {
            if (batchSize < 0) throw new IllegalArgumentException("negative batch");
            return Math.multiplyExact(rowBytes, batchSize);
        }
    }

    private static CompileArtifacts producerlessZeroNodeArtifacts() {
        Shape shape = Shape.of(1);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        GraphValue value = new GraphValue(new ValueId(80), descriptor);
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(value), List.of(), List.of(value.id()), List.of(value.id()), Map.of());
        CompileConstantPlan constants = construct(
                CompileConstantPlan.class,
                new Class<?>[] {List.class, List.class},
                List.of(),
                List.of(new CompileConstantPlan.ConstantSource(
                        value.id(), ScalarValue.float32(1.0f))));
        PublicationPlan publication = construct(
                PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class},
                graph,
                List.of(new ForwardPublicationBinding(new TensorId(9080), value.id())),
                List.of());
        return new CompileArtifacts(CompileMode.FORWARD_ONLY, graph, List.of(), LogicalMemoryPlanning.plan(graph, List.of()), publication, constants, construct(CompileDiagnostics.class, new Class<?>[] {List.class}, List.of()), new DerivativeGraphMetadata(graph, Map.of()));
    }

    private static CompileArtifacts resolvedBindableZeroNodeArtifacts() {
        Shape shape = Shape.of(1);
        TensorDescriptor descriptor = new TensorDescriptor(
                DataType.FLOAT32,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
        GraphValue value = new GraphValue(new ValueId(81), descriptor);
        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(value), List.of(), List.of(value.id()), List.of(value.id()), Map.of());
        CompileConstantPlan constants = construct(
                CompileConstantPlan.class,
                new Class<?>[] {List.class, List.class},
                List.of(new CompileConstantPlan.BindableInput(new TensorId(9081), value.id())),
                List.of());
        PublicationPlan publication = construct(
                PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class},
                graph,
                List.of(new ForwardPublicationBinding(new TensorId(9181), value.id())),
                List.of());
        return new CompileArtifacts(CompileMode.FORWARD_ONLY, graph, List.of(), LogicalMemoryPlanning.plan(graph, List.of()), publication, constants, construct(CompileDiagnostics.class, new Class<?>[] {List.class}, List.of()), new DerivativeGraphMetadata(graph, Map.of()));
    }

    private static CompileArtifacts withAliasedResults(CompileArtifacts source) {
        ValueId output = source.graph().outputs().getFirst();
        PublicationPlan publication = construct(
                PublicationPlan.class,
                new Class<?>[] {CompiledGraphModel.class, List.class, List.class},
                source.graph(),
                List.of(new ForwardPublicationBinding(new TensorId(300), output)),
                List.of(
                        new GradientPublicationBinding(1, 0, new TensorId(301), output),
                        new GradientPublicationBinding(1, 1, new TensorId(302), output)));
        return new CompileArtifacts(CompileMode.FORWARD_AND_BACKWARD, source.graph(), source.partitions(), source.memory(), publication, source.constants(), source.diagnostics(), source.derivatives());
    }

    private static <T extends Throwable> void assertFailure(
            Class<T> type, String message, org.junit.jupiter.api.function.Executable action) {
        assertEquals(message, assertThrows(type, action).getMessage());
    }

    record Fixture(
            CompileArtifacts artifacts,
            List<PlannedPartition> partitions,
            List<CompiledNode> nodes,
            List<GraphValue> values,
            ScalarValue constantValue) {}

    private record ProducerlessFixture(
            CompileArtifacts artifacts,
            PlannedPartition partition,
            GraphValue ordinaryInput,
            GraphValue ordinaryOutput,
            List<GraphValue> constants) {}

    private record FakeInputs(String target) implements BackendAnalysisInputs {}

    private record FakePlan(String route) implements BackendPreparationPlan {}

    private static final class FakeBuffer implements BufferRepresentation {
        @Override
        public void close() {}
    }

    private static final class TestExecutable extends PreparedExecutable {
        private TestExecutable(PreparedMemoryPlan memoryPlan) {
            super(memoryPlan, List.of(), List.of());
        }

        @Override
        protected boolean acceptsBufferRepresentation(
                int selectionIndex, BufferRepresentation representation) {
            return true;
        }

        @Override
        protected boolean acceptsWorkspaceRepresentation(
                int selectionIndex, WorkspaceRepresentation representation) {
            return true;
        }

        @Override
        protected BoundInvocation bindCompatible(
                RunState runState,
                BufferRepresentation[] bufferRepresentations,
                WorkspaceRepresentation[] workspaceRepresentations) {
            throw new UnsupportedOperationException();
        }
    }

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
