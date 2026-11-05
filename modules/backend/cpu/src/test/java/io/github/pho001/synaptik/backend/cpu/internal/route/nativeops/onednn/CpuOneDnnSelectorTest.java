package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PreparationResourceRequirement;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Real-library finite-domain checks; no native availability or successful metadata is simulated. */
final class CpuOneDnnSelectorTest {
    @Test
    void tinyMatmulKeepsBothApisEvenWhenOrdinaryHeuristicRejectsThem() {
        var context = matmulContext(false);
        var ordinary = new CpuPartitionPreparer().analyze(context);
        try (var library = openLibrary()) {
            var selector = new CpuOneDnnSelector(library, CpuOneDnnRoutePlan.Api.AUTO);
            assertSame(ordinary, selector.select(context, ordinary, ordinary));
            var alternatives = selector.alternatives(context, ordinary);
            assertEquals(alternatives.size(), selector.alternativeCount(context, ordinary));
            assertEquals(List.of(2), selector.alternativeCountsByUnit(context, ordinary));
            assertEquals(List.of(CpuOneDnnRoutePlan.Api.PRIMITIVE, CpuOneDnnRoutePlan.Api.GRAPH),
                    alternatives.stream().map(candidate -> candidate.plan().oneDnnPlan().orElseThrow()
                            .units().getFirst().api()).toList());
            for (var candidate : alternatives) {
                var route = candidate.plan().oneDnnPlan().orElseThrow();
                assertEquals(87, selector.candidateFingerprint(candidate.plan()).length());
                assertEquals(0, route.portableCost());
                assertEquals(0, route.selectedCost());
                route.validate(ordinary.plan().units());
                assertTrue(candidate.requirements().containsAll(route.workspaceRequirements()));
                assertEquals(2, route.units().getFirst().inputCopies().size(),
                        "bindable operands require truthful ingress declarations");
            }
            assertEquals(alternatives.stream().map(candidate -> selector.candidateFingerprint(candidate.plan())).toList(),
                    selector.alternatives(context, ordinary).stream()
                            .map(candidate -> selector.candidateFingerprint(candidate.plan())).toList());
            assertNotEquals(selector.candidateFingerprint(alternatives.getFirst().plan()),
                    selector.candidateFingerprint(alternatives.getLast().plan()));
            assertEquals(CpuPartitionPreparationPlan.Route.PORTABLE, ordinary.plan().route());
        }
    }

    @Test
    void chainedMatmulEnumeratesAllOrdinaryNativeAssignmentsAndSharesIngress() {
        var context = matmulContext(true);
        var ordinary = new CpuPartitionPreparer().analyze(context);
        assertEquals(2, ordinary.plan().units().size());
        try (var library = openLibrary()) {
            var selector = new CpuOneDnnSelector(library, CpuOneDnnRoutePlan.Api.AUTO);
            var alternatives = selector.alternatives(context, ordinary);
            assertEquals(8, alternatives.size(), "(ordinary + primitive + Graph)^2 - all ordinary");
            assertEquals(List.of(2, 2), selector.alternativeCountsByUnit(context, ordinary));
            assertEquals(List.of("1:PRIMITIVE", "1:GRAPH", "0:PRIMITIVE", "0:PRIMITIVE,1:PRIMITIVE",
                    "0:PRIMITIVE,1:GRAPH", "0:GRAPH", "0:GRAPH,1:PRIMITIVE", "0:GRAPH,1:GRAPH"),
                    alternatives.stream().map(candidate -> assignment(candidate.plan())).toList());
            var both = alternatives.get(3).plan().oneDnnPlan().orElseThrow();
            var firstCopies = both.units().getFirst().inputCopies();
            var secondCopies = both.units().getLast().inputCopies();
            assertEquals(2, firstCopies.size());
            assertEquals(1, secondCopies.size(), "ordinary-produced intermediate already has native storage");
            assertSame(firstCopies.get(1).workspace(), secondCopies.getFirst().workspace());
            assertTrue(firstCopies.get(1).refresh());
            assertFalse(secondCopies.getFirst().refresh());
            assertEquals(4, both.workspaceRequirements().size(), "two shared ingress spans and two private scratch declarations");
            both.validate(ordinary.plan().units());
            for (var api : List.of(CpuOneDnnRoutePlan.Api.PRIMITIVE, CpuOneDnnRoutePlan.Api.GRAPH)) {
                var constrained = new CpuOneDnnSelector(library, api).alternatives(context, ordinary);
                assertEquals(3, constrained.size(), "(ordinary + selected API)^2 - all ordinary");
                assertEquals(List.of(1, 1),
                        new CpuOneDnnSelector(library, api).alternativeCountsByUnit(context, ordinary));
                assertTrue(constrained.stream().allMatch(candidate -> candidate.plan().oneDnnPlan().orElseThrow()
                        .units().stream().allMatch(unit -> unit.api() == api)));
            }
        }
    }

    @Test
    void phaseOneApiAndResourcesAreFrozenWithoutHeuristicSubstitution() {
        var context = matmulContext(false);
        var ordinary = new CpuPartitionPreparer().analyze(context);
        try (var library = openLibrary()) {
            var selector = new CpuOneDnnSelector(library, CpuOneDnnRoutePlan.Api.AUTO);
            var alternatives = selector.alternatives(context, ordinary);
            assertEquals(2, alternatives.size());
            for (var candidate : alternatives) {
                var frozen = selector.freeze(context, new CpuPartitionPreparer().analyze(context), candidate.plan());
                assertEquals(candidate.plan().oneDnnPlan().orElseThrow().units(),
                        frozen.plan().oneDnnPlan().orElseThrow().units());
                assertEquals(candidate.requirements(), frozen.requirements());
                assertEquals(selector.candidateFingerprint(candidate.plan()), selector.candidateFingerprint(frozen.plan()));
            }
            var graph = alternatives.getLast().plan();
            assertThrows(IllegalArgumentException.class, () -> new CpuOneDnnSelector(library,
                    CpuOneDnnRoutePlan.Api.PRIMITIVE).freeze(context, ordinary, graph));
            var route = alternatives.getFirst().plan().oneDnnPlan().orElseThrow();
            var unit = route.units().getFirst();
            var changedScratch = new CpuOneDnnRoutePlan.Unit(unit.unitPosition(), unit.projection(), unit.api(),
                    unit.scratchpadBytes() + 64, unit.implementation(), unit.inputCopies(),
                    new PreparationResourceRequirement.Workspace(unit.invocationWorkspace().requirementId(),
                            unit.scratchpadBytes() + 64, 64), unit.outputStagingWorkspace());
            var changed = ordinary.plan().withOneDnn(new CpuOneDnnRoutePlan(route.sessionIdentity(),
                    List.of(changedScratch), 0, 0));
            assertThrows(IllegalArgumentException.class, () -> selector.freeze(context, ordinary, changed));
            try (var anotherLibrary = openLibrary()) {
                var otherSession = new CpuOneDnnSelector(anotherLibrary, CpuOneDnnRoutePlan.Api.AUTO);
                assertThrows(IllegalArgumentException.class, () -> otherSession.freeze(context, ordinary, graph));
            }
        }
    }

    @Test
    void qualifiedNonMatmulFamilyEnumeratesOnlySupportedApis() {
        var node = new CompiledNode(new NodeId(0),
                new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0)), List.of(new ValueId(1)));
        var context = context(List.of(node), 2);
        var ordinary = new CpuPartitionPreparer().analyze(context);
        try (var library = openLibrary()) {
            var auto = new CpuOneDnnSelector(library, CpuOneDnnRoutePlan.Api.AUTO);
            var alternatives = auto.alternatives(context, ordinary);
            assertEquals(1, alternatives.size(), "ABS is qualified only for the primitive API");
            assertEquals(List.of(1), auto.alternativeCountsByUnit(context, ordinary));
            assertEquals(List.of(0), new CpuOneDnnSelector(library, CpuOneDnnRoutePlan.Api.GRAPH)
                    .alternativeCountsByUnit(context, ordinary));
            assertEquals(CpuOneDnnRoutePlan.Api.PRIMITIVE,
                    alternatives.getFirst().plan().oneDnnPlan().orElseThrow().units().getFirst().api());
            assertTrue(new CpuOneDnnSelector(library, CpuOneDnnRoutePlan.Api.GRAPH)
                    .alternatives(context, ordinary).isEmpty());
        }
    }

    @Test
    void lowMatmulRetainsDeclaredF32StagingAndExactIngressSpans() {
        var base = matmulContext(false);
        var context = context(base.nodes(), 3, DataType.BFLOAT16);
        var ordinary = new CpuPartitionPreparer().analyze(context);
        try (var library = openLibrary()) {
            var selector = new CpuOneDnnSelector(library, CpuOneDnnRoutePlan.Api.PRIMITIVE);
            var alternatives = selector.alternatives(context, ordinary);
            assertEquals(1, alternatives.size());
            var candidate = alternatives.getFirst();
            var unit = candidate.plan().oneDnnPlan().orElseThrow().units().getFirst();
            assertTrue(unit.projection().requiresOutputNarrowing());
            assertEquals(DataType.BFLOAT16, unit.projection().logicalOutputType());
            assertEquals(DataType.FLOAT32, unit.projection().spec().output().dataType());
            assertEquals(16, unit.outputStagingWorkspace().orElseThrow().byteSize());
            assertTrue(unit.inputCopies().stream().allMatch(copy -> copy.workspace().byteSize() == 8));
            assertEquals(candidate.requirements(),
                    selector.freeze(context, ordinary, candidate.plan()).requirements());
        }
    }

    @Test
    void unqualifiedUnaryNeverBecomesANativeAlternative() {
        var node = new CompiledNode(new NodeId(0),
                new Operation(UnaryElementwiseKind.RELU, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0)), List.of(new ValueId(1)));
        var context = context(List.of(node), 2);
        var ordinary = new CpuPartitionPreparer().analyze(context);
        try (var library = openLibrary()) {
            assertTrue(new CpuOneDnnSelector(library, CpuOneDnnRoutePlan.Api.AUTO)
                    .alternatives(context, ordinary).isEmpty());
            assertEquals(List.of(0), new CpuOneDnnSelector(library, CpuOneDnnRoutePlan.Api.AUTO)
                    .alternativeCountsByUnit(context, ordinary));
        }
    }

    @Test
    void oversizedCompleteNativeDomainIsRejectedInsteadOfReturningAPrefix() {
        var nodes = new ArrayList<CompiledNode>();
        for (int position = 0; position < 9; position++) {
            nodes.add(new CompiledNode(new NodeId(position),
                    new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                    List.of(new ValueId(position == 0 ? 0 : position + 1), new ValueId(1)),
                    List.of(new ValueId(position + 2))));
        }
        var context = context(nodes, 11);
        var ordinary = new CpuPartitionPreparer().analyze(context);
        assertEquals(9, ordinary.plan().units().size());
        try (var library = openLibrary()) {
            var selector = new CpuOneDnnSelector(library, CpuOneDnnRoutePlan.Api.AUTO);
            assertThrows(IllegalArgumentException.class, () -> selector.alternativeCount(context, ordinary));
            var failure = assertThrows(IllegalArgumentException.class,
                    () -> selector.alternatives(context, ordinary));
            assertTrue(failure.getMessage().contains("complete oneDNN assignment domain exceeds 6560"));
        }
    }

    @Test
    void noLocalCompleteEnumerationRetainsOrdinaryTopologiesAndFreezesEveryNativeIdentity() {
        var nodes = List.of(
                new CompiledNode(new NodeId(0),
                        new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE),
                        List.of(new ValueId(0)), List.of(new ValueId(1))),
                new CompiledNode(new NodeId(1),
                        new Operation(UnaryElementwiseKind.SQRT, NoOperationAttrs.INSTANCE),
                        List.of(new ValueId(1)), List.of(new ValueId(2))));
        var context = context(nodes, 3);
        var ordinaryPreparer = new CpuPartitionPreparer();
        assertTrue(ordinaryPreparer.analyze(context).plan().localTuningBatch().isEmpty());
        var ordinary = ordinaryPreparer.completePlanCandidates(context);
        assertTrue(ordinary.complete());
        assertTrue(ordinary.identities().stream().map(identity -> identity.topology().topology())
                .distinct().count() >= 2, "the fixture must exercise split and fused ordinary topologies");
        try (var library = openLibrary()) {
            var selector = new CpuOneDnnSelector(library, CpuOneDnnRoutePlan.Api.AUTO);
            var preparer = new CpuPartitionPreparer(new CpuPartitionLowering(), Optional.of(selector));
            assertTrue(preparer.analyze(context).plan().localTuningBatch().isEmpty(),
                    "non-MATMUL native alternatives belong to complete-plan enumeration, not Phase 1");
            var expanded = preparer.completePlanCandidates(context);
            assertTrue(expanded.complete());
            assertTrue(expanded.identities().containsAll(ordinary.identities()),
                    "native expansion must not delete any retained ordinary topology/representation");
            assertEquals(ordinary.identities().stream().map(identity -> identity.topology())
                            .collect(java.util.stream.Collectors.toSet()),
                    expanded.identities().stream().map(identity -> identity.topology())
                            .collect(java.util.stream.Collectors.toSet()));
            for (var identity : ordinary.identities()) {
                var baseline = ordinaryPreparer.analyzeSelected(context,
                        new CpuPartitionPreparer.SelectedCompletePlan(5, identity, List.of(1)));
                long combinations = 1;
                for (int nativeCount : selector.alternativeCountsByUnit(context, baseline)) {
                    combinations = Math.multiplyExact(combinations, 1L + nativeCount);
                }
                assertEquals(combinations, expanded.identities().stream()
                        .filter(candidate -> candidate.topology().equals(identity.topology())).count(),
                        "each non-MATMUL unit has one ordinary choice plus all permitted native APIs");
            }
            var nativeIdentities = expanded.identities().stream()
                    .filter(identity -> identity.nativeAssignment().isPresent()).toList();
            assertFalse(nativeIdentities.isEmpty());
            assertTrue(nativeIdentities.stream().flatMap(identity ->
                    identity.nativeAssignment().orElseThrow().plan().units().stream())
                    .anyMatch(unit -> unit.api() == CpuOneDnnRoutePlan.Api.GRAPH));
            assertTrue(nativeIdentities.stream().flatMap(identity ->
                    identity.nativeAssignment().orElseThrow().plan().units().stream())
                    .anyMatch(unit -> unit.api() == CpuOneDnnRoutePlan.Api.PRIMITIVE));
            for (var identity : nativeIdentities) {
                var retained = identity.nativeAssignment().orElseThrow();
                var fresh = preparer.analyzeSelected(context,
                        new CpuPartitionPreparer.SelectedCompletePlan(5, identity, List.of(1)));
                assertTrue(fresh.plan().localTuningBatch().isEmpty());
                assertEquals(retained.plan().units(), fresh.plan().oneDnnPlan().orElseThrow().units());
                assertEquals(retained.fingerprint(), selector.candidateFingerprint(fresh.plan()));
                assertTrue(fresh.requirements().containsAll(retained.plan().workspaceRequirements()));
            }
        }
    }

    private static String assignment(CpuPartitionPreparationPlan plan) {
        return plan.oneDnnPlan().orElseThrow().units().stream()
                .map(unit -> unit.unitPosition() + ":" + unit.api())
                .collect(java.util.stream.Collectors.joining(","));
    }

    private static CpuOneDnnLibrary openLibrary() {
        String configured = System.getenv("SYNAPTIK_ONEDNN_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(), "explicit native library required");
        Path path = Path.of(configured);
        assertTrue(path.isAbsolute(), "native library must be an absolute path");
        return CpuOneDnnLibrary.open(path);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> matmulContext(boolean chain) {
        var nodes = new ArrayList<CompiledNode>();
        nodes.add(new CompiledNode(new NodeId(0), new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(0), new ValueId(1)), List.of(new ValueId(2))));

        if (chain) nodes.add(new CompiledNode(new NodeId(1),
                new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                List.of(new ValueId(2), new ValueId(1)), List.of(new ValueId(3))));
        return context(nodes, chain ? 4 : 3);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(List<CompiledNode> nodes, int valueCount) {
        return context(nodes, valueCount, DataType.FLOAT32);
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(
            List<CompiledNode> nodes, int valueCount, DataType type) {
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,
                nodes.stream().map(CompiledNode::id).toList());
        Shape shape = Shape.of(2, 2);
        var descriptor = new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var values = new ArrayList<GraphValue>();
        var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int position = 0; position < valueCount; position++) {
            var id = new ValueId(position);
            boolean produced = nodes.stream().anyMatch(node -> node.outputs().contains(id));
            boolean consumed = nodes.stream().anyMatch(node -> node.inputs().contains(id));
            values.add(new GraphValue(id, descriptor));
            memory.add(new LogicalMemoryRequirement(id, descriptor,
                    produced ? Optional.of(partition) : Optional.empty(),
                    consumed ? List.of(partition) : List.of(), nodes.getLast().outputs().contains(id)));
        }
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(),
                new CpuPartitionAnalysisInputs(false, List.of(),
                        new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.SCALAR, 1, 1, 1)));
    }
}
