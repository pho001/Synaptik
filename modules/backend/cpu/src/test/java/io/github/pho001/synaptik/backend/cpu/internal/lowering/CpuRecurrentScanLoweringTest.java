package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.recurrent.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Descriptor-only admission, resource and constant-graph-size evidence for direct scans. */
public class CpuRecurrentScanLoweringTest {
    @Test void allSignaturesDirectionsAndFlagsUseOneDirectScalarUnit() {
        for (var kind : RecurrentScanKind.values()) for (var direction : RecurrentDirection.values())
            for (boolean bias : List.of(false, true)) for (boolean grad : List.of(false, true)) {
                var context = context(kind, bias, direction, 3, 4, 2, 3, grad);
                var lowered = new CpuPartitionLowering().lower(context);
                assertNull(lowered.portableKernelIr());
                assertThrows(IllegalStateException.class, lowered::kernelIr);
                var analysis = new CpuPartitionPreparer().analyze(context);
                var unit = analysis.plan().units().getFirst();
                assertNull(unit.portablePlan());
                assertEquals(1, analysis.plan().units().size());
                assertEquals(CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR, unit.executionStrategy());
                assertEquals(1, unit.selectedRangeCount()); assertEquals(0, unit.vectorSpeciesBitSize());
                assertEquals(context.nodes().getFirst().outputs().size(), unit.outputCount());
                assertEquals(lowered.recurrentScan(), unit.runtimeFacts().recurrentScan());
                assertEquals(2L * 3 * (kind == RecurrentScanKind.LSTM ? 4 : kind == RecurrentScanKind.GRU_RESET_AFTER ? 3 : 1) * 4,
                        unit.runtimeFacts().workspaceDeclaration().orElseThrow().byteSize());
                var candidates = new CpuPartitionPreparer().completePlanCandidates(context);
                assertFalse(candidates.complete()); assertTrue(candidates.identities().isEmpty());
            }
    }

    @Test void scratchAndNodeCountDoNotGrowWithTimeAndEmptyDomainsOmitScratch() {
        for (int time : List.of(0, 1, 10000)) for (int batch : List.of(0, 4)) {
            var analysis = new CpuPartitionPreparer().analyze(context(RecurrentScanKind.LSTM, true,
                    RecurrentDirection.REVERSE, time, batch, 2, 3, false));
            assertEquals(1, analysis.plan().units().size());
            var workspace = analysis.plan().units().getFirst().runtimeFacts().workspaceDeclaration();
            assertEquals(time > 0 && batch > 0, workspace.isPresent());
            workspace.ifPresent(w -> assertEquals(96, w.byteSize()));
        }
    }

    @Test void providerAndLowererRejectTypesFlagsLayoutsShapesAndOverflowTogether() {
        var context = context(RecurrentScanKind.LSTM, true, RecurrentDirection.FORWARD, 3, 2, 2, 2, false);
        var query = query(context);
        for (int role = 0; role < query.inputs().size(); role++) {
            var original = query.inputs().get(role);
            var changes = new ArrayList<TensorDescriptor>();
            changes.add(new TensorDescriptor(original.dataType(), original.shape(), Optional.empty(), original.requiresGrad()));
            changes.add(new TensorDescriptor(original.dataType(), original.shape(), Optional.of(LayoutDescriptor.of(
                    original.shape(), original.layout().orElseThrow().strides(), 1, true)), original.requiresGrad()));
            changes.add(new TensorDescriptor(role == 1 ? DataType.INT32 : DataType.FLOAT64, original.shape(), original.layout(), false));
            if (role != 1) changes.add(new TensorDescriptor(original.dataType(), original.shape(), original.layout(), true));
            changes.add(descriptor(original.dataType(), Shape.of(1), false));
            for (var change : changes) {
                var in = new ArrayList<>(query.inputs()); in.set(role, change);
                rejected(new OperationCapabilityQuery(query.operation(), in, query.outputs()));
            }
        }
        for (var low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            rejected(new OperationCapabilityQuery(query.operation(), query.inputs().stream().map(d ->
                    d.dataType() == DataType.INT64 ? d : new TensorDescriptor(low, d.shape(), d.layout(), d.requiresGrad())).toList(),
                    query.outputs().stream().map(d -> new TensorDescriptor(low, d.shape(), d.layout(), d.requiresGrad())).toList()));
        }
        var in = new ArrayList<>(query.inputs());
        in.set(0, descriptor(DataType.FLOAT32, Shape.of(1L << 32, 2, 2), false));
        rejected(new OperationCapabilityQuery(query.operation(), in, query.outputs()));
        in.set(0, descriptor(DataType.FLOAT32, Shape.of(50000, 50000, 2), false));
        rejected(new OperationCapabilityQuery(query.operation(), in, query.outputs()));
        for (int port = 0; port < query.outputs().size(); port++) {
            var outputs = new ArrayList<>(query.outputs());
            var d = outputs.get(port);
            outputs.set(port, new TensorDescriptor(d.dataType(), d.shape(), d.layout(), true));
            rejected(new OperationCapabilityQuery(query.operation(), query.inputs(), outputs));
        }
    }

    @Test void repeatedSemanticRolesKeepOneReadBoundaryAndEverySibling() {
        var c = context(RecurrentScanKind.LSTM, false, RecurrentDirection.FORWARD, 2, 2, 2, 2, false);
        var old = c.nodes().getFirst();
        var inputs = new ArrayList<>(old.inputs()); inputs.set(3, inputs.get(2));
        inputs.set(5, inputs.get(4));
        var node = new CompiledNode(old.id(), old.operation(), inputs, old.outputs());
        var projected = project(c.values().stream().map(GraphValue::descriptor).toList(), List.of(node));
        var facts = new CpuPartitionLowering().lower(projected).recurrentScan().orElseThrow();
        assertEquals(4, facts.readCount());
        assertEquals(List.of(0, 1, 2, 2, 3, 3), facts.rolePositions());
        assertEquals(old.outputs(), facts.boundaryValues().subList(4, 7));
    }

    @Test void everyFloatingRoleIndependentlyContributesToExactOutputFlagOr() {
        for (var kind : RecurrentScanKind.values()) {
            var query = query(context(kind, true, RecurrentDirection.REVERSE, 2, 2, 2, 2, false));
            for (int role = 0; role < query.inputs().size(); role++) if (role != 1) {
                var in = new ArrayList<>(query.inputs()); var d = in.get(role);
                in.set(role, new TensorDescriptor(d.dataType(), d.shape(), d.layout(), true));
                var out = query.outputs().stream().map(o -> new TensorDescriptor(o.dataType(), o.shape(), o.layout(), true)).toList();
                assertTrue(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(query.operation(), in, out)));
                rejected(new OperationCapabilityQuery(query.operation(), in, query.outputs()));
            }
        }
    }

    private static void rejected(OperationCapabilityQuery query) {
        assertFalse(new CpuCapabilityProvider().supports(query));
        assertThrows(RuntimeException.class, () -> CpuRecurrentScanLowering.geometry(query));
    }

    public static OperationCapabilityQuery query(PrepareContext<CpuPartitionAnalysisInputs> c) {
        var node = c.nodes().getFirst();
        var map = new HashMap<ValueId, TensorDescriptor>(); c.values().forEach(v -> map.put(v.id(), v.descriptor()));
        return new OperationCapabilityQuery(node.operation(), node.inputs().stream().map(map::get).toList(), node.outputs().stream().map(map::get).toList());
    }

    public static TensorDescriptor descriptor(DataType type, Shape shape, boolean grad) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> context(RecurrentScanKind kind, boolean bias,
            RecurrentDirection direction, int t, int b, int i, int h, boolean grad) {
        int gates = kind == RecurrentScanKind.LSTM ? 4 : kind == RecurrentScanKind.GRU_RESET_AFTER ? 3 : 1;
        var descriptors = new ArrayList<TensorDescriptor>();
        descriptors.add(descriptor(DataType.FLOAT32, Shape.of(t, b, i), grad));
        descriptors.add(descriptor(DataType.INT64, Shape.of(b), false));
        descriptors.add(descriptor(DataType.FLOAT32, Shape.of(b, h), false));
        if (kind == RecurrentScanKind.LSTM) descriptors.add(descriptor(DataType.FLOAT32, Shape.of(b, h), false));
        descriptors.add(descriptor(DataType.FLOAT32, Shape.of(gates * h, i), false));
        descriptors.add(descriptor(DataType.FLOAT32, Shape.of(gates * h, h), false));
        if (bias) descriptors.add(descriptor(DataType.FLOAT32, Shape.of(gates * h), false));
        int reads = descriptors.size();
        descriptors.add(descriptor(DataType.FLOAT32, Shape.of(t, b, h), grad));
        descriptors.add(descriptor(DataType.FLOAT32, Shape.of(b, h), grad));
        if (kind == RecurrentScanKind.LSTM) descriptors.add(descriptor(DataType.FLOAT32, Shape.of(b, h), grad));
        var node = new CompiledNode(new NodeId(0), new Operation(kind, direction),
                java.util.stream.IntStream.range(0, reads).mapToObj(ValueId::new).toList(),
                java.util.stream.IntStream.range(reads, descriptors.size()).mapToObj(ValueId::new).toList());
        return project(descriptors, List.of(node));
    }

    public static PrepareContext<CpuPartitionAnalysisInputs> project(List<TensorDescriptor> descriptors, List<CompiledNode> nodes) {
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, nodes.stream().map(CompiledNode::id).toList());
        var values = new ArrayList<GraphValue>(); var memory = new ArrayList<LogicalMemoryRequirement>();
        for (int p = 0; p < descriptors.size(); p++) {
            var id = new ValueId(p); var d = descriptors.get(p);
            boolean produced = nodes.stream().anyMatch(n -> n.outputs().contains(id));
            boolean consumed = nodes.stream().anyMatch(n -> n.inputs().contains(id));
            values.add(new GraphValue(id, d));
            memory.add(new LogicalMemoryRequirement(id, d, produced ? Optional.of(partition) : Optional.empty(),
                    consumed ? List.of(partition) : List.of(), produced));
        }
        var config = new CpuPartitionAnalysisInputs.PortableExecutionConfig(
                CpuPartitionAnalysisInputs.PortableExecutionConfig.ComputePreference.VECTOR_IF_ELIGIBLE, 4, 4, 1);
        return new PrepareContext<>(partition, nodes, values, memory, Map.of(), new CpuPartitionAnalysisInputs(false, List.of(), config));
    }
}
