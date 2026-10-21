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
            for (boolean bias : List.of(false, true)) for (boolean grad : List.of(false, true))
                for (var type : floatingTypes()) for (int layout = 0; layout < 6; layout++) {
                var context = scalarContext(kind, bias, direction, 3, 4, 2, 3, grad, type, layout);
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
                assertEquals(2L * 3 * (kind == RecurrentScanKind.LSTM ? 4 : kind == RecurrentScanKind.GRU_RESET_AFTER ? 3 : 1)
                                * (type == DataType.FLOAT64 ? 8 : 4),
                        unit.runtimeFacts().workspaceDeclaration().orElseThrow().byteSize());
                assertEquals(type == DataType.FLOAT64 ? 8 : 4,
                        unit.runtimeFacts().workspaceDeclaration().orElseThrow().byteAlignment());
                var candidates = new CpuPartitionPreparer().completePlanCandidates(context);
                assertFalse(candidates.complete()); assertTrue(candidates.identities().isEmpty());
            }
    }

    @Test void scratchAndNodeCountDoNotGrowWithTimeAndEmptyDomainsOmitScratch() {
        for (var type : floatingTypes()) for (int time : List.of(0, 1, 10000)) for (int batch : List.of(0, 4)) {
            var analysis = new CpuPartitionPreparer().analyze(scalarContext(RecurrentScanKind.LSTM, true,
                    RecurrentDirection.REVERSE, time, batch, 2, 3, false, type, 2));
            assertEquals(1, analysis.plan().units().size());
            var workspace = analysis.plan().units().getFirst().runtimeFacts().workspaceDeclaration();
            assertEquals(time > 0 && batch > 0, workspace.isPresent());
            workspace.ifPresent(w -> { assertEquals(type == DataType.FLOAT64 ? 192 : 96, w.byteSize());
                assertEquals(type == DataType.FLOAT64 ? 8 : 4, w.byteAlignment()); });
        }
    }

    @Test void providerAndLowererRejectTypesFlagsLayoutsShapesAndOverflowTogether() {
        var context = context(RecurrentScanKind.LSTM, true, RecurrentDirection.FORWARD, 3, 2, 2, 2, false);
        var query = query(context);
        for (int role = 0; role < query.inputs().size(); role++) {
            var original = query.inputs().get(role);
            var changes = new ArrayList<TensorDescriptor>();
            changes.add(new TensorDescriptor(original.dataType(), original.shape(), Optional.empty(), original.requiresGrad()));
            changes.add(new TensorDescriptor(role == 1 ? DataType.INT32 : DataType.FLOAT64, original.shape(), original.layout(), false));
            if (role != 1) changes.add(new TensorDescriptor(original.dataType(), original.shape(), original.layout(), true));
            changes.add(descriptor(original.dataType(), Shape.of(1), false));
            for (var change : changes) {
                var in = new ArrayList<>(query.inputs()); in.set(role, change);
                rejected(new OperationCapabilityQuery(query.operation(), in, query.outputs()));
            }
        }
        var mixed = new ArrayList<>(query.inputs());
        var state = mixed.get(2);
        mixed.set(2, new TensorDescriptor(DataType.BFLOAT16, state.shape(), state.layout(), false));
        rejected(new OperationCapabilityQuery(query.operation(), mixed, query.outputs()));
        var spanOverflow = new ArrayList<>(query.inputs());
        var x = spanOverflow.getFirst();
        spanOverflow.set(0, new TensorDescriptor(x.dataType(), x.shape(), Optional.of(LayoutDescriptor.of(
                x.shape(), new long[] {Integer.MAX_VALUE, 2, 1}, 3, true)), false));
        rejected(new OperationCapabilityQuery(query.operation(), spanOverflow, query.outputs()));
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
            outputs.set(port, new TensorDescriptor(d.dataType(), d.shape(), Optional.of(LayoutDescriptor.of(
                    d.shape(), d.layout().orElseThrow().strides(), 1, true)), false));
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
        for (var type : floatingTypes()) for (var kind : RecurrentScanKind.values()) {
            var query = query(scalarContext(kind, true, RecurrentDirection.REVERSE, 2, 2, 2, 2, false, type, 2));
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

    public static List<DataType> floatingTypes() {
        return List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16);
    }

    /** Builds independent resolved read geometries, leaving canonical sibling outputs untouched. */
    public static PrepareContext<CpuPartitionAnalysisInputs> scalarContext(RecurrentScanKind kind, boolean bias,
            RecurrentDirection direction, int t, int b, int i, int h, boolean grad, DataType type, int mode) {
        var base = context(kind, bias, direction, t, b, i, h, grad);
        int inputs = base.nodes().getFirst().inputs().size();
        var descriptors = new ArrayList<TensorDescriptor>();
        for (int p = 0; p < base.values().size(); p++) {
            var d = base.values().get(p).descriptor();
            var layout = d.layout().orElseThrow();
            if (p < inputs && mode != 0) {
                long[] strides = layout.strides();
                if (mode == 2) for (int axis = 0; axis < strides.length; axis++) strides[axis] *= 2;
                if (mode == 3) {
                    long stride = 1;
                    for (int axis = 0; axis < strides.length; axis++) { strides[axis] = stride; stride *= d.shape().toLongArray()[axis]; }
                }
                if (mode == 4) Arrays.fill(strides, 0);
                if (mode == 5) Arrays.fill(strides, 1);
                layout = LayoutDescriptor.of(d.shape(), strides, 3, true);
            }
            descriptors.add(new TensorDescriptor(p == 1 ? DataType.INT64 : type, d.shape(), Optional.of(layout), d.requiresGrad()));
        }
        return project(descriptors, base.nodes());
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
