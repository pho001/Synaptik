package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.executable.*;
import io.github.pho001.synaptik.backend.cpu.internal.memory.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.recurrent.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.*;
import io.github.pho001.synaptik.prepare.analysis.*;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.*;
import io.github.pho001.synaptik.runtime.run.*;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Provider-to-finalization qualification, independent equations and exact padding/publication checks. */
final class CpuRecurrentScanConformanceTest {
    @Test void scalarDtypesQualifyAllSignaturesDirectionsAndResolvedReadLayoutsIndependently() {
        for (var type : List.of(DataType.FLOAT32, DataType.FLOAT64, DataType.BFLOAT16, DataType.FLOAT16))
            for (var kind : RecurrentScanKind.values()) for (boolean biased : List.of(false, true))
                for (var direction : RecurrentDirection.values()) for (int layoutMode = 0; layoutMode < 3; layoutMode++) {
                    var base = context(kind, biased, direction, 3, 3, 1, 1, false);
                    var ds = new ArrayList<TensorDescriptor>();
                    int inputCount = base.nodes().getFirst().inputs().size();
                    for (int p = 0; p < base.values().size(); p++) {
                        var d = base.values().get(p).descriptor();
                        var layout = d.layout().orElseThrow();
                        if (p < inputCount && layoutMode > 0) {
                            long[] strides = layout.strides();
                            for (int axis = 0; axis < strides.length; axis++) strides[axis] = layoutMode == 1 ? 2 * strides[axis] : 0;
                            layout = LayoutDescriptor.of(d.shape(), strides, 3, true);
                        }
                        ds.add(new TensorDescriptor(p == 1 ? DataType.INT64 : type, d.shape(), Optional.of(layout), false));
                    }
                    var context = project(ds, base.nodes());
                    var node = context.nodes().getFirst();
                    var query = new io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery(node.operation(),
                            ds.subList(0, inputCount), ds.subList(inputCount, ds.size()));
                    assertTrue(new CpuCapabilityProvider().supports(query), type + " independent provider query");
                    var analysis = new CpuPartitionPreparer().analyze(context);
                    var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
                    try (var f = new Fixture(analysis, 2, true, true, Map.of())) {
                        for (int role = 0; role < inputCount; role++) {
                            f.segment(role).fill((byte) 0x7f); // Poison prefix and strided gaps, including lengths.
                            var d = ds.get(role); long count = d.shape().knownElementCount().orElseThrow();
                            for (int k = 0; k < count; k++) {
                                long address = index(d, k);
                                if (role == 1) f.segment(role).setAtIndex(ValueLayout.JAVA_LONG, address, new long[] {0, 1, 3}[k]);
                                else set(type, f.segment(role), address, role == 0 ? (k + 1) * .04 : role == 2 ? .25
                                        : kind == RecurrentScanKind.LSTM && role == 3 ? .5
                                        : role == (kind == RecurrentScanKind.LSTM ? 4 : 3) ? .8
                                        : role == (kind == RecurrentScanKind.LSTM ? 5 : 4) ? -.3 : .07);
                            }
                        }
                        f.executable.bind(f.state).execute();
                        int out = facts.readCount(), weight = kind == RecurrentScanKind.LSTM ? 4 : 3;
                        for (int b = 0; b < 3; b++) {
                            long length = f.segment(1).getAtIndex(ValueLayout.JAVA_LONG, index(ds.get(1), b));
                            double hidden = get(type, f.segment(2), index(ds.get(2), b));
                            double cell = kind == RecurrentScanKind.LSTM ? get(type, f.segment(3), index(ds.get(3), b)) : 0;
                            for (int visit = 0; visit < length; visit++) {
                                int t = direction == RecurrentDirection.REVERSE ? (int) length - 1 - visit : visit;
                                double[] px = new double[facts.geometry().gates()], ph = new double[px.length];
                                for (int g = 0; g < px.length; g++) {
                                    px[g] = work(type, 0 + work(type, get(type, f.segment(0), index(ds.getFirst(), t * 3 + b))
                                            * get(type, f.segment(weight), index(ds.get(weight), g))));
                                    ph[g] = work(type, 0 + work(type, hidden * get(type, f.segment(weight + 1), index(ds.get(weight + 1), g))));
                                    if (biased) px[g] = work(type, px[g] + get(type, f.segment(weight + 2), index(ds.get(weight + 2), g)));
                                }
                                double next;
                                if (kind == RecurrentScanKind.RNN_TANH) next = work(type, StrictMath.tanh(work(type, px[0] + ph[0])));
                                else if (kind == RecurrentScanKind.GRU_RESET_AFTER) {
                                    double r = sigmoid(type, work(type, px[0] + ph[0])), z = sigmoid(type, work(type, px[1] + ph[1]));
                                    double n = work(type, StrictMath.tanh(work(type, px[2] + work(type, r * ph[2]))));
                                    next = work(type, n + work(type, z * work(type, hidden - n)));
                                } else {
                                    double i = sigmoid(type, work(type, px[0] + ph[0])), forget = sigmoid(type, work(type, px[1] + ph[1]));
                                    double n = work(type, StrictMath.tanh(work(type, px[2] + ph[2]))), o = sigmoid(type, work(type, px[3] + ph[3]));
                                    cell = published(type, work(type, work(type, forget * cell) + work(type, i * n)));
                                    next = work(type, o * work(type, StrictMath.tanh(cell)));
                                }
                                hidden = published(type, next);
                                assertEquals(hidden, get(type, f.segment(out), t * 3 + b), 0d,
                                        "rounded sequential oracle, not a tolerance for public state");
                            }
                            assertEquals(hidden, get(type, f.segment(out + 1), b), 0d);
                            if (kind == RecurrentScanKind.LSTM) assertEquals(cell, get(type, f.segment(out + 2), b), 0d);
                            for (int t = (int) length; t < 3; t++) for (byte v : f.segment(out).asSlice(
                                    (t * 3L + b) * type.byteWidth(), type.byteWidth()).toArray(ValueLayout.JAVA_BYTE)) assertEquals(0, v);
                        }
                    }
                }
    }

    private static long index(TensorDescriptor d, long ordinal) {
        var layout = d.layout().orElseThrow(); var extents = d.shape().toLongArray();
        long result = layout.storageOffset();
        for (int axis = extents.length - 1; axis >= 0; axis--) { result += ordinal % extents[axis] * layout.stride(axis); ordinal /= extents[axis]; }
        return result;
    }
    private static double work(DataType type, double value) { return type == DataType.FLOAT64 ? value : (float) value; }
    private static double published(DataType type, double value) {
        return type == DataType.BFLOAT16 ? BFloat16Bits.toFloat(BFloat16Bits.fromFloat((float) value))
                : type == DataType.FLOAT16 ? Float16Bits.toFloat(Float16Bits.fromFloat((float) value)) : work(type, value);
    }
    private static double sigmoid(DataType type, double value) {
        double e = work(type, StrictMath.exp(value >= 0 ? -value : value));
        return work(type, (value >= 0 ? 1 : e) / work(type, 1 + e));
    }
    private static double get(DataType type, MemorySegment segment, long index) {
        return switch (type) {
            case FLOAT64 -> segment.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
            case FLOAT32 -> segment.getAtIndex(ValueLayout.JAVA_FLOAT, index);
            case BFLOAT16 -> BFloat16Bits.toFloat(segment.getAtIndex(ValueLayout.JAVA_SHORT, index));
            case FLOAT16 -> Float16Bits.toFloat(segment.getAtIndex(ValueLayout.JAVA_SHORT, index));
            default -> throw new AssertionError(type);
        };
    }
    private static void set(DataType type, MemorySegment segment, long index, double value) {
        switch (type) {
            case FLOAT64 -> segment.setAtIndex(ValueLayout.JAVA_DOUBLE, index, value);
            case FLOAT32 -> segment.setAtIndex(ValueLayout.JAVA_FLOAT, index, (float) value);
            case BFLOAT16 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, index, BFloat16Bits.fromFloat((float) value));
            case FLOAT16 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, index, Float16Bits.fromFloat((float) value));
            default -> throw new AssertionError(type);
        }
    }

    @Test void eachFixedSignatureQualifiesDirectFloat32AndQueriesEachLowType() {
        for (var kind : RecurrentScanKind.values()) for (boolean bias : List.of(false, true))
            for (var direction : RecurrentDirection.values()) {
                var context = context(kind, bias, direction, 3, 3, 1, 1, false);
                var node = context.nodes().getFirst();
                var query = new io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery(node.operation(),
                        node.inputs().stream().map(id -> context.values().get((int) id.value()).descriptor()).toList(),
                        node.outputs().stream().map(id -> context.values().get((int) id.value()).descriptor()).toList());
                var provider = new CpuCapabilityProvider(); assertTrue(provider.supports(query));
                for (var low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                    var lowInputs = query.inputs().stream().map(d -> d.dataType() == DataType.INT64 ? d
                            : new TensorDescriptor(low, d.shape(), d.layout(), d.requiresGrad())).toList();
                    var lowOutputs = query.outputs().stream().map(d -> new TensorDescriptor(low, d.shape(), d.layout(), d.requiresGrad())).toList();
                    assertTrue(provider.supports(new io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery(node.operation(), lowInputs, lowOutputs)));
                }
                var analysis = new CpuPartitionPreparer().analyze(context);
                var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
                try (var f = new Fixture(analysis, 2, true, true, Map.of())) {
                    f.lengths(1, 0, 1, 3);
                    f.floats(0, Float.NaN, .2f, -.1f, Float.POSITIVE_INFINITY, Float.NaN, .3f, Float.NaN, Float.NaN, .4f);
                    f.floats(2, -0f, .25f, .25f);
                    int weight = kind == RecurrentScanKind.LSTM ? 4 : 3;
                    if (kind == RecurrentScanKind.LSTM) f.floats(3, -0f, .5f, .5f);
                    f.fill(weight, 1); f.fill(weight + 1, 0); if (bias) f.fill(weight + 2, .07f);
                    f.executable.bind(f.state).execute();
                    int out = facts.readCount();
                    float[] y = f.floats(out), h = f.floats(out + 1);
                    assertEquals(0x80000000, Float.floatToRawIntBits(h[0]));
                    for (int b = 0; b < 3; b++) {
                        int length = new int[] {0, 1, 3}[b]; double expectedH = b == 0 ? -0d : .25, expectedC = b == 0 ? -0d : .5;
                        float roundedHidden = b == 0 ? -0f : .25f, roundedCell = b == 0 ? -0f : .5f;
                        for (int n = 0; n < length; n++) {
                            int t = direction == RecurrentDirection.REVERSE ? length - 1 - n : n;
                            double a = f.floats(0)[t * 3 + b] + (bias ? (double) .07f : 0d);
                            double gate = a >= 0 ? 1 / (1 + Math.exp(-a)) : Math.exp(a) / (1 + Math.exp(a));
                            if (kind == RecurrentScanKind.RNN_TANH) expectedH = Math.tanh(a);
                            else if (kind == RecurrentScanKind.GRU_RESET_AFTER) {
                                double candidate = Math.tanh(a); expectedH = candidate + gate * (expectedH - candidate);
                            } else {
                                expectedC = gate * expectedC + gate * Math.tanh(a); expectedH = gate * Math.tanh(expectedC);
                                // Independent route-specific FLOAT32 sites, alongside the double
                                // finite oracle: this fixture has unit W and zero recurrent U.
                                float px = 0f + f.floats(0)[t * 3 + b] * 1f;
                                if (bias) px = px + .07f;
                                float ph = 0f + roundedHidden * 0f;
                                float activation = px + ph;
                                float exponential = (float) StrictMath.exp(activation >= 0f ? -activation : activation);
                                float denominator = 1f + exponential;
                                float roundedGate = activation >= 0f ? 1f / denominator : exponential / denominator;
                                float roundedCandidate = (float) StrictMath.tanh(activation);
                                float retained = roundedGate * roundedCell, admitted = roundedGate * roundedCandidate;
                                roundedCell = retained + admitted;
                                roundedHidden = roundedGate * (float) StrictMath.tanh(roundedCell);
                            }
                            // Test-only metric: double-equation absolute+relative error <= 3e-6.
                            assertEquals(expectedH, y[t * 3 + b], 3e-6 + 3e-6 * Math.abs(expectedH));
                        }
                        for (int t = length; t < 3; t++) assertEquals(0, Float.floatToRawIntBits(y[t * 3 + b]));
                        if (length > 0) {
                            int last = direction == RecurrentDirection.REVERSE ? 0 : length - 1;
                            assertEquals(Float.floatToRawIntBits(y[last * 3 + b]), Float.floatToRawIntBits(h[b]));
                            if (kind == RecurrentScanKind.LSTM) {
                                assertEquals(expectedC, f.floats(out + 2)[b], 3e-6 + 3e-6 * Math.abs(expectedC));
                                assertEquals(Float.floatToRawIntBits(roundedCell), Float.floatToRawIntBits(f.floats(out + 2)[b]),
                                        "direct FLOAT32 cell state; no cross-backend bitwise promise");
                            }
                        }
                    }
                    // Explicitly inward duplicate-occurrence fixture: public Engine roots remain
                    // identity-unique, but Runtime must retain two publications of one scan port.
                    int finalHiddenBuffer = analysis.plan().boundaryValues().indexOf(facts.boundaryValues().get(out + 1));
                    var first = new io.github.pho001.synaptik.runtime.run.PreparedPublication(
                            f.state.memoryPlan(), finalHiddenBuffer, 0, 0).bind(f.state);
                    var alias = new io.github.pho001.synaptik.runtime.run.PreparedPublication(
                            f.state.memoryPlan(), finalHiddenBuffer, 0, 1).bind(f.state);
                    first.publish(); alias.publish();
                    assertNotSame(first, alias);
                    try (var published = new io.github.pho001.synaptik.runtime.run.RunResult(f.state, List.of(first, alias))) {
                        assertEquals(2, published.resultCount());
                        assertSame(published.publicationRepresentation(0), published.publicationRepresentation(1));
                    }
                }
            }
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
    /** Test-only ordinary shared assignment and run-state fixture; does not bypass CPU finalization. */
    public static final class Fixture implements AutoCloseable {
        public final Arena arena = Arena.ofShared();
        public final Map<ValueId, MemorySegment> storage = new LinkedHashMap<>();
        public final List<CpuBorrowedBuffer> borrowed = new ArrayList<>();
        public final PreparedExecutable executable;
        public final RunState state;
        private final List<MemorySegment> writers = new ArrayList<>();
        private final CpuWorkerGroup workers = new CpuWorkerGroup(4);
        private final Map<ValueId, Long> logicalBytes = new HashMap<>();

        public Fixture(BackendPartitionAnalysis<CpuPartitionPreparationPlan> analysis, int carrier,
                boolean readOnly, boolean sliced, Map<ValueId, MemorySegment> overrides) {
            var buffers = new ArrayList<PreparedMemoryPlan.BufferEntry>(); var workspaces = new ArrayList<PreparedMemoryPlan.WorkspaceEntry>();
            var assignments = new ArrayList<PreparationResourceAssignment>(); var bindings = new ArrayList<List<BufferRepresentationBinding>>();
            var scratch = new ArrayList<io.github.pho001.synaptik.runtime.resource.WorkspaceRepresentation>();
            for (var requirement : analysis.requirements()) {
                if (requirement instanceof PreparationResourceRequirement.Buffer buffer) {
                    var slot = new BufferSlot(buffers.size()); buffers.add(new PreparedMemoryPlan.BufferEntry(slot, buffer.byteSize(), buffer.byteAlignment()));
                    assignments.add(new PreparationResourceAssignment.Buffer(buffer, slot, buffers.size() - 1, 0));
                    var unit = analysis.plan().units().stream().filter(u -> u.boundaryValues().contains(buffer.valueId())).findFirst().orElseThrow();
                    int local = unit.boundaryValues().indexOf(buffer.valueId());
                    var type = unit.runtimeFacts().recurrentScan().map(s -> s.boundaryTypes().get(local))
                            .orElseGet(() -> unit.portablePlan().specialization().boundaryDataTypes().get(local));
                    long capacity = buffer.byteSize() + (sliced ? 16 : 0);
                    MemorySegment segment = carrier == 0 || carrier == 2 && buffers.size() % 2 == 0
                            ? type == DataType.INT64 ? MemorySegment.ofArray(new long[Math.toIntExact(capacity / 8)])
                                : type == DataType.FLOAT64 ? MemorySegment.ofArray(new double[Math.toIntExact(capacity / 8)])
                                : type == DataType.BFLOAT16 || type == DataType.FLOAT16
                                    ? MemorySegment.ofArray(new short[Math.toIntExact(capacity / 2)])
                                    : MemorySegment.ofArray(new float[Math.toIntExact(capacity / 4)])
                            : arena.allocate(capacity, buffer.byteAlignment());
                    if (sliced) segment = segment.asSlice(8, buffer.byteSize());
                    segment = overrides.getOrDefault(buffer.valueId(), segment);
                    storage.put(buffer.valueId(), segment); logicalBytes.put(buffer.valueId(), buffer.byteSize()); writers.add(segment);
                    boolean input = analysis.plan().units().stream().noneMatch(u -> u.boundaryValues().subList(u.boundaryValues().size() - u.outputCount(), u.boundaryValues().size()).contains(buffer.valueId()));
                    var selected = readOnly && input ? segment.asReadOnly() : segment;
                    var representation = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type, selected.byteSize() / type.byteWidth(), selected));
                    borrowed.add(representation); bindings.add(List.of(new BufferRepresentationBinding(representation, RunResourceOwnership.BORROWED)));
                } else if (requirement instanceof PreparationResourceRequirement.Workspace workspace) {
                    var slot = new WorkspaceSlot(workspaces.size()); workspaces.add(new PreparedMemoryPlan.WorkspaceEntry(slot, workspace.byteSize(), workspace.byteAlignment()));
                    assignments.add(new PreparationResourceAssignment.Workspace(workspace, slot, workspaces.size() - 1));
                    scratch.add(CpuContiguousWorkspace.allocate(workspace.byteSize(), workspace.byteAlignment()));
                }
            }
            var memory = new PreparedMemoryPlan(buffers, workspaces);
            executable = new CpuPartitionFinalizer(Optional.empty(), Optional.of(workers)).finalizePartition(
                    new BackendPartitionFinalization<>(analysis, memory, assignments)).executable();
            state = new RunState(memory, bindings, scratch);
        }
        public MemorySegment segment(int id) { return storage.get(new ValueId(id)).asSlice(0, logicalBytes.get(new ValueId(id))); }
        public void floats(int id, float... values) { segment(id).copyFrom(MemorySegment.ofArray(values)); }
        public float[] floats(int id) { return segment(id).toArray(ValueLayout.JAVA_FLOAT); }
        public void fill(int id, float value) { var s = segment(id); for (long p = 0; p < s.byteSize() / 4; p++) s.setAtIndex(ValueLayout.JAVA_FLOAT, p, value); }
        public void lengths(int id, long... values) { segment(id).copyFrom(MemorySegment.ofArray(values)); }
        @Override public void close() { try { state.close(); } finally { try { workers.close(); } finally { arena.close(); } } }
    }
}
