package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRecurrentScanLoweringTest.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRecurrentScanLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.recurrent.*;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.prepare.*;
import io.github.pho001.synaptik.prepare.analysis.*;
import io.github.pho001.synaptik.runtime.execution.PreparedExecutable;
import io.github.pho001.synaptik.runtime.memory.*;
import io.github.pho001.synaptik.runtime.run.*;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Direct-kernel equations, raw movement and no-partial-write qualification on exact segments. */
public class CpuRecurrentScanExecutionTest {
    @Test void independentEquationsQualifyEveryFamilyBiasDirectionAndCarrier() {
        for (var kind : RecurrentScanKind.values()) for (boolean bias : List.of(false, true))
            for (var direction : RecurrentDirection.values()) for (int carrier = 0; carrier < 3; carrier++) {
                var c = context(kind, bias, direction, 3, 4, 2, 3, false);
                var analysis = new CpuPartitionPreparer().analyze(c);
                var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
                try (var f = new Fixture(analysis, carrier, true, true, Map.of())) {
                    var inputs = new ArrayList<float[]>();
                    for (int role = 0; role < c.nodes().getFirst().inputs().size(); role++) {
                        int count = Math.toIntExact(facts.spans().get(role));
                        float[] data = new float[count];
                        for (int p = 0; p < count; p++) data[p] = ((p * 7 + role * 3) % 17 - 8) * .037f;
                        if (role == 0) for (int t = 0; t < 3; t++) for (int b = 0; b < 4; b++)
                            if (t >= new int[] {0, 1, 3, 2}[b]) for (int i = 0; i < 2; i++)
                                data[(t * 4 + b) * 2 + i] = i == 0 ? Float.NaN : Float.POSITIVE_INFINITY;
                        if (role == 2 || kind == RecurrentScanKind.LSTM && role == 3) {
                            data[0] = -0f; data[1] = Float.intBitsToFloat(0x7fc12345); data[2] = Float.NEGATIVE_INFINITY;
                        }
                        inputs.add(data);
                        if (role != 1) f.floats(role, data);
                    }
                    long[] lengths = {0, 1, 3, 2}; f.lengths(1, lengths);
                    var expected = equations(facts.geometry(), inputs, lengths);
                    float[] roundedCells = kind == RecurrentScanKind.LSTM
                            ? roundedFloat32LstmCells(facts.geometry(), inputs, lengths) : null;
                    var bound = f.executable.bind(f.state);
                    bound.execute();
                    int output = facts.readCount();
                    for (int port = 0; port < expected.size(); port++) {
                        float[] actual = f.floats(output + port);
                        for (int p = 0; p < actual.length; p++) {
                            if (port > 0 && p < 3) assertEquals(Float.floatToRawIntBits(inputs.get(port == 1 ? 2 : 3)[p]),
                                    Float.floatToRawIntBits(actual[p]), "zero-length raw state");
                            else finiteOracle(expected.get(port)[p], actual[p]);
                        }
                    }
                    if (roundedCells != null) {
                        float[] actualCells = f.floats(output + 2);
                        for (int p = 0; p < actualCells.length; p++) assertEquals(
                                Float.floatToRawIntBits(roundedCells[p]), Float.floatToRawIntBits(actualCells[p]),
                                "direct sequential FLOAT32 cell-state oracle, not a final-state tolerance");
                    }
                    // Publication is exact movement from the last executed dense hidden output.
                    float[] dense = f.floats(output), hidden = f.floats(output + 1);
                    for (int b = 1; b < 4; b++) {
                        int last = direction == RecurrentDirection.REVERSE ? 0 : (int) lengths[b] - 1;
                        for (int j = 0; j < 3; j++) assertEquals(Float.floatToRawIntBits(dense[(last * 4 + b) * 3 + j]),
                                Float.floatToRawIntBits(hidden[b * 3 + j]));
                    }
                    for (int t = 0; t < 3; t++) for (int b = 0; b < 4; b++) if (t >= lengths[b])
                        for (int j = 0; j < 3; j++) assertEquals(0, Float.floatToRawIntBits(dense[(t * 4 + b) * 3 + j]));
                    assertTrue(f.executable instanceof CpuPreparedPartitionExecutable);
                    assertInstanceOf(CpuRecurrentScanPreparedExecutable.class,
                            ((CpuPreparedPartitionExecutable) f.executable).children().getFirst());
                }
            }
    }

    @Test void lastInvalidLengthPreservesEveryOutputAndRepairReusesBoundInvocation() {
        for (var kind : RecurrentScanKind.values()) {
            var analysis = new CpuPartitionPreparer().analyze(context(kind, true, RecurrentDirection.REVERSE, 3, 4, 2, 2, false));
            var scan = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
            try (var f = new Fixture(analysis, 1, false, false, Map.of())) {
                f.lengths(1, Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE, Long.MAX_VALUE);
                var bound = f.executable.bind(f.state); // poisoned lengths are not inspected during binding
                for (long invalid : new long[] {-1, 4, Long.MAX_VALUE}) {
                    f.lengths(1, 0, 1, 3, invalid);
                    for (int p = scan.readCount(); p < scan.boundaryValues().size(); p++) f.fill(p, 71);
                    var failure = assertThrows(IllegalArgumentException.class, bound::execute);
                    assertTrue(failure.getMessage().contains("row=3 value=" + invalid + " T=3"));
                    for (int p = scan.readCount(); p < scan.boundaryValues().size(); p++)
                        for (float actual : f.floats(p)) assertEquals(71, actual);
                    f.lengths(1, 0, 1, 3, 2); bound.execute();
                }
            }
        }
    }

    @Test void emptyDomainsCopyStatesRawWithoutScratchAndStillValidateAllLengths() {
        for (var kind : RecurrentScanKind.values()) for (int batch : List.of(0, 2)) {
            var analysis = new CpuPartitionPreparer().analyze(context(kind, false, RecurrentDirection.FORWARD, 0, batch, 1, 2, false));
            var scan = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
            try (var f = new Fixture(analysis, 0, false, false, Map.of())) {
                float[] raw = batch == 0 ? new float[0] : new float[] {-0f, Float.intBitsToFloat(0x7fa54321), Float.POSITIVE_INFINITY, -2};
                f.floats(2, raw); if (kind == RecurrentScanKind.LSTM) f.floats(3, raw);
                f.lengths(1, new long[batch]);
                var bound = f.executable.bind(f.state); bound.execute();
                assertTrue(analysis.plan().units().getFirst().runtimeFacts().workspaceDeclaration().isEmpty());
                assertArrayEquals(MemorySegment.ofArray(raw).toArray(ValueLayout.JAVA_INT),
                        f.segment(scan.readCount() + 1).toArray(ValueLayout.JAVA_INT));
                if (kind == RecurrentScanKind.LSTM) assertArrayEquals(MemorySegment.ofArray(raw).toArray(ValueLayout.JAVA_INT),
                        f.segment(scan.readCount() + 2).toArray(ValueLayout.JAVA_INT));
                if (batch > 0) {
                    f.lengths(1, 0, 1); f.fill(scan.readCount() + 1, 71);
                    assertThrows(IllegalArgumentException.class, bound::execute);
                    for (float v : f.floats(scan.readCount() + 1)) assertEquals(71, v);
                }
            }
        }
    }

    @Test void activeSpecialClassesAndSignGuardedSigmoidAreNotRestrictedToFiniteInputs() {
        for (var kind : RecurrentScanKind.values()) for (float value : new float[] {Float.NaN, Float.POSITIVE_INFINITY,
                Float.NEGATIVE_INFINITY, Float.MAX_VALUE, -Float.MAX_VALUE}) {
            var analysis = new CpuPartitionPreparer().analyze(context(kind, true, RecurrentDirection.FORWARD, 1, 1, 1, 1, false));
            var scan = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
            try (var f = new Fixture(analysis, 0, false, false, Map.of())) {
                f.lengths(1, 1); f.floats(0, 1); f.floats(2, 0);
                int bias = scan.rolePositions().size() - 1;
                f.fill(bias, value);
                f.executable.bind(f.state).execute();
                float actual = f.floats(scan.readCount() + 1)[0];
                if (Float.isNaN(value)) assertTrue(Float.isNaN(actual));
                else if (kind == RecurrentScanKind.RNN_TANH) assertEquals(Math.copySign(1f, value), actual);
                else if (kind == RecurrentScanKind.GRU_RESET_AFTER) assertEquals(value > 0 ? 0f : -1f, actual);
                else assertEquals(value > 0 ? (float) StrictMath.tanh(1f) : 0f, actual);
            }
        }
    }

    @Test void oversizedSlicedOutputsKeepPrefixAndCapacityTailSentinels() {
        var analysis = new CpuPartitionPreparer().analyze(context(RecurrentScanKind.LSTM, false,
                RecurrentDirection.FORWARD, 2, 1, 1, 2, false));
        var scan = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
        try (Arena arena = Arena.ofShared()) {
            var overrides = new HashMap<ValueId, MemorySegment>(); var allocations = new ArrayList<MemorySegment>();
            for (int p = scan.readCount(); p < scan.boundaryValues().size(); p++) {
                long bytes = scan.spans().get(p) * 4;
                var storage = arena.allocate(bytes + 16, 8); storage.fill((byte) 0x5a);
                allocations.add(storage); overrides.put(scan.boundaryValues().get(p), storage.asSlice(8));
            }
            try (var f = new Fixture(analysis, 0, false, false, overrides)) {
                f.lengths(1, 1); f.executable.bind(f.state).execute();
                for (var storage : allocations) {
                    for (byte v : storage.asSlice(0, 8).toArray(ValueLayout.JAVA_BYTE)) assertEquals((byte) 0x5a, v);
                    for (byte v : storage.asSlice(storage.byteSize() - 8).toArray(ValueLayout.JAVA_BYTE)) assertEquals((byte) 0x5a, v);
                }
            }
        }
    }

    @Test void rejectsUndersizedMisalignedReadOnlyOverlappingAndDeadBoundaries() {
        var analysis = new CpuPartitionPreparer().analyze(context(RecurrentScanKind.RNN_TANH, false,
                RecurrentDirection.FORWARD, 2, 1, 1, 2, false));
        var scan = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
        try (Arena arena = Arena.ofShared()) {
            var shared = arena.allocate(128, 8);
            for (int variant = 0; variant < 4; variant++) {
                var overrides = new HashMap<ValueId, MemorySegment>();
                if (variant == 0) overrides.put(new ValueId(0), shared.asSlice(0, 4));
                if (variant == 1) overrides.put(new ValueId(0), shared.asSlice(1, 8));
                if (variant == 2) overrides.put(scan.boundaryValues().get(scan.readCount()), shared.asSlice(0, 16).asReadOnly());
                if (variant == 3) {
                    overrides.put(new ValueId(0), shared.asSlice(0, 8));
                    overrides.put(scan.boundaryValues().get(scan.readCount()), shared.asSlice(0, 16));
                }
                try (var f = new Fixture(analysis, 1, false, false, overrides)) {
                    assertThrows(RuntimeException.class, () -> f.executable.bind(f.state));
                }
            }
        }
        var arena = Arena.ofShared(); var storage = arena.allocate(8, 8);
        try (var f = new Fixture(analysis, 1, false, false, Map.of(new ValueId(0), storage))) {
            var bound = f.executable.bind(f.state); arena.close();
            assertThrows(RuntimeException.class, bound::execute);
        }
    }

    @Test void outputScratchOverlapAndWrongThreadAccessFailBeforeWrites() throws Exception {
        var analysis = new CpuPartitionPreparer().analyze(context(RecurrentScanKind.LSTM, false,
                RecurrentDirection.FORWARD, 1, 1, 1, 2, false));
        var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
        try (var f = new Fixture(analysis, 1, false, false, Map.of())) {
            var scratch = (CpuContiguousWorkspace) f.state.workspaceRepresentation(0);
            var bindings = new ArrayList<List<BufferRepresentationBinding>>();
            int output = facts.readCount();
            for (int p = 0; p < f.borrowed.size(); p++) {
                var representation = p == output ? CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT32,
                        facts.spans().get(output), scratch.writableSegment().asSlice(0, facts.spans().get(output) * 4))) : f.borrowed.get(p);
                bindings.add(List.of(new BufferRepresentationBinding(representation, RunResourceOwnership.BORROWED)));
            }
            try (var state = new RunState(f.state.memoryPlan(), bindings, List.of(scratch))) {
                assertThrows(IllegalArgumentException.class, () -> f.executable.bind(state));
            }
        }
        try (Arena confined = Arena.ofConfined(); var f = new Fixture(analysis, 1, false, false,
                Map.of(new ValueId(0), confined.allocate(4, 4)))) {
            f.lengths(1, 1); var bound = f.executable.bind(f.state);
            var failure = new java.util.concurrent.atomic.AtomicReference<Throwable>();
            var thread = new Thread(() -> { try { bound.execute(); } catch (Throwable t) { failure.set(t); } });
            thread.start(); thread.join(5000); assertFalse(thread.isAlive());
            assertInstanceOf(IllegalArgumentException.class, failure.get());
        }
    }

    /** Test-only finite qualification: double equation oracle, absolute+relative error <= 3e-6. */
    private static void finiteOracle(double expected, float actual) {
        assertTrue(Double.isFinite(expected) && Float.isFinite(actual), "finite equation class");
        assertEquals(expected, actual, 3e-6 + 3e-6 * Math.abs(expected), "double equation oracle");
    }

    private static List<double[]> equations(CpuRecurrentScanLowering.Geometry g, List<float[]> data, long[] lengths) {
        int h = g.hiddenSize(), bcount = g.batch(), packed = g.gates() * h, w = g.gates() == 4 ? 4 : 3;
        double[] y = new double[g.time() * bcount * h], hidden = new double[bcount * h], cell = new double[bcount * h];
        for (int p = 0; p < hidden.length; p++) { hidden[p] = data.get(2)[p]; if (g.gates() == 4) cell[p] = data.get(3)[p]; }
        for (int b = 0; b < bcount; b++) for (int n = 0; n < lengths[b]; n++) {
            int t = g.reverse() ? (int) lengths[b] - 1 - n : n;
            double[] px = new double[packed], ph = new double[packed];
            for (int j = 0; j < packed; j++) {
                for (int k = 0; k < g.inputSize(); k++) px[j] += (double) data.get(0)[(t * bcount + b) * g.inputSize() + k] * data.get(w)[j * g.inputSize() + k];
                for (int k = 0; k < h; k++) ph[j] += hidden[b * h + k] * data.get(w + 1)[j * h + k];
                if (g.bias()) px[j] += data.get(w + 2)[j];
            }
            for (int j = 0; j < h; j++) {
                int state = b * h + j; double next;
                if (g.gates() == 1) next = Math.tanh(px[j] + ph[j]);
                else if (g.gates() == 3) {
                    double r = logistic(px[j] + ph[j]), z = logistic(px[h + j] + ph[h + j]);
                    double candidate = Math.tanh(px[2 * h + j] + r * ph[2 * h + j]);
                    next = candidate + z * (hidden[state] - candidate);
                } else {
                    double i = logistic(px[j] + ph[j]), f = logistic(px[h + j] + ph[h + j]);
                    double candidate = Math.tanh(px[2 * h + j] + ph[2 * h + j]);
                    double o = logistic(px[3 * h + j] + ph[3 * h + j]);
                    cell[state] = f * cell[state] + i * candidate; next = o * Math.tanh(cell[state]);
                }
                hidden[state] = next; y[(t * bcount + b) * h + j] = next;
            }
        }
        return g.gates() == 4 ? List.of(y, hidden, cell) : List.of(y, hidden);
    }
    private static double logistic(double x) { return x >= 0 ? 1 / (1 + Math.exp(-x)) : Math.exp(x) / (1 + Math.exp(x)); }

    /**
     * Independently evaluates this direct route's rounded FLOAT32 LSTM sites on test arrays.
     * This supplements the double finite oracle; bit equality is route-specific, not a backend
     * portability promise. Matrices use all contributors in index order, all prior-hidden reads
     * precede replacement, and every primitive/gate/state site rounds to FLOAT32 without FMA.
     * @param geometry checked LSTM dimensions, bias and direction
     * @param data independent test input/state/packed-weight arrays in Model role order
     * @param lengths complete valid row lengths, unchanged by the oracle
     * @return final cell representations, preserving raw initial bits on zero-length rows
     */
    private static float[] roundedFloat32LstmCells(CpuRecurrentScanLowering.Geometry geometry,
            List<float[]> data, long[] lengths) {
        int width = geometry.hiddenSize(), features = geometry.inputSize(), batch = geometry.batch();
        float[] hidden = data.get(2).clone(), cells = data.get(3).clone();
        for (int row = 0; row < batch; row++) for (int visit = 0; visit < lengths[row]; visit++) {
            int time = geometry.reverse() ? (int) lengths[row] - 1 - visit : visit;
            float[][] preactivation = new float[4][width];
            for (int gate = 0; gate < 4; gate++) for (int j = 0; j < width; j++) {
                int packedRow = gate * width + j;
                float inputProjection = 0f, hiddenProjection = 0f;
                for (int k = 0; k < features; k++) {
                    float product = data.get(0)[(time * batch + row) * features + k]
                            * data.get(4)[packedRow * features + k];
                    inputProjection = inputProjection + product;
                }
                for (int k = 0; k < width; k++) {
                    float product = hidden[row * width + k] * data.get(5)[packedRow * width + k];
                    hiddenProjection = hiddenProjection + product;
                }
                if (geometry.bias()) inputProjection = inputProjection + data.get(6)[packedRow];
                preactivation[gate][j] = inputProjection + hiddenProjection;
            }
            float[] nextHidden = new float[width];
            for (int j = 0; j < width; j++) {
                float inputGate = roundedFloat32Sigmoid(preactivation[0][j]);
                float forgetGate = roundedFloat32Sigmoid(preactivation[1][j]);
                float candidate = (float) StrictMath.tanh(preactivation[2][j]);
                float outputGate = roundedFloat32Sigmoid(preactivation[3][j]);
                int state = row * width + j;
                float retained = forgetGate * cells[state], admitted = inputGate * candidate;
                cells[state] = retained + admitted;
                nextHidden[j] = outputGate * (float) StrictMath.tanh(cells[state]);
            }
            System.arraycopy(nextHidden, 0, hidden, row * width, width);
        }
        return cells;
    }

    /**
     * Evaluates the selected direct Java route's branch and rounded FLOAT32 elementary sites.
     * @param value represented preactivation, not mutated
     * @return route-specific FLOAT32 sigmoid result
     */
    private static float roundedFloat32Sigmoid(float value) {
        float exponential = (float) StrictMath.exp(value >= 0f ? -value : value);
        float denominator = 1f + exponential;
        return value >= 0f ? 1f / denominator : exponential / denominator;
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
