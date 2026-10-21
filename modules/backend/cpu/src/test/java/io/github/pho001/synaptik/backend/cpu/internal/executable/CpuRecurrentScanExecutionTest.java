package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRecurrentScanLoweringTest.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRecurrentScanLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.*;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.backend.cpu.internal.route.portable.CpuRecurrentScanCleanJavaOracle;
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
    @Test void everyScalarTypeRetainsActiveSpecialClassesAndGradualSubnormalSites() {
        for (var type : floatingTypes()) for (var kind : RecurrentScanKind.values()) {
            var analysis = new CpuPartitionPreparer().analyze(scalarContext(kind, true, RecurrentDirection.FORWARD,
                    1, 1, 1, 1, false, type, 2));
            var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
            try (var f = new Fixture(analysis, 2, true, true, Map.of())) {
                int weight = kind == RecurrentScanKind.LSTM ? 4 : 3, gates = facts.geometry().gates();
                scatterLengths(facts, f, 1); scatter(facts, f, 0, new double[] {1}); scatter(facts, f, 2, new double[] {0});
                if (kind == RecurrentScanKind.LSTM) scatter(facts, f, 3, new double[] {0});
                scatter(facts, f, weight, new double[gates]); scatter(facts, f, weight + 1, new double[gates]);
                for (double value : new double[] {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY,
                        type == DataType.FLOAT64 ? Double.MAX_VALUE : type == DataType.FLOAT16 ? 65504 : Float.MAX_VALUE,
                        type == DataType.FLOAT64 ? -Double.MAX_VALUE : type == DataType.FLOAT16 ? -65504 : -Float.MAX_VALUE}) {
                    var bias = new double[gates]; Arrays.fill(bias, value); scatter(facts, f, weight + 2, bias);
                    f.executable.bind(f.state).execute();
                    double actual = number(type, f.segment(facts.readCount() + 1), 0);
                    if (Double.isNaN(value)) assertTrue(Double.isNaN(actual));
                    else if (kind == RecurrentScanKind.RNN_TANH) assertEquals(Math.copySign(1d, value), actual);
                    else if (kind == RecurrentScanKind.GRU_RESET_AFTER) assertEquals(value > 0 ? 0d : -1d, actual);
                    else assertEquals(value > 0 ? stored(type, StrictMath.tanh(1d)) : 0d, actual);
                }
            }
        }
        for (var type : floatingTypes()) {
            var analysis = new CpuPartitionPreparer().analyze(scalarContext(RecurrentScanKind.RNN_TANH, false,
                    RecurrentDirection.FORWARD, 1, 1, 1, 1, false, type, 2));
            var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
            try (var f = new Fixture(analysis, 0, true, true, Map.of())) {
                scatterLengths(facts, f, 1); scatter(facts, f, 2, new double[] {0});
                scatter(facts, f, 3, new double[] {1}); scatter(facts, f, 4, new double[] {0});
                setRaw(type, f.segment(0), facts.accessBindings().getFirst().baseElementOffset(), 1);
                f.executable.bind(f.state).execute();
                assertEquals(1, raw(type, f.segment(facts.readCount()), 0), "this gradual route does not choose optional DAZ/FTZ");
                scatter(facts, f, 3, new double[] {.25});
                f.executable.bind(f.state).execute();
                assertEquals(0, raw(type, f.segment(facts.readCount()), 0), "genuine final underflow, positive zero");
            }
        }
    }

    private static double stored(DataType type, double value) {
        return type == DataType.FLOAT64 ? value : type == DataType.FLOAT32 ? (float) value : narrow(type, value);
    }

    @Test void typedReadAliasesCapacityTailsAndConservativeWriteExclusion() {
        for (var type : floatingTypes()) {
            var analysis = new CpuPartitionPreparer().analyze(scalarContext(RecurrentScanKind.LSTM, true,
                    RecurrentDirection.FORWARD, 2, 2, 2, 2, false, type, 2));
            var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
            try (Arena arena = Arena.ofShared()) {
                var sameRead = arena.allocate(facts.spans().get(2) * type.byteWidth(), 8);
                var overrides = new HashMap<ValueId, MemorySegment>();
                overrides.put(new ValueId(2), sameRead); overrides.put(new ValueId(3), sameRead);
                var allocations = new ArrayList<MemorySegment>();
                for (int p = facts.readCount(); p < facts.boundaryValues().size(); p++) {
                    long size = facts.spans().get(p) * type.byteWidth();
                    var allocation = arena.allocate(size + 16, 8); allocation.fill((byte) 0x5a);
                    allocations.add(allocation); overrides.put(new ValueId(p), allocation.asSlice(8));
                }
                try (var f = new Fixture(analysis, 2, true, false, overrides)) {
                    populate(facts, f);
                    var before = sameRead.toArray(ValueLayout.JAVA_BYTE);
                    var oracle = oracleBoundaries(facts, f);
                    CpuRecurrentScanCleanJavaOracle.bind(facts, oracle, workingScratch(facts)).run();
                    f.executable.bind(f.state).execute();
                    assertArrayEquals(before, sameRead.toArray(ValueLayout.JAVA_BYTE), "real read/read alias accepted unchanged");
                    for (int p = facts.readCount(); p < facts.boundaryValues().size(); p++) assertArrayEquals(
                            oracle[p].toArray(ValueLayout.JAVA_BYTE), f.segment(p).toArray(ValueLayout.JAVA_BYTE));
                    for (var allocation : allocations) for (var bytes : List.of(allocation.asSlice(0, 8),
                            allocation.asSlice(allocation.byteSize() - 8))) for (byte value : bytes.toArray(ValueLayout.JAVA_BYTE))
                        assertEquals((byte) 0x5a, value, "physical prefix and selected capacity tail unchanged");
                }
                long inputBytes = facts.spans().getFirst() * type.byteWidth();
                var storage = arena.allocate(Math.max(inputBytes, facts.geometry().scratchBytes()) + 256, 8);
                for (int variant = 0; variant < 6; variant++) {
                    var bad = new HashMap<ValueId, MemorySegment>();
                    int output = facts.readCount();
                    long outputBytes = facts.spans().get(output) * type.byteWidth();
                    if (variant == 0) bad.put(new ValueId(0), storage.asSlice(0, inputBytes - type.byteWidth()));
                    if (variant == 1) bad.put(new ValueId(0), storage.asSlice(1, inputBytes));
                    if (variant == 2) bad.put(new ValueId(output), storage.asSlice(0, outputBytes).asReadOnly());
                    if (variant == 3) { // Whole selected capacity overlap is rejected, not just addressed cells.
                        bad.put(new ValueId(0), storage.asSlice(0, inputBytes + outputBytes));
                        bad.put(new ValueId(output), storage.asSlice(inputBytes, outputBytes));
                    }
                    if (variant == 4) {
                        bad.put(new ValueId(output), storage.asSlice(0, outputBytes));
                        bad.put(new ValueId(output + 1), storage.asSlice(0, facts.spans().get(output + 1) * type.byteWidth()));
                    }
                    if (variant == 5) { // Observable wrong primitive carriers cannot hide behind segment access.
                        var wrong = type == DataType.FLOAT32 ? MemorySegment.ofArray(new int[Math.toIntExact(inputBytes / 4)])
                                : MemorySegment.ofArray(new float[Math.toIntExact((inputBytes + 3) / 4)]);
                        bad.put(new ValueId(0), wrong.asSlice(0, inputBytes));
                    }
                    try (var f = new Fixture(analysis, 1, false, false, bad)) {
                        assertThrows(RuntimeException.class, () -> f.executable.bind(f.state), "type=" + type + " variant=" + variant);
                    }
                }
            }
        }
    }

    @Test void multiStepLowFixturesDetectPrematureGateDoubleWorkAndUnroundedState() {
        for (var type : List.of(DataType.BFLOAT16, DataType.FLOAT16))
            for (var kind : List.of(RecurrentScanKind.GRU_RESET_AFTER, RecurrentScanKind.LSTM)) {
                var witnessed = new boolean[kind == RecurrentScanKind.LSTM ? 5 : 4];
                var analysis = new CpuPartitionPreparer().analyze(scalarContext(kind, true,
                        RecurrentDirection.FORWARD, 2, 1, 3, 2, false, type, 2));
                var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
                try (var f = new Fixture(analysis, 0, false, false, Map.of())) {
                    var random = new Random(0x29c0ffee);
                    // Fixed witnesses; all recurrent matrices have off-diagonal
                    // entries. Exact route oracle uses rounded FLOAT32 sites and stored low states.
                    for (int fixture = 0; fixture < 3; fixture++) {
                        for (int role = 0; role < facts.rolePositions().size(); role++) if (role != 1) {
                            int p = facts.rolePositions().get(role);
                            double[] values = new double[Math.toIntExact(facts.accessBindings().get(p).elementCount())];
                            for (int k = 0; k < values.length; k++) values[k] = (random.nextDouble() - .5) * 3;
                            if (fixture == 2) values = doubleWorkWitness(type, kind).get(role);
                            scatter(facts, f, role, values);
                        }
                        scatterLengths(facts, f, 2);
                        var data = new ArrayList<double[]>();
                        for (int role = 0; role < facts.rolePositions().size(); role++) data.add(role == 1 ? new double[1] : gather(facts, f, role));
                        var exact = lowDiscriminator(type, kind, data, 0);
                        f.executable.bind(f.state).execute();
                        var actual = f.segment(facts.readCount()).toArray(ValueLayout.JAVA_SHORT);
                        assertArrayEquals(exact, actual, "FLOAT32 work and stored-state boundaries");
                        for (int fault = 1; fault < witnessed.length; fault++)
                            witnessed[fault] |= !Arrays.equals(exact, lowDiscriminator(type, kind, data, fault));
                    }
                }
                for (int fault = 1; fault < witnessed.length; fault++) assertTrue(witnessed[fault],
                        type + " " + kind + " fixture must distinguish fault " + fault);
            }
    }

    @Test void allScalarTypesReadLayoutsAndCarriersMatchSpecializedCleanJava() {
        for (var type : floatingTypes()) for (var kind : RecurrentScanKind.values())
            for (boolean bias : List.of(false, true)) for (var direction : RecurrentDirection.values())
                for (int mode = 0; mode < 6; mode++) for (int carrier = 0; carrier < 3; carrier++) {
                    var analysis = new CpuPartitionPreparer().analyze(scalarContext(kind, bias, direction,
                            3, 4, 2, 3, false, type, mode));
                    var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
                    try (var f = new Fixture(analysis, carrier, true, true, Map.of())) {
                        populate(facts, f);
                        var expected = oracleBoundaries(facts, f);
                        var scratch = workingScratch(facts);
                        CpuRecurrentScanCleanJavaOracle.bind(facts, expected, scratch).run();
                        f.executable.bind(f.state).execute();
                        for (int port = 0; port < (kind == RecurrentScanKind.LSTM ? 3 : 2); port++) {
                            int p = facts.readCount() + port;
                            assertArrayEquals(expected[p].toArray(ValueLayout.JAVA_BYTE), f.segment(p).toArray(ValueLayout.JAVA_BYTE),
                                    type + " " + kind + " " + direction + " bias=" + bias + " layout=" + mode + " carrier=" + carrier);
                        }
                        var lengths = gatherLengths(facts, f);
                        for (int b = 0; b < 4; b++) for (int t = 0; t < 3; t++) for (int j = 0; j < 3; j++) {
                            if (t >= lengths[b]) assertEquals(0L, raw(type, f.segment(facts.readCount()), (t * 4L + b) * 3 + j));
                            else if (t == (direction == RecurrentDirection.REVERSE ? 0 : lengths[b] - 1)) assertEquals(
                                    raw(type, f.segment(facts.readCount()), (t * 4L + b) * 3 + j),
                                    raw(type, f.segment(facts.readCount() + 1), b * 3L + j));
                        }
                        // Test-only real-equation oracle, H=3/I=2/T<=3, bounded input magnitudes
                        // below .4 and non-cancelling gates. Absolute+relative error: F64 1e-13,
                        // F32 3e-6, BF16 .012, F16 .002. Exact state bits are tested separately above.
                        var logical = new ArrayList<double[]>();
                        for (int role = 0; role < facts.rolePositions().size(); role++) logical.add(
                                role == 1 ? new double[4] : gather(facts, f, role));
                        var real = realEquations(facts.geometry(), logical, lengths);
                        double tolerance = type == DataType.FLOAT64 ? 1e-13 : type == DataType.FLOAT32 ? 3e-6
                                : type == DataType.BFLOAT16 ? .012 : .002;
                        for (int port = 0; port < real.size(); port++) for (int p = 0; p < real.get(port).length; p++) {
                            double value = number(type, f.segment(facts.readCount() + port), p);
                            assertTrue(Double.isFinite(value));
                            assertEquals(real.get(port)[p], value, tolerance * (1 + Math.abs(real.get(port)[p])));
                        }
                    }
                }
    }

    @Test void stridedLastInvalidLengthsAreAtomicAndBoundInvocationIsReusableForAllTypes() {
        for (var type : floatingTypes()) for (var kind : RecurrentScanKind.values()) for (int mode : List.of(1, 2, 4)) {
            var analysis = new CpuPartitionPreparer().analyze(scalarContext(kind, true, RecurrentDirection.REVERSE,
                    3, 4, 2, 3, false, type, mode));
            var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
            try (var f = new Fixture(analysis, 2, false, true, Map.of())) {
                populate(facts, f);
                var l = facts.accessBindings().get(facts.rolePositions().get(1));
                var lengths = f.segment(facts.rolePositions().get(1));
                var bound = f.executable.bind(f.state);
                for (long invalid : new long[] {-1, 4, Long.MAX_VALUE}) {
                    lengths.setAtIndex(ValueLayout.JAVA_LONG, l.baseElementOffset() + 3 * l.effectiveStrides().getFirst(), invalid);
                    for (int p = facts.readCount(); p < facts.boundaryValues().size(); p++) f.segment(p).fill((byte) 0x5a);
                    assertThrows(IllegalArgumentException.class, bound::execute);
                    for (int p = facts.readCount(); p < facts.boundaryValues().size(); p++)
                        for (byte value : f.segment(p).toArray(ValueLayout.JAVA_BYTE)) assertEquals((byte) 0x5a, value);
                    lengths.setAtIndex(ValueLayout.JAVA_LONG, l.baseElementOffset() + 3 * l.effectiveStrides().getFirst(), 2);
                    bound.execute();
                }
            }
        }
    }

    @Test void zeroLengthAndZeroTimeGatherRawStateBitsForEveryTypeAndLayout() {
        for (var type : floatingTypes()) for (int time : List.of(0, 3)) for (int mode = 0; mode < 6; mode++)
            for (int batch : List.of(0, 2)) {
                var context = scalarContext(RecurrentScanKind.LSTM, false, RecurrentDirection.REVERSE,
                        time, batch, 1, 3, false, type, mode);
                // Empty inputs may carry arbitrary unused offset and strides, even beyond int.
                if (time == 0 || batch == 0) {
                    var ds = new ArrayList<>(context.values().stream().map(v -> v.descriptor()).toList());
                    var d = ds.getFirst();
                    ds.set(0, new io.github.pho001.synaptik.model.tensor.TensorDescriptor(type, d.shape(), Optional.of(
                            io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(d.shape(),
                                    new long[] {Long.MAX_VALUE, 0, 0}, Long.MAX_VALUE, true)), false));
                    context = project(ds, context.nodes());
                }
                var analysis = new CpuPartitionPreparer().analyze(context);
                var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
                try (var f = new Fixture(analysis, 2, true, true, Map.of())) {
                    populate(facts, f);
                    var lengths = facts.accessBindings().get(facts.rolePositions().get(1));
                    for (int b = 0; b < batch; b++) f.segment(facts.rolePositions().get(1)).setAtIndex(ValueLayout.JAVA_LONG,
                            lengths.baseElementOffset() + b * lengths.effectiveStrides().getFirst(), 0);
                    for (int role : List.of(2, 3)) {
                        var a = facts.accessBindings().get(facts.rolePositions().get(role));
                        for (int b = 0; b < batch; b++) for (int j = 0; j < 3; j++) setRaw(type, f.segment(facts.rolePositions().get(role)),
                                address(a, b * 3 + j), j == 0 ? signBit(type) : j == 1 ? 1 : nanPayload(type));
                    }
                    var expected = oracleBoundaries(facts, f);
                    CpuRecurrentScanCleanJavaOracle.bind(facts, expected, workingScratch(facts)).run();
                    f.executable.bind(f.state).execute();
                    for (int p = facts.readCount(); p < facts.boundaryValues().size(); p++) assertArrayEquals(
                            expected[p].toArray(ValueLayout.JAVA_BYTE), f.segment(p).toArray(ValueLayout.JAVA_BYTE));
                }
            }
    }

    @Test void float64RetainsPrecisionAndMinimumSubnormalAtArithmeticSites() {
        var analysis = new CpuPartitionPreparer().analyze(scalarContext(RecurrentScanKind.RNN_TANH, false,
                RecurrentDirection.FORWARD, 1, 2, 1, 1, false, DataType.FLOAT64, 2));
        var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
        try (var f = new Fixture(analysis, 0, true, true, Map.of())) {
            scatter(facts, f, 0, new double[] {1 + Math.scalb(1d, -30), Double.MIN_VALUE});
            scatter(facts, f, 2, new double[] {0, 0}); scatter(facts, f, 3, new double[] {1});
            scatter(facts, f, 4, new double[] {0}); scatterLengths(facts, f, 1, 1);
            f.executable.bind(f.state).execute();
            assertEquals(StrictMath.tanh(1 + Math.scalb(1d, -30)), number(DataType.FLOAT64, f.segment(facts.readCount()), 0));
            assertNotEquals(StrictMath.tanh(1d), number(DataType.FLOAT64, f.segment(facts.readCount()), 0));
            assertEquals(1L, raw(DataType.FLOAT64, f.segment(facts.readCount()), 1));
        }
    }

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

    private static List<double[]> realEquations(CpuRecurrentScanLowering.Geometry g, List<double[]> data, long[] lengths) {
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

    /**
     * Independent two-step H=2/I=3 low-state witness. Faults deliberately substitute: (1) low
     * gates, (2) double work, (3) non-stored carry, (4) an unrounded LSTM cell consumed by hidden.
     * Double temporaries below represent exactly rounded FLOAT32 sites unless fault 2 is selected;
     * these bounded operands have no double-rounding/cancellation ambiguity at those sites.
     */
    private static short[] lowDiscriminator(DataType type, RecurrentScanKind kind, List<double[]> data, int fault) {
        boolean wide = fault == 2, lstm = kind == RecurrentScanKind.LSTM;
        int packed = lstm ? 8 : 6, w = lstm ? 4 : 3;
        double[] hidden = data.get(2).clone(), cell = lstm ? data.get(3).clone() : null;
        short[] result = new short[4];
        for (int t = 0; t < 2; t++) {
            double[] px = new double[packed], ph = new double[packed];
            for (int row = 0; row < packed; row++) {
                for (int k = 0; k < 3; k++) px[row] = site(px[row] + site(data.get(0)[t * 3 + k] * data.get(w)[row * 3 + k], wide), wide);
                for (int k = 0; k < 2; k++) ph[row] = site(ph[row] + site(hidden[k] * data.get(w + 1)[row * 2 + k], wide), wide);
                px[row] = site(px[row] + data.get(w + 2)[row], wide);
            }
            for (int j = 0; j < 2; j++) {
                double a = gate(site(px[j] + ph[j], wide), wide);
                double b = gate(site(px[2 + j] + ph[2 + j], wide), wide);
                if (fault == 1) { a = narrow(type, a); b = narrow(type, b); }
                double next;
                if (lstm) {
                    double candidate = site(StrictMath.tanh(site(px[4 + j] + ph[4 + j], wide)), wide);
                    double output = gate(site(px[6 + j] + ph[6 + j], wide), wide);
                    if (fault == 1) { candidate = narrow(type, candidate); output = narrow(type, output); }
                    double computedCell = site(site(b * cell[j], wide) + site(a * candidate, wide), wide);
                    cell[j] = fault == 3 ? computedCell : narrow(type, computedCell);
                    double consumed = fault == 4 ? computedCell : cell[j];
                    next = site(output * site(StrictMath.tanh(consumed), wide), wide);
                } else {
                    double candidate = site(StrictMath.tanh(site(px[4 + j] + site(a * ph[4 + j], wide), wide)), wide);
                    if (fault == 1) candidate = narrow(type, candidate);
                    next = site(candidate + site(b * site(hidden[j] - candidate, wide), wide), wide);
                }
                hidden[j] = fault == 3 ? next : narrow(type, next);
                result[t * 2 + j] = type == DataType.BFLOAT16 ? BFloat16Bits.fromFloat((float) next) : Float16Bits.fromFloat((float) next);
            }
        }
        return result;
    }

    /** Fixed exact low values from bounded development-only search, seed 0x29c0ffee. */
    private static List<double[]> doubleWorkWitness(DataType type, RecurrentScanKind kind) {
        // BFLOAT16/GRU_RESET_AFTER, search fixture 5207; no search runs in CI.
        if (type == DataType.BFLOAT16 && kind == RecurrentScanKind.GRU_RESET_AFTER) return List.of(
                new double[] {-0.97265625, -0.296875, -0.56640625, -1.3984375, 1.03125, 1.0546875},
                new double[] {0.0},
                new double[] {0.07177734375, -0.3984375},
                new double[] {1.296875, 0.0869140625, 0.9765625, -1.09375, -0.2890625, 1.109375, 1.1484375, 1.2109375, 0.474609375, -0.625, -0.62109375, -1.25, 1.046875, -0.42578125, 1.4140625, -0.33203125, -0.93359375, -0.0203857421875},
                new double[] {0.62109375, -0.2236328125, -0.39453125, 1.234375, -0.984375, 0.6953125, -0.65625, -0.8984375, -0.70703125, 0.251953125, 0.484375, 0.279296875},
                new double[] {-0.380859375, -0.8984375, -0.318359375, -0.734375, 0.921875, 0.380859375});
        // BFLOAT16/LSTM, search fixture 1744; no search runs in CI.
        if (type == DataType.BFLOAT16 && kind == RecurrentScanKind.LSTM) return List.of(
                new double[] {0.177734375, 0.80078125, 1.34375, 0.62890625, 1.4375, 0.76171875},
                new double[] {0.0},
                new double[] {-1.1953125, -1.109375},
                new double[] {0.5703125, -0.330078125},
                new double[] {-0.1591796875, -0.66015625, -0.984375, -0.8515625, -0.9609375, -0.640625, 0.08349609375, 0.59375, 0.119140625, 0.1904296875, 1.1484375, 0.42578125, -0.94921875, -1.3671875, 0.1708984375, 0.79296875, 1.3515625, -0.474609375, -1.1953125, -0.330078125, 0.69140625, 1.0625, -1.4296875, 0.51171875},
                new double[] {-1.15625, 1.34375, 0.00927734375, 0.96484375, 0.06787109375, 1.0234375, -0.890625, 1.09375, 0.1357421875, -0.14453125, 0.8203125, -1.421875, -0.6015625, -0.0810546875, 0.23046875, 0.038330078125},
                new double[] {0.609375, -0.76953125, 0.5625, 0.275390625, 0.65625, -0.06640625, -0.92578125, -1.390625});
        // FLOAT16/GRU_RESET_AFTER, search fixture 2027; no search runs in CI.
        if (type == DataType.FLOAT16 && kind == RecurrentScanKind.GRU_RESET_AFTER) return List.of(
                new double[] {0.775390625, 0.5166015625, -1.259765625, -1.4638671875, 1.2763671875, -0.48681640625},
                new double[] {0.0},
                new double[] {-1.25, -0.8837890625},
                new double[] {-0.3818359375, -1.4306640625, 1.34375, -0.8330078125, 0.78466796875, -0.90869140625, -1.126953125, -0.70263671875, -0.442138671875, -0.1778564453125, -0.7529296875, 1.484375, 0.416259765625, 0.7685546875, 0.152099609375, 0.22021484375, -0.1929931640625, -0.5126953125},
                new double[] {-0.708984375, -0.5400390625, 0.038848876953125, -0.195068359375, -0.93115234375, 0.59375, 0.6474609375, 1.03125, -1.3447265625, -0.587890625, 1.27734375, 0.035888671875},
                new double[] {0.98193359375, 0.66796875, -1.0732421875, 0.217529296875, 0.9541015625, 0.287353515625});
        // FLOAT16/LSTM, search fixture 157; no search runs in CI.
        if (type == DataType.FLOAT16 && kind == RecurrentScanKind.LSTM) return List.of(
                new double[] {-1.1015625, -0.42578125, -0.404541015625, -0.6884765625, 0.279541015625, -1.3046875},
                new double[] {0.0},
                new double[] {-0.73974609375, -0.93994140625},
                new double[] {1.224609375, -0.6943359375},
                new double[] {-0.27587890625, 1.04296875, -0.43603515625, 1.0859375, 0.1510009765625, 1.41015625, -1.107421875, -0.9833984375, 0.1500244140625, -1.1201171875, -1.416015625, 1.0234375, 1.177734375, 0.1434326171875, 1.3828125, 0.6455078125, 0.55126953125, 0.1724853515625, 0.74462890625, 0.10626220703125, 0.5654296875, -0.849609375, -0.33740234375, -0.77685546875},
                new double[] {-1.04296875, 0.88427734375, 1.3447265625, 1.4931640625, 1.2529296875, -0.0260162353515625, -1.123046875, 0.41650390625, 0.48779296875, -0.2034912109375, -0.1656494140625, -1.2939453125, 1.2548828125, -0.48828125, -1.2705078125, -0.53564453125},
                new double[] {1.1943359375, -0.256591796875, -0.7685546875, -0.343017578125, -1.3203125, 1.4208984375, -0.810546875, 0.351806640625});
        throw new AssertionError("low GRU/LSTM witness only");
    }

    private static double site(double value, boolean wide) { return wide ? value : (float) value; }
    private static double gate(double value, boolean wide) {
        double exp = site(StrictMath.exp(value >= 0 ? -value : value), wide);
        return site((value >= 0 ? 1 : exp) / site(1 + exp, wide), wide);
    }
    private static double narrow(DataType type, double value) {
        return type == DataType.BFLOAT16 ? BFloat16Bits.toFloat(BFloat16Bits.fromFloat((float) value))
                : Float16Bits.toFloat(Float16Bits.fromFloat((float) value));
    }

    public static MemorySegment heap(DataType type, long count) {
        int size = Math.toIntExact(count);
        return switch (type) {
            case FLOAT64 -> MemorySegment.ofArray(new double[size]);
            case FLOAT32 -> MemorySegment.ofArray(new float[size]);
            case BFLOAT16, FLOAT16 -> MemorySegment.ofArray(new short[size]);
            case INT64 -> MemorySegment.ofArray(new long[size]);
            default -> throw new AssertionError(type);
        };
    }

    public static MemorySegment workingScratch(CpuRecurrentScanLowering.Facts facts) {
        return facts.geometry().scratchBytes() == 0 ? null : heap(
                facts.geometry().dataType() == DataType.FLOAT64 ? DataType.FLOAT64 : DataType.FLOAT32,
                facts.geometry().scratchBytes() / facts.geometry().workingBytes());
    }

    public static MemorySegment[] oracleBoundaries(CpuRecurrentScanLowering.Facts facts, Fixture f) {
        var result = new MemorySegment[facts.boundaryValues().size()];
        for (int p = 0; p < result.length; p++) result[p] = p < facts.readCount() ? f.segment(p)
                : heap(facts.boundaryTypes().get(p), facts.spans().get(p));
        return result;
    }

    /** Physical gaps are poisoned; only addressed reads receive finite fixture values. */
    public static void populate(CpuRecurrentScanLowering.Facts facts, Fixture f) {
        for (int role = 0; role < facts.rolePositions().size(); role++) {
            int p = facts.rolePositions().get(role);
            var a = facts.accessBindings().get(p);
            if (role == 1) {
                for (long k = 0; k < facts.spans().get(p); k++) f.segment(p).setAtIndex(ValueLayout.JAVA_LONG, k, Long.MAX_VALUE);
            } else {
                for (long k = 0; k < facts.spans().get(p); k++) setRaw(facts.geometry().dataType(), f.segment(p), k, nanPayload(facts.geometry().dataType()));
                var data = new double[Math.toIntExact(a.elementCount())];
                for (int k = 0; k < data.length; k++) data[k] = ((k * 7 + role * 3) % 17 - 8) * .037;
                scatter(facts, f, role, data);
            }
        }
        long[] lengths = new long[facts.geometry().batch()];
        for (int b = 0; b < lengths.length; b++) lengths[b] = Math.min(facts.geometry().time(), new int[] {0, 1, 3, 2}[b % 4]);
        scatterLengths(facts, f, lengths);
        // Poison padded x cells only when not also addressed by a valid overlapping read.
        var x = facts.accessBindings().get(facts.rolePositions().get(0));
        var actualLengths = gatherLengths(facts, f);
        var active = new HashSet<Long>();
        int features = facts.geometry().inputSize(), batch = facts.geometry().batch();
        for (int t = 0; t < facts.geometry().time(); t++) for (int b = 0; b < batch; b++) if (t < actualLengths[b])
            for (int k = 0; k < features; k++) active.add(address(x, ((long) t * batch + b) * features + k));
        for (long k = 0; k < x.elementCount(); k++) if (!active.contains(address(x, k)))
            setRaw(facts.geometry().dataType(), f.segment(facts.rolePositions().get(0)), address(x, k), nanPayload(facts.geometry().dataType()));
    }

    public static void scatter(CpuRecurrentScanLowering.Facts facts, Fixture f, int role, double[] values) {
        int p = facts.rolePositions().get(role);
        var a = facts.accessBindings().get(p);
        assertEquals(a.elementCount(), values.length);
        for (int k = 0; k < values.length; k++) write(facts.geometry().dataType(), f.segment(p), address(a, k), values[k]);
    }

    public static void scatterLengths(CpuRecurrentScanLowering.Facts facts, Fixture f, long... values) {
        int p = facts.rolePositions().get(1);
        var a = facts.accessBindings().get(p);
        for (int b = 0; b < values.length; b++) f.segment(p).setAtIndex(ValueLayout.JAVA_LONG, address(a, b), values[b]);
    }

    public static long[] gatherLengths(CpuRecurrentScanLowering.Facts facts, Fixture f) {
        int p = facts.rolePositions().get(1);
        var a = facts.accessBindings().get(p);
        var values = new long[facts.geometry().batch()];
        for (int b = 0; b < values.length; b++) values[b] = f.segment(p).getAtIndex(ValueLayout.JAVA_LONG, address(a, b));
        return values;
    }

    private static double[] gather(CpuRecurrentScanLowering.Facts facts, Fixture f, int role) {
        int p = facts.rolePositions().get(role);
        var a = facts.accessBindings().get(p);
        var data = new double[Math.toIntExact(a.elementCount())];
        for (int k = 0; k < data.length; k++) data[k] = number(facts.geometry().dataType(), f.segment(p), address(a, k));
        return data;
    }

    private static long address(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.Binding a, long ordinal) {
        long result = a.baseElementOffset();
        for (int axis = a.extents().size() - 1; axis >= 0; axis--) {
            result += (ordinal % a.extents().get(axis)) * a.effectiveStrides().get(axis);
            ordinal /= a.extents().get(axis);
        }
        return result;
    }

    public static double number(DataType type, MemorySegment segment, long index) {
        return switch (type) {
            case FLOAT64 -> segment.getAtIndex(ValueLayout.JAVA_DOUBLE, index);
            case FLOAT32 -> segment.getAtIndex(ValueLayout.JAVA_FLOAT, index);
            case BFLOAT16 -> BFloat16Bits.toFloat(segment.getAtIndex(ValueLayout.JAVA_SHORT, index));
            case FLOAT16 -> Float16Bits.toFloat(segment.getAtIndex(ValueLayout.JAVA_SHORT, index));
            default -> throw new AssertionError(type);
        };
    }

    public static void write(DataType type, MemorySegment segment, long index, double value) {
        switch (type) {
            case FLOAT64 -> segment.setAtIndex(ValueLayout.JAVA_DOUBLE, index, value);
            case FLOAT32 -> segment.setAtIndex(ValueLayout.JAVA_FLOAT, index, (float) value);
            case BFLOAT16 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, index, BFloat16Bits.fromFloat((float) value));
            case FLOAT16 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, index, Float16Bits.fromFloat((float) value));
            default -> throw new AssertionError(type);
        }
    }

    private static long raw(DataType type, MemorySegment segment, long index) {
        return switch (type) {
            case FLOAT64 -> segment.getAtIndex(ValueLayout.JAVA_LONG, index);
            case FLOAT32 -> Integer.toUnsignedLong(segment.getAtIndex(ValueLayout.JAVA_INT, index));
            case BFLOAT16, FLOAT16 -> Short.toUnsignedLong(segment.getAtIndex(ValueLayout.JAVA_SHORT, index));
            default -> throw new AssertionError(type);
        };
    }

    private static void setRaw(DataType type, MemorySegment segment, long index, long bits) {
        switch (type) {
            case FLOAT64 -> segment.setAtIndex(ValueLayout.JAVA_LONG, index, bits);
            case FLOAT32 -> segment.setAtIndex(ValueLayout.JAVA_INT, index, (int) bits);
            case BFLOAT16, FLOAT16 -> segment.setAtIndex(ValueLayout.JAVA_SHORT, index, (short) bits);
            default -> throw new AssertionError(type);
        }
    }

    private static long signBit(DataType type) {
        return type == DataType.FLOAT64 ? Long.MIN_VALUE : type == DataType.FLOAT32 ? 0x80000000L : 0x8000L;
    }

    private static long nanPayload(DataType type) {
        return type == DataType.FLOAT64 ? 0x7ff8123456789abcL : type == DataType.FLOAT32 ? 0x7fc54321L
                : type == DataType.BFLOAT16 ? 0x7fc5L : 0x7e35L;
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
