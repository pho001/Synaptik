package io.github.pho001.synaptik.backend.cpu.internal.route.portable;

import static io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.*;
import static io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRecurrentScanLoweringTest.*;
import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuRecurrentScanExecutionTest.Fixture;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuContiguousWorkspace;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;

/** Eight opt-in, bounded scalar diagnostics; elapsed ratios are never an acceptance or speed gate. */
@EnabledIfEnvironmentVariable(named = "SYNAPTIK_CPU_RECURRENT_SCAN_TIMING", matches = "true")
final class CpuRecurrentScanScalarTimingTest {
    private static volatile double consumed;

    @Test void eightFixedCasesAgainstPreboundSpecializedCleanJavaWithin120Seconds() {
        long deadline = System.nanoTime() + 120_000_000_000L;
        System.out.println("CPU0029 diagnostic ONLY JDK=" + Runtime.version() + " target="
                + System.getProperty("os.name") + "/" + System.getProperty("os.arch")
                + " carriers=typed-heap/native-mixed; setup/allocation excluded; no speed threshold");
        int index = 0, completed = 0;
        for (var type : floatingTypes()) for (int layout : new int[] {0, 2}) {
            var kind = RecurrentScanKind.values()[index++ % 3];
            int time = layout == 0 ? 8 : 24, batch = layout == 0 ? 3 : 4;
            int input = layout == 0 ? 7 : 12, hidden = layout == 0 ? 5 : 9;
            var analysis = new CpuPartitionPreparer().analyze(scalarContext(kind, true, RecurrentDirection.REVERSE,
                    time, batch, input, hidden, false, type, layout));
            var facts = analysis.plan().units().getFirst().runtimeFacts().recurrentScan().orElseThrow();
            try (var f = new Fixture(analysis, 2, true, true, Map.of());
                    var oracleFixture = new Fixture(analysis, 2, true, true, Map.of())) {
                populate(facts, f);
                var inputValues = new double[time * batch * input];
                for (int p = 0; p < inputValues.length; p++) inputValues[p] = ((p * 7) % 17 - 8) * .037;
                scatter(facts, f, 0, inputValues);
                // Keep a genuine moderate recurrent workload, with some padding and an empty row.
                scatterLengths(facts, f, batch == 3 ? new long[] {0, time / 2, time}
                        : new long[] {0, time / 3, time / 2, time});
                // Keep identical read segments. The second identical fixture supplies mutable
                // carrier kinds, physical slicing, spans and alignment matching the route exactly.
                var expected = new java.lang.foreign.MemorySegment[facts.boundaryValues().size()];
                for (int p = 0; p < expected.length; p++) {
                    expected[p] = p < facts.readCount() ? f.segment(p) : oracleFixture.segment(p);
                    assertEquals(f.segment(p).byteSize(), expected[p].byteSize());
                    assertEquals(f.segment(p).isNative(), expected[p].isNative());
                    assertEquals(f.segment(p).isReadOnly(), expected[p].isReadOnly());
                    if (!expected[p].isNative()) {
                        assertEquals(f.segment(p).heapBase().orElseThrow().getClass(), expected[p].heapBase().orElseThrow().getClass());
                        assertEquals(f.segment(p).address(), expected[p].address(), "same physical heap slice offset");
                    }
                    int alignment = facts.boundaryTypes().get(p).byteWidth();
                    assertEquals(0, f.segment(p).address() % alignment);
                    assertEquals(0, expected[p].address() % alignment);
                }
                var routeScratch = (CpuContiguousWorkspace) f.state.workspaceRepresentation(0);
                var oracleScratch = (CpuContiguousWorkspace) oracleFixture.state.workspaceRepresentation(0);
                assertEquals(routeScratch.byteSize(), oracleScratch.byteSize());
                assertEquals(routeScratch.byteAlignment(), oracleScratch.byteAlignment());
                assertEquals(routeScratch.writableSegment().isNative(), oracleScratch.writableSegment().isNative());
                assertEquals(0, oracleScratch.writableSegment().address() % facts.geometry().workingBytes());
                Runnable oracle = CpuRecurrentScanCleanJavaOracle.bind(facts, expected, oracleScratch.writableSegment());
                var bound = f.executable.bind(f.state);
                Runnable route = bound::execute;
                // Preflighted oracle execution omits prepared invocation access/length checks;
                // the route includes them. Ratios compare different call boundaries and cannot
                // isolate invocation overhead or prove kernel parity.
                route.run(); oracle.run();
                for (int p = facts.readCount(); p < expected.length; p++) assertArrayEquals(
                        expected[p].toArray(ValueLayout.JAVA_BYTE), f.segment(p).toArray(ValueLayout.JAVA_BYTE));
                for (int warm = 0; warm < 300 && System.nanoTime() < deadline; warm++) { route.run(); oracle.run(); }
                long[] routeSamples = new long[5], oracleSamples = new long[5];
                int samples = 0;
                for (; samples < 5 && System.nanoTime() < deadline; samples++) {
                    if ((samples & 1) == 0) {
                        routeSamples[samples] = sample(route, 100, deadline);
                        oracleSamples[samples] = sample(oracle, 100, deadline);
                    } else {
                        oracleSamples[samples] = sample(oracle, 100, deadline);
                        routeSamples[samples] = sample(route, 100, deadline);
                    }
                    consumed += number(type, f.segment(facts.readCount() + 1), (long) (batch - 1) * hidden)
                            + number(type, expected[facts.readCount() + 1], (long) (batch - 1) * hidden);
                }
                double routeMean = Arrays.stream(routeSamples).limit(samples).average().orElse(Double.NaN);
                double oracleMean = Arrays.stream(oracleSamples).limit(samples).average().orElse(Double.NaN);
                System.out.println("case=" + completed + " type=" + type + " family=" + kind + " layout=" + layout
                        + " shape=[" + time + "," + batch + "," + input + "] H=" + hidden
                        + " samples=" + samples + " batchIterations=100 routeNs=" + Arrays.toString(routeSamples)
                        + " oracleNs=" + Arrays.toString(oracleSamples) + " meanRatio=" + routeMean / oracleMean
                        + " checksum=" + consumed);
                completed++;
            }
            if (System.nanoTime() >= deadline) {
                System.out.println("INCONCLUSIVE: 120-second deadline, completed=" + completed + "/8");
                return;
            }
        }
        System.out.println("completed=" + completed + "/8; diagnostic only, no performance/parity claim");
    }

    private static long sample(Runnable action, int iterations, long deadline) {
        long start = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            if (System.nanoTime() >= deadline) return System.nanoTime() - start;
            action.run();
        }
        return System.nanoTime() - start;
    }
}
