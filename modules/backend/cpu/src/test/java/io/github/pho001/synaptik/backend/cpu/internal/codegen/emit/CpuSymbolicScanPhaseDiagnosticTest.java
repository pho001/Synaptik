package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuSymbolicScanTimingTest.Fixture;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuSymbolicScanTimingTest.PreparedFamily;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

/**
 * Joint numerical consumers of the actual prepared symbolic derivative and its independent
 * Java controls. Fresh controlled AD has no historical two-entry phase ABI: observations below
 * measure the real joint bound invocation, not fabricated phase helpers or pure-loop costs.
 */
class CpuSymbolicScanPhaseDiagnosticTest {
    @Test void productionJointMatchesAllFiveOutputsAcrossDynamicBatchAndRepeatedPoison() {
        for (int[] dimensions : new int[][] {{3, 4, 3}, {4, 32, 32}}) {
            var family = new PreparedFamily(dimensions[0], dimensions[1], dimensions[2], false);
            for (int batch : new int[] {0, 1, 40, 64}) try (var fixture = new Fixture(family, batch)) {
                fixture.verify();
                var first = snapshot(fixture.actual);
                for (int repetition = 0; repetition < 3; repetition++) {
                    if (repetition < 2) fixture.reset(repetition == 0 ? (byte) 0x7f : (byte) 0x5a);
                    fixture.generated(); fixture.direct(); fixture.assertMatched();
                    assertSnapshot(first, fixture.actual);
                }
            }
        }
    }

    @Test void independentJavaPhaseCompositionStillMatchesItsImmutableJointControl() {
        var family = new PreparedFamily(4, 32, 32, false);
        for (int batch : new int[] {0, 1, 40, 64}) try (var fixture = new Fixture(family, batch)) {
            var writes = new ArrayList<>(fixture.expected);
            writes.add(fixture.savedHistory);
            var inputs = List.of(fixture.initial, fixture.x, fixture.w, fixture.fs, fixture.es);
            var inputBytes = snapshot(inputs);
            fixture.direct();
            var joint = snapshot(writes);
            for (int repetition = 0; repetition < 3; repetition++) {
                if (repetition < 2) for (MemorySegment write : writes)
                    write.fill(repetition == 0 ? (byte) 0x7f : (byte) 0x5a);
                CpuSymbolicScanPhaseOracle.forward(fixture.b, fixture.initial, fixture.x, fixture.w,
                        fixture.carry, fixture.history, fixture.savedHistory);
                CpuSymbolicScanPhaseOracle.reverse(fixture.b, fixture.x, fixture.w, fixture.fs, fixture.es,
                        fixture.savedHistory, fixture.dh, fixture.dx, fixture.dw);
                assertSnapshot(joint, writes);
                assertSnapshot(inputBytes, inputs);
            }
        }
    }

    /**
     * Observes only complete joint consumer/direct pairs under the existing nonpreemptive
     * 30-second admission cap. Setup, checks and reporting count against that cap. An admitted
     * pair finishes both calls even when it overshoots; censored work reports counts, not a partial
     * mean or a throughput claim. There is no timing/JIT/parity assertion and no phase-cost claim.
     */
    @Test
    @EnabledIfSystemProperty(named = "synaptik.cpu.dynamicBatch.performance", matches = "true")
    void boundedJointProductionConsumerObservation() {
        long begin = System.nanoTime(), deadline = begin + 30_000_000_000L;
        int startup = 0, warm = 0, samples = 0;
        long generatedNanos = 0, directNanos = 0;
        boolean censored = false;
        var family = new PreparedFamily(4, 32, 32, false);
        try (var fixture = new Fixture(family, 64)) {
            fixture.verify();
            Runnable direct = fixture.timedDirect();
            for (; startup < 3000 && System.nanoTime() < deadline; startup++) {
                fixture.generated(); direct.run();
            }
            for (; startup == 3000 && warm < 1000 && System.nanoTime() < deadline; warm++) {
                fixture.generated(); direct.run();
            }
            if (startup != 3000 || warm != 1000) censored = true;
            for (; !censored && samples < 900; samples++) {
                if (System.nanoTime() >= deadline) { censored = true; break; }
                boolean generatedFirst = (samples / 300) % 2 == 0;
                long start = System.nanoTime();
                if (generatedFirst) fixture.generated(); else direct.run();
                long middle = System.nanoTime();
                if (generatedFirst) direct.run(); else fixture.generated();
                long end = System.nanoTime();
                generatedNanos += generatedFirst ? middle - start : end - middle;
                directNanos += generatedFirst ? end - middle : middle - start;
                if (end >= deadline) censored = true;
            }
            fixture.assertMatched();
        }
        System.out.println("symbolic joint assigned-slot consumer T4/F32/H32/B64 startupPairs=" + startup
                + " warmPairs=" + warm + " completedSamplePairs=" + samples + " censored=" + censored
                + " elapsedMs=" + (System.nanoTime() - begin) / 1e6
                + (censored || samples != 900 ? " noPartialMean" : " generatedMeanNs=" + generatedNanos / samples
                        + " directMeanNs=" + directNanos / samples)
                + " includesBoundInvocationAndFixtureCosts; noPhaseOrPureLoopClaim");
    }

    /**
     * Snapshots numerical storage independently of later writes.
     * @param segments ordered live numerical publications or inputs
     * @return immutable ordered byte-array snapshots
     */
    private static List<byte[]> snapshot(List<MemorySegment> segments) {
        return segments.stream().map(segment -> segment.toArray(ValueLayout.JAVA_BYTE)).toList();
    }

    /**
     * Compares independent numerical snapshots without relying on storage identity.
     * @param before expected ordered bytes
     * @param segments live ordered numerical storage
     */
    private static void assertSnapshot(List<byte[]> before, List<MemorySegment> segments) {
        for (int output = 0; output < segments.size(); output++)
            assertArrayEquals(before.get(output), segments.get(output).toArray(ValueLayout.JAVA_BYTE));
    }
}
