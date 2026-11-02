package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuSymbolicScanTimingTest.Fixture;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuSymbolicScanTimingTest.PreparedFamily;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Numerical reuse and isolation consumers of the actual prepared symbolic derivative.
 * Fresh controlled AD is not compared with historical unannotated method identities or ABIs.
 * Frozen recipe identity evidence remains with the independent raw-recipe snapshot owners.
 */
class CpuSymbolicScanTimingMethodologyTest {
    @Test void bothFamiliesReuseNumericalMeaningAcross64_40_1_64_0AndIndependentClosure() {
        for (int[] dimensions : new int[][] {{3, 4, 3}, {4, 32, 32}}) {
            var family = new PreparedFamily(dimensions[0], dimensions[1], dimensions[2], false);
            var fixtures = new ArrayList<Fixture>();
            try {
                int[] batches = {64, 40, 1, 64, 0};
                for (int position = 0; position < batches.length; position++) {
                    var fixture = new Fixture(family, batches[position]);
                    fixtures.add(fixture);
                    fixture.verify();
                    var first = snapshot(fixture.actual);
                    fixture.reset((byte) 0x7f);
                    fixture.generated(); fixture.direct(); fixture.assertMatched();
                    assertSnapshot(first, fixture.actual);
                    fixture.generated(); fixture.direct(); fixture.assertMatched();
                    assertSnapshot(first, fixture.actual);
                    if (position == 1) fixtures.getFirst().close();
                    if (position >= 1) fixture.verify();
                    if (fixture.b == 0) {
                        for (byte value : fixture.actual.get(4).toArray(ValueLayout.JAVA_BYTE))
                            assertEquals(0, value, "empty-row parameter gradient remains positive zero");
                    }
                }
            } finally {
                fixtures.forEach(Fixture::close);
            }
        }
    }

    @Test void liveInputChangesAndOutputWritesDoNotAffectAnotherBoundInvocation() {
        for (int[] dimensions : new int[][] {{3, 4, 3}, {4, 32, 32}}) {
            var family = new PreparedFamily(dimensions[0], dimensions[1], dimensions[2], false);
            try (var first = new Fixture(family, 64); var second = new Fixture(family, 64)) {
                first.verify(); second.verify();
                var untouched = snapshot(second.actual);
                first.x.setAtIndex(ValueLayout.JAVA_FLOAT, 0,
                        first.x.getAtIndex(ValueLayout.JAVA_FLOAT, 0) + .125f);
                first.generated(); first.direct(); first.assertMatched();
                assertSnapshot(untouched, second.actual);
                first.reset((byte) 0x33);
                assertSnapshot(untouched, second.actual);
                first.close();
                second.verify();
            }
        }
    }

    @Test void singletonAndEmptyTimeTanhAndSigmoidRemainNumericalConsumers() {
        for (boolean sigmoid : List.of(false, true)) {
            try (var fixture = new Fixture(1, 1, 4, 4, sigmoid)) {
                fixture.verify();
                fixture.generated(); fixture.direct(); fixture.assertMatched();
            }
            try (var fixture = new Fixture(0, 1, 4, 4, sigmoid)) {
                fixture.verify();
                assertEquals(0,fixture.consumer.gradient(1).length,"T=0 sequence gradient has no logical cells");
                fixture.generated(); fixture.direct(); fixture.assertMatched();
            }
        }
    }

    /**
     * Snapshots observable output bytes independently of mutable fixture storage.
     * @param segments exact ordered numerical publications
     * @return immutable ordered byte-array snapshots
     */
    private static List<byte[]> snapshot(List<MemorySegment> segments) {
        return segments.stream().map(segment -> segment.toArray(ValueLayout.JAVA_BYTE)).toList();
    }

    /**
     * Checks that another invocation did not mutate these publications.
     * @param before independent expected snapshots
     * @param segments live ordered numerical publications
     */
    private static void assertSnapshot(List<byte[]> before, List<MemorySegment> segments) {
        for (int output = 0; output < segments.size(); output++)
            assertArrayEquals(before.get(output), segments.get(output).toArray(ValueLayout.JAVA_BYTE));
    }
}
