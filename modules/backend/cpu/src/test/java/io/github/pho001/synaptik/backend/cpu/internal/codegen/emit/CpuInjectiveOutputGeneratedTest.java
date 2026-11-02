package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Generated real-write regression and external-audit entry for large interleaved CPU outputs. */
public class CpuInjectiveOutputGeneratedTest {
    @Test void generatedWritesPreserveEveryInterleavedCellAndUntouchedGap() throws Throwable {
        executeLargeInterleavedWitness();
    }

    @Test void genuinelyOverlappingLargeOutputsRemainRejectedBeforeGeneration() {
        assertThrows(IllegalArgumentException.class,
                () -> new CpuPartitionPreparer().analyze(context(Shape.of(200_001, 3, 3),
                        new long[]{8, 3, 2}, 5, CarrierAccess.INT_ARRAY)));
        assertThrows(IllegalArgumentException.class,
                () -> new CpuPartitionPreparer().analyze(context(Shape.of(200_001, 2, 3),
                        new long[]{7, 3, 2}, 5, CarrierAccess.INT_ARRAY)));
    }

    /**
     * Executes actual generated TILE writes over the audit's 1,200,006-element witness, with a
     * nonzero output offset, split logical ranges and both heap and foreign-memory output carriers.
     * The independent coordinate oracle checks the complete physical output, including its holes.
     * This test-only entry lets an external consumer run the same physical lowering fixture.
     *
     * @throws Throwable if lowering, generation, invocation or a complete-output assertion fails
     */
    public static void executeLargeInterleavedWitness() throws Throwable {
        Shape shape = Shape.of(200_001, 2, 3);
        long[] strides = {8, 3, 2};
        int offset = 5;
        int count = 200_001 * 2 * 3;
        int capacity = offset + 200_000 * 8 + 3 + 4 + 3;
        int[] input = new int[count], expected = new int[capacity];
        Arrays.fill(expected, -1);
        int ordinal = 0;
        for (int i = 0; i < 200_001; i++) for (int j = 0; j < 2; j++) for (int k = 0; k < 3; k++) {
            input[ordinal] = ordinal + 1;
            expected[offset + i * 8 + j * 3 + k * 2] = input[ordinal++];
        }
        for (var carrier : List.of(CarrierAccess.INT_ARRAY, CarrierAccess.MEMORY_SEGMENT)) {
            var selected = context(shape, strides, offset, carrier);
            var plan = new CpuPartitionPreparer().analyze(selected).plan();
            var route = plan.units().getFirst().portablePlan();
            var generator = new CpuClassFileKernelGenerator();
            var artifact = generator.defineClassBytes(route.specialization(), route.kernelIr(),
                    generator.generateClassBytes(route.specialization(), route.kernelIr()));
            int[] output = new int[capacity];
            Arrays.fill(output, -1);
            Object outputCarrier = carrier == CarrierAccess.INT_ARRAY ? output : MemorySegment.ofArray(output);
            for (long[] range : List.of(new long[]{0, 0}, new long[]{0, 3},
                    new long[]{3, 600_003}, new long[]{600_003, count})) {
                long[] geometry = plan.movementGeometry().orElseThrow().pack(new long[2], range[0], range[1]);
                artifact.entryPoint().invokeWithArguments(input, outputCarrier, geometry, range[0], range[1]);
            }
            assertArrayEquals(expected, output, carrier.toString());
        }
    }

    private static PrepareContext<CpuPartitionAnalysisInputs> context(Shape shape, long[] strides,
            long offset, CarrierAccess outputCarrier) {
        var base = CpuNonAffineMovementLoweringTest.context(
                new Operation(TileKind.TILE, new TileAttrs(List.of(1L, 1L, 1L))), List.of(0),
                List.of(CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, shape)),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT32, shape,
                        LayoutDescriptor.of(shape, strides, offset, true)));
        return new PrepareContext<>(base.partition(), base.nodes(), base.values(),
                base.memoryRequirements(), Map.of(),
                new CpuPartitionAnalysisInputs(false, List.of(CarrierAccess.INT_ARRAY, outputCarrier)));
    }
}
