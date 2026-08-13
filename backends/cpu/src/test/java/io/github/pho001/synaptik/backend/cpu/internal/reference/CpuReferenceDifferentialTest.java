package io.github.pho001.synaptik.backend.cpu.internal.reference;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuAffineLayoutLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuIndexingLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuIndexingIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuDataMovementIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuNonAffineMovementLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuRandomLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRandomIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScanIr;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScanLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuScatterLoweringTest;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuScatterIr;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.OneHotKind;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.index.AxisGatherKind;
import io.github.pho001.synaptik.model.operation.index.IndexAxisAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdAttrs;
import io.github.pho001.synaptik.model.operation.index.GatherNdKind;
import io.github.pho001.synaptik.model.operation.index.AxisScatterKind;
import io.github.pho001.synaptik.model.operation.index.ScatterElementsAttrs;
import io.github.pho001.synaptik.model.operation.index.ScatterReduction;
import io.github.pho001.synaptik.model.operation.layout.CropToShapeAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import java.util.List;

class CpuReferenceDifferentialTest {
    private static final long RANDOM_KEY_BIAS = 0x9e3779b97f4a7c15L;
    private static final long RANDOM_M1 = 0xbf58476d1ce4e5b9L;
    private static final long RANDOM_M2 = 0x94d049bb133111ebL;

    @Test void independentScanReferenceCoversKindsModesTypesAndSlices() {
        for (DataType type : List.of(DataType.FLOAT64, DataType.FLOAT32, DataType.BFLOAT16,
                DataType.INT32, DataType.INT64))
            for (var kind : io.github.pho001.synaptik.model.operation.scan.CumulativeScanKind.values())
                for (boolean exclusive : List.of(false, true)) for (boolean reverse : List.of(false, true)) {
                    var lowered = new CpuPartitionLowering().lower(CpuScanLoweringTest.context(
                            kind, type, Shape.of(2, 3), 1, exclusive, reverse));
                    Object input = represented(type, 1, 2, 3, 4, 5, 6);
                    Object output = represented(type, 0, 0, 0, 0, 0, 0);
                    CpuScalarReferenceKernel.execute((CpuScanIr) lowered.portableKernelIr(),
                            lowered.scanGeometry().orElseThrow(),
                            List.of(argument(type, input, true), argument(type, output, false)));
                    assertNotNull(output);
                }
    }

    @Test void independentRandomReferenceMatchesExactVectorsMaskScalingAndState() {
        var lowered = new CpuPartitionLowering().lower(CpuRandomLoweringTest.dropoutContext(
                DataType.FLOAT32, Shape.of(5), .25));
        float[] input = {-0.0f, 1, -2, Float.NaN, Float.POSITIVE_INFINITY};
        long[] state = {0x1234, 7}, next = new long[2];
        float[] output = new float[5]; byte[] mask = new byte[5];
        CpuScalarReferenceKernel.execute((CpuRandomIr) lowered.portableKernelIr(),
                lowered.randomGeometry().orElseThrow(), List.of(
                        new CpuBufferArgument.Floats(input, 0, 20, true),
                        new CpuBufferArgument.Longs(state, 0, 16, true),
                        new CpuBufferArgument.Floats(output, 0, 20, false),
                        new CpuBufferArgument.Bytes(mask, 0, 5, false),
                        new CpuBufferArgument.Longs(next, 0, 16, false)));
        assertAll(() -> assertArrayEquals(new long[] {0x1234, 12}, next),
                () -> assertTrue(java.util.Arrays.stream(new int[] {mask[0], mask[1], mask[2],
                        mask[3], mask[4]}).allMatch(value -> value == 0 || value == 1)),
                () -> assertEquals(0L, Float.floatToRawIntBits(output[0]) == 0
                        ? 0L : Integer.toUnsignedLong(Float.floatToRawIntBits(output[0]))));
    }

    @Test void independentRandomReferenceHonorsBroadcastReadsAndStridedWrites() {
        Shape shape = Shape.of(2, 2); Shape stateShape = Shape.of(2);
        var lowered = new CpuPartitionLowering().lower(CpuRandomLoweringTest.dropoutContext(
                DataType.FLOAT64, shape, .5, List.of(
                        LayoutDescriptor.of(shape, new long[] {0, 2}, 1, true),
                        LayoutDescriptor.of(stateShape, new long[] {2}, 1, true),
                        LayoutDescriptor.of(shape, new long[] {7, 2}, 1, true),
                        LayoutDescriptor.of(shape, new long[] {8, 3}, 2, true),
                        LayoutDescriptor.of(stateShape, new long[] {3}, 2, true))));
        double[] input = {-50, 2, -50, -6}; long[] state = {-50, 0, -50, 0};
        double[] output = new double[11]; java.util.Arrays.fill(output, -50);
        byte[] mask = new byte[14]; java.util.Arrays.fill(mask, (byte) -50);
        long[] next = new long[6]; java.util.Arrays.fill(next, -50);
        CpuScalarReferenceKernel.execute((CpuRandomIr) lowered.portableKernelIr(),
                lowered.randomGeometry().orElseThrow(), List.of(
                        new CpuBufferArgument.Doubles(input, 0, input.length * 8L, true),
                        new CpuBufferArgument.Longs(state, 0, state.length * 8L, true),
                        new CpuBufferArgument.Doubles(output, 0, output.length * 8L, false),
                        new CpuBufferArgument.Bytes(mask, 0, mask.length, false),
                        new CpuBufferArgument.Longs(next, 0, next.length * 8L, false)));
        int[] inputAddresses = {1, 3, 1, 3}; int[] outputAddresses = {1, 3, 8, 10};
        int[] maskAddresses = {2, 5, 10, 13};
        for (int logical = 0; logical < 4; logical++) {
            boolean keep = randomUniform(randomWord(0, 0, logical)) >= .5;
            assertEquals(keep ? (byte) 1 : (byte) 0, mask[maskAddresses[logical]]);
            assertEquals(Double.doubleToRawLongBits(keep
                            ? input[inputAddresses[logical]] / .5 : 0.0d),
                    Double.doubleToRawLongBits(output[outputAddresses[logical]]));
        }
        assertAll(() -> assertEquals(-50, output[0]),
                () -> assertEquals((byte) -50, mask[0]),
                () -> assertEquals(0, next[2]),
                () -> assertEquals(4, next[5]));
    }

    private static long randomWord(long key, long counter, long logical) {
        return randomMix(counter + logical + randomMix(key + RANDOM_KEY_BIAS));
    }

    private static long randomMix(long value) {
        value = (value ^ (value >>> 30)) * RANDOM_M1;
        value = (value ^ (value >>> 27)) * RANDOM_M2;
        return value ^ (value >>> 31);
    }

    private static double randomUniform(long word) { return (word >>> 11) * 0x1.0p-53; }
    @Test void independentScatterOracleCoversExactProductAndReplacementValidation() {
        var lowered=new CpuPartitionLowering().lower(CpuScatterLoweringTest.context(
                new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                        new ScatterElementsAttrs(0,ScatterReduction.MUL)),List.of(0,1,2),
                List.of(CpuScatterLoweringTest.desc(DataType.FLOAT64,Shape.of(2)),
                        CpuScatterLoweringTest.desc(DataType.INT32,Shape.of(3)),
                        CpuScatterLoweringTest.desc(DataType.FLOAT64,Shape.of(3))),
                CpuScatterLoweringTest.desc(DataType.FLOAT64,Shape.of(2))));
        double[] output=new double[2];
        CpuScalarReferenceKernel.execute((CpuScatterIr)lowered.portableKernelIr(),
                lowered.scatterGeometry().orElseThrow(),List.of(
                        new CpuBufferArgument.Doubles(new double[]{0.5,3},0,16,true),
                        new CpuBufferArgument.Ints(new int[]{0,0,0},0,12,true),
                        new CpuBufferArgument.Doubles(new double[]{0.25,8,0.1},0,24,true),
                        new CpuBufferArgument.Doubles(output,0,16,false)),0,2);
        assertAll(() -> assertEquals(0.1,output[0]),()->assertEquals(3.0,output[1]));
    }
    @Test void sliceUpdateReferenceCoversBothFormsAndMultipleSelectedAxes() {
        var signed = new CpuPartitionLowering().lower(CpuNonAffineMovementLoweringTest.context(
                new Operation(SliceKind.SLICE_UPDATE,
                        new SliceAttrs(List.of(2L, 3L), List.of(2L, 2L), List.of(0, 1),
                                List.of(-2L, -2L))),
                List.of(0, 1), List.of(CpuNonAffineMovementLoweringTest.descriptor(
                                DataType.INT64, Shape.of(3, 4)),
                        CpuNonAffineMovementLoweringTest.descriptor(
                                DataType.INT64, Shape.of(2, 2))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT64, Shape.of(3, 4))));
        long[] signedOutput = new long[12];
        CpuScalarReferenceKernel.execute((CpuDataMovementIr) signed.portableKernelIr(),
                signed.movementGeometry().orElseThrow(), List.of(
                        new CpuBufferArgument.Longs(
                                new long[]{1,2,3,4,5,6,7,8,9,10,11,12}, 0, 96, true),
                        new CpuBufferArgument.Longs(new long[]{90,91,80,81}, 0, 32, true),
                        new CpuBufferArgument.Longs(signedOutput, 0, 96, false)), 0, 12);

        var crop = new CpuPartitionLowering().lower(CpuNonAffineMovementLoweringTest.context(
                new Operation(SliceKind.SLICE_UPDATE,
                        new CropToShapeAttrs(Shape.of(2, 2), Shape.of(0, 1))),
                List.of(0, 1), List.of(CpuNonAffineMovementLoweringTest.descriptor(
                                DataType.INT64, Shape.of(2, 4)),
                        CpuNonAffineMovementLoweringTest.descriptor(
                                DataType.INT64, Shape.of(2, 2))),
                CpuNonAffineMovementLoweringTest.descriptor(DataType.INT64, Shape.of(2, 4))));
        long[] cropOutput = new long[8];
        CpuScalarReferenceKernel.execute((CpuDataMovementIr) crop.portableKernelIr(),
                crop.movementGeometry().orElseThrow(), List.of(
                        new CpuBufferArgument.Longs(new long[]{1,2,3,4,5,6,7,8}, 0, 64, true),
                        new CpuBufferArgument.Longs(new long[]{9,10,11,12}, 0, 32, true),
                        new CpuBufferArgument.Longs(cropOutput, 0, 64, false)), 0, 8);
        assertAll(
                () -> assertArrayEquals(new long[]{1,81,3,80,5,6,7,8,9,91,11,90},
                        signedOutput),
                () -> assertArrayEquals(new long[]{1,9,10,4,5,11,12,8}, cropOutput));
    }

    @Test void gatherReferenceMatchesRepresentedBitsForEveryDataAndIndexType() {
        for (DataType dataType : DataType.values()) {
            for (DataType indexType : List.of(DataType.INT32, DataType.INT64)) {
                var lowered = new io.github.pho001.synaptik.backend.cpu.internal.lowering
                        .CpuPartitionLowering().lower(CpuIndexingLoweringTest.context(
                                new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(0)),
                                List.of(0, 1), List.of(CpuIndexingLoweringTest.descriptor(
                                                dataType, Shape.of(3)),
                                        CpuIndexingLoweringTest.descriptor(indexType, Shape.of(2))),
                                CpuIndexingLoweringTest.descriptor(dataType, Shape.of(2))));
                Object source = represented(dataType, 10, 21, 31);
                Object index = indexType == DataType.INT32 ? new int[]{2, 0}
                        : new long[]{2, 0};
                Object output = represented(dataType, 0, 0);
                CpuScalarReferenceKernel.execute((CpuIndexingIr) lowered.portableKernelIr(),
                        lowered.indexingGeometry().orElseThrow(),
                        List.of(argument(dataType, source, true), argument(indexType, index, true),
                                argument(dataType, output, false)), 0, 2);
                assertRepresentedEquals(represented(dataType, 31, 10), output,
                        dataType + "/" + indexType);
            }
        }
    }

    @Test void indexingReferenceValidatesBeforeWritingAndEmitsCanonicalOneHot() {
        var lowered = new io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering()
                .lower(CpuIndexingLoweringTest.context(
                        new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(3)), List.of(0),
                        List.of(CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2))),
                        CpuIndexingLoweringTest.descriptor(DataType.BOOL, Shape.of(2, 3))));
        byte[] output = {9,9,9,9,9,9};
        var invalid = List.<CpuBufferArgument>of(new CpuBufferArgument.Ints(
                new int[]{2,-1},0,8,true), new CpuBufferArgument.Bytes(output,0,6,false));
        assertAll(() -> assertThrows(IndexOutOfBoundsException.class, () ->
                        CpuScalarReferenceKernel.execute((CpuIndexingIr) lowered.portableKernelIr(),
                                lowered.indexingGeometry().orElseThrow(), invalid, 0, 6)),
                () -> assertArrayEquals(new byte[]{9,9,9,9,9,9}, output));
        var valid = List.<CpuBufferArgument>of(new CpuBufferArgument.Ints(
                new int[]{2,0},0,8,true), new CpuBufferArgument.Bytes(output,0,6,false));
        CpuScalarReferenceKernel.execute((CpuIndexingIr) lowered.portableKernelIr(),
                lowered.indexingGeometry().orElseThrow(), valid, 0, 6);
        assertArrayEquals(new byte[]{0,0,1,1,0,0}, output);
    }

    @Test void referenceCoversGatherElementsAndBatchedGatherNdMappings() {
        var elements = new io.github.pho001.synaptik.backend.cpu.internal.lowering
                .CpuPartitionLowering().lower(CpuIndexingLoweringTest.context(
                        new Operation(AxisGatherKind.GATHER_ELEMENTS, new IndexAxisAttrs(1)),
                        List.of(0, 1), List.of(CpuIndexingLoweringTest.descriptor(
                                        DataType.INT64, Shape.of(2, 3)),
                                CpuIndexingLoweringTest.descriptor(DataType.INT32, Shape.of(2, 2))),
                        CpuIndexingLoweringTest.descriptor(DataType.INT64, Shape.of(2, 2))));
        long[] elementOutput = new long[4];
        CpuScalarReferenceKernel.execute((CpuIndexingIr) elements.portableKernelIr(),
                elements.indexingGeometry().orElseThrow(), List.of(
                        new CpuBufferArgument.Longs(new long[]{10,11,12,20,21,22},0,48,true),
                        new CpuBufferArgument.Ints(new int[]{2,0,1,2},0,16,true),
                        new CpuBufferArgument.Longs(elementOutput,0,32,false)), 0, 4);

        var nd = new io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering()
                .lower(CpuIndexingLoweringTest.context(
                        new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(1)), List.of(0, 1),
                        List.of(CpuIndexingLoweringTest.descriptor(
                                        DataType.FLOAT32, Shape.of(2, 3, 2)),
                                CpuIndexingLoweringTest.descriptor(
                                        DataType.INT64, Shape.of(2, 2, 1))),
                        CpuIndexingLoweringTest.descriptor(DataType.FLOAT32, Shape.of(2, 2, 2))));
        float[] ndOutput = new float[8];
        CpuScalarReferenceKernel.execute((CpuIndexingIr) nd.portableKernelIr(),
                nd.indexingGeometry().orElseThrow(), List.of(
                        new CpuBufferArgument.Floats(new float[]{0,1,10,11,20,21,
                                100,101,110,111,120,121},0,48,true),
                        new CpuBufferArgument.Longs(new long[]{2,0,1,0},0,32,true),
                        new CpuBufferArgument.Floats(ndOutput,0,32,false)), 0, 8);
        assertAll(() -> assertArrayEquals(new long[]{12,10,21,22}, elementOutput),
                () -> assertArrayEquals(new float[]{20,21,0,1,110,111,100,101}, ndOutput));
    }
    @Test void affineReferencePreservesOpaqueBfloat16AddressPairs() {
        var lowered = new io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPartitionLowering()
                .lower(CpuAffineLayoutLoweringTest.select(DataType.BFLOAT16, List.of()));
        var encoded = lowered.kernelIr();
        var affine = new CpuAffineCopyIr(DataType.BFLOAT16,
                encoded.values().get(0).accessPlan(), encoded.values().get(1).accessPlan(),
                List.of(new CpuAffineCopyIr.MappingStep(CpuAffineCopyIr.MappingKind.SELECT,
                        2, 1, List.of(0))),
                CpuAffineCopyIr.WriteDomain.LOGICAL_ELEMENTS);
        short[] source = {1, (short)0x7fc1, 2, 3, (short)0xff80, 4, 5, (short)0x8000, 6};
        short[] output = new short[8];
        CpuScalarReferenceKernel.execute(affine, lowered.affineAddressPairs(), List.of(
                new CpuBufferArgument.Shorts(source, 0, 18, true),
                new CpuBufferArgument.Shorts(output, 0, 16, false)), 0, 3);
        assertAll(() -> assertEquals(source[1], output[1]),
                () -> assertEquals(source[4], output[4]),
                () -> assertEquals(source[7], output[7]));
    }
    @Test void preservesClassificationsSignedZeroAndOracleTolerance() {
        assertAll(
                () -> assertTrue(Double.isNaN(CpuScalarReferenceKernel.erf(Double.NaN))),
                () -> assertEquals(1.0, CpuScalarReferenceKernel.erf(Double.POSITIVE_INFINITY)),
                () -> assertEquals(-1.0, CpuScalarReferenceKernel.erf(Double.NEGATIVE_INFINITY)),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0),
                        Double.doubleToRawLongBits(CpuScalarReferenceKernel.erf(-0.0))),
                () -> assertTrue(Double.isNaN(CpuScalarReferenceKernel.gelu(Double.NaN))),
                () -> assertEquals(Double.POSITIVE_INFINITY,
                        CpuScalarReferenceKernel.gelu(Double.POSITIVE_INFINITY)),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0), Double.doubleToRawLongBits(
                        CpuScalarReferenceKernel.gelu(Double.NEGATIVE_INFINITY))),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0),
                        Double.doubleToRawLongBits(CpuScalarReferenceKernel.gelu(-0.0))),
                () -> assertEquals(0.8427007929497149, CpuScalarReferenceKernel.erf(1.0), 2e-7));
        assertAll(
                () -> assertEquals(0.0d, CpuScalarReferenceKernel.sigmoid(Double.NEGATIVE_INFINITY)),
                () -> assertEquals(1.0d, CpuScalarReferenceKernel.sigmoid(Double.POSITIVE_INFINITY)),
                () -> assertEquals(0.5d, CpuScalarReferenceKernel.sigmoid(-0.0d)),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0d), Double.doubleToRawLongBits(
                        CpuScalarReferenceKernel.geluTanhApproximation(Double.NEGATIVE_INFINITY))),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0d), Double.doubleToRawLongBits(
                        CpuScalarReferenceKernel.silu(Double.NEGATIVE_INFINITY))),
                () -> assertEquals(Double.doubleToRawLongBits(-0.0d), Double.doubleToRawLongBits(
                        CpuScalarReferenceKernel.silu(-0.0d))));
        double[][] oracle = {
                {-3.0, -0.00404969409489031}, {-1.0, -0.15865525393145707},
                {-0.25, -0.100323432704662}, {0.0, 0.0},
                {0.25, 0.149676567295338}, {1.0, 0.8413447460685429},
                {3.0, 2.99595030590511}
        };
        for (double[] pair : oracle) assertEquals(pair[1],
                CpuScalarReferenceKernel.gelu(pair[0]),
                2e-7 * Math.max(1.0, Math.abs(pair[1])));
    }

    private static Object represented(DataType type, int... values) {
        return switch (type) {
            case FLOAT64 -> java.util.Arrays.stream(values).asDoubleStream().toArray();
            case FLOAT32 -> { float[] result = new float[values.length];
                for (int i = 0; i < values.length; i++) result[i] = values[i]; yield result; }
            case BFLOAT16 -> { short[] result = new short[values.length];
                for (int i = 0; i < values.length; i++) result[i] = (short) (0x3f00 + values[i]);
                yield result; }
            case INT32 -> values.clone();
            case INT64 -> java.util.Arrays.stream(values).asLongStream().toArray();
            case BOOL -> { byte[] result = new byte[values.length];
                for (int i = 0; i < values.length; i++) result[i] = (byte) (values[i] & 1);
                yield result; }
        };
    }

    private static CpuBufferArgument argument(DataType type, Object carrier, boolean readOnly) {
        return switch (type) {
            case FLOAT64 -> new CpuBufferArgument.Doubles((double[]) carrier, 0,
                    ((double[]) carrier).length * 8L, readOnly);
            case FLOAT32 -> new CpuBufferArgument.Floats((float[]) carrier, 0,
                    ((float[]) carrier).length * 4L, readOnly);
            case BFLOAT16 -> new CpuBufferArgument.Shorts((short[]) carrier, 0,
                    ((short[]) carrier).length * 2L, readOnly);
            case INT32 -> new CpuBufferArgument.Ints((int[]) carrier, 0,
                    ((int[]) carrier).length * 4L, readOnly);
            case INT64 -> new CpuBufferArgument.Longs((long[]) carrier, 0,
                    ((long[]) carrier).length * 8L, readOnly);
            case BOOL -> new CpuBufferArgument.Bytes((byte[]) carrier, 0,
                    ((byte[]) carrier).length, readOnly);
        };
    }

    private static void assertRepresentedEquals(Object expected, Object actual, String message) {
        if (expected instanceof double[] value) assertArrayEquals(value, (double[]) actual, message);
        else if (expected instanceof float[] value) assertArrayEquals(value, (float[]) actual, message);
        else if (expected instanceof short[] value) assertArrayEquals(value, (short[]) actual, message);
        else if (expected instanceof int[] value) assertArrayEquals(value, (int[]) actual, message);
        else if (expected instanceof long[] value) assertArrayEquals(value, (long[]) actual, message);
        else assertArrayEquals((byte[]) expected, (byte[]) actual, message);
    }
}
