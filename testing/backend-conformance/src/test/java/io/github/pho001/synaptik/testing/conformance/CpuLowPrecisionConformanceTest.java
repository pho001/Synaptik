package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.cast.*;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.BinaryComparisonKind;
import io.github.pho001.synaptik.model.operation.elementwise.selection.WhereSelectionKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.ContiguousKind;
import io.github.pho001.synaptik.model.operation.reduction.AggregateReductionKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.*;
import java.lang.foreign.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Public stored results across CPU arithmetic, raw-bit and represented-value boundaries. */
final class CpuLowPrecisionConformanceTest {
    @Test void everyFp8EncodingSurvivesStridedCopyExactUnaryAndWhereSelection() {
        for (var type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            byte[] codes = new byte[256], reverse = new byte[256], mask = new byte[256];
            byte[] strided = new byte[514]; Arrays.fill(strided, (byte) 0x55);
            for (int i = 0; i < 256; i++) {
                codes[i] = (byte) i; reverse[i] = (byte) (255 - i); mask[i] = (byte) (i % 2);
                strided[1 + 2 * i] = codes[i];
            }
            var dense = descriptor(type, Shape.of(256));
            var view = new TensorDescriptor(type, dense.shape(), Optional.of(LayoutDescriptor.of(
                    dense.shape(), new long[] {2}, 1, true)), false);
            assertArrayEquals(codes, operation(new Operation(ContiguousKind.CONTIGUOUS, NoOperationAttrs.INSTANCE),
                    List.of(view), dense, List.of(MemorySegment.ofArray(strided))));
            for (var kind : List.of(UnaryElementwiseKind.NEG, UnaryElementwiseKind.ABS)) {
                byte[] expected = new byte[256];
                for (int i = 0; i < 256; i++) expected[i] = (byte) (kind == UnaryElementwiseKind.NEG ? i ^ 128 : i & 127);
                assertArrayEquals(expected, operation(new Operation(kind, NoOperationAttrs.INSTANCE),
                        List.of(dense), dense, List.of(MemorySegment.ofArray(codes))), type + " " + kind);
            }
            byte[] selected = new byte[256];
            for (int i = 0; i < 256; i++) selected[i] = mask[i] == 0 ? reverse[i] : codes[i];
            assertArrayEquals(selected, operation(new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                    List.of(descriptor(DataType.BOOL, dense.shape()), dense, dense), dense,
                    List.of(MemorySegment.ofArray(mask), MemorySegment.ofArray(codes), MemorySegment.ofArray(reverse))));
            byte[] selfEqual = new byte[256];
            for (int i = 0; i < 256; i++) selfEqual[i] = (byte) (Float.isNaN(LowPrecisionValues.decode(type, i)) ? 0 : 1);
            assertArrayEquals(selfEqual, operation(new Operation(BinaryComparisonKind.EQUAL, NoOperationAttrs.INSTANCE),
                    List.of(dense, dense), descriptor(DataType.BOOL, dense.shape()),
                    List.of(MemorySegment.ofArray(codes), MemorySegment.ofArray(codes))));
        }
    }

    @Test void fp8CastRoundingIncludesTiesSignedUnderflowAndNonSaturatingOverflow() {
        for (var type : List.of(DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2)) {
            boolean e4 = type == DataType.FLOAT8_E4M3;
            float tiny = Math.scalb(1f, e4 ? -9 : -16), step = e4 ? .125f : .25f;
            float edge = e4 ? 464 : 61440;
            float[] values = {0f, -0f, tiny / 2, -tiny / 2, Math.nextUp(tiny / 2),
                    1 + step / 2, 1 + 3 * step / 2, edge, Math.nextDown(edge), Math.nextUp(edge),
                    -Math.nextUp(edge), Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN};
            byte[] expected = new byte[values.length];
            for (int i = 0; i < values.length; i++) expected[i] = (byte) LowPrecisionValues.encode(type, values[i]);
            var shape = Shape.of(values.length);
            assertArrayEquals(expected, operation(new Operation(CastKind.CAST, new CastAttrs(type)),
                    List.of(descriptor(DataType.FLOAT32, shape)), descriptor(type, shape),
                    List.of(MemorySegment.ofArray(values))), type.toString());
        }
    }

    @Test void homogeneousAdditionRoundsOnlyOnceAndUnlikeLowTypesPromoteToFloat32() {
        for (var leftType : LowPrecisionValues.TYPES) {
            float half = Math.scalb(1f, leftType == DataType.BFLOAT16 ? -8 : leftType == DataType.FLOAT16 ? -11
                    : leftType == DataType.FLOAT8_E4M3 ? -4 : -3);
            float[] left = {1, 1, -1, 2}, right = {half, 3 * half, 1, -1};
            var shape = Shape.of(left.length);
            byte[] result = operation(new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                    List.of(descriptor(leftType, shape), descriptor(leftType, shape)), descriptor(leftType, shape),
                    List.of(storage(leftType, left), storage(leftType, right)));
            for (int i = 0; i < left.length; i++) assertEquals(LowPrecisionValues.encode(leftType, left[i] + right[i]),
                    LowPrecisionValues.raw(leftType, MemorySegment.ofArray(result), i), leftType + " index=" + i);
            for (var rightType : LowPrecisionValues.TYPES) if (leftType != rightType) {
                byte[] promoted = operation(new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                        List.of(descriptor(leftType, shape), descriptor(rightType, shape)), descriptor(DataType.FLOAT32, shape),
                        List.of(storage(leftType, left), storage(rightType, right)));
                var numbers = MemorySegment.ofArray(promoted);
                for (int i = 0; i < left.length; i++) {
                    float a = LowPrecisionValues.decode(leftType, LowPrecisionValues.encode(leftType, left[i]));
                    float b = LowPrecisionValues.decode(rightType, LowPrecisionValues.encode(rightType, right[i]));
                    assertEquals(a + b, numbers.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L));
                }
            }
            double[] doubles = {Math.scalb(1d, -40), -Math.scalb(1d, -40), .25, -.25};
            byte[] widened = operation(new Operation(BinaryArithmeticKind.ADD, NoOperationAttrs.INSTANCE),
                    List.of(descriptor(leftType, shape), descriptor(DataType.FLOAT64, shape)), descriptor(DataType.FLOAT64, shape),
                    List.of(storage(leftType, left), MemorySegment.ofArray(doubles)));
            for (int i = 0; i < left.length; i++) assertEquals((double) left[i] + doubles[i],
                    MemorySegment.ofArray(widened).get(ValueLayout.JAVA_DOUBLE_UNALIGNED, i * 8L));
        }
    }

    @Test void arithmeticOverflowAndUnderflowRetainFormatClassAndSign() {
        for (var type : LowPrecisionValues.TYPES) {
            int maxCode = switch (type) {
                case BFLOAT16 -> 0x7f7f; case FLOAT16 -> 0x7bff;
                case FLOAT8_E4M3 -> 0x7e; case FLOAT8_E5M2 -> 0x7b;
                default -> throw new AssertionError(type);
            };
            int normalCode = switch (type) {
                case BFLOAT16 -> 0x80; case FLOAT16 -> 0x400;
                case FLOAT8_E4M3 -> 8; case FLOAT8_E5M2 -> 4;
                default -> throw new AssertionError(type);
            };
            float maximum = LowPrecisionValues.decode(type, maxCode), normal = LowPrecisionValues.decode(type, normalCode);
            float[] left = {maximum, -maximum, normal, -normal}, right = {2, 2, .5f, .5f};
            var input = descriptor(type, Shape.of(4));
            var output = MemorySegment.ofArray(operation(new Operation(BinaryArithmeticKind.MUL, NoOperationAttrs.INSTANCE),
                    List.of(input, input), input, List.of(storage(type, left), storage(type, right))));
            for (int i = 0; i < left.length; i++) {
                float actual = LowPrecisionValues.decode(type, LowPrecisionValues.raw(type, output, i));
                int expected = LowPrecisionValues.encode(type, left[i] * right[i]);
                if (Float.isNaN(LowPrecisionValues.decode(type, expected))) assertTrue(Float.isNaN(actual));
                else if (i < 2) assertEquals(expected, LowPrecisionValues.raw(type, output, i));
                else {
                    int signedZero = i == 3 ? 1 << (type.bitWidth() - 1) : 0;
                    int raw = LowPrecisionValues.raw(type, output, i);
                    assertTrue(raw == expected || raw == signedZero, type + " gradual underflow or permitted FTZ only");
                }
            }
        }
    }

    @Test void representedComparisonsAndExtremaNeverFlushStoredSubnormalsOrPoisonFiniteWinners() {
        for (var type : LowPrecisionValues.TYPES) {
            float tiny = LowPrecisionValues.decode(type, 1);
            var shape = Shape.of(4); var input = descriptor(type, shape);
            float[] values = {tiny, -tiny, -0f, 1};
            byte[] comparison = operation(new Operation(BinaryComparisonKind.GREATER_THAN, NoOperationAttrs.INSTANCE),
                    List.of(input, input), descriptor(DataType.BOOL, shape),
                    List.of(storage(type, values), storage(type, new float[4])));
            assertArrayEquals(new byte[] {1, 0, 0, 1}, comparison, type.toString());
            for (var kind : List.of(AggregateReductionKind.MIN, AggregateReductionKind.MAX)) {
                float[] contributors = kind == AggregateReductionKind.MIN ? new float[] {2, tiny, 1, 3}
                        : new float[] {-2, -tiny, -1, -3};
                byte[] winner = operation(new Operation(kind, NoOperationAttrs.INSTANCE), List.of(input),
                        descriptor(type, Shape.scalar()), List.of(storage(type, contributors)));
                assertEquals(LowPrecisionValues.encode(type, kind == AggregateReductionKind.MIN ? tiny : -tiny),
                        LowPrecisionValues.raw(type, MemorySegment.ofArray(winner), 0), type + " " + kind);
            }
        }
    }

    @Test void emptyAggregatesPublishDeclaredIdentitiesIncludingE4m3UnrepresentableInfinities() {
        for (var type : LowPrecisionValues.TYPES) for (var kind : List.of(AggregateReductionKind.SUM,
                AggregateReductionKind.PROD, AggregateReductionKind.MIN, AggregateReductionKind.MAX)) {
            float identity = switch (kind) {
                case SUM -> 0; case PROD -> 1; case MIN -> Float.POSITIVE_INFINITY;
                case MAX -> Float.NEGATIVE_INFINITY; default -> throw new AssertionError(kind);
            };
            byte[] output = operation(new Operation(kind, NoOperationAttrs.INSTANCE),
                    List.of(descriptor(type, Shape.of(0))), descriptor(type, Shape.scalar()),
                    List.of(storage(type, new float[0])));
            int bits = LowPrecisionValues.raw(type, MemorySegment.ofArray(output), 0);
            if (type == DataType.FLOAT8_E4M3 && Float.isInfinite(identity))
                assertTrue(Float.isNaN(LowPrecisionValues.decode(type, bits)));
            else assertEquals(LowPrecisionValues.encode(type, identity), bits, type + " " + kind);
        }
    }

    private static MemorySegment storage(DataType type, float[] values) {
        var bytes = MemorySegment.ofArray(new byte[values.length * type.byteWidth()]);
        for (int i = 0; i < values.length; i++) {
            int bits = LowPrecisionValues.encode(type, values[i]);
            if (type.byteWidth() == 1) bytes.set(ValueLayout.JAVA_BYTE, i, (byte) bits);
            else bytes.set(ValueLayout.JAVA_SHORT_UNALIGNED, i * 2L, (short) bits);
        }
        return bytes;
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape) {
        return new TensorDescriptor(type, shape, Optional.of(LayoutDescriptor.contiguous(shape)), false);
    }

    private static byte[] operation(Operation operation, List<TensorDescriptor> inputs,
            TensorDescriptor output, List<MemorySegment> data) {
        var ids = new ArrayList<ValueId>(); var values = new ArrayList<GraphValue>();
        var supplied = new HashMap<ValueId, MemorySegment>();
        for (int i = 0; i < inputs.size(); i++) {
            var id = new ValueId(i); ids.add(id); values.add(new GraphValue(id, inputs.get(i))); supplied.put(id, data.get(i));
        }
        var out = new ValueId(inputs.size()); values.add(new GraphValue(out, output));
        var node = new CompiledNode(new NodeId(0), operation, ids, List.of(out));
        var partition = new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID, List.of(node.id()));
        var memory = values.stream().map(value -> new LogicalMemoryRequirement(value.id(), value.descriptor(),
                value.id().equals(out) ? Optional.of(partition) : Optional.empty(),
                value.id().equals(out) ? List.<PlannedPartition>of() : List.of(partition), value.id().equals(out))).toList();
        return execute(new PrepareContext<>(partition, List.of(node), values, memory, Map.of(),
                CpuPartitionAnalysisInputs.DEFAULT), supplied).get(out);
    }

    /**
     * Executes a real portable prepared fixture with fresh physical resources and returns copied buffers.
     * @param context complete independently constructed CPU graph projection
     * @param inputs physical bytes keyed by logical external value identity
     * @return detached raw buffer snapshots keyed by the plan's logical boundary identities
     */
    static Map<ValueId, byte[]> execute(PrepareContext<CpuPartitionAnalysisInputs> context, Map<ValueId, MemorySegment> inputs) {
        var analysis = new CpuPartitionPreparer().analyze(context);
        try (var fixture = new CpuRecurrentScanConformanceTest.Fixture(analysis, 1, false, false, Map.of())) {
            inputs.forEach((id, bytes) -> {
                var target = fixture.storage.get(id);
                if (target != null) MemorySegment.copy(bytes, 0, target, 0, target.byteSize());
            });
            fixture.executable.bind(fixture.state).execute();
            var result = new HashMap<ValueId, byte[]>();
            fixture.storage.forEach((id, segment) -> result.put(id, segment.toArray(ValueLayout.JAVA_BYTE)));
            return result;
        }
    }
}
