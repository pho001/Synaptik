package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxAttrs;
import io.github.pho001.synaptik.model.operation.normalization.SoftmaxKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Device regressions for named normalization sites and guarded categorical-loss contributors. */
class MetalNormalizationLossNativeTest {
    private static final MetalMpsGraphProgram.AttributeKind AXIS = MetalMpsGraphProgram.AttributeKind.AXIS;
    private static final MetalMpsGraphProgram.AttributeKind NORMALIZED =
            MetalMpsGraphProgram.AttributeKind.NORMALIZED_SHAPE_EPSILON;

    @Test
    void genuineEmptyNormalizationDomainsRetainZeroExtentsAndCanonicalPorts() {
        for (var kind : List.of(MetalMpsGraphProgram.NodeKind.LAYER_NORM, MetalMpsGraphProgram.NodeKind.RMS_NORM)) {
            assertEquals(0, unary(kind, NORMALIZED, normalized(2, 1), new long[] {0, 2}, floats()).length);
            assertEquals(0, unary(kind, NORMALIZED, normalized(0, 1), new long[] {2, 0}, floats()).length);
        }
        var inference = node(MetalMpsGraphProgram.NodeKind.BATCH_NORM_INFERENCE,
                new int[] {0, 1, 2, 3, 4}, new int[] {5},
                MetalMpsGraphProgram.AttributeKind.BATCH_NORM_INFERENCE, new long[] {1, 1, bits(1)});
        assertEquals(0, execute(List.of(value(0, 2), value(2), value(2), value(2), value(2), value(0, 2)),
                inference, 5, new byte[][] {floats(), floats(1, 1), floats(0, 0), floats(0, 0), floats(1, 1)})
                .getFirst().length);
        var training = node(MetalMpsGraphProgram.NodeKind.BATCH_NORM_TRAINING,
                new int[] {0, 1, 2, 3, 4}, new int[] {5, 6, 7, 8, 9},
                MetalMpsGraphProgram.AttributeKind.BATCH_NORM_TRAINING,
                new long[] {1, 1, bits(0.5f), 1, bits(1)});
        var results = execute(List.of(value(2, 0), value(0), value(0), value(0), value(0),
                        value(2, 0), value(0), value(0), value(0), value(0)),
                training, 5, new byte[][] {floats(), floats(), floats(), floats(), floats()});
        assertEquals(5, results.size());
        for (float[] result : results) assertEquals(0, result.length);
    }

    @Test
    void trainingRejectsZeroPopulationWhenChannelsRemainNonempty() {
        var training = node(MetalMpsGraphProgram.NodeKind.BATCH_NORM_TRAINING,
                new int[] {0, 1, 2, 3, 4}, new int[] {5, 6, 7, 8, 9},
                MetalMpsGraphProgram.AttributeKind.BATCH_NORM_TRAINING,
                new long[] {1, 1, bits(0.5f), 1, bits(1)});
        assertThrows(IllegalArgumentException.class, () -> MetalNormalizationSupport.validate(
                training, List.of(value(0, 2), value(2), value(2), value(2), value(2),
                        value(0, 2), value(2), value(2), value(2), value(2))));
    }

    @Test
    void admissionAndRawValidationRejectOverlappingPositiveStridesButAdmitTranspose() {
        Shape shape = Shape.of(2, 2);
        var output = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var operation = new Operation(SoftmaxKind.SOFTMAX, new SoftmaxAttrs(1));
        var node = node(MetalMpsGraphProgram.NodeKind.SOFTMAX, new int[] {0}, new int[] {1}, AXIS, new long[] {1});
        var overlap = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 1}, 0, true)), false);
        assertFalse(MetalNormalizationSupport.supports(operation, List.of(overlap), List.of(output)));
        assertThrows(IllegalArgumentException.class, () -> MetalNormalizationSupport.validate(
                node, List.of(MetalMpsGraphProgram.ValueDescriptor.from(overlap), MetalMpsGraphProgram.ValueDescriptor.from(output))));
        var transpose = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1, 2}, 0, true)), false);
        assertTrue(MetalNormalizationSupport.supports(operation, List.of(transpose), List.of(output)));
        MetalNormalizationSupport.validate(node, List.of(
                MetalMpsGraphProgram.ValueDescriptor.from(transpose), MetalMpsGraphProgram.ValueDescriptor.from(output)));
    }

    @Test
    void broadcastAxesDoNotExemptRemainingAxesFromStorageNonoverlap() {
        Shape shape = Shape.of(2, 2, 2);
        var output = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        var operation = new Operation(SoftmaxKind.SOFTMAX, new SoftmaxAttrs(2));
        var node = node(MetalMpsGraphProgram.NodeKind.SOFTMAX,
                new int[] {0}, new int[] {1}, AXIS, new long[] {2});
        var overlap = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {0, 1, 1}, 0, true)), false);
        assertFalse(MetalNormalizationSupport.supports(operation, List.of(overlap), List.of(output)));
        assertThrows(IllegalArgumentException.class, () -> MetalNormalizationSupport.validate(
                node, List.of(MetalMpsGraphProgram.ValueDescriptor.from(overlap),
                        MetalMpsGraphProgram.ValueDescriptor.from(output))));
        // The same broadcast axis is valid when the remaining [2,2] footprint is injective.
        var broadcast = new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {0, 2, 1}, 0, true)), false);
        assertTrue(MetalNormalizationSupport.supports(operation, List.of(broadcast), List.of(output)));
        MetalNormalizationSupport.validate(node, List.of(
                MetalMpsGraphProgram.ValueDescriptor.from(broadcast), MetalMpsGraphProgram.ValueDescriptor.from(output)));
    }

    @Test
    void rawPreflightChecksEveryBatchGradientRoleAndRejectsLowAndMalformedScalars() {
        var node = node(MetalMpsGraphProgram.NodeKind.BATCH_NORM_TRAINING,
                new int[] {0, 1, 2, 3, 4}, new int[] {5, 6, 7, 8, 9},
                MetalMpsGraphProgram.AttributeKind.BATCH_NORM_TRAINING,
                new long[] {1, 1, bits(0.5f), 1, bits(1)});
        for (int role = 0; role < 5; role++) {
            var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
            for (int i = 0; i < 5; i++) values.add(new MetalMpsGraphProgram.ValueDescriptor(
                    DataType.FLOAT32, i == 0 ? new long[] {2, 2} : new long[] {2}, i == role));
            boolean[] flags = {role <= 2, role == 0 || role == 3, role == 0 || role == 4, role == 0, role == 0};
            for (int i = 0; i < 5; i++) values.add(new MetalMpsGraphProgram.ValueDescriptor(
                    DataType.FLOAT32, i == 0 ? new long[] {2, 2} : new long[] {2}, flags[i]));
            MetalNormalizationSupport.validate(node, values);
            for (int output = 0; output < 5; output++) {
                var altered = new ArrayList<>(values);
                altered.set(5 + output, new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, output == 0 ? new long[] {2, 2} : new long[] {2}, !flags[output]));
                assertThrows(IllegalArgumentException.class, () -> MetalNormalizationSupport.validate(node, altered));
            }
        }
        var soft = node(MetalMpsGraphProgram.NodeKind.SOFTMAX, new int[] {0}, new int[] {1}, AXIS, new long[] {0});
        assertThrows(IllegalArgumentException.class, () -> MetalNormalizationSupport.validate(soft, List.of(
                new MetalMpsGraphProgram.ValueDescriptor(DataType.BFLOAT16, new long[] {2}, false),
                new MetalMpsGraphProgram.ValueDescriptor(DataType.BFLOAT16, new long[] {2}, false))));
        var invalidEpsilon = node(MetalMpsGraphProgram.NodeKind.LAYER_NORM, new int[] {0}, new int[] {1},
                NORMALIZED, normalized(2, -1));
        assertThrows(IllegalArgumentException.class, () -> MetalNormalizationSupport.validate(
                invalidEpsilon, List.of(value(2), value(2))));
        var invalidIgnore = node(MetalMpsGraphProgram.NodeKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new int[] {0, 1}, new int[] {2}, MetalMpsGraphProgram.AttributeKind.INDEX_CROSS_ENTROPY,
                new long[] {1, 1, 1, 6, -1});
        assertThrows(IllegalArgumentException.class, () -> MetalNormalizationSupport.validate(invalidIgnore,
                List.of(value(2, 2), new MetalMpsGraphProgram.ValueDescriptor(DataType.INT32, new long[] {2}, false), value(2))));
    }


    @Test
    void lateOutOfRangeIndexRejectsBeforeAnyLossOutputMutation() {
        for (DataType type : List.of(DataType.INT32, DataType.INT64)) {
            var values = List.of(value(2, 2), new MetalMpsGraphProgram.ValueDescriptor(type, new long[] {2}, false), value(2));
            var node = node(MetalMpsGraphProgram.NodeKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                    new int[] {0, 1}, new int[] {2}, MetalMpsGraphProgram.AttributeKind.INDEX_CROSS_ENTROPY,
                    new long[] {1, 1, 1, MetalMpsGraphProgram.dataTypeWire(type),
                        type == DataType.INT32 ? 0xffff_ffffL : -1L});
            execute(values, node, 2, new byte[][] {floats(1, 2, 3, 4), indices(type, 0, 2)}, true);
        }
    }

    @Test
    void softmaxUsesFiniteMaximumShiftAndLiteralNonfiniteRules() {
        var shape = new long[] {2, 3};
        var finite = floats(1000, 1001, 1002, -1002, -1001, -1000);
        var result = unary(MetalMpsGraphProgram.NodeKind.SOFTMAX, AXIS, new long[] {1}, shape, finite);
        for (int row = 0; row < 2; row++) {
            near(0.09003057f, result[row * 3]);
            near(0.24472847f, result[row * 3 + 1]);
            near(0.66524096f, result[row * 3 + 2]);
        }
        result = unary(MetalMpsGraphProgram.NodeKind.SOFTMAX, AXIS, new long[] {1}, shape,
                floats(Float.POSITIVE_INFINITY, 3, Float.NEGATIVE_INFINITY,
                        Float.NEGATIVE_INFINITY, 0, 1));
        assertTrue(Float.isNaN(result[0]));
        positiveZero(result[1]);
        positiveZero(result[2]);
        positiveZero(result[3]);
        near((float) (1 / (1 + Math.E)), result[4]);
        near((float) (Math.E / (1 + Math.E)), result[5]);
        result = unary(MetalMpsGraphProgram.NodeKind.SOFTMAX, AXIS, new long[] {1}, shape,
                floats(Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY,
                        0, Float.NaN, 1));
        for (float value : result) assertTrue(Float.isNaN(value));
        // A negative infinity opts into unshifted EXP; finite overflow remains observable.
        result = unary(MetalMpsGraphProgram.NodeKind.SOFTMAX, AXIS, new long[] {0}, new long[] {2},
                floats(Float.NEGATIVE_INFINITY, 1000));
        positiveZero(result[0]);
        assertTrue(Float.isNaN(result[1]));
    }

    @Test
    void logSoftmaxDoesNotLogRoundedProbabilitiesAndSupportsNonlastAxis() {
        float[] result = unary(MetalMpsGraphProgram.NodeKind.LOG_SOFTMAX, AXIS,
                new long[] {0}, new long[] {2, 2}, floats(1000, 0, 1001, -1000));
        near((float) -Math.log1p(Math.E), result[0]);
        near((float) -Math.log1p(1 / Math.E), result[2]);
        positiveZero(result[1]);
        near(-1000, result[3]);
        result = unary(MetalMpsGraphProgram.NodeKind.LOG_SOFTMAX, AXIS,
                new long[] {0}, new long[] {2}, floats(Float.POSITIVE_INFINITY, 1));
        assertTrue(Float.isNaN(result[0]));
        assertEquals(Float.NEGATIVE_INFINITY, result[1]);
    }

    @Test
    void layerNormPreservesConstantAndNonfiniteGuardsAndAffineSuffixMapping() {
        var kind = MetalMpsGraphProgram.NodeKind.LAYER_NORM;
        float[] result = unary(kind, NORMALIZED, normalized(2, 1), new long[] {2, 2},
                floats(Float.MAX_VALUE, Float.MAX_VALUE, -0.0f, 0));
        for (float value : result) positiveZero(value);
        result = unary(kind, NORMALIZED, normalized(2, 1), new long[] {2, 2},
                floats(Float.POSITIVE_INFINITY, 1, Float.NaN, 2));
        for (float value : result) assertTrue(Float.isNaN(value));
        var values = List.of(value(2, 2, 2), value(2, 2), value(2, 2), value(2, 2, 2));
        var node = node(kind, new int[] {0, 1, 2}, new int[] {3}, NORMALIZED,
                new long[] {2, 2, 2, 1, bits(1)});
        result = execute(values, node, 3, new byte[][] {
            floats(1, 2, 3, 4, 5, 6, 7, 8), floats(1, 2, 3, 4), floats(4, 3, 2, 1)
        }).getFirst();
        float root = (float) Math.sqrt(2.25);
        for (int row = 0; row < 2; row++) for (int f = 0; f < 4; f++) {
            near(((f - 1.5f) / root) * (f + 1) + (4 - f), result[row * 4 + f]);
        }
    }

    @Test
    void rmsNormUsesUncenteredMeanSquareAndOptionalScale() {
        var kind = MetalMpsGraphProgram.NodeKind.RMS_NORM;
        float[] result = unary(kind, NORMALIZED, normalized(2, 1), new long[] {2, 2}, floats(1, 3, 0, -0.0f));
        near(1 / (float) Math.sqrt(6), result[0]);
        near(3 / (float) Math.sqrt(6), result[1]);
        positiveZero(result[2]);
        assertEquals(0x80000000, Float.floatToRawIntBits(result[3]));
        result = execute(List.of(value(2, 2), value(2), value(2, 2)),
                node(kind, new int[] {0, 1}, new int[] {2}, NORMALIZED, normalized(2, 1)),
                2, new byte[][] {floats(1, 3, 2, 4), floats(2, -1)}).getFirst();
        near(2 / (float) Math.sqrt(6), result[0]);
        near(-3 / (float) Math.sqrt(6), result[1]);
        near(4 / (float) Math.sqrt(11), result[2]);
        near(-4 / (float) Math.sqrt(11), result[3]);
    }

    @Test
    void batchNormInferenceUsesAllFiveRolesAndNonlastChannelAxis() {
        var values = List.of(value(2, 3), value(2), value(2), value(2), value(2), value(2, 3));
        var node = node(MetalMpsGraphProgram.NodeKind.BATCH_NORM_INFERENCE,
                new int[] {0, 1, 2, 3, 4}, new int[] {5},
                MetalMpsGraphProgram.AttributeKind.BATCH_NORM_INFERENCE, new long[] {0, 1, bits(1)});
        float[] result = execute(values, node, 5, new byte[][] {
            floats(1, 3, 5, 2, 4, 6), floats(2, 3), floats(-1, 4), floats(3, 4), floats(3, 8)
        }).getFirst();
        float[] expected = {-3, -1, 1, 2, 4, 6};
        for (int i = 0; i < expected.length; i++) near(expected[i], result[i]);
    }

    @Test
    void batchNormTrainingPublishesFiveCanonicalStatisticsFromOnePopulation() {
        var values = List.of(value(2, 2), value(2), value(2), value(2), value(2),
                value(2, 2), value(2), value(2), value(2), value(2));
        var node = node(MetalMpsGraphProgram.NodeKind.BATCH_NORM_TRAINING,
                new int[] {0, 1, 2, 3, 4}, new int[] {5, 6, 7, 8, 9},
                MetalMpsGraphProgram.AttributeKind.BATCH_NORM_TRAINING,
                new long[] {1, 1, bits(0.5f), 1, bits(1)});
        var results = execute(values, node, 5, new byte[][] {
            floats(1, 3, 5, 7), floats(2, 3), floats(-1, 4), floats(10, 20), floats(2, 4)
        });
        float root = (float) Math.sqrt(5);
        near(-4 / root - 1, results.get(0)[0]);
        near(-6 / root + 4, results.get(0)[1]);
        near(4 / root - 1, results.get(0)[2]);
        near(6 / root + 4, results.get(0)[3]);
        near(6.5f, results.get(1)[0]); near(12.5f, results.get(1)[1]);
        near(5, results.get(2)[0]); near(6, results.get(2)[1]);
        near(3, results.get(3)[0]); near(5, results.get(3)[1]);
        near(1 / root, results.get(4)[0]); near(1 / root, results.get(4)[1]);
    }

    @Test
    void batchNormTrainingPreservesActualContributorZeroSignsInSavedAndPublicMeans() {
        var values = List.of(value(2, 1), value(1), value(1), value(1), value(1),
                value(2, 1), value(1), value(1), value(1), value(1));
        var node = node(MetalMpsGraphProgram.NodeKind.BATCH_NORM_TRAINING,
                new int[] {0, 1, 2, 3, 4}, new int[] {5, 6, 7, 8, 9},
                MetalMpsGraphProgram.AttributeKind.BATCH_NORM_TRAINING,
                new long[] {1, 1, bits(0.5f), 1, bits(1)});
        for (float first : new float[] {-0.0f, 0.0f}) for (float second : new float[] {-0.0f, 0.0f}) {
            var results = execute(values, node, 5, new byte[][] {
                floats(first, second), floats(1), floats(0), floats(-0.0f), floats(1)
            });
            int expectedBits = Float.floatToRawIntBits(first + second);
            assertEquals(expectedBits, Float.floatToRawIntBits(results.get(3)[0]), "saved batch mean");
            assertEquals(expectedBits, Float.floatToRawIntBits(results.get(1)[0]), "public next running mean");
        }
    }

    @Test
    void denseCrossEntropyHonorsNonunitMassZeroExclusionAndEveryReduction() {
        float first = (float) (2 * Math.log1p(Math.E));
        float second = (float) (3 * Math.log1p(1 / Math.E));
        for (int reduction = 1; reduction <= 3; reduction++) {
            float[] result = loss(false, DataType.FLOAT32, reduction, false,
                    floats(1000, 1001, -1001, -1000), floats(2, 0, 0, 3));
            if (reduction == 1) { near(first, result[0]); near(second, result[1]); }
            else near(reduction == 2 ? first + second : (first + second) / 2, result[0]);
        }
        float[] result = loss(false, DataType.FLOAT32, 1, false,
                floats(0, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY, Float.NaN), floats(1, 0, 0, 0));
        positiveZero(result[0]); positiveZero(result[1]);
    }

    @Test
    void indexCrossEntropyIgnoresBeforeBoundsOrLogitsAndCountsOnlyIncludedRows() {
        for (DataType type : List.of(DataType.INT32, DataType.INT64)) {
            for (int reduction = 1; reduction <= 3; reduction++) {
                float[] result = loss(true, type, reduction, true,
                        floats(Float.NaN, Float.POSITIVE_INFINITY, 1000, 1001), indices(type, -1, 1));
                float expected = (float) Math.log1p(1 / Math.E);
                if (reduction == 1) { positiveZero(result[0]); near(expected, result[1]); }
                else near(expected, result[0]);
            }
            for (int reduction = 1; reduction <= 3; reduction++) {
                float[] result = loss(true, type, reduction, true,
                        floats(Float.NaN, Float.NaN, Float.NaN, Float.NaN), indices(type, -1, -1));
                if (reduction == 3) assertTrue(Float.isNaN(result[0]));
                else for (float x : result) positiveZero(x);
            }
        }
    }

    private static float[] loss(boolean index, DataType targetType, int reduction, boolean ignore,
            byte[] logits, byte[] target) {
        var values = List.of(value(2, 2), new MetalMpsGraphProgram.ValueDescriptor(targetType,
                index ? new long[] {2} : new long[] {2, 2}, false),
                value(reduction == 1 ? new long[] {2} : new long[0]));
        long[] attrs = index ? (ignore ? new long[] {1, reduction, 1, MetalMpsGraphProgram.dataTypeWire(targetType),
                targetType == DataType.INT32 ? 0xffff_ffffL : -1L} : new long[] {1, reduction, 0})
                : new long[] {1, reduction};
        return execute(values, node(index ? MetalMpsGraphProgram.NodeKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS
                : MetalMpsGraphProgram.NodeKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                new int[] {0, 1}, new int[] {2}, index ? MetalMpsGraphProgram.AttributeKind.INDEX_CROSS_ENTROPY
                : MetalMpsGraphProgram.AttributeKind.DENSE_CROSS_ENTROPY, attrs),
                2, new byte[][] {logits, target}).getFirst();
    }

    private static float[] unary(MetalMpsGraphProgram.NodeKind kind, MetalMpsGraphProgram.AttributeKind attrs,
            long[] words, long[] shape, byte[] input) {
        return execute(List.of(value(shape), value(shape)), node(kind, new int[] {0}, new int[] {1}, attrs, words),
                1, new byte[][] {input}).getFirst();
    }

    private static MetalMpsGraphProgram.Node node(MetalMpsGraphProgram.NodeKind kind, int[] inputs,
            int[] outputs, MetalMpsGraphProgram.AttributeKind attrs, long[] words) {
        return MetalMpsGraphProgram.Node.generic(kind, inputs, outputs, attrs, words);
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(long... dims) {
        return new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, dims, false);
    }

    private static long[] normalized(long features, float epsilon) {
        return new long[] {1, features, 1, bits(epsilon)};
    }

    private static long bits(float value) { return Integer.toUnsignedLong(Float.floatToRawIntBits(value)); }

    private static byte[] floats(float... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.nativeOrder());
        for (float value : values) bytes.putFloat(value);
        return bytes.array();
    }

    private static byte[] indices(DataType type, long... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * type.byteWidth()).order(ByteOrder.nativeOrder());
        for (long value : values) { if (type == DataType.INT32) bytes.putInt((int) value); else bytes.putLong(value); }
        return bytes.array();
    }

    /** Executes one real native custom node with distinct owned value buffers and returns every sibling. */
    private static List<float[]> execute(List<MetalMpsGraphProgram.ValueDescriptor> values,
            MetalMpsGraphProgram.Node node, int inputCount, byte[][] inputs) {
        return execute(values, node, inputCount, inputs, false);
    }

    private static List<float[]> execute(List<MetalMpsGraphProgram.ValueDescriptor> values,
            MetalMpsGraphProgram.Node node, int inputCount, byte[][] inputs, boolean expectRejection) {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(), "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        MetalNativeApi api = MetalNativeApi.open(Path.of(configured).toAbsolutePath().normalize());
        MetalNativeApi.Handle context = null, executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            int[] feedIds = new int[inputCount];
            int[] targetIds = new int[values.size() - inputCount];
            for (int i = 0; i < feedIds.length; i++) feedIds[i] = i;
            for (int i = 0; i < targetIds.length; i++) targetIds[i] = inputCount + i;
            executable = api.createProgramExecutable(context, values, new MetalMpsGraphProgram(List.of(node)),
                    feedIds, targetIds, MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var value : values) buffers.add(api.createBuffer(context, Math.max(1, value.byteCount())));
            try (Arena arena = Arena.ofConfined()) {
                for (int i = 0; i < inputs.length; i++) {
                    MemorySegment source = arena.allocate(inputs[i].length, 1);
                    MemorySegment.copy(inputs[i], 0, source, JAVA_BYTE, 0, inputs[i].length);
                    api.upload(buffers.get(i), 0, source, inputs[i].length);
                }
                MemorySegment table = arena.allocate(ADDRESS, values.size());
                for (int i = 0; i < buffers.size(); i++) table.setAtIndex(ADDRESS, i, buffers.get(i).carrier());
                MemorySegment outputs = arena.allocate(ADDRESS, targetIds.length);
                for (int i = 0; i < targetIds.length; i++) outputs.setAtIndex(ADDRESS, i, buffers.get(targetIds[i]).carrier());
                if (expectRejection) {
                    for (int id : targetIds) {
                        MemorySegment sentinel = arena.allocate(values.get(id).byteCount(), Float.BYTES);
                        sentinel.fill((byte) 0x2a);
                        api.upload(buffers.get(id), 0, sentinel, values.get(id).byteCount());
                    }
                    MetalNativeApi.Handle prepared = executable;
                    assertThrows(MetalNativeApi.NativeFailure.class,
                            () -> api.runExecutable(prepared, values.size(), table, targetIds.length, outputs));
                } else api.runExecutable(executable, values.size(), table, targetIds.length, outputs);
                var result = new ArrayList<float[]>();
                for (int id : targetIds) {
                    int count = Math.toIntExact(values.get(id).elementCount());
                    MemorySegment target = arena.allocate(Math.max(1, values.get(id).byteCount()), Float.BYTES);
                    api.download(buffers.get(id), 0, target, values.get(id).byteCount());
                    ByteBuffer bytes = ByteBuffer.wrap(target.toArray(JAVA_BYTE)).order(ByteOrder.nativeOrder());
                    float[] decoded = new float[count];
                    for (int j = 0; j < count; j++) decoded[j] = bytes.getFloat();
                    if (expectRejection) for (float value : decoded)
                        assertEquals(0x2a2a2a2a, Float.floatToRawIntBits(value), "preflight must precede every loss write");
                    result.add(decoded);
                }
                return result;
            }
        } finally {
            for (int i = buffers.size(); i-- > 0;) api.releaseBuffer(buffers.get(i));
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static void near(float expected, float actual) {
        assertEquals(expected, actual, 3e-4f * Math.max(1, Math.abs(expected)));
    }

    private static void positiveZero(float actual) { assertEquals(0, Float.floatToRawIntBits(actual)); }
}
