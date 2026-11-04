package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.OneDnnApi;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Public preparation and execution regressions for borrowed F32 zero-stride inputs. API preference
 * never forces a native route: valid common broadcasts must prepare and publish the complete
 * logical domain even when their singleton-normalized native operands cannot describe it.
 */
final class EngineOneDnnBroadcastIntegrationTest {
    private static final int COUNT = 65535;

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void scalarArithmeticPreparesAndRefreshesOneBorrowedCell(OneDnnApi api) {
        float[] physical = {2};
        Tensor input = view(Shape.of(COUNT), new long[] {0}, physical);
        List<Tensor> outputs = List.of(input.add(ScalarValue.float32(1)),
                input.sub(ScalarValue.float32(1)), input.mul(ScalarValue.float32(2)),
                input.div(ScalarValue.float32(2)));
        try (Engine engine = configured(api)) {
            // Prepare each public arithmetic site independently, without depending on fusion.
            for (int operation = 0; operation < outputs.size(); operation++) {
                try (var prepared = engine.prepare(engine.compile(List.of(outputs.get(operation))))) {
                    for (float value : new float[] {2, -4}) {
                        physical[0] = value;
                        float expected = switch (operation) {
                            case 0 -> value + 1;
                            case 1 -> value - 1;
                            case 2 -> value * 2;
                            default -> value / 2;
                        };
                        try (RunResult result = engine.run(prepared, List.of(input))) {
                            assertConstant(result, 0, COUNT, expected);
                        }
                        assertEquals(value, physical[0], "borrowed cell remains read-only");
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void bothBroadcastOperandsAndRepeatedInputsSurviveComposedPrograms(OneDnnApi api) {
        Shape shape = Shape.of(255, 257);
        float[] a = {2}, b = {3};
        Tensor left = view(shape, new long[] {0, 0}, a);
        Tensor right = view(shape, new long[] {0, 0}, b);
        Tensor added = left.add(right);
        Tensor repeated = left.add(left);
        List<Tensor> outputs = List.of(added, repeated,
                added.mul(left).sub(ScalarValue.float32(1)),
                left.add(ScalarValue.float32(1)).mul(right));
        try (Engine engine = configured(api);
                var prepared = engine.prepare(engine.compile(outputs))) {
            for (int pass = 0; pass < 2; pass++) {
                a[0] = pass == 0 ? 2 : -4;
                b[0] = pass == 0 ? 3 : 2;
                try (RunResult result = engine.run(prepared, List.of(right, left))) {
                    assertConstant(result, 0, COUNT, a[0] + b[0]);
                    assertConstant(result, 1, COUNT, a[0] + a[0]);
                    assertConstant(result, 2, COUNT, (a[0] + b[0]) * a[0] - 1);
                    assertConstant(result, 3, COUNT, (a[0] + 1) * b[0]);
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void partiallyBroadcastRowsRetainEveryCoordinateAndPredicate(OneDnnApi api) {
        Shape shape = Shape.of(255, 257);
        float[] a = new float[257], b = new float[257];
        for (int column = 0; column < 257; column++) {
            a[column] = column - 128;
            b[column] = 64 - column;
        }
        Tensor left = view(shape, new long[] {0, 1}, a);
        Tensor right = view(shape, new long[] {0, 1}, b);
        try (Engine engine = configured(api);
                var prepared = engine.prepare(engine.compile(List.of(left.sub(right),
                        left.greaterThan(right), left.lessOrEqual(right))));
                RunResult result = engine.run(prepared, List.of(left, right))) {
            var difference = result.materialize(result.publications().get(0), (long) COUNT * Float.BYTES).bytes();
            var greater = result.materialize(result.publications().get(1), COUNT).bytes();
            var lessOrEqual = result.materialize(result.publications().get(2), COUNT).bytes();
            for (int index = 0; index < COUNT; index++) {
                int column = index % 257;
                assertEquals(a[column] - b[column], difference.getFloat());
                assertEquals((byte) (a[column] > b[column] ? 1 : 0), greater.get());
                assertEquals((byte) (a[column] <= b[column] ? 1 : 0), lessOrEqual.get());
            }
        }
    }

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void unaryAndIdentityViewsKeepTheirOriginalReferencedSpan(OneDnnApi api) {
        Tensor input = view(Shape.of(COUNT), new long[] {0}, new float[] {-4});
        List<Tensor> outputs = List.of(input.abs(), input.abs().sqrt(),
                input.sum(new int[0]), input.mean(new int[0]), input.cast(DataType.FLOAT32));
        try (Engine engine = configured(api);
                var prepared = engine.prepare(engine.compile(outputs));
                RunResult result = engine.run(prepared, List.of(input))) {
            assertConstant(result, 0, COUNT, 4);
            assertConstant(result, 1, COUNT, 2);
            for (int port = 2; port < outputs.size(); port++) assertConstant(result, port, COUNT, -4);
        }
    }

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void mseHandlesBothBroadcastRolesAndRepeatedPredictionTargets(OneDnnApi api) {
        Tensor prediction = view(Shape.of(COUNT), new long[] {0}, new float[] {3});
        Tensor target = view(Shape.of(COUNT), new long[] {0}, new float[] {1});
        List<Tensor> outputs = List.of(
                prediction.meanSquaredError(target, LossReduction.NONE),
                prediction.meanSquaredError(target, LossReduction.SUM),
                prediction.meanSquaredError(target, LossReduction.MEAN),
                prediction.meanSquaredError(prediction, LossReduction.NONE),
                prediction.meanSquaredError(target, LossReduction.NONE).add(ScalarValue.float32(1)));
        try (Engine engine = configured(api)) {
            for (int port = 0; port < outputs.size(); port++) {
                var graph = engine.compile(List.of(outputs.get(port)));
                List<Tensor> inputs = graph.inputs().stream().map(metadata ->
                        metadata.tensorId().equals(prediction.id()) ? prediction : target).toList();
                try (var prepared = engine.prepare(graph);
                        RunResult result = engine.run(prepared, inputs)) {
                    int count = port == 1 || port == 2 ? 1 : COUNT;
                    float expected = switch (port) {
                        case 1 -> 4 * COUNT;
                        case 3 -> 0;
                        case 4 -> 5;
                        default -> 4;
                    };
                    assertConstant(result, 0, count, expected);
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void batchNormHandlesBroadcastChannelStatisticsAndRepeatedRoles(OneDnnApi api) {
        Shape shape = Shape.of(257, 255);
        float[] dense = new float[COUNT];
        Arrays.fill(dense, 6);
        Tensor denseInput = view(shape, new long[] {255, 1}, dense);
        Tensor broadcastInput = view(shape, new long[] {0, 0}, new float[] {6});
        Tensor scaleAndMean = view(Shape.of(255), new long[] {0}, new float[] {2});
        Tensor biasAndVariance = view(Shape.of(255), new long[] {0}, new float[] {3});
        try (Engine engine = configured(api)) {
            for (Tensor input : List.of(denseInput, broadcastInput)) {
                Tensor output = input.batchNormInference(1, scaleAndMean, biasAndVariance,
                        scaleAndMean, biasAndVariance, ScalarValue.float32(1));
                try (var prepared = engine.prepare(engine.compile(List.of(output)));
                        RunResult result = engine.run(prepared,
                                List.of(biasAndVariance, input, scaleAndMean))) {
                    assertConstant(result, 0, COUNT, 7);
                }
            }
        }
    }

    /**
     * Opens the configured CPU backend without altering process-wide configuration.
     *
     * @param api non-null API preference; all preferences retain ordinary prepared selection
     * @return newly owned engine; an absent or blank library setting aborts the test
     */
    private static Engine configured(OneDnnApi api) {
        String path = System.getenv("SYNAPTIK_ONEDNN_LIBRARY");
        assumeTrue(path != null && !path.isBlank(), "SYNAPTIK_ONEDNN_LIBRARY is not configured");
        try (Engine.Builder builder = Engine.builder()) {
            return builder.takeOwnership(CpuBackendIntegration.openOneDnn(Path.of(path), api)).build();
        }
    }

    /**
     * Borrows physical heap storage, including layouts whose logical count exceeds storage size.
     *
     * @param shape non-null static logical shape
     * @param strides non-null element strides matching the shape rank, including zero strides
     * @param physical non-null mutable backing cells retained by the caller
     * @return non-gradient F32 tensor with a zero-origin public layout
     */
    private static Tensor view(Shape shape, long[] strides, float[] physical) {
        var layout = LayoutDescriptor.of(shape, strides, 0, true);
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(layout), false), Optional.empty(), Optional.of(new MemorySegmentStorage(
                        DataType.FLOAT32, physical.length, MemorySegment.ofArray(physical))));
    }

    /**
     * Checks the entire logical publication, rather than the one-cell borrowed physical span.
     *
     * @param result non-null live run owning the publication
     * @param port publication position
     * @param count exact positive logical element count
     * @param expected exact represented finite result for every element
     */
    private static void assertConstant(RunResult result, int port, int count, float expected) {
        var value = result.materialize(result.publications().get(port), (long) count * Float.BYTES);
        assertEquals(DataType.FLOAT32, value.dataType());
        var bytes = value.bytes();
        assertEquals(count * Float.BYTES, bytes.remaining());
        for (int index = 0; index < count; index++) {
            assertTrue(bytes.hasRemaining());
            assertEquals(expected, bytes.getFloat(), "logical lane " + index);
        }
    }
}
