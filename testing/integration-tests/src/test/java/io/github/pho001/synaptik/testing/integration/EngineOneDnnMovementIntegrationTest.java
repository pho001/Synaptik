package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.OneDnnApi;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Public movement regressions for an explicitly configured oneDNN CPU composition. Every logical
 * coordinate and represented F32 word is checked, including moves that must retain the ordinary CPU
 * plan when the selected native implementation cannot preserve bits. No assertion forces a route.
 */
public final class EngineOneDnnMovementIntegrationTest {
    private static final int[] RAW_WORDS = {
        0x80000000, 0x00000000, 0x00000001, 0x80000001, 0x007fffff, 0x807fffff,
        0x7f800000, 0xff800000, 0x7fc12345, 0xffc54321, 0x7f812345, 0xff854321,
        0x3f800000, 0xbf800000, 0x7f7fffff, 0xff7fffff
    };

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void squarePermutationUsesComposedCoordinatesAtEveryLane(OneDnnApi api) {
        int side = 256;
        int[] words = new int[side * side];
        for (int row = 0; row < side; row++) {
            for (int column = 0; column < side; column++) {
                words[row * side + column] = Float.floatToRawIntBits(row * 1000f + column);
            }
        }
        Tensor input = tensor(Shape.of(side, side), new long[] {side, 1}, 0, words);
        try (Engine engine = configured(api)) {
            assertTranspose(engine, input, words, side, side, side, 1, 0);
        }
    }

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void squarePermutationAndTransposeReshapePreserveEveryRawWord(OneDnnApi api) {
        int side = 256;
        int[] words = rawWords(side * side);
        Tensor input = tensor(Shape.of(side, side), new long[] {side, 1}, 0, words);
        try (Engine engine = configured(api)) {
            assertTranspose(engine, input, words, side, side, side, 1, 0);
            // Logical reshape follows transposed order, not the original boundary's dense order.
            Tensor reshaped = input.permute(1, 0).reshape(512, 128).contiguous();
            ByteBuffer output = forward(engine, reshaped, input);
            for (int lane = 0; lane < words.length; lane++) {
                int source = (lane % side) * side + lane / side;
                assertEquals(words[source], output.getInt(), "transpose/reshape raw lane " + lane);
            }
            assertEquals(0, output.remaining());
        }
    }

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void nonsquareOffsetAndGappedPermutationsKeepOriginsAndBorrowedStorage(OneDnnApi api) {
        int rows = 257, columns = 255, origin = 7;
        try (Engine engine = configured(api)) {
            for (boolean gapped : new boolean[] {false, true}) {
                int columnStride = gapped ? 2 : 1;
                int rowStride = gapped ? columns * 2 + 3 : columns;
                int[] physical = new int[origin + rows * rowStride + 9];
                Arrays.fill(physical, 0x7fa76543);
                Tensor input = tensor(Shape.of(rows, columns),
                        new long[] {rowStride, columnStride}, origin, physical);
                Tensor output = input.permute(1, 0).contiguous();
                try (var prepared = engine.prepare(engine.compile(List.of(output)))) {
                    for (int run = 0; run < 2; run++) {
                        for (int row = 0; row < rows; row++) {
                            for (int column = 0; column < columns; column++) {
                                physical[origin + row * rowStride + column * columnStride]
                                        = RAW_WORDS[(row * columns + column + run) % RAW_WORDS.length];
                            }
                        }
                        int[] before = physical.clone();
                        try (var result = engine.run(prepared, List.of(input))) {
                            ByteBuffer bytes = result.materialize(
                                    result.publications().getFirst(), (long) rows * columns * Float.BYTES).bytes();
                            assertTransposeBytes(bytes, physical, rows, columns,
                                    rowStride, columnStride, origin);
                        }
                        assertArrayEquals(before, physical, "borrowed input, prefix and holes remain unchanged");
                    }
                }
            }
        }
    }

    @ParameterizedTest
    @EnumSource(OneDnnApi.class)
    void highRankReferenceReorderCandidateRetainsRawCpuMovement(OneDnnApi api) {
        long[] dimensions = {2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4};
        long[] strides = {1, 2, 4, 8, 16, 32, 64, 128, 256, 1024, 4096, 16384};
        int[] words = rawWords(65536);
        Tensor input = tensor(Shape.of(dimensions), strides, 0, words);
        try (Engine engine = configured(api)) {
            ByteBuffer output = forward(engine, input.contiguous(), input);
            for (int lane = 0; lane < words.length; lane++) {
                int remaining = lane;
                long source = 0;
                for (int axis = dimensions.length - 1; axis >= 0; axis--) {
                    source += remaining % dimensions[axis] * strides[axis];
                    remaining /= (int) dimensions[axis];
                }
                assertEquals(words[Math.toIntExact(source)], output.getInt(), "high-rank raw lane " + lane);
            }
            assertEquals(0, output.remaining());
        }
    }

    /**
     * Runs and checks the public permutation without interpreting boundary layout as the mapping.
     * @param engine borrowed open configured CPU engine
     * @param input caller-owned input tensor
     * @param words caller-owned physical F32 words
     * @param rows source logical rows
     * @param columns source logical columns
     * @param rowStride source row stride in elements
     * @param columnStride source column stride in elements
     * @param origin source element origin
     */
    private static void assertTranspose(Engine engine, Tensor input, int[] words,
            int rows, int columns, int rowStride, int columnStride, int origin) {
        assertTransposeBytes(forward(engine, input.permute(1, 0), input),
                words, rows, columns, rowStride, columnStride, origin);
    }

    /**
     * Checks all canonical output coordinates against the independent transpose equation.
     * @param bytes canonical public FLOAT32 output, consumed completely
     * @param words physical source words
     * @param rows source logical rows
     * @param columns source logical columns
     * @param rowStride source row stride in elements
     * @param columnStride source column stride in elements
     * @param origin source element origin
     */
    private static void assertTransposeBytes(ByteBuffer bytes, int[] words,
            int rows, int columns, int rowStride, int columnStride, int origin) {
        for (int row = 0; row < columns; row++) {
            for (int column = 0; column < rows; column++) {
                assertEquals(words[origin + column * rowStride + row * columnStride], bytes.getInt(),
                        "transpose coordinate (" + row + "," + column + ")");
            }
        }
        assertEquals(0, bytes.remaining());
    }

    /**
     * Materializes one public result while all execution owners are open.
     * @param engine borrowed open configured CPU engine
     * @param output requested expression
     * @param input sole caller-owned input
     * @return independent canonical publication bytes after preparation and run close
     */
    private static ByteBuffer forward(Engine engine, Tensor output, Tensor input) {
        try (var prepared = engine.prepare(engine.compile(List.of(output)));
                var result = engine.run(prepared, List.of(input))) {
            return result.materialize(result.publications().getFirst(), Long.MAX_VALUE).bytes();
        }
    }

    /**
     * Opens the optional native composition; only an absent library setting skips the tests.
     * @param api public oneDNN API preference
     * @return newly owned configured CPU engine
     */
    private static Engine configured(OneDnnApi api) {
        String library = System.getenv("SYNAPTIK_ONEDNN_LIBRARY");
        assumeTrue(library != null && !library.isBlank(), "SYNAPTIK_ONEDNN_LIBRARY is not configured");
        try (Engine.Builder builder = Engine.builder()) {
            return builder.takeOwnership(CpuBackendIntegration.openOneDnn(Path.of(library), api)).build();
        }
    }

    /**
     * Borrows integer-backed bytes as FLOAT32 without a conversion that might quiet signaling NaNs.
     * @param shape positive static logical shape
     * @param strides nonnegative physical element strides
     * @param origin physical element origin
     * @param words mutable caller-owned physical storage including prefix and holes
     * @return tensor borrowing the exact represented source bytes
     */
    private static Tensor tensor(Shape shape, long[] strides, long origin, int[] words) {
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.of(shape, strides, origin, true)), false), Optional.empty(),
                Optional.of(new MemorySegmentStorage(DataType.FLOAT32, words.length, MemorySegment.ofArray(words))));
    }

    /**
     * Repeats represented special values without performing floating arithmetic.
     * @param count number of physical F32 words
     * @return fresh mutable represented-bit fixture
     */
    private static int[] rawWords(int count) {
        int[] words = new int[count];
        for (int lane = 0; lane < count; lane++) words[lane] = RAW_WORDS[lane % RAW_WORDS.length];
        return words;
    }
}
