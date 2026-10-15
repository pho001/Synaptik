package io.github.pho001.synaptik.checkpoint.model;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.checkpoint.artifact.ArtifactFormatException;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.nn.module.ModelSpecification;
import io.github.pho001.synaptik.nn.module.ModelSpecification.*;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HexFormat;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Fixed-wire and adversarial metadata evidence, independent of Engine/backend admission. */
class WholeModelDefinitionCodecTest {
    static final List<DataType> FLOATS = List.of(DataType.FLOAT64, DataType.FLOAT32,
            DataType.BFLOAT16, DataType.FLOAT16);

    @Test
    void emptyWireIsExactlyFrozenTwentyOneBytes() throws Exception {
        byte[] encoded = encode(List.of());
        assertEquals("53594e444546303100000001010000000000000000", HexFormat.of().formatHex(encoded));
        assertEquals(21, encoded.length);
        assertTrue(decode(encoded).root().steps().isEmpty());
    }

    @Test
    void everyFamilyAllAttributesGuardPoliciesAndExactEpsilonBitsRoundTrip() throws Exception {
        for (DataType type : FLOATS) {
            List<Step> steps = List.of(new LinearStep(3, 2, true, type, false),
                    new Sequence(List.of(new LinearStep(3, 2, false, type, true))),
                    new LayerNormStep(Shape.of(2, 3), type, epsilon(type)),
                    new EmbeddingStep(7, 3, type),
                    new Conv1dStep(4, 6, 3, true, type, new Conv1dAttrs(2, 3, 2, 2)),
                    new Conv2dStep(4, 6, 2, 3, false, type, new Conv2dAttrs(2, 3, 1, 2, 2, 1, 2)),
                    new Conv3dStep(4, 6, 2, 3, 2, true, type,
                            new Conv3dAttrs(2, 1, 3, 1, 2, 3, 2, 1, 2, 2)),
                    new ActivationStep(ActivationKind.RELU), new ActivationStep(ActivationKind.TANH),
                    new ActivationStep(ActivationKind.SIGMOID));
            ModelSpecification spec = spec(steps);
            byte[] bytes = WholeModelDefinitionCodec.encode(spec);
            assertEquals(bytes.length, WholeModelDefinitionCodec.size(spec));
            var decoded = decode(bytes);
            assertEquals(spec.root(), decoded.root());
            assertEquals(spec.stateDescriptors(), decoded.stateDescriptors());
            assertArrayEquals(bytes, WholeModelDefinitionCodec.encode(decoded));
        }
    }

    @Test
    void wireUsesStableDataTypeIdsAndRawScalarWidths() throws Exception {
        int[] ids = {1, 2, 3, 7};
        for (int i = 0; i < FLOATS.size(); i++) {
            byte[] linear = encode(List.of(new LinearStep(1, 1, true, FLOATS.get(i), true)));
            assertEquals(2, linear[17]);
            assertEquals(ids[i], linear[35]);
            assertEquals(1, linear[34]);
            assertEquals(1, linear[36]);
            var epsilon = epsilon(FLOATS.get(i));
            byte[] norm = encode(List.of(new LayerNormStep(Shape.of(1), FLOATS.get(i), epsilon)));
            int scalar = 31;
            assertEquals(FLOATS.get(i) == DataType.FLOAT16 ? 2 : ids[i], norm[scalar]);
            var buffer = ByteBuffer.wrap(norm); buffer.position(scalar + 1);
            switch (epsilon.dataType()) {
                case FLOAT64 -> assertEquals(Double.doubleToRawLongBits(epsilon.float64Value()), buffer.getLong());
                case FLOAT32 -> assertEquals(Float.floatToRawIntBits(epsilon.float32Value()), buffer.getInt());
                case BFLOAT16 -> assertEquals(epsilon.bfloat16Bits(), buffer.getShort());
                default -> fail("Unexpected epsilon type");
            }
        }
    }

    @Test
    void aliasesRetainNumericEncounterOrderNotLexicalOrder() throws Exception {
        List<Step> nodes = Collections.nCopies(12, new LinearStep(2, 2, true, DataType.FLOAT32, false));
        var spec = new ModelSpecification(new Sequence(nodes), List.of(
                new AliasGroup(List.of("0.weight", "2.weight", "10.weight")),
                new AliasGroup(List.of("0.bias", "11.bias"))));
        var decoded = decode(WholeModelDefinitionCodec.encode(spec));
        assertEquals(spec.aliases(), decoded.aliases());
        assertEquals("10.weight", decoded.stateDescriptors().get(20).path());
    }

    @Test
    void rejectsEveryTruncatedPrefixAndTrailingBytes() {
        byte[] bytes = encode(List.of(new LinearStep(3, 2, true, DataType.FLOAT32, true)));
        for (int i = 0; i < bytes.length; i++) {
            byte[] truncated = Arrays.copyOf(bytes, i);
            assertThrows(ArtifactFormatException.class, () -> decode(truncated), "prefix " + i);
        }
        assertThrows(ArtifactFormatException.class, () -> decode(Arrays.copyOf(bytes, bytes.length + 1)));
    }

    @Test
    void rejectsUnknownVersionsKindsTypesAndNoncanonicalFlags() {
        byte[] bytes = encode(List.of(new LinearStep(1, 1, true, DataType.FLOAT32, true)));
        for (int[] mutation : List.of(new int[] {0, 0}, new int[] {11, 2}, new int[] {12, 2},
                new int[] {17, 9}, new int[] {34, 2}, new int[] {35, 0}, new int[] {35, 4},
                new int[] {35, 5}, new int[] {35, 6}, new int[] {35, 8}, new int[] {36, 2})) {
            byte[] bad = bytes.clone(); bad[mutation[0]] = (byte) mutation[1];
            assertThrows(ArtifactFormatException.class, () -> decode(bad), Arrays.toString(mutation));
        }
        byte[] activation = encode(List.of(new ActivationStep(ActivationKind.RELU)));
        activation[18] = 4;
        assertThrows(ArtifactFormatException.class, () -> decode(activation));
    }

    @Test
    void rejectsNegativeHugeChildRankAndAliasCountsBeforeAllocation() {
        byte[] empty = encode(List.of());
        for (int count : new int[] {-1, 4096, Integer.MAX_VALUE}) {
            byte[] children = empty.clone(); ByteBuffer.wrap(children).putInt(13, count);
            assertThrows(ArtifactFormatException.class, () -> decode(children));
            byte[] groups = empty.clone(); ByteBuffer.wrap(groups).putInt(17, count);
            assertThrows(ArtifactFormatException.class, () -> decode(groups));
        }
        byte[] norm = encode(List.of(new LayerNormStep(Shape.of(1), DataType.FLOAT64, epsilon(DataType.FLOAT64))));
        for (int rank : new int[] {-1, 0, 2, Integer.MAX_VALUE}) {
            byte[] bad = norm.clone(); ByteBuffer.wrap(bad).putInt(19, rank);
            assertThrows(ArtifactFormatException.class, () -> decode(bad));
        }
        // Enough extent bytes, but missing the complete FLOAT64 scalar suffix.
        assertThrows(ArtifactFormatException.class, () -> decode(Arrays.copyOf(norm, 36)));
    }

    @Test
    void semanticRankAboveSixteenAndExactTreeLimitsAreNotBackendLimits() throws Exception {
        long[] extents = new long[32]; Arrays.fill(extents, 1);
        byte[] norm = encode(List.of(new LayerNormStep(Shape.of(extents), DataType.FLOAT32, epsilon(DataType.FLOAT32))));
        assertEquals(32, ((LayerNormStep) decode(norm).root().steps().getFirst()).normalizedShape().rank());
        Sequence deep = new Sequence(List.of());
        for (int i = 1; i < 64; i++) deep = new Sequence(List.of(deep));
        byte[] boundary = WholeModelDefinitionCodec.encode(new ModelSpecification(deep, List.of()));
        assertDoesNotThrow(() -> decode(boundary));
        ByteBuffer tooDeep = ByteBuffer.allocate(boundary.length + 5);
        tooDeep.put(boundary, 0, 12).put((byte) 1).putInt(1).put(boundary, 12, boundary.length - 12);
        assertThrows(ArtifactFormatException.class, () -> decode(tooDeep.array()));
        assertEquals(4095, decode(encode(Collections.nCopies(4095,
                new ActivationStep(ActivationKind.RELU)))).root().steps().size());
    }

    @Test
    void definitionCeilingAppliesToDrySizeAndRawDecode() {
        long[] extents = new long[131072]; Arrays.fill(extents, 1);
        var spec = spec(List.of(new LayerNormStep(Shape.of(extents), DataType.FLOAT32, epsilon(DataType.FLOAT32))));
        assertThrows(IllegalArgumentException.class, () -> WholeModelDefinitionCodec.size(spec));
        assertThrows(IllegalArgumentException.class, () -> WholeModelDefinitionCodec.encode(spec));
        assertThrows(ArtifactFormatException.class, () -> decode(new byte[1048577]));
    }

    @Test
    void aliasesRejectMalformedUtf8LengthsUnknownDuplicateAndReorderedPaths() throws Exception {
        var steps = Collections.<Step>nCopies(2, new LinearStep(1, 1, false, DataType.FLOAT32, false));
        var spec = new ModelSpecification(new Sequence(steps), List.of(new AliasGroup(List.of("0.weight", "1.weight"))));
        byte[] valid = WholeModelDefinitionCodec.encode(spec);
        int groupCount = 17 + 20 * 2;
        for (int count : new int[] {-1, 0, 1, Integer.MAX_VALUE}) {
            byte[] bad = valid.clone(); ByteBuffer.wrap(bad).putInt(groupCount + 4, count);
            assertThrows(ArtifactFormatException.class, () -> decode(bad));
        }
        int firstLength = groupCount + 8;
        for (int length : new int[] {-1, 0, Integer.MAX_VALUE}) {
            byte[] bad = valid.clone(); ByteBuffer.wrap(bad).putInt(firstLength, length);
            assertThrows(ArtifactFormatException.class, () -> decode(bad));
        }
        for (int first : new int[] {0xc0, '1', '9'}) {
            byte[] bad = valid.clone(); bad[firstLength + 4] = (byte) first;
            assertThrows(ArtifactFormatException.class, () -> decode(bad));
        }
        byte[] duplicate = valid.clone(); duplicate[firstLength + 16] = '0';
        assertThrows(ArtifactFormatException.class, () -> decode(duplicate));
        assertEquals(spec.aliases(), decode(valid).aliases());
    }

    @Test
    void ownerSemanticFailuresCarryTheirCause() {
        byte[] linear = encode(List.of(new LinearStep(1, 1, false, DataType.FLOAT32, false)));
        ByteBuffer.wrap(linear).putLong(18, 0);
        assertNotNull(assertThrows(ArtifactFormatException.class, () -> decode(linear)).getCause());
        byte[] overflow = encode(List.of(new LinearStep(1, 1, false, DataType.FLOAT64, false)));
        ByteBuffer.wrap(overflow).putLong(18, Long.MAX_VALUE);
        assertInstanceOf(ArithmeticException.class,
                assertThrows(ArtifactFormatException.class, () -> decode(overflow)).getCause());
    }

    @Test
    void scalarWireRejectsEveryDisallowedEpsilonTypeAndNonFiniteOrNonpositiveBits() {
        for (int id : new int[] {0, 4, 5, 6, 7, 8}) {
            ByteBuffer bytes = ByteBuffer.allocate(44);
            bytes.putLong(0x53594e4445463031L).putInt(1).put((byte) 1).putInt(1)
                    .put((byte) 3).put((byte) 2).putInt(1).putLong(1).put((byte) id);
            assertThrows(ArtifactFormatException.class, () -> decode(bytes.array()));
        }
        for (int bits : new int[] {0, 0x80000000, 0xbf800000, 0x7f800000, 0x7fc01234}) {
            byte[] bytes = encode(List.of(new LayerNormStep(Shape.of(1), DataType.FLOAT32, epsilon(DataType.FLOAT32))));
            ByteBuffer.wrap(bytes).putInt(32, bits);
            assertNotNull(assertThrows(ArtifactFormatException.class, () -> decode(bytes)).getCause());
        }
    }

    static ScalarValue epsilon(DataType type) {
        return switch (type) {
            case FLOAT64 -> ScalarValue.float64(Double.longBitsToDouble(0x3ee4f8b588e368f1L));
            case BFLOAT16 -> ScalarValue.bfloat16Bits((short) 0x3729);
            default -> ScalarValue.float32(Float.intBitsToFloat(0x3727c5ad));
        };
    }
    static ModelSpecification spec(List<Step> steps) { return new ModelSpecification(new Sequence(steps), List.of()); }
    static byte[] encode(List<Step> steps) { return WholeModelDefinitionCodec.encode(spec(steps)); }
    static ModelSpecification decode(byte[] bytes) throws ArtifactFormatException {
        return WholeModelDefinitionCodec.decode(ByteBuffer.wrap(bytes));
    }
}
