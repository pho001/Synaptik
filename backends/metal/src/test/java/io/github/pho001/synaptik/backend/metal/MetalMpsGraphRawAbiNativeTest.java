package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalMpsGraphRawAbiNativeTest {
    @Test
    void nativeAbiSevenAcceptsSchemaTwentyAndRejectsStaleNineteenBeforeCreation()
            throws Throwable {
        Path library = configuredLibrary();
        try (RawAbi abi = new RawAbi(library)) {
            byte[] valid = validNegImage();
            assertEquals(0, abi.create(valid, valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 0, 0), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 0, 0x38314d53), valid.length));
            for (int schema = 12; schema <= 19; schema++)
                assertEquals(1, abi.create(rewriteInt(valid, 4, schema), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 4, 21), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 8, 0), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 8, 64), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 8, 132), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 12, valid.length - 1), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 16, 0), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 16, 1), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 16, 3), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 16, 4), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 20, 1), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 20, 0x554e4b4e), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 20, 2), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 24, 1), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 60, 1), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 84, 32), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 104, 0), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 108, 1), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 112, 16_384), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, nodeOffset(2), 116), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, nodeOffset(2) + 4, 3), valid.length));
            assertEquals(1, abi.create(valid, valid.length - 1));
            assertEquals(1, abi.create(valid, Integer.MIN_VALUE));
            assertEquals(1, abi.create(
                    rewriteInt(valid, 8, Integer.MIN_VALUE), Integer.MIN_VALUE));
        }
    }

    @Test
    void schemaTwentyRejectsCorruptFusionRecordsManifestAndDigest() throws Throwable {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.FLOOR,
                        new int[] {0},
                        new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.NONE,
                        new long[0]),
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.CEIL,
                        new int[] {1},
                        new int[] {2},
                        MetalMpsGraphProgram.AttributeKind.NONE,
                        new long[0])));
        byte[] valid = program.encodedProgramImage(
                List.of(descriptor(4), descriptor(4), descriptor(4)),
                new int[] {0},
                new int[] {2},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        int coreBytes = MetalMpsGraphProgram.HEADER_BYTES
                + readInt(valid, 28) * MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES
                + readInt(valid, 32) * MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES
                + readInt(valid, 44) * Long.BYTES
                + readInt(valid, 48) * Long.BYTES
                + readInt(valid, 52) * Integer.BYTES
                + readInt(valid, 56) * Long.BYTES;
        int memberOffset = coreBytes
                + readInt(valid, 60) * MetalMpsGraphProgram.STEP_DESCRIPTOR_BYTES;
        int bindingOffset = memberOffset + readInt(valid, 64) * Integer.BYTES;
        int materializedOffset = bindingOffset
                + readInt(valid, 68) * MetalMpsGraphProgram.BINDING_DESCRIPTOR_BYTES;
        int instructionOffset = materializedOffset + readInt(valid, 72) * Integer.BYTES;
        int manifestOffset = instructionOffset
                + readInt(valid, 76) * MetalMpsGraphProgram.INSTRUCTION_DESCRIPTOR_BYTES;
        int digestOffset = manifestOffset + readInt(valid, 80);

        var malformed = new ArrayList<byte[]>();
        malformed.add(appendOutOfRangeMemberStep(valid));
        malformed.add(rewriteInt(valid, coreBytes, 2));
        malformed.add(rewriteInt(valid, instructionOffset + 8, 0));
        malformed.add(rewriteInt(valid, instructionOffset + 8, 5));
        malformed.add(rewriteInt(valid, instructionOffset + 12, 2));
        malformed.add(rewriteInt(valid, instructionOffset + 16, 2));
        malformed.add(rewriteInt(valid, instructionOffset + 24, 0));
        malformed.add(rewriteInt(valid, instructionOffset + 36, 1));
        malformed.add(rewriteLong(valid, instructionOffset + 48, 1L));
        byte[] manifestForgery = valid.clone();
        manifestForgery[manifestOffset] ^= 1;
        malformed.add(manifestForgery);
        byte[] digestForgery = valid.clone();
        digestForgery[digestOffset] ^= 1;
        malformed.add(digestForgery);
        malformed.add(rewriteManifest(
                valid,
                "fixed-corpus-bytes 84603\n",
                "fixed-corpus-bytes 84604\n"));

        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(valid, valid.length));
            assertEquals(
                    1,
                    abi.create(rewriteInt(valid, 20, 1), valid.length),
                    "predecessor generator schema must remain rejected");
            int secondNodeOffset = nodeOffset(3) + MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES;
            byte[] generatedNegMember = rewriteInt(
                    valid,
                    secondNodeOffset,
                    MetalMpsGraphProgram.NodeKind.NEG.wireIdentity());
            byte[] generatedAbsMember = rewriteInt(
                    valid,
                    secondNodeOffset,
                    MetalMpsGraphProgram.NodeKind.ABS.wireIdentity());
            assertEquals(
                    1,
                    abi.create(generatedNegMember, generatedNegMember.length),
                    "valid NEG node cannot inhabit a generated member range");
            assertEquals(
                    1,
                    abi.create(generatedAbsMember, generatedAbsMember.length),
                    "valid ABS node cannot inhabit a generated member range");
            for (byte[] image : malformed)
                assertEquals(1, abi.create(image, image.length));
        }
    }

    @Test
    void schemaTwentyKeepsArithmeticAndRawFloat16CustomOnly() throws Throwable {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.neg(0, 1)));
        var float16 = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT16, new long[] {3}, false);
        byte[] custom = program.encodedProgramImage(
                List.of(float16, float16),
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        byte[] mpsGraph = program.encodedProgramImage(
                List.of(float16, float16),
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.MPSGRAPH);
        var movementProgram = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.contiguous(0, 1)));
        byte[] movementCustom = movementProgram.encodedProgramImage(
                List.of(float16, float16),
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        byte[] movementMpsGraph = movementProgram.encodedProgramImage(
                List.of(float16, float16),
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.MPSGRAPH);

        assertEquals(6L, float16.byteCount());
        assertEquals(7, readInt(custom, MetalMpsGraphProgram.HEADER_BYTES));
        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(custom, custom.length));
            assertEquals(0, abi.createUnaligned(custom));
            assertEquals(13, abi.create(mpsGraph, mpsGraph.length));
            assertEquals(0, abi.create(movementCustom, movementCustom.length));
            assertEquals(13, abi.create(movementMpsGraph, movementMpsGraph.length));
            assertEquals(
                    1,
                    abi.create(
                            rewriteInt(custom, MetalMpsGraphProgram.HEADER_BYTES, 8),
                            custom.length));
            assertEquals(
                    1,
                    abi.create(
                            rewriteLong(
                                    custom,
                                    MetalMpsGraphProgram.HEADER_BYTES + 32,
                                    4L),
                            custom.length));
        }
    }

    @Test
    void packagedNativeCreatesLowCastThenFloat32NegCustomPartition() throws Throwable {
        Path library = configuredLibrary();
        var cast = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.CAST,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.CAST_TARGET,
                new long[] {MetalMpsGraphProgram.dataTypeWire(DataType.FLOAT32)});
        var mixed = new MetalMpsGraphProgram(List.of(
                cast, MetalMpsGraphProgram.Node.neg(1, 2)));
        var floatOnly = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.neg(0, 1)));
        byte[] floatOnlyImage = floatOnly.encodedProgramImage(
                List.of(descriptor(4), descriptor(4)),
                new int[] {0}, new int[] {1}, MetalPreparedRoute.MPSGRAPH);

        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(floatOnlyImage, floatOnlyImage.length),
                    "FLOAT32-only MPSGraph route remains available");
            for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                        descriptor(low, 4), descriptor(4), descriptor(4));
                MetalPointwiseFusionPlan plan = mixed.resolvePlan(
                        values,
                        new int[] {0}, new int[] {2}, MetalPreparedRoute.CUSTOM_PROGRAM, null);
                assertEquals(List.of(
                                MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM,
                                MetalPointwiseFusionPlan.StepKind.MPSGRAPH_BOUNDARY),
                        plan.steps().stream().map(MetalPointwiseFusionPlan.Step::kind).toList(),
                        low.toString());
                assertEquals(0, plan.generatedUnitCount(), low.toString());
                byte[] custom = mixed.encodedProgramImage(
                        values,
                        new int[] {0}, new int[] {2}, MetalPreparedRoute.CUSTOM_PROGRAM);
                assertEquals(MetalPreparedRoute.CUSTOM_PROGRAM.wireIdentity(),
                        readInt(custom, 16), low.toString());
                assertEquals(0, abi.create(custom, custom.length), low.toString());
                byte[] forbidden = mixed.encodedProgramImage(
                        values,
                        new int[] {0}, new int[] {2}, MetalPreparedRoute.MPSGRAPH);
                assertEquals(13, abi.create(forbidden, forbidden.length), low.toString());
            }
        }
    }

    @Test
    void packagedNativeCreatesLowCastThenFloat32ExpMpsGraphBoundary() throws Throwable {
        var cast = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.CAST,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.CAST_TARGET,
                new long[] {MetalMpsGraphProgram.dataTypeWire(DataType.FLOAT32)});
        var exp = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.EXP,
                new int[] {1}, new int[] {2},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
        var program = new MetalMpsGraphProgram(List.of(cast, exp));
        try (RawAbi abi = new RawAbi(configuredLibrary())) {
            for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                        descriptor(low, 4), descriptor(4), descriptor(4));
                MetalPointwiseFusionPlan plan = program.resolvePlan(
                        values, new int[] {0}, new int[] {2},
                        MetalPreparedRoute.CUSTOM_PROGRAM, null);
                assertEquals(List.of(
                                MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM,
                                MetalPointwiseFusionPlan.StepKind.MPSGRAPH_BOUNDARY),
                        plan.steps().stream().map(MetalPointwiseFusionPlan.Step::kind).toList(),
                        low.toString());
                assertEquals(0, plan.generatedUnitCount(), low.toString());
                byte[] image = program.encodedProgramImage(
                        values, new int[] {0}, new int[] {2},
                        MetalPreparedRoute.CUSTOM_PROGRAM);
                assertEquals(0, abi.create(image, image.length), low.toString());
            }
        }
    }

    @Test
    void schemaTwentyAuthenticatesTypedLowExpCustomStepsAndRejectsForgedImages()
            throws Throwable {
        var exp = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.EXP,
                        new int[] {0}, new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.NONE, new long[0])));
        try (RawAbi abi = new RawAbi(configuredLibrary())) {
            for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                var value = descriptor(low, 7, 17);
                var values = List.of(value, value);
                var plan = exp.resolvePlan(values, new int[] {0}, new int[] {1},
                        MetalPreparedRoute.CUSTOM_PROGRAM, null);
                assertEquals(List.of(MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM),
                        plan.steps().stream().map(MetalPointwiseFusionPlan.Step::kind).toList());
                assertEquals(0, plan.generatedUnitCount());
                byte[] image = exp.encodedProgramImage(values, new int[] {0}, new int[] {1},
                        MetalPreparedRoute.CUSTOM_PROGRAM);
                assertEquals(0, abi.create(image, image.length), low.name());
                byte[] wrongRoute = exp.encodedProgramImage(values, new int[] {0}, new int[] {1},
                        MetalPreparedRoute.MPSGRAPH);
                assertNotEquals(0, abi.create(wrongRoute, wrongRoute.length), low.name());
                int inputDescriptor = MetalMpsGraphProgram.HEADER_BYTES;
                int outputDescriptor = inputDescriptor
                        + MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES;
                assertEquals(1, abi.create(rewriteInt(image, 4, 19), image.length));
                assertEquals(1, abi.create(rewriteInt(image, inputDescriptor,
                        MetalMpsGraphProgram.dataTypeWire(DataType.FLOAT32)), image.length));
                assertEquals(1, abi.create(rewriteInt(image, outputDescriptor,
                        MetalMpsGraphProgram.dataTypeWire(DataType.FLOAT32)), image.length));
                assertEquals(1, abi.create(rewriteInt(image, inputDescriptor + 16,
                        readInt(image, inputDescriptor + 16) | 1), image.length));
                assertEquals(1, abi.create(rewriteInt(image, outputDescriptor + 16,
                        readInt(image, outputDescriptor + 16) | 1), image.length));
                assertEquals(1, abi.create(rewriteLong(image,
                        inputDescriptor + 32, 0L), image.length));
                assertEquals(1, abi.create(rewriteInt(image, nodeOffset(2),
                        MetalMpsGraphProgram.NodeKind.EXPM1.wireIdentity()), image.length));
            }
        }
    }

    /**
     * A low feed cast to FLOAT32 does not authorize fused FLOAT32 work later in that partition.
     * The forged images retain a valid typed fusion extension and a matching manifest digest, so
     * rejection exercises native low-partition authorization rather than checksum corruption.
     */
    @Test
    void checksumConsistentMixedLowFusionExtensionIsRejected() throws Throwable {
        Path library = configuredLibrary();
        var cast = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.CAST,
                new int[] {0},
                new int[] {1},
                MetalMpsGraphProgram.AttributeKind.CAST_TARGET,
                new long[] {MetalMpsGraphProgram.dataTypeWire(DataType.FLOAT32)});
        var pointwiseProgram = new MetalMpsGraphProgram(List.of(
                cast,
                unaryNode(MetalMpsGraphProgram.NodeKind.FLOOR, 1, 2),
                unaryNode(MetalMpsGraphProgram.NodeKind.CEIL, 2, 3)));
        var anchorProgram = new MetalMpsGraphProgram(List.of(
                cast,
                MetalMpsGraphProgram.Node.matmul(1, 2, 3),
                MetalMpsGraphProgram.Node.binary(
                        MetalMpsGraphProgram.NodeKind.ADD, 3, 4, 5)));
        List<MetalMpsGraphProgram.ValueDescriptor> pointwiseValues = List.of(
                descriptor(DataType.FLOAT64, 4), descriptor(4), descriptor(4), descriptor(4));
        List<MetalMpsGraphProgram.ValueDescriptor> anchorValues = List.of(
                descriptor(DataType.FLOAT64, 3), descriptor(3), descriptor(3, 2),
                descriptor(2), descriptor(2), descriptor(2));
        byte[] pointwise = pointwiseProgram.encodedProgramImage(
                pointwiseValues,
                new int[] {0}, new int[] {3}, MetalPreparedRoute.CUSTOM_PROGRAM);
        byte[] anchor = anchorProgram.encodedProgramImage(
                anchorValues,
                new int[] {0, 2, 4}, new int[] {5}, MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(3, readInt(pointwise,
                extensionCoreBytes(pointwise) + MetalMpsGraphProgram.STEP_DESCRIPTOR_BYTES));
        assertEquals(4, readInt(anchor,
                extensionCoreBytes(anchor) + MetalMpsGraphProgram.STEP_DESCRIPTOR_BYTES));

        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(pointwise, pointwise.length));
            assertEquals(0, abi.create(anchor, anchor.length));
            for (DataType low : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                var lowPointwiseValues = new ArrayList<>(pointwiseValues);
                lowPointwiseValues.set(0, descriptor(low, 4));
                byte[] canonicalPointwise = pointwiseProgram.encodedProgramImage(
                        lowPointwiseValues,
                        new int[] {0}, new int[] {3}, MetalPreparedRoute.CUSTOM_PROGRAM);
                assertEquals(0, abi.create(canonicalPointwise, canonicalPointwise.length),
                        low + " canonical mixed-low pointwise partition");

                var lowAnchorValues = new ArrayList<>(anchorValues);
                lowAnchorValues.set(0, descriptor(low, 3));
                byte[] canonicalAnchor = anchorProgram.encodedProgramImage(
                        lowAnchorValues,
                        new int[] {0, 2, 4}, new int[] {5}, MetalPreparedRoute.CUSTOM_PROGRAM);
                assertEquals(0, abi.create(canonicalAnchor, canonicalAnchor.length),
                        low + " canonical mixed-low anchor partition");

                for (byte[] allFloatImage : List.of(pointwise, anchor)) {
                    int rank = allFloatImage == pointwise ? 4 : 3;
                    byte[] canonicalLow = allFloatImage == pointwise
                            ? canonicalPointwise : canonicalAnchor;
                    byte[] forged = rewriteManifest(
                            rewriteInt(allFloatImage, MetalMpsGraphProgram.HEADER_BYTES,
                                    MetalMpsGraphProgram.dataTypeWire(low)),
                            "value 0 " + MetalMpsGraphProgram.dataTypeWire(DataType.FLOAT64)
                                    + " 1 0 0 " + rank + "\n",
                            "value 0 " + MetalMpsGraphProgram.dataTypeWire(low)
                                    + " 1 0 0 " + rank + "\n");
                    assertArrayEquals(
                            Arrays.copyOfRange(canonicalLow, MetalMpsGraphProgram.HEADER_BYTES,
                                    extensionCoreBytes(canonicalLow)),
                            Arrays.copyOfRange(forged, MetalMpsGraphProgram.HEADER_BYTES,
                                    extensionCoreBytes(forged)),
                            "forgery must preserve the accepted mixed-low core");
                    int manifestOffset = extensionManifestOffset(forged);
                    int digestOffset = manifestOffset + readInt(forged, 80);
                    assertArrayEquals(
                            MessageDigest.getInstance("SHA-256").digest(
                                    Arrays.copyOfRange(forged, manifestOffset, digestOffset)),
                            Arrays.copyOfRange(forged, digestOffset, forged.length));
                    assertEquals(1, abi.create(forged, forged.length),
                            low + " must not enter "
                                    + (allFloatImage == pointwise
                                            ? "generated-pointwise" : "anchor-epilogue")
                                    + " execution despite an updated manifest digest");
                }
            }
        }
    }

    @Test
    void nativeGeneralMatmulAnchorRejectsMalformedContractionGeometry() throws Throwable {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                MetalMpsGraphProgram.Node.binary(
                        MetalMpsGraphProgram.NodeKind.ADD, 2, 3, 4)));
        List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                descriptor(3),
                descriptor(3, 2),
                descriptor(2),
                descriptor(2),
                descriptor(2));
        int[] feeds = {0, 1, 3};
        int[] targets = {4};
        MetalPointwiseFusionPlan fusion = MetalPointwiseFusionPlanner.plan(
                program,
                values,
                feeds,
                targets,
                MetalPreparedRoute.CUSTOM_PROGRAM);
        byte[] valid = program.encodedProgramImage(
                values,
                feeds,
                targets,
                MetalPreparedRoute.CUSTOM_PROGRAM,
                fusion);
        int dimensionsOffset = MetalMpsGraphProgram.HEADER_BYTES
                + values.size() * MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES
                + program.nodes().size() * MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES;
        byte[] malformed = rewriteLong(valid, dimensionsOffset + Long.BYTES, 4L);
        malformed = rewriteLong(
                malformed,
                MetalMpsGraphProgram.HEADER_BYTES
                        + MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES
                        + 32,
                8L);

        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(valid, valid.length));
            assertEquals(1, abi.create(malformed, malformed.length));
        }
    }

    @Test
    void rankZeroEpiloguesRequireAuthenticatedAnchorMembership() throws Throwable {
        Path library = configuredLibrary();
        var validProgram = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                MetalMpsGraphProgram.Node.scalarValue(
                        MetalMpsGraphProgram.NodeKind.SCALAR_MUL,
                        2,
                        3,
                        0x4000_0000),
                unaryNode(MetalMpsGraphProgram.NodeKind.RELU, 3, 4)));
        List<MetalMpsGraphProgram.ValueDescriptor> validValues = List.of(
                descriptor(3),
                descriptor(3),
                descriptor(),
                descriptor(),
                descriptor());
        int[] validFeeds = {0, 1};
        int[] validTargets = {4};
        MetalPointwiseFusionPlan validFusion = MetalPointwiseFusionPlanner.plan(
                validProgram,
                validValues,
                validFeeds,
                validTargets,
                MetalPreparedRoute.CUSTOM_PROGRAM);
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                validValues,
                validProgram,
                validFeeds,
                validTargets,
                MetalPreparedRoute.CUSTOM_PROGRAM,
                validFusion);
        byte[] valid = validProgram.encodedProgramImage(
                validValues,
                validFeeds,
                validTargets,
                MetalPreparedRoute.CUSTOM_PROGRAM,
                validFusion);

        var standaloneMulProgram = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                unaryNode(MetalMpsGraphProgram.NodeKind.RELU, 2, 3),
                MetalMpsGraphProgram.Node.scalarValue(
                        MetalMpsGraphProgram.NodeKind.SCALAR_MUL,
                        4,
                        5,
                        0x4000_0000)));
        List<MetalMpsGraphProgram.ValueDescriptor> standaloneMulValues = List.of(
                descriptor(3),
                descriptor(3),
                descriptor(),
                descriptor(),
                descriptor(),
                descriptor());
        int[] standaloneMulFeeds = {0, 1, 4};
        int[] standaloneMulTargets = {3, 5};
        MetalPointwiseFusionPlan standaloneMulFusion = MetalPointwiseFusionPlanner.plan(
                standaloneMulProgram,
                standaloneMulValues,
                standaloneMulFeeds,
                standaloneMulTargets,
                MetalPreparedRoute.CUSTOM_PROGRAM);
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        standaloneMulValues,
                        standaloneMulProgram,
                        standaloneMulFeeds,
                        standaloneMulTargets,
                        MetalPreparedRoute.CUSTOM_PROGRAM,
                        standaloneMulFusion));
        byte[] standaloneMul = standaloneMulProgram.encodedProgramImage(
                standaloneMulValues,
                standaloneMulFeeds,
                standaloneMulTargets,
                MetalPreparedRoute.CUSTOM_PROGRAM,
                standaloneMulFusion);

        var standaloneReluProgram = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                MetalMpsGraphProgram.Node.binary(
                        MetalMpsGraphProgram.NodeKind.ADD, 2, 3, 4),
                unaryNode(MetalMpsGraphProgram.NodeKind.RELU, 5, 6)));
        List<MetalMpsGraphProgram.ValueDescriptor> standaloneReluValues = List.of(
                descriptor(3),
                descriptor(3),
                descriptor(),
                descriptor(),
                descriptor(),
                descriptor(),
                descriptor());
        int[] standaloneReluFeeds = {0, 1, 3, 5};
        int[] standaloneReluTargets = {4, 6};
        MetalPointwiseFusionPlan standaloneReluFusion = MetalPointwiseFusionPlanner.plan(
                standaloneReluProgram,
                standaloneReluValues,
                standaloneReluFeeds,
                standaloneReluTargets,
                MetalPreparedRoute.CUSTOM_PROGRAM);
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        standaloneReluValues,
                        standaloneReluProgram,
                        standaloneReluFeeds,
                        standaloneReluTargets,
                        MetalPreparedRoute.CUSTOM_PROGRAM,
                        standaloneReluFusion));
        byte[] standaloneRelu = standaloneReluProgram.encodedProgramImage(
                standaloneReluValues,
                standaloneReluFeeds,
                standaloneReluTargets,
                MetalPreparedRoute.CUSTOM_PROGRAM,
                standaloneReluFusion);

        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(valid, valid.length));
            assertEquals(8, abi.create(standaloneMul, standaloneMul.length));
            assertEquals(8, abi.create(standaloneRelu, standaloneRelu.length));
        }
    }

    @Test
    void schemaTwentyAuthenticatesCapStopPrecedenceAndFirstRejectedNode() throws Throwable {
        Path library = configuredLibrary();
        var nodes = new ArrayList<MetalMpsGraphProgram.Node>();
        MetalMpsGraphProgram.NodeKind[] kinds = {
            MetalMpsGraphProgram.NodeKind.FLOOR,
            MetalMpsGraphProgram.NodeKind.CEIL,
            MetalMpsGraphProgram.NodeKind.SIGN,
            MetalMpsGraphProgram.NodeKind.RELU
        };
        for (int unit = 0; unit < 33; unit++) {
            int input = unit * 9;
            for (int relative = 0; relative < 8; relative++) {
                nodes.add(unaryNode(
                        kinds[relative % kinds.length],
                        input + relative,
                        input + relative + 1));
            }
            if (unit != 32) nodes.add(MetalMpsGraphProgram.Node.neg(input + 8, input + 9));
        }
        List<MetalMpsGraphProgram.ValueDescriptor> values = new ArrayList<>();
        for (int value = 0; value < 297; value++) values.add(descriptor(4));
        byte[] valid = new MetalMpsGraphProgram(nodes).encodedProgramImage(
                values,
                new int[] {0},
                new int[] {296},
                MetalPreparedRoute.CUSTOM_PROGRAM);

        assertEquals(32, readInt(valid, 88));
        assertEquals(256, readInt(valid, 76));
        assertEquals(288, readInt(valid, 104));
        assertEquals(1, readInt(valid, 108));
        byte[] wrongPosition = rewriteInt(valid, 104, 287);
        wrongPosition = rewriteManifest(wrongPosition, "stop 288 1\n", "stop 287 1\n");
        byte[] wrongPrecedence = rewriteInt(valid, 108, 2);
        wrongPrecedence = rewriteManifest(wrongPrecedence, "stop 288 1\n", "stop 288 2\n");

        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(valid, valid.length));
            assertEquals(1, abi.create(wrongPosition, wrongPosition.length));
            assertEquals(1, abi.create(wrongPrecedence, wrongPrecedence.length));
        }
    }

    @Test
    void schemaTwentyRejectsFunctionCapAndCompactVirtualValueForgeries() throws Throwable {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                unaryNode(MetalMpsGraphProgram.NodeKind.FLOOR, 0, 1),
                unaryNode(MetalMpsGraphProgram.NodeKind.CEIL, 1, 2)));
        byte[] valid = program.encodedProgramImage(
                List.of(descriptor(4), descriptor(4), descriptor(4)),
                new int[] {0},
                new int[] {2},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        int coreBytes = extensionCoreBytes(valid);
        int memberOffset = coreBytes
                + readInt(valid, 60) * MetalMpsGraphProgram.STEP_DESCRIPTOR_BYTES;
        int bindingOffset = memberOffset + readInt(valid, 64) * Integer.BYTES;
        int materializedOffset = bindingOffset
                + readInt(valid, 68) * MetalMpsGraphProgram.BINDING_DESCRIPTOR_BYTES;

        byte[] overFunctionCap = rewriteInt(
                valid, coreBytes + 7 * Integer.BYTES, 16_385);
        overFunctionCap = rewriteManifest(
                overFunctionCap,
                "step 0 3 0 2 0 2 0 2 402 0\n",
                "step 0 3 0 2 0 2 0 2 16385 0\n");

        byte[] virtualInsteadOfTarget = rewriteInt(
                valid, materializedOffset + Integer.BYTES, 1);
        virtualInsteadOfTarget = rewriteInt(
                virtualInsteadOfTarget,
                bindingOffset + MetalMpsGraphProgram.BINDING_DESCRIPTOR_BYTES + 16,
                1);
        virtualInsteadOfTarget = rewriteManifest(
                virtualInsteadOfTarget, "materialized 1 2\n", "materialized 1 1\n");
        virtualInsteadOfTarget = rewriteManifest(
                virtualInsteadOfTarget,
                "binding 1 0 1 2 1 2\n",
                "binding 1 0 1 2 1 1\n");

        byte[] virtualInsteadOfFeed = rewriteInt(valid, materializedOffset, 1);
        virtualInsteadOfFeed = rewriteInt(virtualInsteadOfFeed, bindingOffset + 16, 1);
        virtualInsteadOfFeed = rewriteManifest(
                virtualInsteadOfFeed, "materialized 0 0\n", "materialized 0 1\n");
        virtualInsteadOfFeed = rewriteManifest(
                virtualInsteadOfFeed,
                "binding 0 0 0 1 0 0\n",
                "binding 0 0 0 1 0 1\n");

        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(valid, valid.length));
            assertEquals(1, abi.create(overFunctionCap, overFunctionCap.length));
            assertEquals(1, abi.create(
                    virtualInsteadOfTarget, virtualInsteadOfTarget.length));
            assertEquals(1, abi.create(virtualInsteadOfFeed, virtualInsteadOfFeed.length));
        }
    }

    @Test
    void nativeParsesUnpaddedOddReferencePoolBeforeAttributes() throws Throwable {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.generic(
                        MetalMpsGraphProgram.NodeKind.ADD,
                        new int[] {0, 0},
                        new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.NONE,
                        new long[0]),
                MetalMpsGraphProgram.Node.scalarValue(
                        MetalMpsGraphProgram.NodeKind.SCALAR_ADD,
                        1,
                        2,
                        0x3f80_0000)));
        byte[] image = program.encodedProgramImage(
                List.of(descriptor(4), descriptor(4), descriptor(4)),
                new int[] {0},
                new int[] {2},
                MetalPreparedRoute.MPSGRAPH);

        assertEquals(400, image.length);
        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(image, image.length));
        }
    }

    @Test
    void nativePointwiseEligibilityUsesExplicitRankAndUnsignedElementBounds() throws Throwable {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                unaryNode(MetalMpsGraphProgram.NodeKind.FLOOR, 0, 1),
                unaryNode(MetalMpsGraphProgram.NodeKind.CEIL, 1, 2)));
        long[] rankSixteen = new long[16];
        Arrays.fill(rankSixteen, 1L);
        byte[] rankSixteenImage = program.encodedProgramImage(
                List.of(
                        descriptor(rankSixteen),
                        descriptor(rankSixteen),
                        descriptor(rankSixteen)),
                new int[] {0},
                new int[] {2},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        byte[] scalarImage = program.encodedProgramImage(
                List.of(descriptor(), descriptor(), descriptor()),
                new int[] {0},
                new int[] {2},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        byte[] maximumImage = program.encodedProgramImage(
                List.of(
                        descriptor(65_535L, 65_537L),
                        descriptor(65_535L, 65_537L),
                        descriptor(65_535L, 65_537L)),
                new int[] {0},
                new int[] {2},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        byte[] onePastImage = program.encodedProgramImage(
                List.of(
                        descriptor(65_536L, 65_536L),
                        descriptor(65_536L, 65_536L),
                        descriptor(65_536L, 65_536L)),
                new int[] {0},
                new int[] {2},
                MetalPreparedRoute.CUSTOM_PROGRAM);

        assertEquals(1, readInt(rankSixteenImage, 88));
        assertEquals(0, readInt(scalarImage, 88));
        assertEquals(1, readInt(maximumImage, 88));
        assertEquals(0, readInt(onePastImage, 88));
        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(rankSixteenImage, rankSixteenImage.length));
            assertEquals(1, abi.create(scalarImage, scalarImage.length));
            assertEquals(0, abi.create(maximumImage, maximumImage.length));
            assertEquals(0, abi.create(onePastImage, onePastImage.length));
        }
    }

    @Test
    void nativeAbiSevenRejectsMalformedStorageLayoutDescriptors() throws Throwable {
        Path library = configuredLibrary();
        try (RawAbi abi = new RawAbi(library)) {
            byte[] valid = validSelectStorageLayoutImage();
            assertEquals(0, abi.create(valid, valid.length));

            int inputDescriptor = MetalMpsGraphProgram.HEADER_BYTES;
            int outputDescriptor =
                    inputDescriptor + MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES;
            int dimensionsOffset = nodeOffset(2) + MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES;
            int stridesOffset = dimensionsOffset + 3 * Long.BYTES;
            assertEquals(1, abi.create(
                    rewriteInt(valid, inputDescriptor + 12, -1), valid.length));
            assertEquals(1, abi.create(
                    rewriteInt(valid, inputDescriptor + 16, 10), valid.length));
            assertEquals(1, abi.create(
                    rewriteInt(valid, inputDescriptor + 20, 1), valid.length));
            assertEquals(1, abi.create(
                    rewriteLong(valid, inputDescriptor + 24, -1L), valid.length));
            assertEquals(1, abi.create(
                    rewriteLong(valid, inputDescriptor + 32, 7L), valid.length));
            assertEquals(8, abi.create(
                    rewriteLong(valid, dimensionsOffset, 0L), valid.length));
            assertEquals(1, abi.create(
                    rewriteLong(valid, stridesOffset, 0L), valid.length));
            assertEquals(1, abi.create(
                    rewriteLong(valid, stridesOffset, -1L), valid.length));
            assertEquals(1, abi.create(
                    rewriteLong(
                            rewriteLong(valid, stridesOffset, 1L),
                            stridesOffset + Long.BYTES,
                            1L),
                    valid.length));
            assertEquals(1, abi.create(
                    rewriteInt(valid, outputDescriptor + 20, 1), valid.length));
        }
    }

    @Test
    void task0066JavaAndNativeSchemaValidatorsAcceptScalarExactBoolCustomPrograms() throws Throwable {
        List<MetalMpsGraphProgram.NodeKind> exactBoolKinds = List.of(
                MetalMpsGraphProgram.NodeKind.IS_FINITE,
                MetalMpsGraphProgram.NodeKind.IS_NAN,
                MetalMpsGraphProgram.NodeKind.IS_INF,
                MetalMpsGraphProgram.NodeKind.LOGICAL_AND,
                MetalMpsGraphProgram.NodeKind.LOGICAL_OR,
                MetalMpsGraphProgram.NodeKind.LOGICAL_NOT,
                MetalMpsGraphProgram.NodeKind.WHERE);
        Path library = configuredLibrary();
        try (RawAbi abi = new RawAbi(library)) {
            for (MetalMpsGraphProgram.NodeKind kind : exactBoolKinds) {
                ScalarCase scalar = scalarCase(kind);
        MetalNativeApi.ProgramExecutableAbi.validateCreate (
            scalar.values(),
            scalar.program(),
            scalar.feeds(),
            scalar.targets(),MetalPreparedRoute.CUSTOM_PROGRAM);
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                                    scalar.values(),
                                    scalar.program(),
                                    scalar.feeds(),
                                    scalar.targets(),
                    MetalPreparedRoute.MPSGRAPH),
                            kind + " direct MPSGraph route");
                    byte[] custom = scalar.program().encodedProgramImage(
                            scalar.values(), scalar.feeds(), scalar.targets(),
                    MetalPreparedRoute.CUSTOM_PROGRAM);
        byte[] direct =
            scalar
                .program()
                .encodedProgramImage(
                    scalar.values(), scalar.feeds(), scalar.targets(), MetalPreparedRoute.MPSGRAPH);
                    assertEquals(0, abi.create(custom, custom.length),
                            kind + " custom route");
        assertEquals(13, abi.create(direct, direct.length), kind + " direct route");
            }
        }
    }
    @Test
    void JavaAndNativeRejectMalformedExactRawUnaryPrograms() throws Throwable {
        Path library = configuredLibrary();
        try (RawAbi abi = new RawAbi(library)) {
            for (MetalMpsGraphProgram.NodeKind kind : List.of(
                    MetalMpsGraphProgram.NodeKind.FLOOR,
                    MetalMpsGraphProgram.NodeKind.CEIL,
                    MetalMpsGraphProgram.NodeKind.SIGN)) {
                var node = MetalMpsGraphProgram.Node.generic(
                        kind,
                        new int[] {0},
                        new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.NONE,
                        new long[0]);
                var program = new MetalMpsGraphProgram(List.of(node));
                List<MetalMpsGraphProgram.ValueDescriptor> rankZero =
                        List.of(scalarDescriptor(DataType.FLOAT32), scalarDescriptor(DataType.FLOAT32));
                for (MetalPreparedRoute route :
                        List.of(MetalPreparedRoute.CUSTOM_PROGRAM, MetalPreparedRoute.MPSGRAPH)) {
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                                    rankZero,
                                    program,
                                    new int[] {0},
                                    new int[] {1},
                                    route));
                    byte[] image = program.encodedProgramImage(
                            rankZero, new int[] {0}, new int[] {1}, route);
                    assertEquals(1, abi.create(image, image.length), kind + " rank zero " + route);
                }

                List<MetalMpsGraphProgram.ValueDescriptor> wrongType = List.of(
                        new MetalMpsGraphProgram.ValueDescriptor(
                                DataType.INT32, new long[] {4}, false),
                        new MetalMpsGraphProgram.ValueDescriptor(
                                DataType.INT32, new long[] {4}, false));
                assertThrows(
                        IllegalArgumentException.class,
                        () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                                wrongType,
                                program,
                                new int[] {0},
                                new int[] {1},
                                MetalPreparedRoute.MPSGRAPH));
                byte[] wrongTypeImage = program.encodedProgramImage(
                        wrongType,
                        new int[] {0},
                        new int[] {1},
                        MetalPreparedRoute.MPSGRAPH);
                assertEquals(1, abi.create(wrongTypeImage, wrongTypeImage.length));

                byte[] valid = program.encodedProgramImage(
                        List.of(descriptor(4), descriptor(4)),
                        new int[] {0},
                        new int[] {1},
                        MetalPreparedRoute.MPSGRAPH);
                byte[] wrongAttribute = rewriteInt(valid, nodeOffset(2) + Integer.BYTES, 7);
                assertEquals(1, abi.create(wrongAttribute, wrongAttribute.length));
            }
        }
    }
    @Test
    void task0066KeepsCastValidationExactAndRejectsMpsGraphFallback() throws Throwable {
        Path library = configuredLibrary();
        var node = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.CAST,
                new int[] {0},
                new int[] {1},
                MetalMpsGraphProgram.AttributeKind.CAST_TARGET,
                new long[] {6});
        var program = new MetalMpsGraphProgram(List.of(node));
        List<MetalMpsGraphProgram.ValueDescriptor> valid = List.of(
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.INT32, new long[] {4}, false),
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.INT64, new long[] {4}, false));
        List<MetalMpsGraphProgram.ValueDescriptor> wrongOutput = List.of(
                valid.getFirst(),
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {4}, false));
        try (RawAbi abi = new RawAbi(library)) {MetalPreparedRoute route =MetalPreparedRoute.CUSTOM_PROGRAM;
                MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        valid,
                        program,
                        new int[] {0},
                        new int[] {1},
                        route);
                byte[] validImage = program.encodedProgramImage(
                        valid, new int[] {0}, new int[] {1}, route);
                assertEquals(0, abi.create(validImage, validImage.length));

                assertThrows(
                        IllegalArgumentException.class,
                        () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                                wrongOutput,
                                program,
                                new int[] {0},
                                new int[] {1},
                                route));
                byte[] wrongTypeImage = program.encodedProgramImage(
                        wrongOutput, new int[] {0}, new int[] {1}, route);
                assertEquals(1, abi.create(wrongTypeImage, wrongTypeImage.length));

                byte[] wrongAttribute = rewriteInt(
                        validImage, nodeOffset(2) + Integer.BYTES, 0);
                assertEquals(1, abi.create(wrongAttribute, wrongAttribute.length));

      assertThrows(
          IllegalArgumentException.class,
          () ->
              MetalNativeApi.ProgramExecutableAbi.validateCreate(
                  valid,
                  program,
                  new int[] {0
            },
                  new int[] {1},
                  MetalPreparedRoute.MPSGRAPH));
      byte[] mpsGraphImage =
          program.encodedProgramImage(
              valid, new int[] {0}, new int[] {1}, MetalPreparedRoute.MPSGRAPH);
      assertEquals(13, abi.create(mpsGraphImage, mpsGraphImage.length));
        }
    }

    @Test
    void task0066RejectsIntegerImageUnfoldAndMpsGraphFallback() throws Throwable {
        Path library = configuredLibrary();
        var node = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.UNFOLD2D,
                new int[] {0},
                new int[] {1},
                MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                new long[] {2, 2, 1, 1, 0, 0, 1, 1, 0});
        var program = new MetalMpsGraphProgram(List.of(node));
        List<MetalMpsGraphProgram.ValueDescriptor> integers = List.of(
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.INT32, new long[] {1, 1, 2, 2}, false),
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.INT32, new long[] {1, 4, 1}, false));
        try (RawAbi abi = new RawAbi(library)) {
            for (MetalPreparedRoute route :
                    List.of(MetalPreparedRoute.CUSTOM_PROGRAM, MetalPreparedRoute.MPSGRAPH)) {
                assertThrows(
                        IllegalArgumentException.class,
                        () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                                integers,
                                program,
                                new int[] {0},
                                new int[] {1},
                                route));
                byte[] image = program.encodedProgramImage(
                        integers, new int[] {0}, new int[] {1}, route);
                assertEquals(
            route == MetalPreparedRoute.CUSTOM_PROGRAM ?1 : 13, abi.create(image, image.length));
            }
        }
    }

    @Test
    void task0066KeepsWideCustomOriginsButRejectsMpsGraph() throws Throwable {
        Path library = configuredLibrary();
        long uint32Max = 0xffff_ffffL;
        long hostilePositions = 1_073_741_825L;
        try (RawAbi abi = new RawAbi(library)) {
            long wide = 1L << 32;
            var feasibleWideUnfold2d = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.UNFOLD2D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                            new long[] {1, 2, 1, wide, 0, wide, 1, 1, 0})));
            assertTask0059WindowOriginRejected(
                    abi,
                    feasibleWideUnfold2d,
                    List.of(
                            descriptor(DataType.FLOAT32, 1, 1, 1, 1),
                            descriptor(DataType.FLOAT32, 1, 2, 2)),
                    "UNFOLD2D feasible wide custom origin");
            var feasibleWideFold2d = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.FOLD2D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.FOLD_WINDOW_2D,
                            new long[] {
                                4, 1, 1, 1, 1,
                                1, 2, 1, wide, 0, wide, 1, 1, 0
                            })));
            assertTask0059WindowOriginRejected(
                    abi,
                    feasibleWideFold2d,
                    List.of(
                            descriptor(DataType.FLOAT32, 1, 2, 2),
                            descriptor(DataType.FLOAT32, 1, 1, 1, 1)),
                    "FOLD2D feasible wide custom origin");

            var unfold2d = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.UNFOLD2D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                            new long[] {1, 1, 1, 4, 0, 0, 1, 1, 1})));
            assertTask0059WindowOriginRejected(
                    abi,
                    unfold2d,
                    List.of(
                            descriptor(DataType.FLOAT32, 1, 1, 1, uint32Max),
                            descriptor(DataType.FLOAT32, 1, 1, hostilePositions)),
                    "UNFOLD2D");

            var unfold3d = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.UNFOLD3D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.WINDOW_3D,
                            new long[] {
                                1, 1, 1, 1, 1, 4, 0, 0, 0, 1, 1, 1, 1
                            })));
            assertTask0059WindowOriginRejected(
                    abi,
                    unfold3d,
                    List.of(
                            descriptor(DataType.FLOAT32, 1, 1, 1, 1, uint32Max),
                            descriptor(DataType.FLOAT32, 1, 1, hostilePositions)),
                    "UNFOLD3D");

            var fold2d = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.FOLD2D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.FOLD_WINDOW_2D,
                            new long[] {
                                4, 1, 1, 1, uint32Max,
                                1, 1, 1, 4, 0, 0, 1, 1, 1
                            })));
            assertTask0059WindowOriginRejected(
                    abi,
                    fold2d,
                    List.of(
                            descriptor(DataType.FLOAT32, 1, 1, hostilePositions),
                            descriptor(DataType.FLOAT32, 1, 1, 1, uint32Max)),
                    "FOLD2D");

            var fold3d = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.FOLD3D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.FOLD_WINDOW_3D,
                            new long[] {
                                5, 1, 1, 1, 1, uint32Max,
                                1, 1, 1, 1, 1, 4, 0, 0, 0, 1, 1, 1, 1
                            })));
            assertTask0059WindowOriginRejected(
                    abi,
                    fold3d,
                    List.of(
                            descriptor(DataType.FLOAT32, 1, 1, hostilePositions),
                            descriptor(DataType.FLOAT32, 1, 1, 1, 1, uint32Max)),
                    "FOLD3D");
        }
    }

    @Test
    void javaAndNativeAcceptMatchingGradButRejectDisallowedScalarShapes() throws Throwable {
        Path library = configuredLibrary();
        try (RawAbi abi = new RawAbi(library)) {
            for (MetalMpsGraphProgram.NodeKind kind : List.of(
                    MetalMpsGraphProgram.NodeKind.SCALAR_ADD,
                    MetalMpsGraphProgram.NodeKind.SCALAR_SUB,
                    MetalMpsGraphProgram.NodeKind.SCALAR_MUL,
                    MetalMpsGraphProgram.NodeKind.SCALAR_DIV,
                    MetalMpsGraphProgram.NodeKind.RECIPROCAL)) {
                MetalMpsGraphProgram.Node node = kind == MetalMpsGraphProgram.NodeKind.RECIPROCAL
                        ? MetalMpsGraphProgram.Node.generic(
                                kind,
                                new int[] {0},
                                new int[] {1},
                                MetalMpsGraphProgram.AttributeKind.NONE,
                                new long[0])
                        : MetalMpsGraphProgram.Node.scalarValue(
                                kind, 0, 1, 0x8000_0001);
                var program = new MetalMpsGraphProgram(List.of(node));
                List<MetalMpsGraphProgram.ValueDescriptor> grad = List.of(
                        new MetalMpsGraphProgram.ValueDescriptor(
                                DataType.FLOAT32, new long[] {4}, true),
                        new MetalMpsGraphProgram.ValueDescriptor(
                                DataType.FLOAT32, new long[] {4}, true));
                MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        grad,
                        program,
                        new int[] {0},
                        new int[] {1},
                        MetalPreparedRoute.MPSGRAPH);
                byte[] gradImage = program.encodedProgramImage(
                        grad, new int[] {0}, new int[] {1}, MetalPreparedRoute.MPSGRAPH);
                assertEquals(0, abi.create(gradImage, gradImage.length), kind + " grad");

                List<MetalMpsGraphProgram.ValueDescriptor> rankZero =
                        List.of(scalarDescriptor(DataType.FLOAT32), scalarDescriptor(DataType.FLOAT32));
                assertThrows(
                        IllegalArgumentException.class,
                        () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                                rankZero,
                                program,
                                new int[] {0},
                                new int[] {1},
                                MetalPreparedRoute.MPSGRAPH));
                byte[] scalarImage = program.encodedProgramImage(
                        rankZero,
                        new int[] {0},
                        new int[] {1},
                        MetalPreparedRoute.MPSGRAPH);
                assertEquals(8, abi.create(scalarImage, scalarImage.length), kind + " rank zero");
                if (kind != MetalMpsGraphProgram.NodeKind.RECIPROCAL) {
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> MetalMpsGraphProgram.Node.generic(
                                    kind,
                                    new int[] {0},
                                    new int[] {1},
                                    MetalMpsGraphProgram.AttributeKind.SCALAR_VALUE,
                                    new long[] {2L, 0x8000_0001L}));
                    List<MetalMpsGraphProgram.ValueDescriptor> values =
                            List.of(descriptor(4), descriptor(4));
                    byte[] validImage = program.encodedProgramImage(
                            values,
                            new int[] {0},
                            new int[] {1},
                            MetalPreparedRoute.MPSGRAPH);
                    byte[] wrongTypeImage = rewriteLong(
                            validImage, validImage.length - 2 * Long.BYTES, 2L);
                    assertEquals(
                            1,
                            abi.create(wrongTypeImage, wrongTypeImage.length),
                            kind + " scalar type");
                }
            }
        }
    }

    @Test
    void nativeAbiSevenReturnsDedicatedStatusForStructurallyValidUnsupportedOperation()
            throws Throwable {
        Path library = configuredLibrary();
        var attention = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.SCALED_DOT_PRODUCT_ATTENTION,
                new int[] {0, 1, 2},
                new int[] {3},
                MetalMpsGraphProgram.AttributeKind.ATTENTION,
                new long[] {0L, 0L});
        var program = new MetalMpsGraphProgram(List.of(attention));
        var values = List.of(
                descriptor(2, 4), descriptor(2, 4), descriptor(2, 4), descriptor(2, 4));
        byte[] image = program.encodedProgramImage(values, new int[] {0, 1, 2}, new int[] {3});
        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(13, abi.create(image, image.length));
            assertEquals(1, abi.create(
                    rewriteLong(image, image.length - Long.BYTES, 2L), image.length));
            int referencesOffset = MetalMpsGraphProgram.HEADER_BYTES
                    + values.size() * MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES
                    + MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES
                    + 8 * Long.BYTES
                    + 8 * Long.BYTES;
            assertEquals(1, abi.create(
                    rewriteInt(image, referencesOffset + 4 * Integer.BYTES, 3), image.length));
            assertEquals(1, abi.create(
                    rewriteInt(image, referencesOffset + 7 * Integer.BYTES, 0), image.length));
        }
    }

    @Test
    void expWire55AcceptsOnlyCanonicalNoGradientFloat32DirectImages() throws Throwable {
        Path library = configuredLibrary();
        byte[] valid = unaryImage(MetalMpsGraphProgram.NodeKind.EXP);
        int valuesOffset = MetalMpsGraphProgram.HEADER_BYTES;
        int outputOffset = valuesOffset + MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES;
        int nodeOffset = outputOffset + MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES;
        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(valid, valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 4, 19), valid.length),
                    "stale schema-nineteen image must reject");
            assertEquals(1, abi.create(rewriteInt(valid, 16, 3), valid.length),
                    "EXP has no custom-program route");
            assertEquals(1, abi.create(rewriteInt(valid, valuesOffset,
                    MetalMpsGraphProgram.dataTypeWire(DataType.BFLOAT16)), valid.length),
                    "BFLOAT16 input must reject");
            assertEquals(1, abi.create(rewriteInt(valid, outputOffset,
                    MetalMpsGraphProgram.dataTypeWire(DataType.FLOAT64)), valid.length),
                    "FLOAT64 output must reject");
            assertEquals(1, abi.create(rewriteInt(valid, valuesOffset + 16, 3), valid.length),
                    "input gradient flag must reject");
            assertEquals(1, abi.create(rewriteInt(valid, outputOffset + 16, 3), valid.length),
                    "output gradient flag must reject");
            assertEquals(1, abi.create(rewriteInt(valid, valuesOffset + 16, 0), valid.length),
                    "missing input layout must reject");
            assertEquals(1, abi.create(rewriteInt(valid, outputOffset + 16, 0), valid.length),
                    "missing output layout must reject");
            assertEquals(1, abi.create(rewriteInt(valid, nodeOffset + 4, 1), valid.length),
                    "EXP attributes must remain absent");
            byte[] mismatchedShape = new MetalMpsGraphProgram(List.of(
                    unaryNode(MetalMpsGraphProgram.NodeKind.EXP, 0, 1)))
                    .encodedProgramImage(
                            List.of(descriptor(4), descriptor(5)),
                            new int[] {0}, new int[] {1});
            assertEquals(1, abi.create(mismatchedShape, mismatchedShape.length));
            byte[] scalar = new MetalMpsGraphProgram(List.of(
                    unaryNode(MetalMpsGraphProgram.NodeKind.EXP, 0, 1)))
                    .encodedProgramImage(
                            List.of(descriptor(), descriptor()),
                            new int[] {0}, new int[] {1});
            assertEquals(1, abi.create(scalar, scalar.length));
        }
    }

    @Test
    void sigmoidWire64AcceptsOnlyCanonicalNoGradientFloat32DirectImages() throws Throwable {
        Path library = configuredLibrary();
        byte[] valid = unaryImage(MetalMpsGraphProgram.NodeKind.SIGMOID);
        int valuesOffset = MetalMpsGraphProgram.HEADER_BYTES;
        int outputOffset = valuesOffset + MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES;
        int nodeOffset = outputOffset + MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES;
        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(valid, valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, valuesOffset,
                    MetalMpsGraphProgram.dataTypeWire(DataType.BFLOAT16)), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, outputOffset,
                    MetalMpsGraphProgram.dataTypeWire(DataType.FLOAT16)), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, valuesOffset + 16, 3), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, outputOffset + 16, 3), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, valuesOffset + 16, 0), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, outputOffset + 16, 0), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, nodeOffset + 4, 1), valid.length));
            byte[] scalar = new MetalMpsGraphProgram(List.of(
                    unaryNode(MetalMpsGraphProgram.NodeKind.SIGMOID, 0, 1)))
                    .encodedProgramImage(List.of(descriptor(), descriptor()),
                            new int[] {0}, new int[] {1});
            assertEquals(1, abi.create(scalar, scalar.length));
        }
    }
    @Test
    void task0066AuthenticatesTransposeStorageAndMatmulGradientMetadataOnCustomRoute()
            throws Throwable {
        Path library = configuredLibrary();
        var sourceShape = io.github.pho001.synaptik.model.shape.Shape.of(3, 2);
        var transposeShape = io.github.pho001.synaptik.model.shape.Shape.of(2, 3);
        var source = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32, sourceShape.toLongArray(), false);
        var transpose = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32,
                transposeShape.toLongArray(),
                java.util.Optional.of(
                        io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(
                                transposeShape, new long[] {1, 2}, 0L, true)),
                false,
                true);
        var right = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32, new long[] {3, 4}, false);
        var output = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32, new long[] {2, 4}, false);
        var permute = MetalMpsGraphProgram.Node.permutation(0, 1, List.of(1, 0));
        var matmul = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.MATMUL,
                new int[] {1, 2},
                new int[] {3},
                MetalMpsGraphProgram.AttributeKind.NONE,
                new long[0]);
        var program = new MetalMpsGraphProgram(List.of(permute, matmul));
        List<MetalMpsGraphProgram.ValueDescriptor> values =
                List.of(source, transpose, right, output);
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                values,
                program,
                new int[] {0, 2},
                new int[] {3},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        byte[] valid = program.encodedProgramImage(
                values, new int[] {0, 2}, new int[] {3}, MetalPreparedRoute.CUSTOM_PROGRAM);

        int transposeDescriptor =
                MetalMpsGraphProgram.HEADER_BYTES + MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES;
        int dimensionCount = readInt(valid, 44);
        int referenceCount = readInt(valid, 52);
        int strideCount = readInt(valid, 48);
        int dimensionsOffset = nodeOffset(4)
                + 2 * MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES;
        int stridesOffset = dimensionsOffset + dimensionCount * Long.BYTES;
        int referencesOffset = stridesOffset + strideCount * Long.BYTES;
        int attributesOffset = referencesOffset + referenceCount * Integer.BYTES;
        int transposeStrideIndex = readInt(valid, transposeDescriptor + 12);

        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(valid, valid.length));
            assertEquals(1, abi.create(
                    rewriteInt(valid, transposeDescriptor + 16, 6), valid.length),
                    "non-dense provenance cannot authenticate an alias");
            assertEquals(1, abi.create(
                    rewriteInt(valid, transposeDescriptor + 16, 10), valid.length),
                    "dense storage must carry view provenance");

            byte[] offsetForgery = rewriteLong(
                    rewriteLong(valid, transposeDescriptor + 24, 1L),
                    transposeDescriptor + 32,
                    7L);
            assertEquals(1, abi.create(offsetForgery, offsetForgery.length));

            byte[] strideForgery = rewriteInt(valid, transposeDescriptor + 20, 1);
            strideForgery = rewriteLong(
                    strideForgery,
                    stridesOffset + transposeStrideIndex * Long.BYTES,
                    3L);
            strideForgery = rewriteLong(
                    strideForgery,
                    stridesOffset + (transposeStrideIndex + 1) * Long.BYTES,
                    1L);
            assertEquals(1, abi.create(strideForgery, strideForgery.length));
            assertEquals(1, abi.create(
                    rewriteLong(valid, transposeDescriptor + 32, 5L), valid.length));

            byte[] wrongPermutation =
                    rewriteLong(valid, attributesOffset + Long.BYTES, 0L);
            wrongPermutation = rewriteLong(
                    wrongPermutation, attributesOffset + 2 * Long.BYTES, 1L);
            assertEquals(1, abi.create(wrongPermutation, wrongPermutation.length));
            int transposeDimensionIndex = readInt(valid, transposeDescriptor + 8);
            assertNotEquals(0, abi.create(
                    rewriteLong(
                            valid,
                            dimensionsOffset + transposeDimensionIndex * Long.BYTES,
                            3L),
                    valid.length));
            assertEquals(1, abi.create(
                    rewriteInt(
                            valid,
                            transposeDescriptor,
                            MetalMpsGraphProgram.dataTypeWire(DataType.INT32)),
                    valid.length));
            assertNotEquals(0, abi.create(
                    rewriteLong(valid, dimensionsOffset, Long.MAX_VALUE), valid.length));
            assertEquals(1, abi.create(
                    rewriteInt(valid, transposeDescriptor + 16, 15), valid.length),
                    "PERMUTE and MATMUL gradient metadata must be preserved exactly");

            var direct = new MetalMpsGraphProgram(List.of(matmulNode(0, 1, 2)));
            List<MetalMpsGraphProgram.ValueDescriptor> directValues =
                    List.of(descriptor(2, 3), descriptor(3, 4), descriptor(2, 4));
            byte[] directImage = direct.encodedProgramImage(
                    directValues, new int[] {0, 1}, new int[] {2});
            assertEquals(0, abi.create(directImage, directImage.length));
            assertEquals(1, abi.create(
                    rewriteInt(
                            directImage,
                            MetalMpsGraphProgram.HEADER_BYTES + 16,
                            readInt(directImage, MetalMpsGraphProgram.HEADER_BYTES + 16) | 1),
                    directImage.length),
                    "FLOAT32 MATMUL output grad must equal the input-grad OR");
            List<List<MetalMpsGraphProgram.ValueDescriptor>> noGradientDomains = List.of(
                    List.of(
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.INT32, new long[] {2, 3}, false),
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.INT64, new long[] {3, 4}, false),
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.INT64, new long[] {2, 4}, false)),
                    List.of(
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.BFLOAT16, new long[] {2, 3}, false),
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32, new long[] {3, 4}, false),
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32, new long[] {2, 4}, false)));
            for (List<MetalMpsGraphProgram.ValueDescriptor> domain : noGradientDomains) {
                byte[] image = direct.encodedProgramImage(
                        domain,
                        new int[] {0, 1},
                        new int[] {2},
                        MetalPreparedRoute.CUSTOM_PROGRAM);
                assertEquals(0, abi.create(image, image.length));
                assertEquals(1, abi.create(
                        rewriteInt(
                                image,
                                MetalMpsGraphProgram.HEADER_BYTES + 16,
                                readInt(image, MetalMpsGraphProgram.HEADER_BYTES + 16) | 1),
                        image.length),
                        domain.getFirst().dataType() + " MATMUL must reject gradient flags");
            }
        }

        var wrongStride = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32,
                transposeShape.toLongArray(),
                java.util.Optional.of(
                        io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(
                                transposeShape, new long[] {3, 1}, 0L, true)),
                false,
                true);
        var wrongOffset = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32,
                transposeShape.toLongArray(),
                java.util.Optional.of(
                        io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(
                                transposeShape, new long[] {1, 2}, 1L, true)),
                false,
                true);
        for (MetalMpsGraphProgram.ValueDescriptor forged : List.of(wrongStride, wrongOffset)) {
            assertThrows(IllegalArgumentException.class, () ->
                    MetalNativeApi.ProgramExecutableAbi.validateCreate(
                            List.of(source, forged, right, output),
                            program,
                            new int[] {0, 2},
                            new int[] {3},
                            MetalPreparedRoute.CUSTOM_PROGRAM));
        }
        var direct = new MetalMpsGraphProgram(List.of(matmulNode(0, 1, 2)));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        List.of(
                                new MetalMpsGraphProgram.ValueDescriptor(
                                        DataType.FLOAT32, new long[] {2, 3}, true),
                                descriptor(3, 4),
                                descriptor(2, 4)),
                        direct,
                        new int[] {0, 1},
                        new int[] {2},
                        MetalPreparedRoute.MPSGRAPH));
    }

    @Test
    void task0062JavaAndNativeMsePreflightsStayInParity() throws Throwable {
        Path library = configuredLibrary();
        var node = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.MEAN_SQUARED_ERROR,
                new int[] {0, 1},
                new int[] {2},
                MetalMpsGraphProgram.AttributeKind.MSE,
                new long[] {1L});
        var program = new MetalMpsGraphProgram(List.of(node));
        List<MetalMpsGraphProgram.ValueDescriptor> valid = List.of(
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {2, 3}, true),
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {2, 3}, false),
                new MetalMpsGraphProgram.ValueDescriptor(
                        DataType.FLOAT32, new long[] {2, 3}, true));
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                valid,
                program,
                new int[] {0, 1},
                new int[] {2},
                MetalPreparedRoute.MPSGRAPH);
        byte[] validImage = program.encodedProgramImage(
                valid, new int[] {0, 1}, new int[] {2}, MetalPreparedRoute.MPSGRAPH);

        List<List<MetalMpsGraphProgram.ValueDescriptor>> invalidValues = List.of(
                List.of(
                        new MetalMpsGraphProgram.ValueDescriptor(
                                DataType.FLOAT32, new long[] {2, 3}, true),
                        new MetalMpsGraphProgram.ValueDescriptor(
                                DataType.FLOAT32, new long[] {2, 3}, false),
                        new MetalMpsGraphProgram.ValueDescriptor(
                                DataType.FLOAT32, new long[] {2, 3}, false)),
                List.of(
                        new MetalMpsGraphProgram.ValueDescriptor(
                                DataType.FLOAT32, new long[] {2, 3}, false),
                        new MetalMpsGraphProgram.ValueDescriptor(
                                DataType.FLOAT64, new long[] {2, 3}, false),
                        new MetalMpsGraphProgram.ValueDescriptor(
                                DataType.FLOAT32, new long[] {2, 3}, false)),
                List.of(
                        new MetalMpsGraphProgram.ValueDescriptor(
                                DataType.FLOAT32, new long[] {2, 3}, false),
                        new MetalMpsGraphProgram.ValueDescriptor(
                                DataType.FLOAT32, new long[] {3, 2}, false),
                        new MetalMpsGraphProgram.ValueDescriptor(
                                DataType.FLOAT32, new long[] {2, 3}, false)),
                List.of(
                        scalarDescriptor(DataType.FLOAT32),
                        scalarDescriptor(DataType.FLOAT32),
                        scalarDescriptor(DataType.FLOAT32)));

        try (RawAbi abi = new RawAbi(library)) {
            assertEquals(0, abi.create(validImage, validImage.length));
            for (List<MetalMpsGraphProgram.ValueDescriptor> invalid : invalidValues) {
                assertThrows(
                        IllegalArgumentException.class,
                        () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                                invalid,
                                program,
                                new int[] {0, 1},
                                new int[] {2},
                                MetalPreparedRoute.MPSGRAPH));
                byte[] image = program.encodedProgramImage(
                        invalid,
                        new int[] {0, 1},
                        new int[] {2},
                        MetalPreparedRoute.MPSGRAPH);
                assertEquals(1, abi.create(image, image.length));
            }

            int dimensionCount = readInt(validImage, 44);
            int referenceCount = readInt(validImage, 52);
            int strideCount = readInt(validImage, 48);
            int dimensionsOffset = nodeOffset(3)
                    + MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES;
            int referencesOffset = dimensionsOffset
                    + dimensionCount * Long.BYTES
                    + strideCount * Long.BYTES;
            int attributesOffset =
                    referencesOffset + referenceCount * Integer.BYTES;
            assertEquals(1, abi.create(
                    rewriteLong(validImage, attributesOffset, 4L),
                    validImage.length));
            assertEquals(1, abi.create(
                    rewriteInt(
                            validImage,
                            nodeOffset(3) + Integer.BYTES,
                            MetalMpsGraphProgram.AttributeKind.NONE.wireIdentity()),
                    validImage.length));

            var sumProgram = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.MEAN_SQUARED_ERROR,
                            new int[] {0, 1},
                            new int[] {2},
                            MetalMpsGraphProgram.AttributeKind.MSE,
                            new long[] {2L})));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                            List.of(descriptor(2, 3), descriptor(2, 3), descriptor(2, 3)),
                            sumProgram,
                            new int[] {0, 1},
                            new int[] {2},
                            MetalPreparedRoute.MPSGRAPH));
            byte[] wrongSumShape = sumProgram.encodedProgramImage(
                    List.of(descriptor(2, 3), descriptor(2, 3), descriptor(2, 3)),
                    new int[] {0, 1},
                    new int[] {2},
                    MetalPreparedRoute.MPSGRAPH);
            assertEquals(1, abi.create(wrongSumShape, wrongSumShape.length));
    }
  }

  @Test
  void task0066JavaAndNativeRejectDirectMpsGraphForSelectedOccurrences() throws Throwable {
    Path library = configuredLibrary();
    var program =
        new MetalMpsGraphProgram(
            List.of(
                MetalMpsGraphProgram.Node.generic(
                    MetalMpsGraphProgram.NodeKind.CAST,
                    new int[] {0},
                    new int[] {1},
                    MetalMpsGraphProgram.AttributeKind.CAST_TARGET,
                    new long[] {MetalMpsGraphProgram.dataTypeWire(DataType.FLOAT32)})));
    List<MetalMpsGraphProgram.ValueDescriptor> values =
        List.of(descriptor(DataType.FLOAT64, 2), descriptor(DataType.FLOAT32, 2));
    MetalNativeApi.ProgramExecutableAbi.validateCreate(
        values,
        program,
        new int[] {0},
        new int[] {1},
        MetalPreparedRoute.CUSTOM_PROGRAM);
    assertThrows(
        IllegalArgumentException.class,
        () ->
            MetalNativeApi.ProgramExecutableAbi.validateCreate(
                values,
                program,
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.MPSGRAPH));
    try (RawAbi abi = new RawAbi(library)) {
      byte[] custom =
          program.encodedProgramImage(
              values, new int[] {0}, new int[] {1}, MetalPreparedRoute.CUSTOM_PROGRAM);
      byte[] direct =
          program.encodedProgramImage(
              values, new int[] {0}, new int[] {1}, MetalPreparedRoute.MPSGRAPH);
      assertEquals(0, abi.create(custom, custom.length));
      assertEquals(13, abi.create(direct, direct.length));
        }
    }


    @Test
    void task0069RawNativeRejectsEveryCustomL1Boundary() throws Throwable {
        Path library = configuredLibrary();
        var validProgram = task0069L1Program(
                false, MetalMpsGraphProgram.ReductionForm.MULTI_AXIS, List.of(0));
        List<MetalMpsGraphProgram.ValueDescriptor> validValues =
                List.of(descriptor(4), descriptor());
        try (RawAbi abi = new RawAbi(library)) {
            byte[] valid = validProgram.encodedProgramImage(
                    validValues,
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            assertEquals(0, abi.create(valid, valid.length));

            byte[] direct = validProgram.encodedProgramImage(
                    validValues,
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.MPSGRAPH);
            assertEquals(13, abi.create(direct, direct.length));

            var retainedProgram = task0069L1Program(
                    true, MetalMpsGraphProgram.ReductionForm.MULTI_AXIS, List.of(0));
            byte[] validRetained = retainedProgram.encodedProgramImage(
                    List.of(descriptor(4), descriptor(1)),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            assertEquals(0, abi.create(validRetained, validRetained.length));

            var malformed = new ArrayList<byte[]>();
            malformed.add(task0069L1Program(
                            false,
                            MetalMpsGraphProgram.ReductionForm.SINGLE_AXIS,
                            List.of(0))
                    .encodedProgramImage(
                            validValues,
                            new int[] {0},
                            new int[] {1},
                            MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(task0069L1Program(
                            false,
                            MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                            List.of())
                    .encodedProgramImage(
                            validValues,
                            new int[] {0},
                            new int[] {1},
                            MetalPreparedRoute.CUSTOM_PROGRAM));
            byte[] duplicateAxes = Arrays.copyOf(valid, valid.length + Long.BYTES);
            ByteBuffer duplicateAxesBuffer =
                    ByteBuffer.wrap(duplicateAxes).order(ByteOrder.LITTLE_ENDIAN);
            duplicateAxesBuffer.putInt(2 * Integer.BYTES, duplicateAxes.length);
            duplicateAxesBuffer.putInt(9 * Integer.BYTES, 5);
            duplicateAxesBuffer.putInt(nodeOffset(2) + 7 * Integer.BYTES, 5);
            duplicateAxesBuffer.putLong(valid.length - 2 * Long.BYTES, 2L);
            duplicateAxesBuffer.putLong(valid.length, 0L);
            malformed.add(duplicateAxes);
            malformed.add(validProgram.encodedProgramImage(
                    List.of(
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32, new long[] {4}, true),
                            descriptor()),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(validProgram.encodedProgramImage(
                    List.of(
                            descriptor(4),
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32, new long[0], true)),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(validProgram.encodedProgramImage(
                    List.of(descriptor(DataType.FLOAT64, 4), descriptor()),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(validProgram.encodedProgramImage(
                    List.of(descriptor(2, 2), descriptor()),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(validProgram.encodedProgramImage(
                    List.of(descriptor(4), descriptor(1)),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(retainedProgram.encodedProgramImage(
                    List.of(descriptor(4), descriptor()),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(validProgram.encodedProgramImage(
                    List.of(descriptor(4), descriptor(DataType.FLOAT64)),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(validProgram.encodedProgramImage(
                    List.of(descriptor(0xffff_ffffL / Float.BYTES + 1L), descriptor()),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            Shape inputShape = Shape.of(4);
            malformed.add(validProgram.encodedProgramImage(
                    List.of(
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32,
                                    inputShape.toLongArray(),
                                    java.util.Optional.of(LayoutDescriptor.of(
                                            inputShape, new long[] {2}, 0, true)),
                                    false,
                                    false),
                            descriptor()),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            for (byte[] image : malformed) {
                assertEquals(1, abi.create(image, image.length));
            }
        }
    }

    @Test
    void task0069RawNativeRejectsEveryCustomVarianceBoundary() throws Throwable {
        Path library = configuredLibrary();
        var validProgram = task0069VarianceProgram(false, List.of(0), 0L);
        List<MetalMpsGraphProgram.ValueDescriptor> validValues =
                List.of(descriptor(1), descriptor());
        try (RawAbi abi = new RawAbi(library)) {
            byte[] valid = validProgram.encodedProgramImage(
                    validValues,
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            assertEquals(0, abi.create(valid, valid.length));
            byte[] retained = task0069VarianceProgram(true, List.of(0), 0L)
                    .encodedProgramImage(
                            List.of(descriptor(1), descriptor(1)),
                            new int[] {0},
                            new int[] {1},
                            MetalPreparedRoute.CUSTOM_PROGRAM);
            assertEquals(0, abi.create(retained, retained.length));
            byte[] direct = validProgram.encodedProgramImage(
                    validValues,
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.MPSGRAPH);
            assertEquals(13, abi.create(direct, direct.length));

            int nodeDescriptor = nodeOffset(validValues.size());
            int outputDescriptor =
                    MetalMpsGraphProgram.HEADER_BYTES
                            + MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES;
            int dimensionsOffset =
                    nodeDescriptor + MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES;
            int stridesOffset =
                    dimensionsOffset + readInt(valid, 44) * Long.BYTES;
            int referencesOffset =
                    stridesOffset + readInt(valid, 48) * Long.BYTES;
            int attributesOffset =
                    referencesOffset + readInt(valid, 52) * Integer.BYTES;
            int nodeReferencesOffset = referencesOffset + 2 * Integer.BYTES;
            int retainedDimensionsOffset =
                    nodeDescriptor + MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES;
            int retainedStridesOffset =
                    retainedDimensionsOffset + readInt(retained, 44) * Long.BYTES;
            int retainedOutputStrideIndex = readInt(retained, outputDescriptor + 12);

            var malformed = new ArrayList<byte[]>();
            malformed.add(rewriteInt(
                    valid,
                    nodeDescriptor + Integer.BYTES,
                    MetalMpsGraphProgram.AttributeKind.NONE.wireIdentity()));
            malformed.add(rewriteInt(
                    valid, nodeDescriptor + 7 * Integer.BYTES, 3));
            malformed.add(rewriteLong(
                    valid, attributesOffset + 2 * Long.BYTES, 2L));
            malformed.add(validProgram.encodedProgramImage(
                    List.of(descriptor(1), descriptor(DataType.FLOAT64)),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(rewriteLong(
                    retained, outputDescriptor + 3 * Long.BYTES, 1L));
            malformed.add(rewriteLong(
                    retained,
                    retainedStridesOffset + retainedOutputStrideIndex * Long.BYTES,
                    2L));
            malformed.add(rewriteInt(retained, outputDescriptor + 5 * Integer.BYTES, 3));
            malformed.add(rewriteInt(
                    valid, nodeReferencesOffset + Integer.BYTES, 0));
            malformed.add(task0069VarianceProgram(false, List.of(), 0L)
                    .encodedProgramImage(
                            validValues,
                            new int[] {0},
                            new int[] {1},
                            MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(task0069VarianceProgram(false, List.of(1), 0L)
                    .encodedProgramImage(
                            validValues,
                            new int[] {0},
                            new int[] {1},
                            MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(task0069VarianceProgram(false, List.of(0), 1L)
                    .encodedProgramImage(
                            validValues,
                            new int[] {0},
                            new int[] {1},
                            MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(validProgram.encodedProgramImage(
                    List.of(descriptor(2), descriptor()),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(validProgram.encodedProgramImage(
                    List.of(descriptor(1, 1), descriptor(1)),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(validProgram.encodedProgramImage(
                    List.of(descriptor(1), descriptor(1)),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(task0069VarianceProgram(true, List.of(0), 0L)
                    .encodedProgramImage(
                            validValues,
                            new int[] {0},
                            new int[] {1},
                            MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(validProgram.encodedProgramImage(
                    List.of(
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32, new long[] {1L}, true),
                            descriptor()),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(validProgram.encodedProgramImage(
                    List.of(
                            descriptor(1),
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32, new long[0], true)),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            malformed.add(validProgram.encodedProgramImage(
                    List.of(descriptor(DataType.FLOAT64, 1), descriptor()),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            Shape singleton = Shape.of(1);
            malformed.add(validProgram.encodedProgramImage(
                    List.of(
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32,
                                    singleton.toLongArray(),
                                    java.util.Optional.of(LayoutDescriptor.of(
                                            singleton, new long[] {2L}, 0L, true)),
                                    false,
                                    false),
                            descriptor()),
                    new int[] {0},
                    new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            for (byte[] image : malformed) {
                assertEquals(1, abi.create(image, image.length));
            }
            byte[] zeroExtent = rewriteLong(valid, dimensionsOffset, 0L);
            assertEquals(8, abi.create(zeroExtent, zeroExtent.length));
        }
    }

    @Test
    void task0069RawNativeRejectsEveryCustomScatterAddBoundary() throws Throwable {
        Path library = configuredLibrary();
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.scatterAdd(0, 1, 2, 3, 0)));
        List<MetalMpsGraphProgram.ValueDescriptor> validValues = List.of(
                descriptor(5),
                descriptor(DataType.INT32, 3),
                descriptor(3),
                descriptor(5));
        try (RawAbi abi = new RawAbi(library)) {
            byte[] valid = program.encodedProgramImage(
                    validValues,
                    new int[] {0, 1, 2},
                    new int[] {3},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            assertEquals(0, abi.create(valid, valid.length));
            int nodeDescriptor = nodeOffset(validValues.size());
            int dimensionsOffset =
                    nodeDescriptor + MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES;
            int stridesOffset =
                    dimensionsOffset + readInt(valid, 44) * Long.BYTES;
            int referencesOffset =
                    stridesOffset + readInt(valid, 48) * Long.BYTES;
            int attributesOffset =
                    referencesOffset + readInt(valid, 52) * Integer.BYTES;
            int nodeReferencesOffset = referencesOffset + 4 * Integer.BYTES;

            byte[] directMpsGraph = program.encodedProgramImage(
                    validValues,
                    new int[] {0, 1, 2},
                    new int[] {3},
                    MetalPreparedRoute.MPSGRAPH);
            var indirectIndexProgram = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.contiguous(1, 4),
                    MetalMpsGraphProgram.Node.scatterAdd(0, 4, 2, 3, 0)));
            byte[] indexNotFeed = indirectIndexProgram.encodedProgramImage(
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, 3),
                            descriptor(3),
                            descriptor(5),
                            descriptor(DataType.INT32, 3)),
                    new int[] {0, 1, 2},
                    new int[] {3},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            assertEquals(13, abi.create(directMpsGraph, directMpsGraph.length));
            assertEquals(1, abi.create(indexNotFeed, indexNotFeed.length));

            byte[] nonzeroAxis = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.scatterAdd(0, 1, 2, 3, 1)))
                    .encodedProgramImage(
                            validValues,
                            new int[] {0, 1, 2},
                            new int[] {3},
                            MetalPreparedRoute.CUSTOM_PROGRAM);
            List<byte[]> malformedStructure = List.of(
                    nonzeroAxis,
                    rewriteInt(
                            valid,
                            nodeDescriptor + Integer.BYTES,
                            MetalMpsGraphProgram.AttributeKind.NONE.wireIdentity()),
                    rewriteInt(valid, nodeDescriptor + 7 * Integer.BYTES, 0),
                    rewriteInt(valid, nodeDescriptor + 7 * Integer.BYTES, 2),
                    rewriteLong(valid, attributesOffset, -1L),
                    rewriteInt(valid, nodeReferencesOffset + Integer.BYTES, 0),
                    rewriteInt(valid, nodeReferencesOffset + 2 * Integer.BYTES, 0),
                    rewriteInt(valid, nodeReferencesOffset + 2 * Integer.BYTES, 1),
                    rewriteInt(valid, nodeReferencesOffset + 3 * Integer.BYTES, 0),
                    rewriteInt(valid, nodeReferencesOffset + 3 * Integer.BYTES, 1),
                    rewriteInt(valid, nodeReferencesOffset + 3 * Integer.BYTES, 2));
            for (int row = 0; row < malformedStructure.size(); row++) {
                byte[] image = malformedStructure.get(row);
                assertEquals(1, abi.create(image, image.length),
                        "malformed ScatterAdd structure row " + row);
            }
            byte[] zeroD = rewriteLong(
                    rewriteLong(valid, dimensionsOffset, 0L),
                    dimensionsOffset + 3 * Long.BYTES,
                    0L);
            byte[] zeroU = rewriteLong(
                    rewriteLong(valid, dimensionsOffset + Long.BYTES, 0L),
                    dimensionsOffset + 2 * Long.BYTES,
                    0L);
            assertEquals(8, abi.create(zeroD, zeroD.length));
            assertEquals(8, abi.create(zeroU, zeroU.length));

            Shape dataShape = Shape.of(5);
            Shape updateShape = Shape.of(3);
            long excessive = 0xffff_ffffL / Float.BYTES + 1L;
            long int64ByteOverflow = 0xffff_ffffL / Long.BYTES + 1L;
            long countOverflow = 0x1_0000_0000L;
            List<List<MetalMpsGraphProgram.ValueDescriptor>> malformedValues = List.of(
                    List.of(
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32, new long[] {5}, true),
                            descriptor(DataType.INT32, 3),
                            descriptor(3),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.INT32, new long[] {3}, true),
                            descriptor(3),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, 3),
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32, new long[] {3}, true),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, 3),
                            descriptor(3),
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32, new long[] {5}, true)),
                    List.of(
                            descriptor(1, 5),
                            descriptor(DataType.INT32, 3),
                            descriptor(3),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, 1, 3),
                            descriptor(3),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, 3),
                            descriptor(1, 3),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, 3),
                            descriptor(3),
                            descriptor(1, 5)),
                    List.of(
                            descriptor(DataType.FLOAT64, 5),
                            descriptor(DataType.INT32, 3),
                            descriptor(3),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.FLOAT32, 3),
                            descriptor(3),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, 3),
                            descriptor(DataType.FLOAT64, 3),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, 3),
                            descriptor(3),
                            descriptor(DataType.FLOAT64, 5)),
                    List.of(
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32,
                                    dataShape.toLongArray(),
                                    java.util.Optional.of(LayoutDescriptor.of(
                                            dataShape, new long[] {2}, 0, true)),
                                    false,
                                    false),
                            descriptor(DataType.INT32, 3),
                            descriptor(3),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.INT32,
                                    updateShape.toLongArray(),
                                    java.util.Optional.of(LayoutDescriptor.of(
                                            updateShape, new long[] {2}, 0, true)),
                                    false,
                                    false),
                            descriptor(3),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, 3),
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32,
                                    updateShape.toLongArray(),
                                    java.util.Optional.of(LayoutDescriptor.of(
                                            updateShape, new long[] {2}, 0, true)),
                                    false,
                                    false),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, 3),
                            descriptor(3),
                            new MetalMpsGraphProgram.ValueDescriptor(
                                    DataType.FLOAT32,
                                    dataShape.toLongArray(),
                                    java.util.Optional.of(LayoutDescriptor.of(
                                            dataShape, new long[] {2}, 0, true)),
                                    false,
                                    false)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, 4),
                            descriptor(3),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, 3),
                            descriptor(3),
                            descriptor(6)),
                    List.of(
                            descriptor(excessive),
                            descriptor(DataType.INT64, 3),
                            descriptor(3),
                            descriptor(excessive)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, excessive),
                            descriptor(excessive),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT64, int64ByteOverflow),
                            descriptor(int64ByteOverflow),
                            descriptor(5)),
                    List.of(
                            descriptor(5),
                            descriptor(DataType.INT32, countOverflow),
                            descriptor(countOverflow),
                            descriptor(5)));
            for (int row = 0; row < malformedValues.size(); row++) {
                List<MetalMpsGraphProgram.ValueDescriptor> values = malformedValues.get(row);
                byte[] image = program.encodedProgramImage(
                        values,
                        new int[] {0, 1, 2},
                        new int[] {3},
                        MetalPreparedRoute.CUSTOM_PROGRAM);
                assertEquals(1, abi.create(image, image.length),
                        "malformed ScatterAdd value row " + row);
            }
        }
    }

    @Test
    void task0063JavaAndRawNativePreflightsAgreeAtUint32AndOnHostileImages()
            throws Throwable {
        Path library = configuredLibrary();
        long uint32Max = 0xffff_ffffL;
        try (RawAbi abi = new RawAbi(library)) {
            for (MetalMpsGraphProgram.NodeKind kind : List.of(
                    MetalMpsGraphProgram.NodeKind.SORT,
                    MetalMpsGraphProgram.NodeKind.ARGSORT,
                    MetalMpsGraphProgram.NodeKind.TOP_K,
                    MetalMpsGraphProgram.NodeKind.ARG_MAX,
                    MetalMpsGraphProgram.NodeKind.ARG_MIN)) {
                Task0063Case exact = task0063Case(kind, 1L, uint32Max);
                MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        exact.values(),
                        exact.program(),
                        new int[] {0},
                        exact.targets(),
                        MetalPreparedRoute.CUSTOM_PROGRAM);
                byte[] exactImage = exact.program().encodedProgramImage(
                        exact.values(),
                        new int[] {0},
                        exact.targets(),
                        MetalPreparedRoute.CUSTOM_PROGRAM);
                assertEquals(0, abi.create(exactImage, exactImage.length),
                        kind + " exact UINT32_MAX");

                Task0063Case onePast = task0063Case(kind, 1L, uint32Max + 1L);
                assertTask0063RejectedByJavaAndNative(abi, onePast, kind + " one past");
            }

            Task0063Case productOverflow = task0063Case(
                    MetalMpsGraphProgram.NodeKind.SORT, 65_536L, 65_536L);
            assertTask0063RejectedByJavaAndNative(
                    abi, productOverflow, "product one past UINT32_MAX");

            Task0063Case exactSort = task0063Case(
                    MetalMpsGraphProgram.NodeKind.SORT, 1L, uint32Max);
            byte[] exactSortImage = exactSort.program().encodedProgramImage(
                    exactSort.values(),
                    new int[] {0},
                    exactSort.targets(),
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            int firstSpan = MetalMpsGraphProgram.HEADER_BYTES + 32;
            assertEquals(1, abi.create(
                    rewriteLong(exactSortImage, firstSpan, uint32Max + 1L),
                    exactSortImage.length), "malformed byte span");
            int strideOffset = MetalMpsGraphProgram.HEADER_BYTES
                    + exactSort.values().size() * MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES
                    + MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES
                    + 4 * Long.BYTES;
            assertEquals(1, abi.create(
                    rewriteLong(exactSortImage, strideOffset, uint32Max + 1L),
                    exactSortImage.length), "one-past canonical stride");

            Task0063Case topK = task0063Case(
                    MetalMpsGraphProgram.NodeKind.TOP_K, 6L);
            byte[] validTopK = topK.program().encodedProgramImage(
                    topK.values(),
                    new int[] {0},
                    topK.targets(),
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            assertEquals(0, abi.create(validTopK, validTopK.length));

            List<List<MetalMpsGraphProgram.ValueDescriptor>> malformedCompanions =
                    List.of(
                            List.of(
                                    descriptor(DataType.INT32, 6),
                                    descriptor(DataType.INT32, 1),
                                    descriptor(DataType.INT32, 1)),
                            List.of(
                                    descriptor(DataType.INT32, 6),
                                    descriptor(DataType.INT32, 1),
                                    descriptor(DataType.INT64, 2)),
                            List.of(
                                    descriptor(DataType.INT32, 6),
                                    descriptor(DataType.INT32, 1),
                                    new MetalMpsGraphProgram.ValueDescriptor(
                                            DataType.INT64, new long[] {1}, true)));
            for (List<MetalMpsGraphProgram.ValueDescriptor> values
                    : malformedCompanions) {
                assertThrows(
                        IllegalArgumentException.class,
                        () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                                values,
                                topK.program(),
                                new int[] {0},
                                topK.targets(),
                                MetalPreparedRoute.CUSTOM_PROGRAM));
                byte[] malformed = topK.program().encodedProgramImage(
                        values,
                        new int[] {0},
                        topK.targets(),
                        MetalPreparedRoute.CUSTOM_PROGRAM);
                assertEquals(1, abi.create(malformed, malformed.length));
            }

            int referencesOffset = MetalMpsGraphProgram.HEADER_BYTES
                    + topK.values().size() * MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES
                    + MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES
                    + 6 * Long.BYTES;
            assertEquals(1, abi.create(
                    rewriteInt(
                            validTopK,
                            referencesOffset + 4 * Integer.BYTES,
                            topK.values().size()),
                    validTopK.length), "out-of-range TOP_K companion");
        }
    }

    @Test
    void task0064JavaAndRawNativePreflightsRejectHostileGeometryBeforeExecutionResources()
            throws Throwable {
        Path library = configuredLibrary();
        long uint32Max = 0xffff_ffffL;
        try (RawAbi abi = new RawAbi(library)) {
            var convolution = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.CONV2D,
                            new int[] {0, 1},
                            new int[] {2},
                            MetalMpsGraphProgram.AttributeKind.CONV_2D,
                            new long[] {1, 1, 0, 0, 1, 1, 1})));
            List<MetalMpsGraphProgram.ValueDescriptor> exactConvolution = List.of(
                    descriptor(DataType.FLOAT32, 1, 1, 1, uint32Max),
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1),
                    descriptor(DataType.FLOAT32, 1, 1, 1, uint32Max));
            assertTask0064AcceptedByJavaAndNative(
                    abi, convolution, exactConvolution,
                    new int[] {0, 1}, new int[] {2}, "convolution exact UINT32_MAX");

            List<MetalMpsGraphProgram.ValueDescriptor> overLimitConvolution = List.of(
                    descriptor(DataType.FLOAT32, 1, 1, 1, uint32Max + 1L),
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1),
                    descriptor(DataType.FLOAT32, 1, 1, 1, uint32Max + 1L));
            assertTask0064RejectedByJavaAndNative(
                    abi, convolution, overLimitConvolution,
                    new int[] {0, 1}, new int[] {2}, "convolution one past UINT32_MAX");

            List<MetalMpsGraphProgram.ValueDescriptor> wrongConvolutionCarrier = List.of(
                    descriptor(DataType.FLOAT64, 1, 1, 1, 4),
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1),
                    descriptor(DataType.FLOAT32, 1, 1, 1, 4));
            assertTask0064RejectedByJavaAndNative(
                    abi, convolution, wrongConvolutionCarrier,
                    new int[] {0, 1}, new int[] {2}, "convolution FLOAT64 input");

            var maximum = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.MAX_POOL2D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                            new long[] {1, 1, 1, 1, 0, 0, 1, 1, 0})));
            List<MetalMpsGraphProgram.ValueDescriptor> exactMaximum = List.of(
                    descriptor(DataType.BFLOAT16, 1, 1, 1, uint32Max),
                    descriptor(DataType.BFLOAT16, 1, 1, 1, uint32Max));
            assertTask0064AcceptedByJavaAndNative(
                    abi, maximum, exactMaximum,
                    new int[] {0}, new int[] {1}, "maximum pool exact UINT32_MAX");

            List<MetalMpsGraphProgram.ValueDescriptor> wrongMaximumShape = List.of(
                    descriptor(DataType.BFLOAT16, 1, 1, 1, 4),
                    descriptor(DataType.BFLOAT16, 1, 1, 1, 3));
            assertTask0064RejectedByJavaAndNative(
                    abi, maximum, wrongMaximumShape,
                    new int[] {0}, new int[] {1}, "maximum pool wrong output geometry");

            var convolution3d = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.CONV3D,
                            new int[] {0, 1},
                            new int[] {2},
                            MetalMpsGraphProgram.AttributeKind.CONV_3D,
                            new long[] {1, 1, 1, 0, 0, 0, 1, 1, 1, 1})));
            List<MetalMpsGraphProgram.ValueDescriptor> convolution3dValues = List.of(
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1, 1),
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1, 1),
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1, 1));
            assertTask0064AcceptedByJavaAndNative(
                    abi, convolution3d, convolution3dValues,
                    new int[] {0, 1}, new int[] {2}, "Conv3d custom-only route");

            var average2d = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.AVERAGE_POOL2D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                            new long[] {1, 1, 1, 1, 0, 0, 1, 1, 0})));
            List<MetalMpsGraphProgram.ValueDescriptor> average2dValues = List.of(
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1),
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1));
            assertTask0064AcceptedByJavaAndNative(
                    abi, average2d, average2dValues,
                    new int[] {0}, new int[] {1}, "AveragePool2d custom-only route");

            var maximum3d = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.MAX_POOL3D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.WINDOW_3D,
                            new long[] {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0})));
            List<MetalMpsGraphProgram.ValueDescriptor> maximum3dValues = List.of(
                    descriptor(DataType.BFLOAT16, 1, 1, 1, 1, 1),
                    descriptor(DataType.BFLOAT16, 1, 1, 1, 1, 1));
            assertTask0064AcceptedByJavaAndNative(
                    abi, maximum3d, maximum3dValues,
                    new int[] {0}, new int[] {1}, "MaxPool3d custom-only route");

            var average3d = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.AVERAGE_POOL3D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.WINDOW_3D,
                            new long[] {1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0})));
            List<MetalMpsGraphProgram.ValueDescriptor> average3dValues = List.of(
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1, 1),
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1, 1));
            assertTask0064AcceptedByJavaAndNative(
                    abi, average3d, average3dValues,
                    new int[] {0}, new int[] {1}, "AveragePool3d custom-only route");

            var terminalAtLimit = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.MAX_POOL2D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                            new long[] {1, 1, 1, 3, 0, 0, 1, 1, 1})));
            List<MetalMpsGraphProgram.ValueDescriptor> terminalAtLimitValues = List.of(
                    descriptor(DataType.BFLOAT16, 1, 1, 1, uint32Max),
                    descriptor(DataType.BFLOAT16, 1, 1, 1, 1_431_655_766L));
            assertTask0064AcceptedByJavaAndNative(
                    abi, terminalAtLimit, terminalAtLimitValues,
                    new int[] {0}, new int[] {1}, "ceil terminal origin exact UINT32_MAX");

            var terminalOnePast = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.MAX_POOL2D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                            new long[] {1, 1, 1, 4, 0, 0, 1, 1, 1})));
            List<MetalMpsGraphProgram.ValueDescriptor> terminalOnePastValues = List.of(
                    descriptor(DataType.BFLOAT16, 1, 1, 1, uint32Max),
                    descriptor(DataType.BFLOAT16, 1, 1, 1, 1_073_741_825L));
            assertTask0064RejectedByJavaAndNative(
                    abi, terminalOnePast, terminalOnePastValues,
                    new int[] {0}, new int[] {1}, "ceil terminal origin one past UINT32_MAX");

            var boundedPoolWork = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.MAX_POOL2D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                            new long[] {1, 65_536, 1, 1, 0, 32_767, 1, 1, 0})));
            List<MetalMpsGraphProgram.ValueDescriptor> boundedPoolWorkValues = List.of(
                    descriptor(DataType.BFLOAT16, 1, 1, 1, 2),
                    descriptor(DataType.BFLOAT16, 1, 1, 1, 1));
            assertTask0064AcceptedByJavaAndNative(
                    abi, boundedPoolWork, boundedPoolWorkValues,
                    new int[] {0}, new int[] {1}, "pool kernel-position cap");

            var amplifiedPoolWork = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.AVERAGE_POOL2D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.WINDOW_2D,
                            new long[] {1, 65_537, 1, 1, 0, 32_768, 1, 1, 0})));
            List<MetalMpsGraphProgram.ValueDescriptor> amplifiedPoolWorkValues = List.of(
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1),
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1));
            assertTask0064RejectedByJavaAndNative(
                    abi, amplifiedPoolWork,
                    amplifiedPoolWorkValues, new int[] {0}, new int[] {1},
                    "pool kernel-position cap one past");

            var amplifiedPool3dWork = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.MAX_POOL3D,
                            new int[] {0},
                            new int[] {1},
                            MetalMpsGraphProgram.AttributeKind.WINDOW_3D,
                            new long[] {
                                1, 1, 65_537, 1, 1, 1, 0, 0, 32_768, 1, 1, 1, 0
                            })));
            List<MetalMpsGraphProgram.ValueDescriptor> amplifiedPool3dWorkValues = List.of(
                    descriptor(DataType.BFLOAT16, 1, 1, 1, 1, 1),
                    descriptor(DataType.BFLOAT16, 1, 1, 1, 1, 1));
            assertTask0064RejectedByJavaAndNative(
                    abi, amplifiedPool3dWork,
                    amplifiedPool3dWorkValues, new int[] {0}, new int[] {1},
                    "Pool3d kernel-position cap one past");

        }
    }

    private static void assertTask0059WindowOriginRejected(
            RawAbi abi,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            String message) throws Throwable {
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                values,
                program,
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        byte[] customImage = program.encodedProgramImage(
                values, new int[] {0}, new int[] {1}, MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(0, abi.create(customImage, customImage.length),
                message + " native custom route");

        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        values,
                        program,
                        new int[] {0},
                        new int[] {1},
                        MetalPreparedRoute.MPSGRAPH),
                message + " Java MPSGraph route");
        byte[] mpsGraphImage = program.encodedProgramImage(
                values, new int[] {0}, new int[] {1}, MetalPreparedRoute.MPSGRAPH);
        assertEquals(
        13, abi.create(mpsGraphImage, mpsGraphImage.length),
                message + " native MPSGraph route");
    }

    private static void assertTask0064AcceptedByJavaAndNative(
            RawAbi abi,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            String message) throws Throwable {
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                values, program, feeds, targets, MetalPreparedRoute.CUSTOM_PROGRAM);
        byte[] image = program.encodedProgramImage(
                values, feeds, targets, MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(0, abi.create(image, image.length), message);
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        values, program, feeds, targets, MetalPreparedRoute.MPSGRAPH),
                message + " Java MPSGraph route");
        byte[] mpsGraphImage = program.encodedProgramImage(
                values, feeds, targets, MetalPreparedRoute.MPSGRAPH);
        assertNotEquals(0, abi.create(mpsGraphImage, mpsGraphImage.length),
                message + " native MPSGraph route");
    }

    private static void assertTask0064RejectedByJavaAndNative(
            RawAbi abi,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            String message) throws Throwable {
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        values, program, feeds, targets,
                        MetalPreparedRoute.CUSTOM_PROGRAM),
                message);
        byte[] image = program.encodedProgramImage(
                values, feeds, targets, MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(1, abi.create(image, image.length), message);
    }

    @Test
    void task0065JavaAndNativePreflightAgreeOnZeroFeedsRolesAndUint32SpanBoundaries()
            throws Throwable {
        Path library = configuredLibrary();
        try (RawAbi abi = new RawAbi(library)) {
            var initial = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.INITIAL_STATE,
                            new int[0],
                            new int[] {0},
                            MetalMpsGraphProgram.AttributeKind.GRAPH_RNG_STATE,
                            new long[] {-1L, Long.MIN_VALUE})));
            List<MetalMpsGraphProgram.ValueDescriptor> initialValues =
                    List.of(descriptor(DataType.INT64, 2));
            MetalNativeApi.ProgramExecutableAbi.validateCreate(
                    initialValues,
                    initial,
                    new int[0],
                    new int[] {0},
                    MetalPreparedRoute.CUSTOM_PROGRAM);

            byte[] initialImage = initial.encodedProgramImage(
                    initialValues,
                    new int[0],
                    new int[] {0},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            assertEquals(0, abi.create(initialImage, initialImage.length), "zero-feed initializer");
            assertThrows(
                    IllegalArgumentException.class,
                    () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                            initialValues,
                            initial,
                            new int[0],
                            new int[] {0},
                            MetalPreparedRoute.MPSGRAPH));

            var dropout = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.DROPOUT,
                            new int[] {0, 1},
                            new int[] {2, 3, 4},
                            MetalMpsGraphProgram.AttributeKind.DROPOUT,
                            new long[] {Double.doubleToRawLongBits(0.5d)})));
            List<MetalMpsGraphProgram.ValueDescriptor> exactMaximum = List.of(
                    descriptor(DataType.FLOAT32, 0xffff_ffffL),
                    descriptor(DataType.INT64, 2),
                    descriptor(DataType.FLOAT32, 0xffff_ffffL),
                    descriptor(DataType.BOOL, 0xffff_ffffL),
                    descriptor(DataType.INT64, 2));
            assertTask0065AcceptedByJavaAndNative(
                    abi, dropout, exactMaximum, "exact UINT32_MAX dropout");
            assertThrows(
                    IllegalArgumentException.class,
                    () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                            exactMaximum,
                            dropout,
                            new int[] {0, 1},
                            new int[] {2},
                            MetalPreparedRoute.MPSGRAPH),
                    "dropout has no direct MPSGraph route");

            List<MetalMpsGraphProgram.ValueDescriptor> extentOver = List.of(
                    descriptor(DataType.FLOAT32, 0x1_0000_0000L),
                    descriptor(DataType.INT64, 2),
                    descriptor(DataType.FLOAT32, 0x1_0000_0000L),
                    descriptor(DataType.BOOL, 0x1_0000_0000L),
                    descriptor(DataType.INT64, 2));
            assertTask0065RejectedByJavaAndNative(
                    abi, dropout, extentOver, "one-past UINT32 extent");
            List<MetalMpsGraphProgram.ValueDescriptor> productOver = List.of(
                    descriptor(DataType.FLOAT32, 65_536, 65_536),
                    descriptor(DataType.INT64, 2),
                    descriptor(DataType.FLOAT32, 65_536, 65_536),
                    descriptor(DataType.BOOL, 65_536, 65_536),
                    descriptor(DataType.INT64, 2));
            assertTask0065RejectedByJavaAndNative(
                    abi, dropout, productOver, "one-past UINT32 element count");

            Shape one = Shape.of(1);
            var onePastSpanInput = new MetalMpsGraphProgram.ValueDescriptor(
                    DataType.FLOAT32,
                    one.toLongArray(),
                    java.util.Optional.of(LayoutDescriptor.of(
                            one, new long[] {1L}, 0xffff_ffffL, true)),
                    false,
                    false);
            List<MetalMpsGraphProgram.ValueDescriptor> spanOver = List.of(
                    onePastSpanInput,
                    descriptor(DataType.INT64, 2),
                    descriptor(DataType.FLOAT32, 1),
                    descriptor(DataType.BOOL, 1),
                    descriptor(DataType.INT64, 2));
            assertTask0065RejectedByJavaAndNative(
                    abi, dropout, spanOver, "one-past UINT32 referenced span");

            List<MetalMpsGraphProgram.ValueDescriptor> small = List.of(
                    descriptor(DataType.FLOAT32, 4),
                    descriptor(DataType.INT64, 2),
                    descriptor(DataType.FLOAT32, 4),
                    descriptor(DataType.BOOL, 4),
                    descriptor(DataType.INT64, 2));
            byte[] valid = dropout.encodedProgramImage(
                    small,
                    new int[] {0, 1},
                    new int[] {2},
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            assertEquals(0, abi.create(valid, valid.length));
            int stateFlags = MetalMpsGraphProgram.HEADER_BYTES
                    + MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES + 16;
            int outputFlags = MetalMpsGraphProgram.HEADER_BYTES
                    + 2 * MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES + 16;
            int maskFlags = MetalMpsGraphProgram.HEADER_BYTES
                    + 3 * MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES + 16;
            assertEquals(1, abi.create(
                    rewriteInt(valid, stateFlags, readInt(valid, stateFlags) | 1),
                    valid.length), "state gradient");
            assertEquals(1, abi.create(
                    rewriteInt(valid, outputFlags, readInt(valid, outputFlags) | 1),
                    valid.length), "primary gradient mismatch");
            assertEquals(1, abi.create(
                    rewriteInt(valid, maskFlags, readInt(valid, maskFlags) | 1),
                    valid.length), "mask gradient");
            assertEquals(1, abi.create(
                    rewriteLong(valid, valid.length - Long.BYTES,
                            Double.doubleToRawLongBits(1.0d)),
                    valid.length), "invalid probability");
        }
    }

    private static void assertTask0065AcceptedByJavaAndNative(
            RawAbi abi,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            String message) throws Throwable {
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                values,
                program,
                new int[] {0, 1},
                new int[] {2},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        byte[] image = program.encodedProgramImage(
                values,
                new int[] {0, 1},
                new int[] {2},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(0, abi.create(image, image.length), message);
    }

    private static void assertTask0065RejectedByJavaAndNative(
            RawAbi abi,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            String message) throws Throwable {
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        values,
                        program,
                        new int[] {0, 1},
                        new int[] {2},
                        MetalPreparedRoute.CUSTOM_PROGRAM),
                message);
        byte[] image = program.encodedProgramImage(
                values,
                new int[] {0, 1},
                new int[] {2},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(1, abi.create(image, image.length), message);
    }

    private static Task0063Case task0063Case(
            MetalMpsGraphProgram.NodeKind kind, long... inputDimensions) {
        int axis = inputDimensions.length - 1;
        MetalMpsGraphProgram.Node node;
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>();
        values.add(descriptor(DataType.INT32, inputDimensions));
        switch (kind) {
            case SORT -> {
                node = MetalMpsGraphProgram.Node.generic(
                        kind,
                        new int[] {0},
                        new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.SORT,
                        new long[] {axis, 0});
                values.add(descriptor(DataType.INT32, inputDimensions));
            }
            case ARGSORT -> {
                node = MetalMpsGraphProgram.Node.generic(
                        kind,
                        new int[] {0},
                        new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.SORT,
                        new long[] {axis, 0});

                values.add(descriptor(DataType.INT64, inputDimensions));
            }
            case TOP_K -> {
                node = MetalMpsGraphProgram.Node.generic(
                        kind,
                        new int[] {0},
                        new int[] {1, 2},
                        MetalMpsGraphProgram.AttributeKind.TOP_K,
                        new long[] {axis, 1, 1, 1});
                long[] outputDimensions = inputDimensions.clone();
                outputDimensions[axis] = 1L;
                values.add(descriptor(DataType.INT32, outputDimensions));
                values.add(descriptor(DataType.INT64, outputDimensions));
            }
            case ARG_MAX, ARG_MIN -> {
                node = MetalMpsGraphProgram.Node.generic(
                        kind,
                        new int[] {0},
                        new int[] {1},
                        MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA,
                        new long[] {axis, 0, kind == MetalMpsGraphProgram.NodeKind.ARG_MAX ? 1 : 2});
                values.add(descriptor(
                        DataType.INT64,
                        Arrays.copyOf(inputDimensions, inputDimensions.length - 1)));
            }
            default -> throw new IllegalArgumentException("not a Task0063 kind: " + kind);
        }
        return new Task0063Case(
                new MetalMpsGraphProgram(List.of(node)),
                List.copyOf(values),
                new int[] {1});
    }

    private static MetalMpsGraphProgram task0069L1Program(
            boolean keepDimensions,
            MetalMpsGraphProgram.ReductionForm form,
            List<Integer> axes) {
        return new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.reduction(
                MetalMpsGraphProgram.NodeKind.L1_NORM,
                0,
                1,
                form,
                axes,
                keepDimensions)));
    }

    private static MetalMpsGraphProgram task0069VarianceProgram(
            boolean keepDimensions, List<Integer> axes, long correction) {
        return new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.statisticalReduction(
                        MetalMpsGraphProgram.NodeKind.VARIANCE,
                        0,
                        1,
                        axes,
                        keepDimensions,
                        correction)));
    }

    private static void assertTask0063RejectedByJavaAndNative(
            RawAbi abi, Task0063Case fixture, String message) throws Throwable {
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                        fixture.values(),
                        fixture.program(),
                        new int[] {0},
                        fixture.targets(),
                        MetalPreparedRoute.CUSTOM_PROGRAM),
                message);
        byte[] image = fixture.program().encodedProgramImage(
                fixture.values(),
                new int[] {0},
                fixture.targets(),
                MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(1, abi.create(image, image.length), message);
    }

    private static byte[] validNegImage() {
        var program = new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.neg(0, 1)));
        return program.encodedProgramImage(
                List.of(descriptor(4), descriptor(4)), new int[] {0}, new int[] {1});
    }

    private static byte[] validSelectStorageLayoutImage() {
        var inputShape = io.github.pho001.synaptik.model.shape.Shape.of(2, 3);
        var outputShape = io.github.pho001.synaptik.model.shape.Shape.of(3);
        var input = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32,
                inputShape.toLongArray(),
                java.util.Optional.of(
                        io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(
                                inputShape, new long[] {4, 1}, 1L, true)),
                false,
                false);
        var output = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT32,
                outputShape.toLongArray(),
                java.util.Optional.of(
                        io.github.pho001.synaptik.model.layout.LayoutDescriptor.of(
                                outputShape, new long[] {1}, 5L, true)),
                false,
                false);
        var select = MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.SELECT,
                new int[] {0},
                new int[] {1},
                MetalMpsGraphProgram.AttributeKind.SELECT,
                new long[] {0, 1});
        return new MetalMpsGraphProgram(List.of(select)).encodedProgramImage(
                List.of(input, output),
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.CUSTOM_PROGRAM);
    }

    private static byte[] unaryImage(MetalMpsGraphProgram.NodeKind kind) {
        var node = MetalMpsGraphProgram.Node.generic(
                kind, new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.NONE, new long[0]);
        return new MetalMpsGraphProgram(List.of(node)).encodedProgramImage(
                List.of(descriptor(4), descriptor(4)), new int[] {0}, new int[] {1});
    }
    private static MetalMpsGraphProgram.Node unaryNode(
            MetalMpsGraphProgram.NodeKind kind, int input, int output) {
        return MetalMpsGraphProgram.Node.generic(
                kind,
                new int[] {input},
                new int[] {output},
                MetalMpsGraphProgram.AttributeKind.NONE,
                new long[0]);
    }

    private static int extensionCoreBytes(byte[] image) {
        return MetalMpsGraphProgram.HEADER_BYTES
                + readInt(image, 28) * MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES
                + readInt(image, 32) * MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES
                + readInt(image, 44) * Long.BYTES
                + readInt(image, 48) * Long.BYTES
                + readInt(image, 52) * Integer.BYTES
                + readInt(image, 56) * Long.BYTES;
    }
    private static byte[] appendOutOfRangeMemberStep(byte[] image) {
        int stepCount = readInt(image, 60);
        int memberCount = readInt(image, 64);
        int bindingCount = readInt(image, 68);
        int instructionCount = readInt(image, 76);
        int memberOffset =
                extensionCoreBytes(image) + stepCount * MetalMpsGraphProgram.STEP_DESCRIPTOR_BYTES;
        int bindingOffset = memberOffset + memberCount * Integer.BYTES;
        byte[] result =
                new byte[image.length + MetalMpsGraphProgram.STEP_DESCRIPTOR_BYTES + Integer.BYTES];
        System.arraycopy(image, 0, result, 0, memberOffset);
        ByteBuffer out = ByteBuffer.wrap(result).order(ByteOrder.LITTLE_ENDIAN);
        out.position(memberOffset)
                .putInt(1)
                .putInt(memberCount)
                .putInt(1)
                .putInt(bindingCount)
                .putInt(0)
                .putInt(instructionCount)
                .putInt(0)
                .putInt(0)
                .putInt(0)
                .putInt(0);
        int expandedMemberOffset =
                memberOffset + MetalMpsGraphProgram.STEP_DESCRIPTOR_BYTES;
        System.arraycopy(
                image,
                memberOffset,
                result,
                expandedMemberOffset,
                memberCount * Integer.BYTES);
        out.putInt(expandedMemberOffset + memberCount * Integer.BYTES, readInt(image, 32));
        int expandedBindingOffset =
                expandedMemberOffset + (memberCount + 1) * Integer.BYTES;
        System.arraycopy(
                image,
                bindingOffset,
                result,
                expandedBindingOffset,
                image.length - bindingOffset);
        out.putInt(12, result.length);
        out.putInt(60, stepCount + 1);
        out.putInt(64, memberCount + 1);
        return result;
    }


    /**
     * Replaces one manifest fragment and rebuilds its SHA-256 digest and image length.
     *
     * @param image schema-20 custom-program image with a canonical manifest
     * @param expected fragment required to occur exactly once
     * @param replacement ASCII replacement fragment
     * @return a new image with the replacement manifest and matching digest
     * @throws IllegalArgumentException if the fragment is absent or repeated
     */
    private static byte[] rewriteManifest(byte[] image, String expected, String replacement) {
        int manifestOffset = extensionManifestOffset(image);
        int manifestBytes = readInt(image, 80);
        String manifest = new String(
                image, manifestOffset, manifestBytes, StandardCharsets.US_ASCII);
        int occurrence = manifest.indexOf(expected);
        if (occurrence < 0 || manifest.indexOf(expected, occurrence + 1) >= 0) {
            throw new IllegalArgumentException("manifest fragment must occur exactly once");
        }
        byte[] replacementManifest =
                manifest.replace(expected, replacement).getBytes(StandardCharsets.US_ASCII);
        byte[] result = new byte[manifestOffset + replacementManifest.length + 32];
        System.arraycopy(image, 0, result, 0, manifestOffset);
        System.arraycopy(
                replacementManifest,
                0,
                result,
                manifestOffset,
                replacementManifest.length);
        byte[] digest;
        try {
            digest = MessageDigest.getInstance("SHA-256").digest(replacementManifest);
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException("SHA-256 unavailable", exception);
        }
        System.arraycopy(
                digest,
                0,
                result,
                manifestOffset + replacementManifest.length,
                digest.length);
        ByteBuffer header = ByteBuffer.wrap(result).order(ByteOrder.LITTLE_ENDIAN);
        header.putInt(12, result.length);
        header.putInt(80, replacementManifest.length);
        return result;
    }

    /**
     * Locates the manifest after all counted custom-program extension records.
     *
     * @param image schema-20 custom-program image with complete extension counts
     * @return byte offset of the manifest in {@code image}
     */
    private static int extensionManifestOffset(byte[] image) {
        return extensionCoreBytes(image)
                + readInt(image, 60) * MetalMpsGraphProgram.STEP_DESCRIPTOR_BYTES
                + readInt(image, 64) * Integer.BYTES
                + readInt(image, 68) * MetalMpsGraphProgram.BINDING_DESCRIPTOR_BYTES
                + readInt(image, 72) * Integer.BYTES
                + readInt(image, 76) * MetalMpsGraphProgram.INSTRUCTION_DESCRIPTOR_BYTES;
    }


    private static MetalMpsGraphProgram.ValueDescriptor descriptor(long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, dimensions, false);
    }

    private static MetalMpsGraphProgram.ValueDescriptor descriptor(
            DataType dataType, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(dataType, dimensions, false);
    }

    private static MetalMpsGraphProgram.Node matmulNode(int left, int right, int output) {
        return MetalMpsGraphProgram.Node.matmul(left, right, output);
    }

    private static int readInt(byte[] source, int offset) {
        return ByteBuffer.wrap(source).order(ByteOrder.LITTLE_ENDIAN).getInt(offset);
    }

    private static ScalarCase scalarCase(MetalMpsGraphProgram.NodeKind kind) {
        DataType[] inputTypes;
        DataType outputType;
        switch (kind) {
            case IS_FINITE, IS_NAN, IS_INF -> {
                inputTypes = new DataType[] {DataType.FLOAT32};
                outputType = DataType.BOOL;
            }
            case LOGICAL_NOT -> {
                inputTypes = new DataType[] {DataType.BOOL};
                outputType = DataType.BOOL;
            }
            case LOGICAL_AND, LOGICAL_OR -> {
                inputTypes = new DataType[] {DataType.BOOL, DataType.BOOL};
                outputType = DataType.BOOL;
            }
            case WHERE -> {
                inputTypes =
                        new DataType[] {DataType.BOOL, DataType.FLOAT32, DataType.FLOAT32};
                outputType = DataType.FLOAT32;
            }
            default -> throw new IllegalArgumentException("not an exact BOOL operation: " + kind);
        }
        var values = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>(inputTypes.length + 1);
        for (DataType inputType : inputTypes) values.add(scalarDescriptor(inputType));
        values.add(scalarDescriptor(outputType));
        int[] feeds = java.util.stream.IntStream.range(0, inputTypes.length).toArray();
        var node = MetalMpsGraphProgram.Node.generic(
                kind,
                feeds,
                new int[] {inputTypes.length},
                MetalMpsGraphProgram.AttributeKind.NONE,
                new long[0]);
        return new ScalarCase(
                new MetalMpsGraphProgram(List.of(node)),
                List.copyOf(values),
                feeds,
                new int[] {inputTypes.length});
    }

    private static MetalMpsGraphProgram.ValueDescriptor scalarDescriptor(DataType dataType) {
        return new MetalMpsGraphProgram.ValueDescriptor(dataType, new long[0], false);
    }

    private record ScalarCase(
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets) {}

    private record Task0063Case(
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] targets) {
        private Task0063Case {
            targets = targets.clone();
        }

        @Override
        public int[] targets() {
            return targets.clone();
        }
    }

    private static int nodeOffset(int valueCount) {
        return MetalMpsGraphProgram.HEADER_BYTES
                + valueCount * MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES;
    }

    private static byte[] rewriteInt(byte[] source, int offset, int value) {
        byte[] result = source.clone();
        ByteBuffer.wrap(result).order(ByteOrder.LITTLE_ENDIAN).putInt(offset, value);
        return result;
    }


    private static byte[] rewriteLong(byte[] source, int offset, long value) {
        byte[] result = source.clone();
        ByteBuffer.wrap(result).order(ByteOrder.LITTLE_ENDIAN).putLong(offset, value);
        return result;
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        return Path.of(configured).toAbsolutePath().normalize();
    }

    private static final class RawAbi implements AutoCloseable {
        private final Arena arena = Arena.ofShared();
        private final MetalNativeApi api;
        private final MetalNativeApi.Handle context;
        private final MethodHandle create;

        private RawAbi(Path library) {
            api = MetalNativeApi.open(library);
            context = api.createContext();
            MemorySegment symbol = SymbolLookup.libraryLookup(library, arena)
                    .find("synaptik_metal_mpsgraph_executable_create")
                    .orElseThrow();
            create = Linker.nativeLinker().downcallHandle(symbol,
                    FunctionDescriptor.of(JAVA_INT, ADDRESS, ADDRESS, JAVA_INT, ADDRESS));
        }

        private int create(byte[] image, int byteCount) throws Throwable {
            return create(image, byteCount, 0);
        }

        private int createUnaligned(byte[] image) throws Throwable {
            return create(image, image.length, 1);
        }

        private int create(byte[] image, int byteCount, int leadingBytes) throws Throwable {
            MemorySegment allocation = arena.allocate(image.length + leadingBytes, Long.BYTES);
            MemorySegment program = allocation.asSlice(leadingBytes, image.length);
            MemorySegment.copy(MemorySegment.ofArray(image), 0, program, 0, image.length);
            MemorySegment output = arena.allocate(ADDRESS);
            output.set(ADDRESS, 0, MemorySegment.NULL);
            int status = (int) create.invokeExact(
                    context.carrier(), program, byteCount, output);
            MemorySegment carrier = output.get(ADDRESS, 0);
            if (status != 0) {
                assertEquals(0L, carrier.address(),
                        "rejected program must not publish a native executable");
            }
            if (carrier.address() != 0L) {
                api.releaseExecutable(new MetalNativeApi.Handle(carrier));
            }
            return status;
        }

        @Override
        public void close() {
            api.releaseContext(context);
            api.close();
            arena.close();
        }
    }
}
