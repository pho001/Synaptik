package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalMpsGraphRawAbiNativeTest {
    @Test
    void nativeAbiFiveAcceptsCanonicalSchemaFifteenAndRejectsMalformedImages() throws Throwable {
        Path library = configuredLibrary();
        try (RawAbi abi = new RawAbi(library)) {
            byte[] valid = validNegImage();
            assertEquals(0, abi.create(valid, valid.length));

            assertEquals(1, abi.create(rewriteInt(valid, 0, 0), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 4, 12), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 4, 13), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 4, 14), valid.length));
            assertEquals(1, abi.create(
                    rewriteInt(rewriteInt(valid, 4, 13), 40, 0), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 8, valid.length - 1), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 40, 0), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 40, 1), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 40, 4), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 44, 1), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, nodeOffset(2), 116), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, nodeOffset(2) + 4, 3), valid.length));
            assertEquals(1, abi.create(valid, valid.length - 1));
            assertEquals(1, abi.create(valid, Integer.MIN_VALUE));
            assertEquals(1, abi.create(
                    rewriteInt(valid, 8, Integer.MIN_VALUE), Integer.MIN_VALUE));
        }
    }

    @Test
    void nativeAbiFiveRejectsMalformedStorageLayoutDescriptors() throws Throwable {
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
    void javaAndNativeSchemaValidatorsRejectRankZeroExactBoolPrograms() throws Throwable {
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
                for (MetalPreparedRoute route :
                        List.of(MetalPreparedRoute.CUSTOM_PROGRAM, MetalPreparedRoute.MPSGRAPH)) {
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                                    NumericalProfile.STRICT_IEEE,
                                    scalar.values(),
                                    scalar.program(),
                                    scalar.feeds(),
                                    scalar.targets(),
                                    route),
                            kind + " " + route + " Java preflight");
                    byte[] image = scalar.program().encodedProgramImage(
                            scalar.values(), scalar.feeds(), scalar.targets(), route);
                    assertEquals(1, abi.create(image, image.length),
                            kind + " " + route + " native decode");
                }
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
                    MetalMpsGraphProgram.NodeKind.SIGN,
                    MetalMpsGraphProgram.NodeKind.RELU)) {
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
                            () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                                    NumericalProfile.STRICT_IEEE,
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
                        () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                                NumericalProfile.ACCELERATOR,
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
    void task0059JavaAndNativeRejectMalformedCastTypeAndAttributeImages() throws Throwable {
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
        try (RawAbi abi = new RawAbi(library)) {
            for (MetalPreparedRoute route :
                    List.of(MetalPreparedRoute.CUSTOM_PROGRAM, MetalPreparedRoute.MPSGRAPH)) {
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.STRICT_IEEE,
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
                        () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                                NumericalProfile.STRICT_IEEE,
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
            }
        }
    }

    @Test
    void task0059JavaAndNativeRejectIntegerImageUnfoldCarriers() throws Throwable {
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
                        () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                                NumericalProfile.STRICT_IEEE,
                                integers,
                                program,
                                new int[] {0},
                                new int[] {1},
                                route));
                byte[] image = program.encodedProgramImage(
                        integers, new int[] {0}, new int[] {1}, route);
                assertEquals(1, abi.create(image, image.length));
            }
        }
    }

    @Test
    void task0059JavaAndNativeKeepWideCustomOriginsButRejectMpsGraph() throws Throwable {
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
    void javaAndNativeRejectGradOrScalarShapeForAdmittedScalarRecipes() throws Throwable {
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
                assertThrows(
                        IllegalArgumentException.class,
                        () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                                NumericalProfile.ACCELERATOR,
                                grad,
                                program,
                                new int[] {0},
                                new int[] {1},
                                MetalPreparedRoute.MPSGRAPH));
                byte[] gradImage = program.encodedProgramImage(
                        grad, new int[] {0}, new int[] {1}, MetalPreparedRoute.MPSGRAPH);
                assertEquals(1, abi.create(gradImage, gradImage.length), kind + " grad");

                List<MetalMpsGraphProgram.ValueDescriptor> rankZero =
                        List.of(scalarDescriptor(DataType.FLOAT32), scalarDescriptor(DataType.FLOAT32));
                assertThrows(
                        IllegalArgumentException.class,
                        () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                                NumericalProfile.ACCELERATOR,
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
    void nativeAbiFiveReturnsDedicatedStatusForStructurallyValidUnsupportedOperation()
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
            for (MetalMpsGraphProgram.NodeKind kind : List.of(
                    MetalMpsGraphProgram.NodeKind.EXP,
                    MetalMpsGraphProgram.NodeKind.SIGMOID)) {
                byte[] blocked = unaryImage(kind);
                assertEquals(13, abi.create(blocked, blocked.length));
            }
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
    void task0061AuthenticatesTransposeStorageAndMatmulGradientMetadataInBothPreflights()
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
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.ACCELERATOR,
                values,
                program,
                new int[] {0, 2},
                new int[] {3},
                MetalPreparedRoute.MPSGRAPH);
        byte[] valid = program.encodedProgramImage(
                values, new int[] {0, 2}, new int[] {3}, MetalPreparedRoute.MPSGRAPH);

        int transposeDescriptor =
                MetalMpsGraphProgram.HEADER_BYTES + MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES;
        int dimensionCount = readInt(valid, 20);
        int referenceCount = readInt(valid, 24);
        int strideCount = readInt(valid, 44);
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
                    MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                            NumericalProfile.ACCELERATOR,
                            List.of(source, forged, right, output),
                            program,
                            new int[] {0, 2},
                            new int[] {3},
                            MetalPreparedRoute.MPSGRAPH));
        }
        var direct = new MetalMpsGraphProgram(List.of(matmulNode(0, 1, 2)));
        assertThrows(IllegalArgumentException.class, () ->
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
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
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.ACCELERATOR,
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
                        () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                                NumericalProfile.ACCELERATOR,
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

            int dimensionCount = readInt(validImage, 20);
            int referenceCount = readInt(validImage, 24);
            int strideCount = readInt(validImage, 44);
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
                    () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                            NumericalProfile.ACCELERATOR,
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
                MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.STRICT_IEEE,
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
                        () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                                NumericalProfile.STRICT_IEEE,
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
                    abi, NumericalProfile.ACCELERATOR, convolution, exactConvolution,
                    new int[] {0, 1}, new int[] {2}, "convolution exact UINT32_MAX");

            List<MetalMpsGraphProgram.ValueDescriptor> overLimitConvolution = List.of(
                    descriptor(DataType.FLOAT32, 1, 1, 1, uint32Max + 1L),
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1),
                    descriptor(DataType.FLOAT32, 1, 1, 1, uint32Max + 1L));
            assertTask0064RejectedByJavaAndNative(
                    abi, NumericalProfile.ACCELERATOR, convolution, overLimitConvolution,
                    new int[] {0, 1}, new int[] {2}, "convolution one past UINT32_MAX");

            List<MetalMpsGraphProgram.ValueDescriptor> wrongConvolutionCarrier = List.of(
                    descriptor(DataType.FLOAT64, 1, 1, 1, 4),
                    descriptor(DataType.FLOAT32, 1, 1, 1, 1),
                    descriptor(DataType.FLOAT32, 1, 1, 1, 4));
            assertTask0064RejectedByJavaAndNative(
                    abi, NumericalProfile.ACCELERATOR, convolution, wrongConvolutionCarrier,
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
                    abi, NumericalProfile.STRICT_IEEE, maximum, exactMaximum,
                    new int[] {0}, new int[] {1}, "maximum pool exact UINT32_MAX");

            List<MetalMpsGraphProgram.ValueDescriptor> wrongMaximumShape = List.of(
                    descriptor(DataType.BFLOAT16, 1, 1, 1, 4),
                    descriptor(DataType.BFLOAT16, 1, 1, 1, 3));
            assertTask0064RejectedByJavaAndNative(
                    abi, NumericalProfile.STRICT_IEEE, maximum, wrongMaximumShape,
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
                    abi, NumericalProfile.ACCELERATOR, convolution3d, convolution3dValues,
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
                    abi, NumericalProfile.ACCELERATOR, average2d, average2dValues,
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
                    abi, NumericalProfile.STRICT_IEEE, maximum3d, maximum3dValues,
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
                    abi, NumericalProfile.ACCELERATOR, average3d, average3dValues,
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
                    abi, NumericalProfile.STRICT_IEEE, terminalAtLimit, terminalAtLimitValues,
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
                    abi, NumericalProfile.STRICT_IEEE, terminalOnePast, terminalOnePastValues,
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
                    abi, NumericalProfile.STRICT_IEEE, boundedPoolWork, boundedPoolWorkValues,
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
                    abi, NumericalProfile.ACCELERATOR, amplifiedPoolWork,
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
                    abi, NumericalProfile.STRICT_IEEE, amplifiedPool3dWork,
                    amplifiedPool3dWorkValues, new int[] {0}, new int[] {1},
                    "Pool3d kernel-position cap one past");

        }
    }

    private static void assertTask0059WindowOriginRejected(
            RawAbi abi,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            String message) throws Throwable {
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.STRICT_IEEE,
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
                () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.STRICT_IEEE,
                        values,
                        program,
                        new int[] {0},
                        new int[] {1},
                        MetalPreparedRoute.MPSGRAPH),
                message + " Java MPSGraph route");
        byte[] mpsGraphImage = program.encodedProgramImage(
                values, new int[] {0}, new int[] {1}, MetalPreparedRoute.MPSGRAPH);
        assertEquals(1, abi.create(mpsGraphImage, mpsGraphImage.length),
                message + " native MPSGraph route");
    }

    private static void assertTask0064AcceptedByJavaAndNative(
            RawAbi abi,
            NumericalProfile profile,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            String message) throws Throwable {
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                profile, values, program, feeds, targets, MetalPreparedRoute.CUSTOM_PROGRAM);
        byte[] image = program.encodedProgramImage(
                values, feeds, targets, MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(0, abi.create(image, image.length), message);
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        profile, values, program, feeds, targets, MetalPreparedRoute.MPSGRAPH),
                message + " Java MPSGraph route");
        byte[] mpsGraphImage = program.encodedProgramImage(
                values, feeds, targets, MetalPreparedRoute.MPSGRAPH);
        assertNotEquals(0, abi.create(mpsGraphImage, mpsGraphImage.length),
                message + " native MPSGraph route");
    }

    private static void assertTask0064RejectedByJavaAndNative(
            RawAbi abi,
            NumericalProfile profile,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            String message) throws Throwable {
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        profile, values, program, feeds, targets,
                        MetalPreparedRoute.CUSTOM_PROGRAM),
                message);
        byte[] image = program.encodedProgramImage(
                values, feeds, targets, MetalPreparedRoute.CUSTOM_PROGRAM);
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

    private static void assertTask0063RejectedByJavaAndNative(
            RawAbi abi, Task0063Case fixture, String message) throws Throwable {
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.STRICT_IEEE,
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
            MemorySegment program = arena.allocate(image.length, Long.BYTES);
            MemorySegment.copy(MemorySegment.ofArray(image), 0, program, 0, image.length);
            MemorySegment output = arena.allocate(ADDRESS);
            output.set(ADDRESS, 0, MemorySegment.NULL);
            int status = (int) create.invokeExact(
                    context.carrier(), program, byteCount, output);
            MemorySegment carrier = output.get(ADDRESS, 0);
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
