package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static org.junit.jupiter.api.Assertions.assertEquals;
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
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalMpsGraphRawAbiNativeTest {
    @Test
    void nativeAbiFiveAcceptsCanonicalSchemaFourteenAndRejectsMalformedImages() throws Throwable {
        Path library = configuredLibrary();
        try (RawAbi abi = new RawAbi(library)) {
            byte[] valid = validNegImage();
            assertEquals(0, abi.create(valid, valid.length));

            assertEquals(1, abi.create(rewriteInt(valid, 0, 0), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 4, 12), valid.length));
            assertEquals(1, abi.create(rewriteInt(valid, 4, 13), valid.length));
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
                    + 8 * Long.BYTES;
            assertEquals(1, abi.create(
                    rewriteInt(image, referencesOffset + 4 * Integer.BYTES, 3), image.length));
            assertEquals(1, abi.create(
                    rewriteInt(image, referencesOffset + 7 * Integer.BYTES, 0), image.length));
        }
    }

    private static byte[] validNegImage() {
        var program = new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.neg(0, 1)));
        return program.encodedProgramImage(
                List.of(descriptor(4), descriptor(4)), new int[] {0}, new int[] {1});
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
