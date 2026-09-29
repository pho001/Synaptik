package io.github.pho001.synaptik.backend.metal;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.Float16Bits;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.PartitionDag;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import io.github.pho001.synaptik.runtime.resource.BufferRepresentation;
import io.github.pho001.synaptik.trace.TraceEvent;
import io.github.pho001.synaptik.trace.TracePayload;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionAccuracy;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionCertificate;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionCertificateKey;
import io.github.pho001.synaptik.trace.certificate.LowPrecisionDeterminism;
import io.github.pho001.synaptik.trace.payload.LowPrecisionTraceMetadata;
import io.github.pho001.synaptik.trace.payload.TraceRouteKind;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class MetalFloat16ProtocolTest {
    @Test
    void schemaTraceAndPreflightKeepArithmeticCustomAndAllowRawMovementGraph() {
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.neg(0, 1)));
        var float16 = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.FLOAT16, new long[] {3}, false);
        List<MetalMpsGraphProgram.ValueDescriptor> float16Values =
                List.of(float16, float16);
        byte[] image = program.encodedProgramImage(
                NumericalProfile.STRICT_IEEE,
                float16Values,
                new int[] {0},
                new int[] {1});
        ByteBuffer header = ByteBuffer.wrap(image).order(ByteOrder.LITTLE_ENDIAN);

        assertEquals(0x39314d53, header.getInt(0));
        assertEquals(19, header.getInt(Integer.BYTES));
        assertEquals(7, header.getInt(MetalMpsGraphProgram.HEADER_BYTES));
        assertEquals(7, header.getInt(
                MetalMpsGraphProgram.HEADER_BYTES
                        + MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES));
        assertEquals(3L, float16.elementCount());
        assertEquals(6L, float16.byteCount());
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.STRICT_IEEE,
                float16Values,
                program,
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.STRICT_IEEE,
                        float16Values,
                        program,
                        new int[] {0},
                        new int[] {1},
                        MetalPreparedRoute.MPSGRAPH));
        var movementProgram = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.contiguous(0, 1)));
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.STRICT_IEEE,
                float16Values,
                movementProgram,
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                NumericalProfile.STRICT_IEEE,
                float16Values,
                movementProgram,
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.MPSGRAPH);
        var wrongScalarType = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.scalarValue(
                        MetalMpsGraphProgram.NodeKind.SCALAR_ADD,
                        0,
                        1,
                        DataType.BFLOAT16,
                        lowWord(DataType.BFLOAT16, 1.0f))));
        assertThrows(
                IllegalArgumentException.class,
                () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                        NumericalProfile.ACCELERATOR,
                        float16Values,
                        wrongScalarType,
                        new int[] {0},
                        new int[] {1},
                        MetalPreparedRoute.CUSTOM_PROGRAM));
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    var splitClamp = new MetalMpsGraphProgram(List.of(
                            MetalMpsGraphProgram.Node.generic(
                                    MetalMpsGraphProgram.NodeKind.CLAMP,
                                    new int[] {0},
                                    new int[] {1},
                                    MetalMpsGraphProgram.AttributeKind.CLAMP_RANGE,
                                    new long[] {
                                        MetalMpsGraphProgram.dataTypeWire(DataType.FLOAT16),
                                        lowWord(DataType.FLOAT16, -1.0f),
                                        MetalMpsGraphProgram.dataTypeWire(DataType.BFLOAT16),
                                        lowWord(DataType.BFLOAT16, 1.0f)
                                    })));
                    MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                            NumericalProfile.ACCELERATOR,
                            float16Values,
                            splitClamp,
                            new int[] {0},
                            new int[] {1},
                            MetalPreparedRoute.CUSTOM_PROGRAM);
                });

        TensorDescriptor descriptor = descriptor(DataType.FLOAT16, Shape.of(3));
        TensorDescriptor reshaped = new TensorDescriptor(
                DataType.FLOAT16,
                Shape.of(3),
                Optional.of(LayoutDescriptor.of(Shape.of(3), new long[] {1}, 0L, true)),
                false);
        var provider = new MetalCapabilityProvider();
        for (NumericalProfile profile : NumericalProfile.values()) {
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    profile,
                    new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                    List.of(descriptor),
                    List.of(descriptor))));
            assertTrue(provider.supports(new OperationCapabilityQuery(
                    profile,
                    new Operation(
                            ShapeTransformKind.RESHAPE,
                            new TargetShapeAttrs(Shape.of(3))),
                    List.of(descriptor),
                    List.of(reshaped))));
        }

        List<TraceEvent<? extends TracePayload>> events = new ArrayList<>();
        var trace = new MetalTraceProducer(events::add);
        assertNotNull(trace.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalPreparedRoute.CUSTOM_PROGRAM,
                program,
                float16Values,
                new int[] {0},
                new int[] {1},
                0));
        var bfloat16 = new MetalMpsGraphProgram.ValueDescriptor(
                DataType.BFLOAT16, new long[] {3}, false);
        assertNotNull(trace.prepareUnit(
                NumericalProfile.STRICT_IEEE,
                MetalPreparedRoute.CUSTOM_PROGRAM,
                program,
                List.of(bfloat16, bfloat16),
                new int[] {0},
                new int[] {1},
                0));

        MetalPreparationStructure float16Structure =
                (MetalPreparationStructure) events.get(0).payload();
        MetalPreparationStructure bfloat16Structure =
                (MetalPreparationStructure) events.get(1).payload();
        assertEquals(19, float16Structure.schemaVersion());
        assertNotEquals(
                bfloat16Structure.canonicalDigest(),
                float16Structure.canonicalDigest());
    }

    @Test
    void certifiedRawBoundaryAdmitsOddDenseStorageAndRejectsOffsetsHolesAndZeroCounts() {
        assertTrue(MetalLowPrecisionRouteCertification.isCanonicalNonEmptyBoundary(
                LayoutDescriptor.contiguous(Shape.of(257))));
        assertFalse(MetalLowPrecisionRouteCertification.isCanonicalNonEmptyBoundary(
                LayoutDescriptor.of(Shape.of(257), new long[] {1}, 1L, true)));
        assertFalse(MetalLowPrecisionRouteCertification.isCanonicalNonEmptyBoundary(
                LayoutDescriptor.of(Shape.of(257), new long[] {2}, 0L, true)));
        assertFalse(MetalLowPrecisionRouteCertification.isCanonicalNonEmptyBoundary(
                LayoutDescriptor.contiguous(Shape.of(0))));
    }

    @Test
    void acceleratorExecutesPointwiseScalarReductionAndMatmulForBothRaw16Carriers() {
        Path library = configuredLibrary();
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            var program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.neg(0, 2),
                    MetalMpsGraphProgram.Node.binary(
                            MetalMpsGraphProgram.NodeKind.ADD, 2, 1, 3),
                    MetalMpsGraphProgram.Node.scalarValue(
                            MetalMpsGraphProgram.NodeKind.SCALAR_ADD,
                            3, 4, type, lowWord(type, 1.0f)),
                    MetalMpsGraphProgram.Node.reduction(
                            MetalMpsGraphProgram.NodeKind.SUM,
                            4,
                            5,
                            MetalMpsGraphProgram.ReductionForm.FULL,
                            List.of(),
                            false),
                    MetalMpsGraphProgram.Node.matmul(6, 7, 8),
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.MEAN_SQUARED_ERROR,
                            new int[] {9, 10},
                            new int[] {11},
                            MetalMpsGraphProgram.AttributeKind.MSE,
                            new long[] {1}),
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.MEAN_SQUARED_ERROR,
                            new int[] {9, 10},
                            new int[] {12},
                            MetalMpsGraphProgram.AttributeKind.MSE,
                            new long[] {3}),
                    MetalMpsGraphProgram.Node.reduction(
                            MetalMpsGraphProgram.NodeKind.L1_NORM,
                            13,
                            14,
                            MetalMpsGraphProgram.ReductionForm.MULTI_AXIS,
                            List.of(0),
                            false),
                    MetalMpsGraphProgram.Node.statisticalReduction(
                            MetalMpsGraphProgram.NodeKind.VARIANCE,
                            15,
                            16,
                            List.of(0),
                            false,
                            0)));
            List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                    typed(type, 4),
                    typed(type, 4),
                    typed(type, 4),
                    typed(type, 4),
                    typed(type, 4),
                    typed(type),
                    typed(type, 2, 2),
                    typed(type, 2, 2),
                    typed(type, 2, 2),
                    typed(type, 4),
                    typed(type, 4),
                    typed(type, 4),
                    typed(type),
                    typed(type, 3),
                    typed(type),
                    typed(type, 1),
                    typed(type));
            int[] feeds = {0, 1, 6, 7, 9, 10, 13, 15};
            int[] targets = {2, 3, 4, 5, 8, 11, 12, 14, 16};
            assertThrows(
                    IllegalArgumentException.class,
                    () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                            NumericalProfile.STRICT_IEEE,
                            values,
                            program,
                            feeds,
                            targets,
                            MetalPreparedRoute.CUSTOM_PROGRAM));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> MetalNativeApi.MpsGraphExecutableAbi.validateCreate(
                            NumericalProfile.ACCELERATOR,
                            values,
                            program,
                            feeds,
                            targets,
                            MetalPreparedRoute.MPSGRAPH));

            List<byte[]> actual = execute(
                    library,
                    program,
                    values,
                    feeds,
                    targets,
                    List.of(
                            lowWords(type, 1.0f, -2.0f, 0.5f, -0.0f),
                            lowWords(type, 2.0f, 4.0f, 0.5f, 0.0f),
                            lowWords(type, 1.0f, 2.0f, 3.0f, 4.0f),
                            lowWords(type, 5.0f, 6.0f, 7.0f, 8.0f),
                            lowWords(type, 1.0f, 2.0f, 3.0f, 4.0f),
                            lowWords(type, 0.0f, 1.0f, 1.0f, 2.0f),
                            lowWords(type, -2.0f, 3.0f, -4.0f),
                            lowWords(type, 5.0f)));
            assertArrayEquals(
                    lowWords(type, -1.0f, 2.0f, -0.5f, 0.0f), actual.get(0), type + " NEG");
            assertArrayEquals(
                    lowWords(type, 1.0f, 6.0f, 0.0f, 0.0f), actual.get(1), type + " ADD");
            assertArrayEquals(
                    lowWords(type, 2.0f, 7.0f, 1.0f, 1.0f),
                    actual.get(2),
                    type + " scalar ADD");
            assertArrayEquals(lowWords(type, 11.0f), actual.get(3), type + " SUM");
            assertArrayEquals(
                    lowWords(type, 19.0f, 22.0f, 43.0f, 50.0f),
                    actual.get(4),
                    type + " MATMUL");
            assertArrayEquals(
                    lowWords(type, 1.0f, 1.0f, 4.0f, 4.0f),
                    actual.get(5),
                    type + " MSE NONE");
            assertArrayEquals(lowWords(type, 2.5f), actual.get(6), type + " MSE MEAN");
            assertArrayEquals(lowWords(type, 9.0f), actual.get(7), type + " L1");
            assertArrayEquals(lowWords(type, 0.0f), actual.get(8), type + " VARIANCE");
        }
    }

    @Test
    void acceleratorExecutesRemainingPointwiseExtremaScanAndSelectionFamilies() {
        Path library = configuredLibrary();
        for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            long two = lowWord(type, 2.0f);
            long negativeOne = lowWord(type, -1.0f);
            long positiveOne = lowWord(type, 1.0f);
            var program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.abs(0, 2),
                    MetalMpsGraphProgram.Node.binary(
                            MetalMpsGraphProgram.NodeKind.TENSOR_MIN, 0, 1, 3),
                    MetalMpsGraphProgram.Node.binary(
                            MetalMpsGraphProgram.NodeKind.TENSOR_MAX, 0, 1, 4),
                    MetalMpsGraphProgram.Node.binary(
                            MetalMpsGraphProgram.NodeKind.GT, 0, 1, 5),
                    MetalMpsGraphProgram.Node.scalarValue(
                            MetalMpsGraphProgram.NodeKind.SCALAR_MIN, 0, 6, type, two),
                    MetalMpsGraphProgram.Node.scalarValue(
                            MetalMpsGraphProgram.NodeKind.SCALAR_MAX, 0, 7, type, two),
                    MetalMpsGraphProgram.Node.clamp(
                            0, 8, type, negativeOne, positiveOne),
                    noAttribute(MetalMpsGraphProgram.NodeKind.FLOOR, 0, 9),
                    noAttribute(MetalMpsGraphProgram.NodeKind.CEIL, 0, 10),
                    noAttribute(MetalMpsGraphProgram.NodeKind.SIGN, 0, 11),
                    noAttribute(MetalMpsGraphProgram.NodeKind.RELU, 0, 12),
                    noAttribute(MetalMpsGraphProgram.NodeKind.RECIPROCAL, 0, 13),
                    MetalMpsGraphProgram.Node.reduction(
                            MetalMpsGraphProgram.NodeKind.MEAN,
                            0, 14, MetalMpsGraphProgram.ReductionForm.FULL,
                            List.of(), false),
                    MetalMpsGraphProgram.Node.reduction(
                            MetalMpsGraphProgram.NodeKind.REDUCTION_MIN,
                            0, 15, MetalMpsGraphProgram.ReductionForm.FULL,
                            List.of(), false),
                    MetalMpsGraphProgram.Node.reduction(
                            MetalMpsGraphProgram.NodeKind.REDUCTION_MAX,
                            0, 16, MetalMpsGraphProgram.ReductionForm.FULL,
                            List.of(), false),
                    MetalMpsGraphProgram.Node.scan(
                            MetalMpsGraphProgram.NodeKind.CUM_SUM, 0, 17, 0, false, false),
                    MetalMpsGraphProgram.Node.scan(
                            MetalMpsGraphProgram.NodeKind.CUM_PROD, 0, 18, 0, false, false),
                    noAttribute(MetalMpsGraphProgram.NodeKind.IS_FINITE, 0, 20),
                    MetalMpsGraphProgram.Node.generic(
                            MetalMpsGraphProgram.NodeKind.WHERE,
                            new int[] {19, 0, 1},
                            new int[] {21},
                            MetalMpsGraphProgram.AttributeKind.NONE,
                            new long[0])));
            var lowVector = typed(type, 4);
            var boolVector = typed(DataType.BOOL, 4);
            List<MetalMpsGraphProgram.ValueDescriptor> values = List.of(
                    lowVector, lowVector,
                    lowVector, lowVector, lowVector, boolVector,
                    lowVector, lowVector, lowVector, lowVector, lowVector, lowVector, lowVector,
                    lowVector, typed(type), typed(type), typed(type), lowVector, lowVector,
                    boolVector, boolVector, lowVector);
            int[] targets = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21};
            List<byte[]> actual = execute(
                    library,
                    program,
                    values,
                    new int[] {0, 1, 19},
                    targets,
                    List.of(
                            lowWords(type, -1.5f, -0.0f, 2.5f, 4.0f),
                            lowWords(type, 1.0f, -2.0f, 3.0f, 0.0f),
                            new byte[] {1, 0, 1, 0}));
            assertArrayEquals(lowWords(type, 1.5f, 0.0f, 2.5f, 4.0f), actual.get(0));
            assertArrayEquals(lowWords(type, -1.5f, -2.0f, 2.5f, 0.0f), actual.get(1));
            assertArrayEquals(lowWords(type, 1.0f, -0.0f, 3.0f, 4.0f), actual.get(2));
            assertArrayEquals(new byte[] {0, 1, 0, 1}, actual.get(3));
            assertArrayEquals(lowWords(type, -1.5f, -0.0f, 2.0f, 2.0f), actual.get(4));
            assertArrayEquals(lowWords(type, 2.0f, 2.0f, 2.5f, 4.0f), actual.get(5));
            assertArrayEquals(lowWords(type, -1.0f, -0.0f, 1.0f, 1.0f), actual.get(6));
            assertArrayEquals(lowWords(type, -2.0f, -0.0f, 2.0f, 4.0f), actual.get(7));
            assertArrayEquals(lowWords(type, -1.0f, -0.0f, 3.0f, 4.0f), actual.get(8));
            assertArrayEquals(lowWords(type, -1.0f, -0.0f, 1.0f, 1.0f), actual.get(9));
            assertArrayEquals(lowWords(type, 0.0f, 0.0f, 2.5f, 4.0f), actual.get(10));
            assertArrayEquals(
                    lowWords(type, -2.0f / 3.0f, Float.NEGATIVE_INFINITY, 0.4f, 0.25f),
                    actual.get(11));
            assertArrayEquals(lowWords(type, 1.25f), actual.get(12));
            assertArrayEquals(lowWords(type, -1.5f), actual.get(13));
            assertArrayEquals(lowWords(type, 4.0f), actual.get(14));
            assertArrayEquals(lowWords(type, -1.5f, -1.5f, 1.0f, 5.0f), actual.get(15));
            assertArrayEquals(lowWords(type, -1.5f, 0.0f, 0.0f, 0.0f), actual.get(16));
            assertArrayEquals(new byte[] {1, 1, 1, 1}, actual.get(17));
            assertArrayEquals(lowWords(type, -1.5f, -2.0f, 2.5f, 0.0f), actual.get(18));
        }
    }

    @Test
    void nativeResourcesRoundTripOddUnalignedFloat16PayloadAndZeroLength() {
        Path library = configuredLibrary();
        byte[] canonical = {
            0x7e, 0x01,
            (byte) 0x80, 0x00,
            (byte) 0xfe, 0x42
        };
        byte[] physical = ByteOrder.nativeOrder() == ByteOrder.LITTLE_ENDIAN
                ? new byte[] {
                    0x01, 0x7e,
                    0x00, (byte) 0x80,
                    0x42, (byte) 0xfe
                }
                : canonical.clone();
        TensorDescriptor descriptor = descriptor(DataType.FLOAT16, Shape.of(3));

        try (Arena arena = Arena.ofConfined();
                MetalBackendIntegration integration = MetalBackendIntegration.open(
                        new MetalBackendConfiguration(library))) {
            MemorySegment sourceAllocation = arena.allocate(7, Short.BYTES);
            MemorySegment source = sourceAllocation.asSlice(1, physical.length);
            assertEquals(1L, source.address() & 1L);
            MemorySegment.copy(MemorySegment.ofArray(physical), 0, source, 0, physical.length);
            var storage = new MemorySegmentStorage(DataType.FLOAT16, 3, source);

            try (BufferRepresentation representation = integration.borrow(storage)) {
                assertTrue(integration.acceptsStorageLayoutTransfer(representation, descriptor));
                MemorySegment destinationAllocation = arena.allocate(7, Short.BYTES);
                MemorySegment destination = destinationAllocation.asSlice(1, physical.length);
                assertEquals(1L, destination.address() & 1L);
                integration.bindStorageLayoutDownload(representation, descriptor)
                        .accept(destination);
                assertArrayEquals(physical, destination.toArray(JAVA_BYTE));
                assertArrayEquals(
                        canonical,
                        integration.copyToCanonicalHostBytes(
                                representation, descriptor, canonical.length));
                assertThrows(
                        IllegalArgumentException.class,
                        () -> integration.copyToCanonicalHostBytes(
                                representation, descriptor, canonical.length - 1L));
            }

            MemorySegment shortSource = source.asSlice(0, 4);
            var shortStorage = new MemorySegmentStorage(DataType.FLOAT16, 2, shortSource);
            try (BufferRepresentation shortRepresentation = integration.borrow(shortStorage)) {
                assertFalse(integration.acceptsStorageLayoutTransfer(
                        shortRepresentation, descriptor));
                assertThrows(
                        IllegalArgumentException.class,
                        () -> integration.bindStorageLayoutDownload(
                                shortRepresentation, descriptor));
            }
            assertThrows(
                    IllegalArgumentException.class,
                    () -> new MemorySegmentStorage(
                            DataType.FLOAT16, 3, source.asSlice(0, 5)));

            var zeroStorage = new MemorySegmentStorage(
                    DataType.FLOAT16, 0, sourceAllocation.asSlice(0, 0));
            try (BufferRepresentation zero = integration.borrow(zeroStorage)) {
                assertEquals(0L, ((MetalBufferRepresentation) zero).byteSize());
            }
        }

        try (MetalDeviceContext context = MetalDeviceContext.open(library)) {
            for (DataType type : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
                PrepareContext<MetalNegAnalysisInputs> prepareContext =
                        lowPrecisionPrepareContext(context, type);
                var analysis = new MetalNegPartitionPreparer().analyze(prepareContext);
                assertEquals(MetalPreparedRoute.CUSTOM_PROGRAM, analysis.plan().route());
                assertEquals(
                        List.of(MetalNegTuningBatch.Candidate.CUSTOM_PROGRAM),
                        new MetalNegRouteCandidateGenerator()
                                .generate(
                                        prepareContext,
                                        analysis.plan(),
                                        MetalNegTuningBatch.Candidate.values().length)
                                .candidates());
                assertEquals(List.of(new ValueId(71_001)), analysis.plan().internalValueIds());
                assertArrayEquals(new int[] {1}, analysis.plan().internalValueIndices());
                assertEquals(
                        List.of(MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM,
                                MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM),
                        analysis.plan().pointwiseFusionPlan().steps().stream()
                                .map(MetalPointwiseFusionPlan.Step::kind)
                                .toList());
            }
        }
    }

    @Test
    void exactEnvironmentCertificateAddsGraphWithoutRemovingCustomBaseline() {
        Path library = configuredLibrary();
        try (MetalDeviceContext context = MetalDeviceContext.open(library)) {
            List<TraceEvent<? extends TracePayload>> events = new ArrayList<>();
            var trace = new MetalTraceProducer(events::add);
            PrepareContext<MetalNegAnalysisInputs> prepareContext =
                    certifiedReshapePrepareContext(context, DataType.FLOAT16, trace);
            var preparer = new MetalNegPartitionPreparer();
            var custom = preparer.analyze(prepareContext);
            assertEquals(MetalPreparedRoute.CUSTOM_PROGRAM, custom.plan().route());
            MetalNegTuningBatch batch = new MetalNegRouteCandidateGenerator()
                    .generate(prepareContext, custom.plan(), 3);
            assertEquals(
                    List.of(
                            MetalNegTuningBatch.Candidate.CUSTOM_PROGRAM,
                            MetalNegTuningBatch.Candidate.MPSGRAPH),
                    batch.candidates());
            var codec = new MetalNegTuningCodec();
            byte[] customCandidate =
                    codec.encodeCandidate(MetalNegTuningBatch.Candidate.CUSTOM_PROGRAM);
            byte[] graphCandidate =
                    codec.encodeCandidate(MetalNegTuningBatch.Candidate.MPSGRAPH);
            assertFalse(Arrays.equals(customCandidate, graphCandidate));
            byte[] float16GraphDecision = codec.encodeDecision(new MetalNegTuningDecision(
                    MetalNegTuningBatch.CANDIDATE_SCHEMA_VERSION,
                    batch.compatibility(),
                    MetalNegTuningBatch.Candidate.MPSGRAPH));

            PrepareContext<MetalNegAnalysisInputs> bfloat16Context =
                    certifiedReshapePrepareContext(context, DataType.BFLOAT16);
            var bfloat16Analysis = preparer.analyze(bfloat16Context);
            MetalNegTuningBatch bfloat16Batch = new MetalNegRouteCandidateGenerator()
                    .generate(bfloat16Context, bfloat16Analysis.plan(), 3);
            assertNotEquals(
                    batch.compatibility().workload(),
                    bfloat16Batch.compatibility().workload());
            assertFalse(Arrays.equals(
                    codec.encodeCompatibility(batch.compatibility()),
                    codec.encodeCompatibility(bfloat16Batch.compatibility())));
            assertTrue(codec.decodeDecision(float16GraphDecision, bfloat16Batch).isEmpty());

            try (MetalDeviceContext otherContext = MetalDeviceContext.open(library)) {
                PrepareContext<MetalNegAnalysisInputs> otherPrepareContext =
                        certifiedReshapePrepareContext(otherContext, DataType.FLOAT16);
                var otherAnalysis = preparer.analyze(otherPrepareContext);
                MetalNegTuningBatch otherBatch = new MetalNegRouteCandidateGenerator()
                        .generate(otherPrepareContext, otherAnalysis.plan(), 3);
                assertEquals(
                        batch.compatibility().workload(),
                        otherBatch.compatibility().workload());
                assertNotEquals(
                        batch.compatibility().target(),
                        otherBatch.compatibility().target());
                assertTrue(codec.decodeDecision(float16GraphDecision, otherBatch).isEmpty());
            }
            var graph = preparer.analyzeForTesting(
                    prepareContext, MetalPreparedRoute.MPSGRAPH);
            assertEquals(MetalPreparedRoute.MPSGRAPH, graph.plan().route());
            assertTrue(graph.plan().internalValueIds().isEmpty());
            assertEquals(2, graph.plan().declarations().size());
            MetalLowPrecisionRouteCertification.Qualification qualification =
                    MetalLowPrecisionRouteCertification.find(graph.plan()).orElseThrow();

            PrepareContext<MetalNegAnalysisInputs> sliceContext =
                    certifiedSlicePrepareContext(context, DataType.FLOAT16);
            var sliceCustom = preparer.analyze(sliceContext);
            assertEquals(
                    List.of(
                            MetalNegTuningBatch.Candidate.CUSTOM_PROGRAM,
                            MetalNegTuningBatch.Candidate.MPSGRAPH),
                    new MetalNegRouteCandidateGenerator()
                            .generate(sliceContext, sliceCustom.plan(), 3)
                            .candidates());
            var sliceGraph = preparer.analyzeForTesting(
                    sliceContext, MetalPreparedRoute.MPSGRAPH);
            assertFalse(sliceGraph.plan().programValueDescriptors().get(1).densePhysical());
            assertTrue(MetalLowPrecisionRouteCertification.find(sliceGraph.plan()).isPresent());

            List<LowPrecisionTraceMetadata> certificateEvents = events.stream()
                    .map(TraceEvent::payload)
                    .filter(LowPrecisionTraceMetadata.class::isInstance)
                    .map(LowPrecisionTraceMetadata.class::cast)
                    .toList();
            assertEquals(2, certificateEvents.size());
            LowPrecisionTraceMetadata customTrace = certificateEvents.get(0);
            assertEquals(TraceRouteKind.CUSTOM_KERNEL, customTrace.selectedRoute());
            assertEquals(LowPrecisionCertificate.Status.NOT_CERTIFIED,
                    customTrace.certificateStatus());
            assertTrue(customTrace.certificateKey().isEmpty());
            assertTrue(customTrace.accuracy().isEmpty());
            assertTrue(customTrace.determinism().isEmpty());

            LowPrecisionTraceMetadata graphTrace = certificateEvents.get(1);
            assertEquals(TraceRouteKind.GRAPH_EXECUTABLE, graphTrace.selectedRoute());
            assertEquals(LowPrecisionCertificateKey.NO_ACCUMULATOR,
                    graphTrace.accumulatorDtype());
            assertEquals(LowPrecisionCertificate.Status.CERTIFIED,
                    graphTrace.certificateStatus());
            assertEquals(qualification.expected().key(),
                    graphTrace.certificateKey().orElseThrow());
            assertEquals(MetalLowPrecisionCertificateStore.ROUTE,
                    graphTrace.certificateKey().orElseThrow().route());
            assertEquals(LowPrecisionAccuracy.Verdict.PASS,
                    graphTrace.accuracy().orElseThrow().verdict());
            assertEquals(qualification.certificate().accuracy().evidenceDigest(),
                    graphTrace.accuracy().orElseThrow().evidenceDigest());
            assertEquals(LowPrecisionDeterminism.Verdict.PASS,
                    graphTrace.determinism().orElseThrow().verdict());
            assertEquals(qualification.certificate().determinism().evidenceDigest(),
                    graphTrace.determinism().orElseThrow().evidenceDigest());

            long byteCount = 65536L * Short.BYTES;
            for (MetalNegPreparationPlan executionPlan :
                    List.of(custom.plan(), graph.plan())) {
                try (MetalMpsGraphExecutableResource executable =
                                context.createMpsGraphExecutable(executionPlan);
                        MetalBufferRepresentation input = context.createBuffer(byteCount);
                        MetalBufferRepresentation output = context.createBuffer(byteCount);
                        Arena arena = Arena.ofConfined()) {
                    short[] expectedWords = {
                        (short) 0x0000, (short) 0x8000, (short) 0x0001, (short) 0x03ff,
                        (short) 0x7c00, (short) 0xfc00, (short) 0x7e01, (short) 0x7d01
                    };
                    MemorySegment source = arena.allocate(byteCount, Short.BYTES);
                    for (int index = 0; index < expectedWords.length; index++) {
                        source.setAtIndex(JAVA_SHORT, index, expectedWords[index]);
                    }
                    input.upload(0L, source, 0L, byteCount);
                    MemorySegment poison = arena.allocate(byteCount, 1);
                    poison.fill((byte) 0xa5);
                    output.upload(0L, poison, 0L, byteCount);

                    int inputCount = executable.inputRequiredBytes().length;
                    assertEquals(
                            executionPlan.route() == MetalPreparedRoute.CUSTOM_PROGRAM ? 2 : 1,
                            inputCount);
                    MemorySegment inputHandles = arena.allocate(ADDRESS, inputCount);
                    inputHandles.setAtIndex(ADDRESS, 0, input.executionHandle().carrier());
                    if (inputCount == 2) {
                        inputHandles.setAtIndex(ADDRESS, 1, output.executionHandle().carrier());
                    }
                    MemorySegment outputHandles = arena.allocate(ADDRESS);
                    outputHandles.set(ADDRESS, 0L, output.executionHandle().carrier());
                    executable.run(inputCount, inputHandles, 1, outputHandles);

                    MemorySegment actual = arena.allocate(
                            (long) expectedWords.length * Short.BYTES, Short.BYTES);
                    output.download(0L, actual, 0L, actual.byteSize());
                    for (int index = 0; index < expectedWords.length; index++) {
                        assertEquals(expectedWords[index], actual.getAtIndex(JAVA_SHORT, index));
                    }
                }
            }
        }
    }

    private static MetalMpsGraphProgram.Node noAttribute(
            MetalMpsGraphProgram.NodeKind kind, int input, int output) {
        return MetalMpsGraphProgram.Node.generic(
                kind,
                new int[] {input},
                new int[] {output},
                MetalMpsGraphProgram.AttributeKind.NONE,
                new long[0]);
    }

    private static MetalMpsGraphProgram.ValueDescriptor typed(
            DataType type, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, false);
    }

    private static List<byte[]> execute(
            Path library,
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets,
            List<byte[]> inputBytes) {
        MetalNativeApi api = MetalNativeApi.open(library);
        MetalNativeApi.Handle context = null;
        MetalNativeApi.Handle executable = null;
        var buffers = new ArrayList<MetalNativeApi.Handle>();
        try {
            context = api.createContext();
            executable = api.createMpsGraphExecutable(
                    context,
                    NumericalProfile.ACCELERATOR,
                    values,
                    program,
                    feeds,
                    targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            for (var value : values) {
                buffers.add(api.createBuffer(context, value.byteCount()));
            }
            for (int index = 0; index < feeds.length; index++) {
                try (Arena arena = Arena.ofConfined()) {
                    byte[] bytes = inputBytes.get(index);
                    MemorySegment source = arena.allocate(bytes.length, 1);
                    source.copyFrom(MemorySegment.ofArray(bytes));
                    api.upload(buffers.get(feeds[index]), 0L, source, bytes.length);
                }
            }
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment addresses = arena.allocate(ADDRESS, buffers.size());
                for (int index = 0; index < buffers.size(); index++) {
                    addresses.setAtIndex(ADDRESS, index, buffers.get(index).carrier());
                }
                MemorySegment outputs = arena.allocate(ADDRESS, targets.length);
                for (int index = 0; index < targets.length; index++) {
                    outputs.setAtIndex(ADDRESS, index, buffers.get(targets[index]).carrier());
                }
                api.runExecutable(executable, buffers.size(), addresses, targets.length, outputs);
                var result = new ArrayList<byte[]>(targets.length);
                for (int target : targets) {
                    long byteCount = values.get(target).byteCount();
                    MemorySegment destination = arena.allocate(byteCount, 1);
                    api.download(buffers.get(target), 0L, destination, byteCount);
                    result.add(destination.toArray(JAVA_BYTE));
                }
                return List.copyOf(result);
            }
        } finally {
            for (int index = buffers.size(); index-- > 0;) {
                api.releaseBuffer(buffers.get(index));
            }
            if (executable != null) api.releaseExecutable(executable);
            if (context != null) api.releaseContext(context);
            api.close();
        }
    }

    private static byte[] lowWords(DataType type, float... values) {
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Short.BYTES)
                .order(ByteOrder.nativeOrder());
        for (float value : values) bytes.putShort((short) lowWord(type, value));
        return bytes.array();
    }

    private static long lowWord(DataType type, float value) {
        return switch (type) {
            case BFLOAT16 -> BFloat16Bits.fromFloat(value) & 0xffffL;
            case FLOAT16 -> Float16Bits.fromFloat(value) & 0xffffL;
            default -> throw new IllegalArgumentException("expected a raw16 carrier");
        };
    }

    private static PrepareContext<MetalNegAnalysisInputs> lowPrecisionPrepareContext(
            MetalDeviceContext context, DataType type) {
        TensorDescriptor descriptor = descriptor(type, Shape.of(3));
        ValueId feed = new ValueId(71_000);
        ValueId internal = new ValueId(71_001);
        ValueId target = new ValueId(71_002);
        CompiledNode first = new CompiledNode(
                new NodeId(71_000),
                new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                List.of(feed),
                List.of(internal));
        CompiledNode second = new CompiledNode(
                new NodeId(71_001),
                new Operation(UnaryElementwiseKind.NEG, NoOperationAttrs.INSTANCE),
                List.of(internal),
                List.of(target));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID, List.of(first.id(), second.id()));
        return new PrepareContext<>(
                NumericalProfile.STRICT_IEEE,
                new PartitionDag(partition, List.of(first, second)),
                List.of(
                        new GraphValue(feed, descriptor),
                        new GraphValue(internal, descriptor),
                        new GraphValue(target, descriptor)),
                List.of(
                        new LogicalMemoryRequirement(
                                feed,
                                descriptor,
                                Optional.empty(),
                                List.of(partition),
                                false),
                        new LogicalMemoryRequirement(
                                internal,
                                descriptor,
                                Optional.of(partition),
                                List.of(partition),
                                false),
                        new LogicalMemoryRequirement(
                                target,
                                descriptor,
                                Optional.of(partition),
                                List.of(),
                                true)),
                Map.of(),
                new MetalNegAnalysisInputs(context));
    }

    private static PrepareContext<MetalNegAnalysisInputs> certifiedReshapePrepareContext(
            MetalDeviceContext context, DataType type) {
        return certifiedReshapePrepareContext(context, type, null);
    }

    private static PrepareContext<MetalNegAnalysisInputs> certifiedReshapePrepareContext(
            MetalDeviceContext context, DataType type, MetalTraceProducer traceProducer) {
        Shape inputShape = Shape.of(65536);
        Shape outputShape = Shape.of(256, 256);
        TensorDescriptor input = descriptor(type, inputShape);
        TensorDescriptor output = new TensorDescriptor(
                type,
                outputShape,
                Optional.of(LayoutDescriptor.of(
                        outputShape,
                        LayoutDescriptor.contiguous(outputShape).strides(),
                        0L,
                        true)),
                false);
        ValueId feed = new ValueId(72_000);
        ValueId target = new ValueId(72_001);
        CompiledNode reshape = new CompiledNode(
                new NodeId(72_000),
                new Operation(
                        ShapeTransformKind.RESHAPE,
                        new TargetShapeAttrs(outputShape)),
                List.of(feed),
                List.of(target));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID, List.of(reshape.id()));
        return new PrepareContext<>(
                NumericalProfile.STRICT_IEEE,
                new PartitionDag(partition, List.of(reshape)),
                List.of(new GraphValue(feed, input), new GraphValue(target, output)),
                List.of(
                        new LogicalMemoryRequirement(
                                feed, input, Optional.empty(), List.of(partition), false),
                        new LogicalMemoryRequirement(
                                target, output, Optional.of(partition), List.of(), true)),
                Map.of(),
                new MetalNegAnalysisInputs(context, traceProducer));
    }

    private static PrepareContext<MetalNegAnalysisInputs> certifiedSlicePrepareContext(
            MetalDeviceContext context, DataType type) {
        Shape shape = Shape.of(65536);
        TensorDescriptor input = descriptor(type, shape);
        TensorDescriptor output = new TensorDescriptor(
                type,
                shape,
                Optional.of(LayoutDescriptor.of(shape, new long[] {1}, 0L, true)),
                false);
        ValueId feed = new ValueId(73_000);
        ValueId target = new ValueId(73_001);
        CompiledNode slice = new CompiledNode(
                new NodeId(73_000),
                new Operation(
                        SliceKind.SLICE,
                        new SliceAttrs(
                                List.of(0L), List.of(65536L), List.of(0), List.of(1L))),
                List.of(feed),
                List.of(target));
        PlannedPartition partition = new PlannedPartition(
                MetalCapabilityProvider.METAL_BACKEND_ID, List.of(slice.id()));
        return new PrepareContext<>(
                NumericalProfile.STRICT_IEEE,
                new PartitionDag(partition, List.of(slice)),
                List.of(new GraphValue(feed, input), new GraphValue(target, output)),
                List.of(
                        new LogicalMemoryRequirement(
                                feed, input, Optional.empty(), List.of(partition), false),
                        new LogicalMemoryRequirement(
                                target, output, Optional.of(partition), List.of(), true)),
                Map.of(),
                new MetalNegAnalysisInputs(context));
    }

    private static TensorDescriptor descriptor(DataType dataType, Shape shape) {
        return new TensorDescriptor(
                dataType,
                shape,
                Optional.of(LayoutDescriptor.contiguous(shape)),
                false);
    }

    private static Path configuredLibrary() {
        String configured = System.getenv("SYNAPTIK_METAL_TEST_LIBRARY");
        assumeTrue(configured != null && !configured.isBlank(),
                "SYNAPTIK_METAL_TEST_LIBRARY is not set");
        return Path.of(configured).toAbsolutePath().normalize();
    }
}
