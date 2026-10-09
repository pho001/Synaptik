package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalPointwiseFusionPlannerTest {
    @Test
    void lowValueAnywhereInPartitionDisablesF32PointwiseFusionAndPreview() {
        var f32Program = new MetalMpsGraphProgram(List.of(
                unary(MetalMpsGraphProgram.NodeKind.FLOOR, 0, 1),
                unary(MetalMpsGraphProgram.NodeKind.CEIL, 1, 2)));
        List<MetalMpsGraphProgram.ValueDescriptor> f32Values = descriptors(3, 4L);
        MetalPointwiseFusionPlan f32 = MetalPointwiseFusionPlanner.plan(
                f32Program, f32Values,
                new int[] {0}, new int[] {2}, MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(List.of(MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE),
                f32.steps().stream().map(MetalPointwiseFusionPlan.Step::kind).toList());

        for (DataType lowType : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            var program = new MetalMpsGraphProgram(List.of(
                    unary(MetalMpsGraphProgram.NodeKind.FLOOR, 0, 1),
                    unary(MetalMpsGraphProgram.NodeKind.CEIL, 1, 2),
                    MetalMpsGraphProgram.Node.neg(3, 4)));
            var values = new ArrayList<>(f32Values);
            values.add(descriptor(lowType, 4L));
            values.add(descriptor(lowType, 4L));
            int[] feeds = {0, 3};
            int[] targets = {2, 4};
            MetalPointwiseFusionPlan mixed = program.resolvePlan(
                    values, feeds, targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM, null);

            assertEquals(List.of(
                            MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM,
                            MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM,
                            MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM),
                    mixed.steps().stream().map(MetalPointwiseFusionPlan.Step::kind).toList());
            assertArrayEquals(new int[] {0, 1, 2, 3, 4},
                    mixed.materializedProgramValueIndices());
            byte[] preview = program.encodedProgramImage(
                    values, feeds, targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            ByteBuffer header = ByteBuffer.wrap(preview).order(ByteOrder.LITTLE_ENDIAN);
            assertEquals(0, header.getInt(76), "instruction count for " + lowType);
            assertEquals(0, header.getInt(88), "generated unit count for " + lowType);
            assertThrows(IllegalArgumentException.class, () -> program.encodedProgramImage(
                    values, feeds, targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM, f32));
        }
    }

    @Test
    void lowValueAnywhereInPartitionDisablesF32AnchorEpilogueFusion() {
        var f32Program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 3),
                MetalMpsGraphProgram.Node.binary(MetalMpsGraphProgram.NodeKind.ADD, 2, 3, 4)));
        List<MetalMpsGraphProgram.ValueDescriptor> f32Values = List.of(
                descriptor(DataType.FLOAT32, 2L, 2L),
                descriptor(DataType.FLOAT32, 2L, 2L),
                descriptor(DataType.FLOAT32, 2L),
                descriptor(DataType.FLOAT32, 2L, 2L),
                descriptor(DataType.FLOAT32, 2L, 2L));
        MetalPointwiseFusionPlan f32 = MetalPointwiseFusionPlanner.plan(
                f32Program, f32Values,
                new int[] {0, 1, 2}, new int[] {4}, MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(List.of(MetalPointwiseFusionPlan.StepKind.ANCHOR_EPILOGUE),
                f32.steps().stream().map(MetalPointwiseFusionPlan.Step::kind).toList());

        for (DataType lowType : List.of(DataType.BFLOAT16, DataType.FLOAT16)) {
            var program = new MetalMpsGraphProgram(List.of(
                    MetalMpsGraphProgram.Node.matmul(0, 1, 3),
                    MetalMpsGraphProgram.Node.binary(
                            MetalMpsGraphProgram.NodeKind.ADD, 2, 3, 4),
                    MetalMpsGraphProgram.Node.neg(5, 6)));
            var values = new ArrayList<>(f32Values);
            values.add(descriptor(lowType, 4L));
            values.add(descriptor(lowType, 4L));
            int[] feeds = {0, 1, 2, 5};
            int[] targets = {4, 6};
            MetalPointwiseFusionPlan mixed = program.resolvePlan(
                    values, feeds, targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM, null);

            assertEquals(3, mixed.steps().size());
            assertTrue(mixed.steps().stream().noneMatch(step ->
                    step.kind() == MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE
                            || step.kind() == MetalPointwiseFusionPlan.StepKind.ANCHOR_EPILOGUE));
            assertEquals(0, mixed.instructions().size());
            byte[] preview = program.encodedProgramImage(
                    values, feeds, targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM);
            assertEquals(0, ByteBuffer.wrap(preview).order(ByteOrder.LITTLE_ENDIAN).getInt(76));
            assertThrows(IllegalArgumentException.class, () -> program.encodedProgramImage(
                    values, feeds, targets,
                    MetalPreparedRoute.CUSTOM_PROGRAM, f32));
        }
    }

    @Test
    void nineNodeChainUsesSevenThenTwoWithoutVirtualMaterialization() {
        var fixture = chain(9);
        MetalPointwiseFusionPlan plan = plan(fixture);

        assertEquals(2, plan.generatedUnitCount());
        assertEquals(List.of(7, 2), plan.steps().stream()
                .map(MetalPointwiseFusionPlan.Step::memberCount).toList());
        assertArrayEquals(new int[] {0, 7, 9}, plan.materializedProgramValueIndices());
        assertArrayEquals(new int[] {2}, plan.targetMaterializedSlots());
        assertEquals(9, plan.instructions().size());
        String manifest = new String(plan.canonicalManifest(), StandardCharsets.US_ASCII);
        assertTrue(manifest.contains("pointmeta 0 4 4 1 0 0\n"));
        assertTrue(manifest.contains("pointmeta 1 4 4 1 0 0\n"));
        assertFalse(manifest.contains("\nfeed "));
    }

    @Test
    void sourceCountCoversEveryUnsignedStepOrdinalDecimalWidthBoundary() {
        int[] ordinals = {
            9, 10,
            99, 100,
            999, 1_000,
            9_999, 10_000,
            99_999, 100_000,
            999_999, 1_000_000,
            9_999_999, 10_000_000,
            99_999_999, 100_000_000,
            999_999_999, 1_000_000_000,
            -1
        };
        int[] expected = {
            402, 403,
            403, 404,
            404, 405,
            405, 406,
            406, 407,
            407, 408,
            408, 409,
            409, 410,
            410, 411,
            411
        };
        List<MetalPointwiseFusionPlan.Opcode> opcodes =
                List.of(MetalPointwiseFusionPlan.Opcode.FLOOR,
                        MetalPointwiseFusionPlan.Opcode.CEIL);
        for (int index = 0; index < ordinals.length; index++) {
            assertEquals(
                    expected[index],
                    MetalPointwiseFusionPlanner.functionUtf8Bytes(ordinals[index], opcodes),
                    Integer.toUnsignedString(ordinals[index]));
        }
    }

    @Test
    void task0069VarianceIsAnExactFixedBarrierAndOtherVarianceIsMpsGraph() {
        var nodes = List.of(
                unary(MetalMpsGraphProgram.NodeKind.FLOOR, 0, 1),
                unary(MetalMpsGraphProgram.NodeKind.CEIL, 1, 2),
                MetalMpsGraphProgram.Node.statisticalReduction(
                        MetalMpsGraphProgram.NodeKind.VARIANCE, 2, 3, List.of(0), true, 0L),
                unary(MetalMpsGraphProgram.NodeKind.SIGN, 3, 4),
                unary(MetalMpsGraphProgram.NodeKind.RELU, 4, 5));
        var values = List.of(
                descriptor(1L), descriptor(1L), descriptor(1L),
                descriptor(1L), descriptor(1L), descriptor(1L));
        MetalPointwiseFusionPlan exact = MetalPointwiseFusionPlanner.plan(
                new MetalMpsGraphProgram(nodes),
                values,
                new int[] {0},
                new int[] {5},
                MetalPreparedRoute.CUSTOM_PROGRAM);

        assertEquals(List.of(
                        MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE,
                        MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM,
                        MetalPointwiseFusionPlan.StepKind.GENERATED_POINTWISE),
                exact.steps().stream().map(MetalPointwiseFusionPlan.Step::kind).toList());
        assertArrayEquals(new int[] {0, 2, 3, 5}, exact.materializedProgramValueIndices());

        var nonSingletonValues = List.of(descriptor(2L), descriptor());
        var nonSingleton = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.statisticalReduction(
                        MetalMpsGraphProgram.NodeKind.VARIANCE,
                        0,
                        1,
                        List.of(0),
                        false,
                        0L)));
        MetalPointwiseFusionPlan boundary = MetalPointwiseFusionPlanner.plan(
                nonSingleton,
                nonSingletonValues,
                new int[] {0},
                new int[] {1},
                MetalPreparedRoute.CUSTOM_PROGRAM);
        assertEquals(MetalPointwiseFusionPlan.StepKind.MPSGRAPH_BOUNDARY,
                boundary.steps().getFirst().kind());
    }

    @Test
    void thirtyThirdGeneratedUnitStopsAtTheAuthenticatedUnitCap() {
        var nodes = new ArrayList<MetalMpsGraphProgram.Node>();
        for (int unit = 0; unit < 33; unit++) {
            int input = unit * 3;
            nodes.add(unary(MetalMpsGraphProgram.NodeKind.FLOOR, input, input + 1));
            nodes.add(unary(MetalMpsGraphProgram.NodeKind.CEIL, input + 1, input + 2));
            if (unit != 32) nodes.add(MetalMpsGraphProgram.Node.neg(input + 2, input + 3));
        }
        var fixture = new Fixture(
                new MetalMpsGraphProgram(nodes), descriptors(99, 4L), new int[] {0}, new int[] {98});
        MetalPointwiseFusionPlan plan = plan(fixture);

        assertEquals(32, plan.generatedUnitCount());
        assertEquals(64, plan.instructions().size());
        assertEquals(MetalPointwiseFusionPlan.CapReason.UNIT_COUNT, plan.capReason());
        assertEquals(96, plan.firstRejectedNodePosition());
        assertEquals(MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM,
                plan.steps().get(plan.steps().size() - 2).kind());
        assertEquals(MetalPointwiseFusionPlan.StepKind.FIXED_CUSTOM,
                plan.steps().getLast().kind());
    }

    @Test
    void pointwiseGeometryRequiresRankOneThroughSixteenAndUnsignedElementCount() {
        var twoNodes = new MetalMpsGraphProgram(List.of(
                unary(MetalMpsGraphProgram.NodeKind.FLOOR, 0, 1),
                unary(MetalMpsGraphProgram.NodeKind.CEIL, 1, 2)));

        long[] rankSixteen = new long[16];
        java.util.Arrays.fill(rankSixteen, 1L);
        var rankSixteenPlan = plan(new Fixture(
                twoNodes, descriptors(3, rankSixteen), new int[] {0}, new int[] {2}));
        assertEquals(1, rankSixteenPlan.generatedUnitCount());

        var unsignedMaximumPlan = plan(new Fixture(
                twoNodes,
                descriptors(3, 65_535L, 65_537L),
                new int[] {0},
                new int[] {2}));
        assertEquals(1, unsignedMaximumPlan.generatedUnitCount());

        var scalarPlan = plan(new Fixture(
                twoNodes, descriptors(3), new int[] {0}, new int[] {2}));
        assertEquals(0, scalarPlan.generatedUnitCount());
        assertEquals(2, scalarPlan.steps().size());

        var onePastUnsignedPlan = plan(new Fixture(
                twoNodes,
                descriptors(3, 65_536L, 65_536L),
                new int[] {0},
                new int[] {2}));
        assertEquals(0, onePastUnsignedPlan.generatedUnitCount());
        assertEquals(2, onePastUnsignedPlan.steps().size());

        assertThrows(IllegalArgumentException.class, () -> descriptor(0L));
        assertThrows(IllegalArgumentException.class, () -> descriptor(new long[17]));
    }

    @Test
    void schemaTwentyPacksOddReferencePoolDirectlyBeforeAttributes() {
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
        List<MetalMpsGraphProgram.ValueDescriptor> values = descriptors(3, 4L);
        byte[] image = program.encodedProgramImage(
                values,
                new int[] {0},
                new int[] {2},
                MetalPreparedRoute.MPSGRAPH);
        int attributesOffset = MetalMpsGraphProgram.HEADER_BYTES
                + values.size() * MetalMpsGraphProgram.VALUE_DESCRIPTOR_BYTES
                + 2 * MetalMpsGraphProgram.NODE_DESCRIPTOR_BYTES
                + 3 * Long.BYTES
                + 3 * Long.BYTES
                + 7 * Integer.BYTES;
        ByteBuffer bytes = ByteBuffer.wrap(image).order(ByteOrder.LITTLE_ENDIAN);

        assertEquals(384, attributesOffset);
        assertEquals(400, image.length);
        assertEquals(1L, bytes.getLong(attributesOffset));
        assertEquals(0x3f80_0000L, bytes.getLong(attributesOffset + Long.BYTES));
    }

    @Test
    void plainMpsGraphImageHasNoFusionExtensionSections() {
        var fixture = new Fixture(
                new MetalMpsGraphProgram(List.of(MetalMpsGraphProgram.Node.neg(0, 1))),
                descriptors(2, 4L),
                new int[] {0},
                new int[] {1});
        byte[] image = fixture.program().encodedProgramImage(
                fixture.values(),
                fixture.feeds(),
                fixture.targets(),
                MetalPreparedRoute.MPSGRAPH);
        ByteBuffer header = ByteBuffer.wrap(image).order(ByteOrder.LITTLE_ENDIAN);

        assertEquals(MetalMpsGraphProgram.SCHEMA_VERSION, header.getInt(4));
        assertEquals(MetalMpsGraphProgram.HEADER_BYTES, header.getInt(8));
        assertEquals(MetalPreparedRoute.MPSGRAPH.wireIdentity(), header.getInt(16));
        assertEquals(0, header.getInt(20));
        assertEquals(0, header.getInt(24));
        for (int offset = 60; offset <= 100; offset += Integer.BYTES)
            assertEquals(0, header.getInt(offset));
        assertEquals(-1, header.getInt(104));
        for (int offset = 108; offset <= 120; offset += Integer.BYTES)
            assertEquals(0, header.getInt(offset));
    }

    private static MetalPointwiseFusionPlan plan(Fixture fixture) {
        return MetalPointwiseFusionPlanner.plan(
                fixture.program(),
                fixture.values(),
                fixture.feeds(),
                fixture.targets(),
                MetalPreparedRoute.CUSTOM_PROGRAM);
    }

    private static Fixture chain(int length) {
        var nodes = new ArrayList<MetalMpsGraphProgram.Node>(length);
        MetalMpsGraphProgram.NodeKind[] kinds = {
            MetalMpsGraphProgram.NodeKind.FLOOR,
            MetalMpsGraphProgram.NodeKind.CEIL,
            MetalMpsGraphProgram.NodeKind.SIGN,
            MetalMpsGraphProgram.NodeKind.RELU
        };
        for (int index = 0; index < length; index++)
            nodes.add(unary(kinds[index % kinds.length], index, index + 1));
        return new Fixture(
                new MetalMpsGraphProgram(nodes),
                descriptors(length + 1, 4L),
                new int[] {0},
                new int[] {length});
    }

    private static MetalMpsGraphProgram.Node unary(
            MetalMpsGraphProgram.NodeKind kind, int input, int output) {
        return MetalMpsGraphProgram.Node.generic(
                kind,
                new int[] {input},
                new int[] {output},
                MetalMpsGraphProgram.AttributeKind.NONE,
                new long[0]);
    }

    private static List<MetalMpsGraphProgram.ValueDescriptor> descriptors(int count, long... shape) {
        var result = new ArrayList<MetalMpsGraphProgram.ValueDescriptor>(count);
        for (int index = 0; index < count; index++) result.add(descriptor(shape));
        return List.copyOf(result);
    }

    private static MetalMpsGraphProgram.ValueDescriptor descriptor(long... shape) {
        return new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, shape, false);
    }

    private static MetalMpsGraphProgram.ValueDescriptor descriptor(
            DataType type, long... shape) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, shape, false);
    }

    private record Fixture(
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets) {
        private Fixture {
            values = List.copyOf(values);
            feeds = feeds.clone();
            targets = targets.clone();
        }

        @Override
        public int[] feeds() {
            return feeds.clone();
        }

        @Override
        public int[] targets() {
            return targets.clone();
        }
    }
}
