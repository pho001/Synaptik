package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalAnchorEpiloguePlannerTest {
    @Test
    void matmulBindsOrderedScalarAddReluAsOneAuthenticatedStep() {
        var values = List.of(
                f32(2, 3),
                f32(3, 4),
                f32(4),
                f32(2, 4),
                f32(2, 4),
                f32(2, 4),
                f32(2, 4));
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 3),
                MetalMpsGraphProgram.Node.scalarValue(
                        MetalMpsGraphProgram.NodeKind.SCALAR_MUL,
                        3,
                        4,
                        0x8000_0000),
                MetalMpsGraphProgram.Node.binary(MetalMpsGraphProgram.NodeKind.ADD, 2, 4, 5),
                unary(MetalMpsGraphProgram.NodeKind.RELU, 5, 6)));

        MetalPointwiseFusionPlan plan = plan(program, values, new int[] {0, 1, 2}, new int[] {6});

        assertEquals(1, plan.steps().size());
        MetalPointwiseFusionPlan.Step step = plan.steps().getFirst();
        assertEquals(MetalPointwiseFusionPlan.StepKind.ANCHOR_EPILOGUE, step.kind());
        assertEquals(1, step.anchorKindWire());
        assertEquals(4, step.memberCount());
        assertArrayEquals(new int[] {0, 1, 2, 6}, plan.materializedProgramValueIndices());
        assertEquals(List.of(1, 2, 3), plan.instructions().stream()
                .map(instruction -> assertInstanceOf(
                        MetalPointwiseFusionPlan.AnchorInstruction.class,
                        instruction).opcode().wire())
                .toList());
        MetalPointwiseFusionPlan.AnchorInstruction scalar =
                (MetalPointwiseFusionPlan.AnchorInstruction) plan.instructions().get(0);
        MetalPointwiseFusionPlan.AnchorInstruction add =
                (MetalPointwiseFusionPlan.AnchorInstruction) plan.instructions().get(1);
        assertEquals(0x8000_0000, scalar.raw0());
        assertEquals(1, add.raw0());
        assertEquals(2, add.raw1());
        String manifest = new String(plan.canonicalManifest(), StandardCharsets.US_ASCII);
        assertTrue(manifest.contains("step 0 4 0 4 0 4 0 3 0 1\n"));
        assertTrue(manifest.contains(
                "instruction 1 0 2 2 2 2 2 4 4294967295 5 0 0 0 0000000000000001 0000000000000002\n"));
    }

    @Test
    void convExternalAddUsesOrdinaryRightAlignedBroadcasting() {
        assertTrue(hasSingleConvAnchor(f32(3)));
        assertTrue(hasSingleConvAnchor(f32(1, 2, 1, 1)));
        assertFalse(hasSingleConvAnchor(f32(2)));
    }

    @Test
    void convNeverAbsorbsScalarMultiply() {
        var values = List.of(
                f32(1, 1, 2, 3),
                f32(2, 1, 1, 1),
                f32(1, 2, 2, 3),
                f32(1, 2, 2, 3));
        var program = new MetalMpsGraphProgram(List.of(
                conv(0, 1, 2),
                MetalMpsGraphProgram.Node.scalarValue(
                        MetalMpsGraphProgram.NodeKind.SCALAR_MUL,
                        2,
                        3,
                        0x3f80_0000)));

        MetalPointwiseFusionPlan plan = plan(program, values, new int[] {0, 1}, new int[] {3});

        assertTrue(plan.steps().stream()
                .noneMatch(step -> step.kind() == MetalPointwiseFusionPlan.StepKind.ANCHOR_EPILOGUE));
    }

    @Test
    void publicationAndFanoutKeepTheOriginalStepComposition() {
        var values = List.of(
                f32(2, 3), f32(3, 4), f32(2, 4), f32(2, 4), f32(2, 4));
        var nodes = List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                unary(MetalMpsGraphProgram.NodeKind.RELU, 2, 3),
                MetalMpsGraphProgram.Node.neg(2, 4));
        var program = new MetalMpsGraphProgram(nodes);

        MetalPointwiseFusionPlan fanout = plan(program, values, new int[] {0, 1}, new int[] {3, 4});
        assertTrue(fanout.steps().stream()
                .noneMatch(step -> step.kind() == MetalPointwiseFusionPlan.StepKind.ANCHOR_EPILOGUE));

        var publishedProgram = new MetalMpsGraphProgram(nodes.subList(0, 2));
        MetalPointwiseFusionPlan published = plan(
                publishedProgram,
                values.subList(0, 4),
                new int[] {0, 1},
                new int[] {2, 3});
        assertTrue(published.steps().stream()
                .noneMatch(step -> step.kind() == MetalPointwiseFusionPlan.StepKind.ANCHOR_EPILOGUE));
    }

    @Test
    void strictProfileNeverAdmitsAnAnchorEpilogue() {
        var values = List.of(f32(2, 3), f32(3, 4), f32(2, 4), f32(2, 4));
        var program = new MetalMpsGraphProgram(List.of(
                MetalMpsGraphProgram.Node.matmul(0, 1, 2),
                unary(MetalMpsGraphProgram.NodeKind.RELU, 2, 3)));

        MetalPointwiseFusionPlan plan = MetalPointwiseFusionPlanner.plan(
                NumericalProfile.STRICT_IEEE,
                program,
                values,
                new int[] {0, 1},
                new int[] {3},
                MetalPreparedRoute.CUSTOM_PROGRAM);

        assertTrue(plan.steps().stream()
                .noneMatch(step -> step.kind() == MetalPointwiseFusionPlan.StepKind.ANCHOR_EPILOGUE));
    }

    private static boolean hasSingleConvAnchor(MetalMpsGraphProgram.ValueDescriptor addend) {
        var values = List.of(
                f32(1, 1, 2, 3),
                f32(2, 1, 1, 1),
                f32(1, 2, 2, 3),
                addend,
                f32(1, 2, 2, 3));
        var program = new MetalMpsGraphProgram(List.of(
                conv(0, 1, 2),
                MetalMpsGraphProgram.Node.binary(MetalMpsGraphProgram.NodeKind.ADD, 2, 3, 4)));
        MetalPointwiseFusionPlan plan = plan(program, values, new int[] {0, 1, 3}, new int[] {4});
        return plan.steps().size() == 1
                && plan.steps().getFirst().kind()
                        == MetalPointwiseFusionPlan.StepKind.ANCHOR_EPILOGUE
                && plan.steps().getFirst().anchorKindWire() == 2;
    }

    private static MetalMpsGraphProgram.Node conv(int input, int weight, int output) {
        return MetalMpsGraphProgram.Node.generic(
                MetalMpsGraphProgram.NodeKind.CONV2D,
                new int[] {input, weight},
                new int[] {output},
                MetalMpsGraphProgram.AttributeKind.CONV_2D,
                new long[] {1, 1, 0, 0, 1, 1, 1});
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

    private static MetalPointwiseFusionPlan plan(
            MetalMpsGraphProgram program,
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            int[] feeds,
            int[] targets) {
        return MetalPointwiseFusionPlanner.plan(
                NumericalProfile.ACCELERATOR,
                program,
                values,
                feeds,
                targets,
                MetalPreparedRoute.CUSTOM_PROGRAM);
    }

    private static MetalMpsGraphProgram.ValueDescriptor f32(long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(DataType.FLOAT32, dimensions, false);
    }
}
