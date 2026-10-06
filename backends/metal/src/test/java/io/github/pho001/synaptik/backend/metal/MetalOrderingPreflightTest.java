package io.github.pho001.synaptik.backend.metal;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.List;
import org.junit.jupiter.api.Test;

class MetalOrderingPreflightTest {
    private static final long UINT32_MAX = 0xffff_ffffL;

    @Test
    void JavaAbiAcceptsTheExactUnsignedBoundaryWithoutAllocatingResources() {
        var maximum = List.of(
                value(DataType.BOOL, UINT32_MAX),
                value(DataType.BOOL, UINT32_MAX));
        var sort = program(node(MetalMpsGraphProgram.NodeKind.SORT,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.SORT, 0, 0));
        assertDoesNotThrow(() -> validate(maximum, sort, new int[] {0}, new int[] {1}));

        var exactStride = List.of(
                value(DataType.BOOL, 1, UINT32_MAX),
                value(DataType.INT64, 1, UINT32_MAX));
        var argsort = program(node(MetalMpsGraphProgram.NodeKind.ARGSORT,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.SORT, 1, 1));
        assertDoesNotThrow(() -> validate(
                exactStride, argsort, new int[] {0}, new int[] {1}));

        var topValues = List.of(
                value(DataType.BOOL, UINT32_MAX),
                value(DataType.BOOL, UINT32_MAX),
                value(DataType.INT64, UINT32_MAX));
        var top = program(node(MetalMpsGraphProgram.NodeKind.TOP_K,
                new int[] {0}, new int[] {1, 2},
                MetalMpsGraphProgram.AttributeKind.TOP_K,
                0, UINT32_MAX, 1, 0));
        assertDoesNotThrow(() -> validate(
                topValues, top, new int[] {0}, new int[] {1, 2}));
    }

    @Test
    void JavaAbiRejectsOnePastAndProductOverflowBeforeNativeInvocation() {
        var onePast = List.of(
                value(DataType.BOOL, UINT32_MAX + 1L),
                value(DataType.BOOL, UINT32_MAX + 1L));
        var sort = program(node(MetalMpsGraphProgram.NodeKind.SORT,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.SORT, 0, 0));
        assertThrows(IllegalArgumentException.class,
                () -> validate(onePast, sort, new int[] {0}, new int[] {1}));

        var productOverflow = List.of(
                value(DataType.INT32, 65_536, 65_536),
                value(DataType.INT64, 65_536, 65_536));
        var argsort = program(node(MetalMpsGraphProgram.NodeKind.ARGSORT,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.SORT, 0, 0));
        assertThrows(IllegalArgumentException.class,
                () -> validate(productOverflow, argsort, new int[] {0}, new int[] {1}));
    }

    @Test
    void JavaAbiAuthenticatesEveryOutputRoleAttributeAndCustomOnlyRoute() {
        var descriptors = List.of(
                value(DataType.FLOAT32, 2, 4),
                value(DataType.FLOAT32, 2, 2),
                value(DataType.INT64, 2, 2));
        var valid = program(node(MetalMpsGraphProgram.NodeKind.TOP_K,
                new int[] {0}, new int[] {1, 2},
                MetalMpsGraphProgram.AttributeKind.TOP_K, 1, 2, 0, 1));
        for (NumericalProfile profile : NumericalProfile.values()) {
            assertDoesNotThrow(() -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                    profile, descriptors, valid, new int[] {0}, new int[] {1},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            assertDoesNotThrow(() -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                    profile, descriptors, valid, new int[] {0}, new int[] {2},
                    MetalPreparedRoute.CUSTOM_PROGRAM));
            assertThrows(IllegalArgumentException.class,
                    () -> MetalNativeApi.ProgramExecutableAbi.validateCreate(
                            profile, descriptors, valid, new int[] {0}, new int[] {1, 2},
                            MetalPreparedRoute.MPSGRAPH));
        }

        var wrongIndexType = List.of(
                value(DataType.FLOAT32, 2, 4),
                value(DataType.FLOAT32, 2, 2),
                value(DataType.INT32, 2, 2));
        assertThrows(IllegalArgumentException.class,
                () -> validate(wrongIndexType, valid, new int[] {0}, new int[] {1, 2}));

        var wrongValueShape = List.of(
                value(DataType.FLOAT32, 2, 4),
                value(DataType.FLOAT32, 2, 3),
                value(DataType.INT64, 2, 3));
        assertThrows(IllegalArgumentException.class,
                () -> validate(wrongValueShape, valid, new int[] {0}, new int[] {1, 2}));

        var badAxis = program(node(MetalMpsGraphProgram.NodeKind.TOP_K,
                new int[] {0}, new int[] {1, 2},
                MetalMpsGraphProgram.AttributeKind.TOP_K, 2, 2, 1, 1));
        assertThrows(IllegalArgumentException.class,
                () -> validate(descriptors, badAxis, new int[] {0}, new int[] {1, 2}));
    }

    @Test
    void JavaAbiRejectsBooleanArgExtremaAndAcceptsScalarIndexOutput() {
        var arg = program(node(MetalMpsGraphProgram.NodeKind.ARG_MIN,
                new int[] {0}, new int[] {1},
                MetalMpsGraphProgram.AttributeKind.ARG_EXTREMA, 0, 0, 2));
        assertDoesNotThrow(() -> validate(
                List.of(value(DataType.INT64, 4), value(DataType.INT64)),
                arg, new int[] {0}, new int[] {1}));
        assertThrows(IllegalArgumentException.class, () -> validate(
                List.of(value(DataType.BOOL, 4), value(DataType.INT64)),
                arg, new int[] {0}, new int[] {1}));
    }

    private static MetalMpsGraphProgram.ValueDescriptor value(
            DataType type, long... dimensions) {
        return new MetalMpsGraphProgram.ValueDescriptor(type, dimensions, false);
    }

    private static MetalMpsGraphProgram.Node node(
            MetalMpsGraphProgram.NodeKind kind,
            int[] inputs,
            int[] outputs,
            MetalMpsGraphProgram.AttributeKind attributes,
            long... words) {
        return MetalMpsGraphProgram.Node.generic(kind, inputs, outputs, attributes, words);
    }

    private static MetalMpsGraphProgram program(MetalMpsGraphProgram.Node node) {
        return new MetalMpsGraphProgram(List.of(node));
    }

    private static void validate(
            List<MetalMpsGraphProgram.ValueDescriptor> values,
            MetalMpsGraphProgram program,
            int[] feeds,
            int[] targets) {
        MetalNativeApi.ProgramExecutableAbi.validateCreate(
                NumericalProfile.STRICT_IEEE,
                values,
                program,
                feeds,
                targets,
                MetalPreparedRoute.CUSTOM_PROGRAM);
    }
}
