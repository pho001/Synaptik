package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuGeneratorSchema;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Validates active Model and backend identities against the retained pre-cutover allocation record. */
final class LowPrecisionIdentityContractTest {
    private static final String IDENTITIES = "/low-precision-identity-allocations-v1.tsv";

    @Test
    void modelCpuAndMetalProfileFreeIdentitiesRetainHistoricalAllocations() throws Exception {
        Map<String, String[]> rows = rows(resource(IDENTITIES), 7, 2);
        assertEquals(13, rows.size());
        assertArrayEquals(
                new DataType[] {
                    DataType.FLOAT64,
                    DataType.FLOAT32,
                    DataType.BFLOAT16,
                    DataType.INT32,
                    DataType.INT64,
                    DataType.BOOL,
                    DataType.FLOAT16
                },
                DataType.values());

        assertEquals(69, CpuGeneratorSchema.CURRENT_VERSION);

        assertEquals(
                20,
                staticInt(
                        "io.github.pho001.synaptik.backend.metal.MetalMpsGraphProgram",
                        "SCHEMA_VERSION"));
        assertEquals(
                7,
                staticInt(
                        "io.github.pho001.synaptik.backend.metal.MetalNativeApi", "ABI_VERSION"));
        assertEquals(
                30,
                staticInt(
                        "io.github.pho001.synaptik.backend.metal.MetalNegTuningBatch",
                        "CANDIDATE_SCHEMA_VERSION"));
        assertEquals(
                30,
                staticInt(
                        "io.github.pho001.synaptik.backend.metal.MetalNegTuningBatch",
                        "COMPATIBILITY_SCHEMA_VERSION"));
        assertEquals(
                30,
                staticInt(
                        "io.github.pho001.synaptik.backend.metal.MetalNegTuningBatch",
                        "ROUTE_POLICY_VERSION"));
        assertEquals(
                30,
                staticInt(
                        "io.github.pho001.synaptik.backend.metal.MetalNegTuningCodec",
                        "CODEC_VERSION"));
        assertEquals(
                30,
                staticInt(
                        "io.github.pho001.synaptik.backend.metal.MetalNegRouteCandidateGenerator",
                        "WORKLOAD_SIGNATURE_VERSION"));
        assertEquals(
                30,
                staticInt(
                        "io.github.pho001.synaptik.backend.metal.MetalNegRouteCandidateGenerator",
                        "EXACT_DEFAULT_POLICY"));

        assertAllocation(
                rows, "generator-schema", "cpu", "68", "NONE", "ACTIVATED_CPU_FLOAT16");
        assertAllocation(rows, "attribute-wires", "metal", "0..41", "UNCHANGED", "FROZEN");
        assertAllocation(
                rows, "candidate-schema", "metal", "29", "NONE",
                "ACTIVATED_METAL_CUSTOM_ONLY_LOW_PRECISION");
        assertAllocation(
                rows, "codec-version", "metal", "29", "NONE",
                "ACTIVATED_METAL_CUSTOM_ONLY_LOW_PRECISION");
        assertAllocation(
                rows, "compatibility-schema", "metal", "29", "NONE",
                "ACTIVATED_METAL_CUSTOM_ONLY_LOW_PRECISION");
        assertAllocation(
                rows, "dtype-wire-FLOAT16", "metal", "7", "NONE", "ACTIVATED_METAL_FLOAT16");
        assertAllocation(
                rows, "native-abi", "metal", "7", "NONE",
                "ACTIVATED_METAL_CUSTOM_ONLY_LOW_PRECISION");
        assertAllocation(rows, "operation-wires", "metal", "1..115", "UNCHANGED", "FROZEN");
        assertAllocation(rows, "program-schema", "metal", "19", "NONE", "ACTIVATED_METAL_FLOAT16");
        assertAllocation(
                rows, "route-policy-version", "metal", "29", "NONE",
                "ACTIVATED_METAL_CUSTOM_ONLY_LOW_PRECISION");
        assertAllocation(rows, "route-wires", "metal", "1..3", "UNCHANGED", "FROZEN");
        assertAllocation(
                rows, "workload-policy-version", "metal", "29", "NONE",
                "ACTIVATED_METAL_CUSTOM_ONLY_LOW_PRECISION");
        assertAllocation(
                rows,
                "DataType-FLOAT16",
                "model",
                "ORDINAL_6",
                "APPEND_AFTER_BOOL",
                "ACTIVATED_MODEL_ONLY");

        Class<?> program =
                Class.forName("io.github.pho001.synaptik.backend.metal.MetalMpsGraphProgram");
        Method dataTypeWire = program.getDeclaredMethod("dataTypeWire", DataType.class);
        dataTypeWire.setAccessible(true);
        DataType[] wiredTypes = {
            DataType.FLOAT64,
            DataType.FLOAT32,
            DataType.BFLOAT16,
            DataType.INT32,
            DataType.INT64,
            DataType.BOOL,
            DataType.FLOAT16
        };
        int[] expectedWires = {4, 1, 5, 2, 6, 3, 7};
        for (int index = 0; index < wiredTypes.length; index++) {
            assertEquals(expectedWires[index], dataTypeWire.invoke(null, wiredTypes[index]));
        }
        assertWireSequence(nestedEnum(program, "NodeKind"), 1, 115);
        assertWireSequence(nestedEnum(program, "AttributeKind"), 0, 42);
        assertWireSequence(
                Class.forName("io.github.pho001.synaptik.backend.metal.MetalPreparedRoute"), 1, 3);
    }

    private static int staticInt(String className, String fieldName) throws Exception {
        Class<?> type = Class.forName(className);
        Field field = type.getDeclaredField(fieldName);
        field.setAccessible(true);
        return field.getInt(null);
    }

    private static void assertAllocation(
            Map<String, String[]> rows,
            String identity,
            String scope,
            String current,
            String reserved,
            String status) {
        String[] row = rows.get(identity);
        assertEquals(scope, row[1]);
        assertEquals(current, row[3]);
        assertEquals(reserved, row[4]);
        assertEquals(status, row[5]);
    }

    private static Class<?> nestedEnum(Class<?> owner, String simpleName) {
        return Arrays.stream(owner.getDeclaredClasses())
                .filter(type -> type.getSimpleName().equals(simpleName))
                .findFirst().orElseThrow();
    }

    private static void assertWireSequence(Class<?> enumType, int first, int count)
            throws Exception {
        Object[] constants = enumType.getEnumConstants();
        assertEquals(count, constants.length);
        Method wireIdentity = enumType.getDeclaredMethod("wireIdentity");
        wireIdentity.setAccessible(true);
        for (int index = 0; index < constants.length; index++) {
            assertEquals(first + index, wireIdentity.invoke(constants[index]));
        }
    }

    private static Map<String, String[]> rows(String text, int columns, int keyColumn) {
        assertTrue(text.endsWith("\n"));
        assertFalse(text.contains("\r"));
        String[] lines = text.split("\n", -1);
        var result = new LinkedHashMap<String, String[]>();
        String prior = null;
        for (int index = 1; index < lines.length - 1; index++) {
            String line = lines[index];
            String[] fields = line.split("\t", -1);
            assertEquals(columns, fields.length, "contract row " + index);
            assertEquals("1", fields[0], "contract schema at row " + index);
            byte[] bytes = line.getBytes(StandardCharsets.UTF_8);
            String[] duplicate = result.putIfAbsent(fields[keyColumn], fields);
            if (duplicate != null) {
                assertArrayEquals(
                        String.join("\t", duplicate).getBytes(StandardCharsets.UTF_8),
                        bytes,
                        "duplicate key changed bytes: " + fields[keyColumn]);
                throw new AssertionError("duplicate contract key: " + fields[keyColumn]);
            }
            if (prior != null) {
                assertTrue(
                        prior.compareTo(line) < 0,
                        "contract rows must be strictly sorted at row " + index);
            }
            prior = line;
        }
        return Map.copyOf(result);
    }

    private static String resource(String name) throws Exception {
        try (InputStream input = LowPrecisionIdentityContractTest.class.getResourceAsStream(name)) {
            if (input == null) throw new AssertionError("missing resource " + name);
            return new String(input.readAllBytes(), StandardCharsets.UTF_8);
        }
    }
}
