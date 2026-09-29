package io.github.pho001.synaptik.testing.conformance;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.metal.MetalCapabilityProvider;
import io.github.pho001.synaptik.config.compile.NumericalProfile;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationAttrs;
import io.github.pho001.synaptik.model.operation.attention.*;
import io.github.pho001.synaptik.model.operation.convolution.*;
import io.github.pho001.synaptik.model.operation.elementwise.binary.*;
import io.github.pho001.synaptik.model.operation.elementwise.cast.*;
import io.github.pho001.synaptik.model.operation.elementwise.classification.*;
import io.github.pho001.synaptik.model.operation.elementwise.comparison.*;
import io.github.pho001.synaptik.model.operation.elementwise.logical.*;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.*;
import io.github.pho001.synaptik.model.operation.elementwise.selection.*;
import io.github.pho001.synaptik.model.operation.elementwise.unary.*;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.linalg.*;
import io.github.pho001.synaptik.model.operation.loss.*;
import io.github.pho001.synaptik.model.operation.normalization.*;
import io.github.pho001.synaptik.model.operation.ordering.*;
import io.github.pho001.synaptik.model.operation.pooling.*;
import io.github.pho001.synaptik.model.operation.random.*;
import io.github.pho001.synaptik.model.operation.recurrent.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.RecordComponent;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;

/**
 * Freezes canonical current F32 capability witnesses and the separately declared low-precision
 * target disposition without turning either artifact into runtime policy.
 *
 * <p>The providers expose a predicate rather than an enumerable domain. Consequently each ledger
 * row is one real, fully described occurrence submitted to the actual provider. The 115-row base
 * is the current complete Metal operation-kind inventory, with F32 data roles wherever the kind
 * admits them and natural BOOL, index, or state roles retained. Ordered F32 cast pairs are added
 * separately. Existing provider boundary tests remain authoritative for the larger shape and
 * attribute domains; this snapshot detects drift in these named witnesses only.</p>
 *
 * <p>The F32 provider output ends at {@code supported}. The corresponding mapped BF16 and FP16
 * occurrences are queried independently and recorded in {@code bf16_current_supported} and
 * {@code fp16_current_supported}; neither answer is inferred from F32. The following BF16/FP16
 * disposition, target-profile, and exclusion columns are the approved migration contract and do
 * not substitute for those observed provider answers.</p>
 */
final class LowPrecisionCapabilityLedgerTest {
    private static final String RESOURCE = "/low-precision-capability-ledger-v1.tsv";
    private static final String HEADER = "schema\tbackend\tprofile\toccurrence\tbasis\tkind\tattrs"
            + "\tinputs\toutputs\tinput_arity\toutput_arity\tsupported"
            + "\tbf16_current_supported\tfp16_current_supported\tbf16_target\tfp16_target"
            + "\ttarget_profile\texclusion\n";

    @Test
    void checkedLedgerEqualsCanonicalCurrentProviderAnswers() throws Exception {
        String actual = renderLedger();
        dumpIfRequested(actual);
        String expected = resource(RESOURCE);
        assertCanonicalRows(expected);
        assertCanonicalRows(actual);
        assertLedgerEquals(expected, actual);
    }

    @Test
    void snapshotContainsBothProfilesBackendsAndExplicitSupportOutcomes() throws Exception {
        List<Seed> seeds = seeds();
        String ledger = resource(RESOURCE);
        Map<String, byte[]> rows = rowsByKey(ledger);
        assertEquals(seeds.size() * 4, rows.size());
        for (String backend : List.of("cpu", "metal")) {
            for (NumericalProfile profile : NumericalProfile.values()) {
                long count = rows.keySet().stream()
                        .filter(key -> key.startsWith(backend + "|" + profile + "|"))
                        .count();
                assertEquals(seeds.size(), count, backend + " " + profile);
            }
        }
        assertTrue(ledger.contains("\ttrue\ttrue\ttrue\tPRESERVE_CURRENT\t"));
        assertTrue(ledger.contains("\tfalse\tfalse\tfalse\tEXCLUDED\t"));
        assertTrue(ledger.contains("\tCURRENT_F32_UNSUPPORTED\n"));
        assertTrue(ledger.contains("\tSTRICT_SNAPSHOT_ONLY\n"));
        List<String[]> acceleratorClosure = ledger.lines()
                .skip(1)
                .map(line -> line.split("\t", -1))
                .filter(fields -> fields[2].equals("ACCELERATOR"))
                .filter(fields -> fields[4].equals("F32_OCCURRENCE"))
                .filter(fields -> fields[11].equals("true"))
                .toList();
        assertFalse(acceleratorClosure.isEmpty());
        assertTrue(acceleratorClosure.stream().allMatch(fields ->
                fields[12].equals("true")
                        && fields[13].equals("true")
                        && fields[14].equals("PRESERVE_CURRENT")
                        && fields[15].equals("PRESERVE_CURRENT")));
    }

    @Test
    void metalAcceleratorClosesEverySupportedFloat32OccurrenceForBothLowTypes() {
        var provider = new MetalCapabilityProvider();
        int supported = 0;
        for (Seed seed : seeds()) {
            OperationCapabilityQuery query = new OperationCapabilityQuery(
                    NumericalProfile.ACCELERATOR,
                    seed.query().operation(),
                    seed.query().inputs(),
                    seed.query().outputs());
            if (hasFloat32Role(query) && provider.supports(query)) {
                supported++;
                assertTrue(
                        provider.supports(toBfloat16(query)),
                        "BFLOAT16 " + seed.id());
                assertTrue(
                        provider.supports(toFloat16(query)),
                        "FLOAT16 " + seed.id());
            }
        }
        assertTrue(supported > 0);
    }

    @Test
    void metalStrictLowSupportNeverExceedsCurrentStrictFloat32Support() {
        var provider = new MetalCapabilityProvider();
        for (Seed seed : seeds()) {
            OperationCapabilityQuery query = new OperationCapabilityQuery(
                    NumericalProfile.STRICT_IEEE,
                    seed.query().operation(),
                    seed.query().inputs(),
                    seed.query().outputs());
            if (!hasFloat32Role(query)) {
                continue;
            }
            boolean lowSupported =
                    provider.supports(toBfloat16(query)) || provider.supports(toFloat16(query));
            if (lowSupported) {
                assertTrue(provider.supports(query), seed.id());
            }
        }
    }
    @Test
    void cpuStrictIeeeSupportsFloat16ForEverySupportedFloat32Occurrence() {
        assertFloat16Closure(NumericalProfile.STRICT_IEEE);
    }

    @Test
    void cpuAcceleratorSupportsFloat16ForEverySupportedFloat32Occurrence() {
        assertFloat16Closure(NumericalProfile.ACCELERATOR);
    }

    private static void assertFloat16Closure(NumericalProfile profile) {
        var provider = new CpuCapabilityProvider();
        int supported = 0;
        for (Seed seed : seeds()) {
            OperationCapabilityQuery query = new OperationCapabilityQuery(
                    profile, seed.query().operation(), seed.query().inputs(), seed.query().outputs());
            if (hasFloat32Role(query) && provider.supports(query)) {
                supported++;
                assertTrue(provider.supports(toFloat16(query)), profile + " " + seed.id());
            }
        }
        assertEquals(114, supported, profile.name());
    }


    private static String renderLedger() {
        List<Seed> seeds = seeds();
        Map<String, BackendCapabilityProvider> providers = new TreeMap<>();
        providers.put("cpu", new CpuCapabilityProvider());
        providers.put("metal", new MetalCapabilityProvider());
        var lines = new ArrayList<String>();
        for (var backend : providers.entrySet()) {
            for (NumericalProfile profile : NumericalProfile.values()) {
                for (Seed seed : seeds) {
                    OperationCapabilityQuery query = new OperationCapabilityQuery(
                            profile,
                            seed.query().operation(),
                            seed.query().inputs(),
                            seed.query().outputs());
                    boolean supported = backend.getValue().supports(query);
                    String bfloat16Supported = "NOT_APPLICABLE";
                    if (hasFloat32Role(query)) {
                        bfloat16Supported = Boolean.toString(
                                backend.getValue().supports(toBfloat16(query)));
                    }
                    String float16Supported = "NOT_APPLICABLE";
                    if (hasFloat32Role(query)) {
                        float16Supported = Boolean.toString(
                                backend.getValue().supports(toFloat16(query)));
                    }
                    Target target = target(
                            backend.getKey(),
                            profile,
                            query,
                            supported,
                            bfloat16Supported,
                            float16Supported);
                    lines.add(row(
                            backend.getKey(),
                            profile,
                            seed,
                            query,
                            supported,
                            bfloat16Supported,
                            float16Supported,
                            target));
                }
            }
        }
        lines.sort(String::compareTo);
        return HEADER + String.join("\n", lines) + "\n";
    }

    private static String row(
            String backend,
            NumericalProfile profile,
            Seed seed,
            OperationCapabilityQuery query,
            boolean supported,
            String bfloat16Supported,
            String float16Supported,
            Target target) {
        return String.join(
                "\t",
                "1",
                backend,
                profile.name(),
                seed.id(),
                basis(query),
                kind(query.operation()),
                canonical(query.operation().attrs()),
                descriptors(query.inputs()),
                descriptors(query.outputs()),
                Integer.toString(query.inputs().size()),
                Integer.toString(query.outputs().size()),
                Boolean.toString(supported),
                bfloat16Supported,
                float16Supported,
                target.bfloat16(),
                target.float16(),
                target.profile(),
                target.exclusion());
    }

    private static Target target(
            String backend,
            NumericalProfile profile,
            OperationCapabilityQuery query,
            boolean supported,
            String bfloat16Supported,
            String float16Supported) {
        if (profile == NumericalProfile.STRICT_IEEE) {
            return new Target("PRESERVE_CURRENT", "PRESERVE_CURRENT", "NONE",
                    "STRICT_SNAPSHOT_ONLY");
        }
        if (!hasFloat32Role(query)) {
            return new Target("NOT_APPLICABLE", "NOT_APPLICABLE", "NONE",
                    "NON_F32_CONTROL");
        }
        boolean currentBfloat16 = Boolean.parseBoolean(bfloat16Supported);
        boolean currentFloat16 = Boolean.parseBoolean(float16Supported);
        if (!supported) {
            return new Target(
                    currentBfloat16 ? "PRESERVE_CURRENT" : "EXCLUDED",
                    "EXCLUDED",
                    "NONE",
                    currentBfloat16
                            ? "CURRENT_F32_UNSUPPORTED_PRESERVE_BF16"
                            : "CURRENT_F32_UNSUPPORTED");
        }
        String targetProfile = isExactSubstrate(query.operation())
                ? "PROFILE_COMMON_EXACT" : "ACCELERATOR";
        String bfloat16;
        String float16;
        String exclusion = "NONE";
        if (currentBfloat16) {
            bfloat16 = "PRESERVE_CURRENT";
        } else if (backend.equals("metal")
                || query.operation().kind() == DropoutKind.DROPOUT) {
            bfloat16 = "REQUIRE_F32_PARITY";
        } else {
            bfloat16 = "NEEDS_AUDIT";
            exclusion = "BF16_CURRENT_FALSE_NEEDS_AUDIT";
        }
        float16 = currentFloat16 ? "PRESERVE_CURRENT" : "REQUIRE_F32_PARITY";
        return new Target(bfloat16, float16, targetProfile, exclusion);
    }

    private static OperationCapabilityQuery toBfloat16(OperationCapabilityQuery query) {
        return new OperationCapabilityQuery(
                query.numericalProfile(),
                new Operation(query.operation().kind(), toBfloat16(query.operation().attrs())),
                query.inputs().stream().map(LowPrecisionCapabilityLedgerTest::toBfloat16).toList(),
                query.outputs().stream().map(LowPrecisionCapabilityLedgerTest::toBfloat16).toList());
    }

    private static TensorDescriptor toBfloat16(TensorDescriptor descriptor) {
        DataType type = descriptor.dataType() == DataType.FLOAT32
                ? DataType.BFLOAT16 : descriptor.dataType();
        return new TensorDescriptor(
                type, descriptor.shape(), descriptor.layout(), descriptor.requiresGrad());
    }

    private static OperationAttrs toBfloat16(OperationAttrs attrs) {
        if (attrs instanceof ScalarValueAttrs scalar) {
            return new ScalarValueAttrs(toBfloat16(scalar.value()));
        }
        if (attrs instanceof ClampRangeAttrs clamp) {
            return new ClampRangeAttrs(
                    toBfloat16(clamp.minValue()), toBfloat16(clamp.maxValue()));
        }
        if (attrs instanceof CastAttrs cast) {
            DataType target = cast.targetDataType() == DataType.FLOAT32
                    ? DataType.BFLOAT16 : cast.targetDataType();
            return new CastAttrs(target);
        }
        if (attrs instanceof PadAttrs pad) {
            return new PadAttrs(pad.before(), pad.after(), toBfloat16(pad.constantValue()));
        }
        if (attrs instanceof ScaledDotProductAttentionAttrs attention) {
            return new ScaledDotProductAttentionAttrs(
                    attention.scale().map(LowPrecisionCapabilityLedgerTest::toBfloat16),
                    attention.causal());
        }
        if (attrs instanceof BatchNormInferenceAttrs batchNorm) {
            return new BatchNormInferenceAttrs(
                    batchNorm.channelAxis(), toBfloat16(batchNorm.epsilon()));
        }
        if (attrs instanceof BatchNormTrainingAttrs batchNorm) {
            return new BatchNormTrainingAttrs(
                    batchNorm.channelAxis(),
                    toBfloat16(batchNorm.momentum()),
                    toBfloat16(batchNorm.epsilon()));
        }
        if (attrs instanceof LayerNormAttrs layerNorm) {
            return new LayerNormAttrs(
                    layerNorm.normalizedShape(), toBfloat16(layerNorm.epsilon()));
        }
        if (attrs instanceof RmsNormAttrs rmsNorm) {
            return new RmsNormAttrs(
                    rmsNorm.normalizedShape(), toBfloat16(rmsNorm.epsilon()));
        }
        return attrs;
    }

    private static ScalarValue toBfloat16(ScalarValue value) {
        return value.dataType() == DataType.FLOAT32
                ? ScalarValue.bfloat16(value.float32Value()) : value;
    }
    private static OperationCapabilityQuery toFloat16(OperationCapabilityQuery query) {
        return new OperationCapabilityQuery(
                query.numericalProfile(),
                new Operation(query.operation().kind(), toFloat16(query.operation().attrs())),
                query.inputs().stream().map(LowPrecisionCapabilityLedgerTest::toFloat16).toList(),
                query.outputs().stream().map(LowPrecisionCapabilityLedgerTest::toFloat16).toList());
    }

    private static TensorDescriptor toFloat16(TensorDescriptor descriptor) {
        DataType type = descriptor.dataType() == DataType.FLOAT32
                ? DataType.FLOAT16 : descriptor.dataType();
        return new TensorDescriptor(
                type, descriptor.shape(), descriptor.layout(), descriptor.requiresGrad());
    }

    private static OperationAttrs toFloat16(OperationAttrs attrs) {
        if (attrs instanceof ScalarValueAttrs scalar) {
            return new ScalarValueAttrs(toFloat16(scalar.value()));
        }
        if (attrs instanceof ClampRangeAttrs clamp) {
            return new ClampRangeAttrs(
                    toFloat16(clamp.minValue()), toFloat16(clamp.maxValue()));
        }
        if (attrs instanceof CastAttrs cast) {
            DataType target = cast.targetDataType() == DataType.FLOAT32
                    ? DataType.FLOAT16 : cast.targetDataType();
            return new CastAttrs(target);
        }
        if (attrs instanceof PadAttrs pad) {
            return new PadAttrs(pad.before(), pad.after(), toFloat16(pad.constantValue()));
        }
        return attrs;
    }

    private static ScalarValue toFloat16(ScalarValue value) {
        return value.dataType() == DataType.FLOAT32
                ? ScalarValue.float16(value.float32Value()) : value;
    }


    private static boolean isExactSubstrate(Operation operation) {
        return operation.kind() == CastKind.CAST
                || operation.kind() instanceof ShapeTransformKind
                || operation.kind() instanceof AxisTransformKind
                || operation.kind() == ContiguousKind.CONTIGUOUS
                || operation.kind() == SelectKind.SELECT
                || operation.kind() == SliceKind.SLICE
                || operation.kind() == SliceKind.SLICE_UPDATE
                || operation.kind() == PadKind.PAD
                || operation.kind() instanceof TensorCompositionKind
                || operation.kind() == TileKind.TILE
                || operation.kind() instanceof AxisGatherKind
                || operation.kind() == GatherNdKind.GATHER_ND
                || operation.kind() == AxisScatterKind.SCATTER_ELEMENTS
                || operation.kind() == ScatterNdKind.SCATTER_ND
                || operation.kind() instanceof WindowTransformKind;
    }

    private static boolean hasFloat32Role(OperationCapabilityQuery query) {
        return java.util.stream.Stream.concat(query.inputs().stream(), query.outputs().stream())
                .anyMatch(descriptor -> descriptor.dataType() == DataType.FLOAT32);
    }

    private static String basis(OperationCapabilityQuery query) {
        return hasFloat32Role(query) ? "F32_OCCURRENCE" : "NON_F32_CONTROL";
    }

    private static String kind(Operation operation) {
        Enum<?> kind = (Enum<?>) operation.kind();
        return kind.getClass().getName() + "." + kind.name();
    }

    private static String descriptors(List<TensorDescriptor> descriptors) {
        var result = new ArrayList<String>(descriptors.size());
        for (int index = 0; index < descriptors.size(); index++) {
            TensorDescriptor descriptor = descriptors.get(index);
            String layout = descriptor.layout().map(value -> "{kind=" + value.kind()
                    + ",strides=" + array(value.strides())
                    + ",offset=" + value.storageOffset()
                    + ",view=" + value.isView()
                    + ",span=" + value.referencedElementSpan() + "}").orElse("UNRESOLVED");
            result.add(index + ":{type=" + descriptor.dataType()
                    + ",shape=" + array(descriptor.shape().toLongArray())
                    + ",layout=" + layout
                    + ",grad=" + descriptor.requiresGrad() + "}");
        }
        return "[" + String.join(";", result) + "]";
    }

    private static String canonical(Object value) {
        if (value == null) return "null";
        if (value == NoOperationAttrs.INSTANCE) return NoOperationAttrs.class.getName() + "{}";
        if (value instanceof ScalarValue scalar) return scalar(scalar);
        if (value instanceof Shape shape) return "Shape" + array(shape.toLongArray());
        if (value instanceof Optional<?> optional) {
            return optional.map(element -> "Optional[" + canonical(element) + "]")
                    .orElse("Optional.empty");
        }
        if (value instanceof List<?> list) {
            return "[" + list.stream().map(LowPrecisionCapabilityLedgerTest::canonical)
                    .reduce((left, right) -> left + "," + right).orElse("") + "]";
        }
        if (value instanceof long[] array) return array(array);
        if (value instanceof int[] array) return array(array);
        if (value instanceof Enum<?> enumeration) {
            return enumeration.getDeclaringClass().getName() + "." + enumeration.name();
        }
        if (value instanceof Double number) {
            return "f64:0x" + String.format("%016X", Double.doubleToRawLongBits(number));
        }
        if (value instanceof Float number) {
            return "f32:0x" + String.format("%08X", Float.floatToRawIntBits(number));
        }
        if (value instanceof Number || value instanceof Boolean) return value.toString();
        if (value.getClass().isRecord()) return canonicalRecord(value);
        throw new AssertionError("unsupported canonical value " + value.getClass().getName());
    }

    private static String canonicalRecord(Object value) {
        RecordComponent[] components = value.getClass().getRecordComponents();
        var fields = new ArrayList<String>(components.length);
        for (RecordComponent component : components) {
            try {
                fields.add(component.getName() + "=" + canonical(component.getAccessor().invoke(value)));
            } catch (IllegalAccessException | InvocationTargetException failure) {
                throw new AssertionError("cannot encode " + value.getClass().getName()
                        + "." + component.getName(), failure);
            }
        }
        return value.getClass().getName() + "{" + String.join(",", fields) + "}";
    }

    private static String scalar(ScalarValue value) {
        String bits = switch (value.dataType()) {
            case FLOAT64 -> String.format("%016X",
                    Double.doubleToRawLongBits(value.float64Value()));
            case FLOAT32 -> String.format("%08X",
                    Float.floatToRawIntBits(value.float32Value()));
            case BFLOAT16 -> String.format("%04X", value.bfloat16Bits() & 0xffff);
            case FLOAT16 -> String.format("%04X", value.float16Bits() & 0xffff);
            case INT64 -> String.format("%016X", value.int64Value());
            case INT32 -> String.format("%08X", value.int32Value());
            case BOOL -> value.booleanValue() ? "01" : "00";
        };
        return "Scalar{" + value.dataType() + ":0x" + bits + "}";
    }

    private static String array(long[] values) {
        return Arrays.toString(values).replace(" ", "");
    }

    private static String array(int[] values) {
        return Arrays.toString(values).replace(" ", "");
    }

    private static void assertCanonicalRows(String ledger) {
        assertTrue(ledger.endsWith("\n"), "canonical ledger must end in one newline");
        assertFalse(ledger.contains("\r"), "canonical ledger must use LF");
        assertEquals(ledger, new String(ledger.getBytes(StandardCharsets.UTF_8),
                StandardCharsets.UTF_8));
        rowsByKey(ledger);
    }

    private static Map<String, byte[]> rowsByKey(String ledger) {
        String[] lines = ledger.split("\n", -1);
        assertEquals(HEADER.stripTrailing(), lines[0]);
        var rows = new LinkedHashMap<String, byte[]>();
        String priorLine = null;
        for (int index = 1; index < lines.length - 1; index++) {
            String line = lines[index];
            String[] fields = line.split("\t", -1);
            assertEquals(18, fields.length, "ledger row " + index);
            String key = fields[1] + "|" + fields[2] + "|" + fields[3];
            byte[] bytes = line.getBytes(StandardCharsets.UTF_8);
            byte[] duplicate = rows.putIfAbsent(key, bytes);
            if (duplicate != null) {
                assertArrayEquals(duplicate, bytes, "duplicate key changed bytes: " + key);
                throw new AssertionError("duplicate canonical ledger key: " + key);
            }
            if (priorLine != null) assertTrue(priorLine.compareTo(line) < 0,
                    "ledger rows must be strictly sorted at row " + index);
            priorLine = line;
        }
        return Map.copyOf(rows);
    }

    private static void dumpIfRequested(String actual) throws Exception {
        String path = System.getProperty("synaptik.low.precision.ledger.dump");
        if (path == null || path.isBlank())
            path = System.getenv("SYNAPTIK_LOW_PRECISION_LEDGER_DUMP");
        if (path != null && !path.isBlank())
            Files.writeString(Path.of(path), actual, StandardCharsets.UTF_8);
    }

    private static void assertLedgerEquals(String expected, String actual) {
        Map<String, byte[]> expectedRows = rowsByKey(expected);
        Map<String, byte[]> actualRows = rowsByKey(actual);
        for (String key : expectedRows.keySet()) {
            if (!actualRows.containsKey(key)) {
                throw new AssertionError("capability ledger missing row " + key + "\nexpected: "
                        + new String(expectedRows.get(key), StandardCharsets.UTF_8));
            }
            byte[] expectedBytes = expectedRows.get(key);
            byte[] actualBytes = actualRows.get(key);
            if (!Arrays.equals(expectedBytes, actualBytes)) {
                throw new AssertionError("capability ledger changed row " + key
                        + "\nexpected: " + new String(expectedBytes, StandardCharsets.UTF_8)
                        + "\nactual:   " + new String(actualBytes, StandardCharsets.UTF_8));
            }
        }
        for (String key : actualRows.keySet()) {
            if (!expectedRows.containsKey(key)) {
                throw new AssertionError("capability ledger added row " + key + "\nactual: "
                        + new String(actualRows.get(key), StandardCharsets.UTF_8));
            }
        }
        assertArrayEquals(expected.getBytes(StandardCharsets.UTF_8),
                actual.getBytes(StandardCharsets.UTF_8), "canonical ledger bytes");
    }

    private static String resource(String name) throws Exception {
        try (InputStream input = LowPrecisionCapabilityLedgerTest.class.getResourceAsStream(name)) {
            if (input == null) throw new AssertionError("missing resource " + name);
            return new String(input.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    private static List<Seed> seeds() {
        var result = new ArrayList<Seed>();
        for (Row row : Row.values()) {
            if (row == Row.CAST) {
                for (DataType type : DataType.values()) {
                    result.add(seed(row, "FLOAT32-to-" + type,
                            cast(DataType.FLOAT32, type)));
                    if (type != DataType.FLOAT32) {
                        result.add(seed(row, type + "-to-FLOAT32",
                                cast(type, DataType.FLOAT32)));
                    }
                }
            } else {
                result.add(seed(row, "base", representative(row)));
            }
        }
        assertEquals(127, result.size(), "115 operations plus twelve additional F32 cast pairs");
        return List.copyOf(result);
    }

    private static Seed seed(Row row, String variant, OperationCapabilityQuery query) {
        return new Seed(String.format("%03d-%s/%s", row.ordinal() + 1, row.name(), variant), query);
    }

    private static OperationCapabilityQuery representative(Row kind) {
        return switch (kind) {
            case NEG, ABS, RECIPROCAL, LOG, LOG1P, EXP, EXPM1, ERF, SQRT, RSQRT, FLOOR, CEIL,
                    SIGN, RELU, SIGMOID, TANH, GELU, GELU_TANH_APPROXIMATION, SILU -> unary(kind);
            case ADD, SUB, MUL, DIV, TENSOR_MIN, TENSOR_MAX, TENSOR_POW -> binary(kind);
            case SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_DIV, SCALAR_POW, SCALAR_MIN,
                    SCALAR_MAX, CLAMP -> scalar(kind);
            case SUM, MEAN, REDUCTION_MIN, REDUCTION_MAX, PROD, ALL, ANY, ARG_MAX, ARG_MIN,
                    LOG_SUM_EXP, VARIANCE, STANDARD_DEVIATION, L1_NORM, L2_NORM -> reduction(kind);
            case GT, GE, LT, LE, EQ, NE -> comparison(kind);
            case IS_FINITE, IS_NAN, IS_INF -> classification(kind);
            case LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> logical(kind);
            case RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE, CONTIGUOUS -> affine(kind);
            case GATHER, ONE_HOT, SCATTER_ELEMENTS, GATHER_ELEMENTS, SCATTER_ADD, GATHER_ND,
                    SCATTER_ND -> indexing(kind);
            case UNFOLD_AXIS, FOLD_AXIS, UNFOLD2D, FOLD2D, UNFOLD3D, FOLD3D -> window(kind);
            case SELECT, PAD, SLICE, SLICE_UPDATE, CONCAT, STACK, TILE -> movement(kind);
            case MATMUL -> query(new Operation(MatmulKind.MATMUL, NoOperationAttrs.INSTANCE),
                    List.of(f32(2, 3), f32(3, 4)), List.of(f32(2, 4)));
            case WHERE -> query(new Operation(WhereSelectionKind.WHERE, NoOperationAttrs.INSTANCE),
                    List.of(bool(3), f32(2, 1), f32(1, 3)), List.of(f32(2, 3)));
            case CAST -> throw new AssertionError("CAST is expanded as ordered F32 pairs");
            case CUM_SUM, CUM_PROD -> query(new Operation(kind == Row.CUM_SUM
                            ? CumulativeScanKind.CUM_SUM : CumulativeScanKind.CUM_PROD,
                    new CumulativeScanAttrs(1, false, false)), List.of(f32(2, 3)), List.of(f32(2, 3)));
            case SCALED_DOT_PRODUCT_ATTENTION -> attention();
            case CONV2D, CONV3D -> convolution(kind);
            case MEAN_SQUARED_ERROR, DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                    INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> loss(kind);
            case BATCH_NORM_INFERENCE, BATCH_NORM_TRAINING, LAYER_NORM, RMS_NORM, SOFTMAX,
                    LOG_SOFTMAX -> normalization(kind);
            case SORT, ARGSORT, TOP_K -> ordering(kind);
            case MAX_POOL2D, AVERAGE_POOL2D, MAX_POOL3D, AVERAGE_POOL3D -> pooling(kind);
            case DROPOUT, INITIAL_STATE -> random(kind);
            case RNN_TANH, GRU_RESET_AFTER, LSTM -> recurrent(kind);
        };
    }

    private static OperationCapabilityQuery unary(Row row) {
        return query(new Operation(UnaryElementwiseKind.valueOf(row.name()),
                NoOperationAttrs.INSTANCE), List.of(f32(2, 3)), List.of(f32(2, 3)));
    }

    private static OperationCapabilityQuery binary(Row row) {
        BinaryArithmeticKind kind = switch (row) {
            case TENSOR_MIN -> BinaryArithmeticKind.MIN;
            case TENSOR_MAX -> BinaryArithmeticKind.MAX;
            case TENSOR_POW -> BinaryArithmeticKind.POW;
            default -> BinaryArithmeticKind.valueOf(row.name());
        };
        return query(new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(f32(2, 1, 3), f32(1, 4, 3)), List.of(f32(2, 4, 3)));
    }

    private static OperationCapabilityQuery scalar(Row row) {
        ScalarElementwiseKind kind = row == Row.CLAMP ? ScalarElementwiseKind.CLAMP
                : ScalarElementwiseKind.valueOf(row.name().replace("SCALAR_", ""));
        OperationAttrs attrs = kind == ScalarElementwiseKind.CLAMP
                ? new ClampRangeAttrs(ScalarValue.float32(-1.0f), ScalarValue.float32(1.0f))
                : new ScalarValueAttrs(ScalarValue.float32(2.0f));
        return query(new Operation(kind, attrs), List.of(f32(2, 3)), List.of(f32(2, 3)));
    }

    private static OperationCapabilityQuery reduction(Row row) {
        AggregateReductionKind kind = switch (row) {
            case REDUCTION_MIN -> AggregateReductionKind.MIN;
            case REDUCTION_MAX -> AggregateReductionKind.MAX;
            default -> AggregateReductionKind.valueOf(row.name());
        };
        OperationAttrs attrs = switch (kind) {
            case ARG_MAX, ARG_MIN -> new ArgExtremaAttrs(1, false,
                    ArgExtremaTiePolicy.FIRST_INDEX);
            case L1_NORM -> new MultiAxisReductionAttrs(List.of(0), false);
            case VARIANCE -> new StatisticalReductionAttrs(List.of(0), false, 0);
            case LOG_SUM_EXP, L2_NORM -> new MultiAxisReductionAttrs(List.of(1), false);
            case STANDARD_DEVIATION -> new StatisticalReductionAttrs(List.of(1), false, 0);
            default -> NoOperationAttrs.INSTANCE;
        };
        TensorDescriptor input;
        TensorDescriptor output;
        if (kind == AggregateReductionKind.ALL || kind == AggregateReductionKind.ANY) {
            input = bool(2, 3);
            output = boolScalar();
        } else if (kind == AggregateReductionKind.PROD) {
            input = i32(2, 3);
            output = i32Scalar();
        } else if (kind == AggregateReductionKind.ARG_MAX
                || kind == AggregateReductionKind.ARG_MIN) {
            input = f32(2, 3);
            output = descriptor(DataType.INT64, Shape.of(2), false);
        } else if (kind == AggregateReductionKind.L1_NORM) {
            input = f32(3);
            output = f32Scalar();
        } else if (kind == AggregateReductionKind.VARIANCE) {
            input = f32(1);
            output = f32Scalar();
        } else if (attrs instanceof MultiAxisReductionAttrs
                || attrs instanceof StatisticalReductionAttrs) {
            input = f32(2, 3);
            output = f32(2);
        } else {
            input = f32(2, 3);
            output = f32Scalar();
        }
        return query(new Operation(kind, attrs), List.of(input), List.of(output));
    }

    private static OperationCapabilityQuery comparison(Row row) {
        BinaryComparisonKind kind = switch (row) {
            case GT -> BinaryComparisonKind.GREATER_THAN;
            case GE -> BinaryComparisonKind.GREATER_OR_EQUAL;
            case LT -> BinaryComparisonKind.LESS_THAN;
            case LE -> BinaryComparisonKind.LESS_OR_EQUAL;
            case EQ -> BinaryComparisonKind.EQUAL;
            case NE -> BinaryComparisonKind.NOT_EQUAL;
            default -> throw new AssertionError(row);
        };
        return query(new Operation(kind, NoOperationAttrs.INSTANCE),
                List.of(f32(2, 1, 3), f32(1, 4, 3)), List.of(bool(2, 4, 3)));
    }

    private static OperationCapabilityQuery classification(Row row) {
        return query(new Operation(FloatingClassificationKind.valueOf(row.name()),
                NoOperationAttrs.INSTANCE), List.of(f32(2, 3)), List.of(bool(2, 3)));
    }

    private static OperationCapabilityQuery logical(Row row) {
        BooleanLogicalKind kind = BooleanLogicalKind.valueOf(row.name().replace("LOGICAL_", ""));
        List<TensorDescriptor> inputs = kind == BooleanLogicalKind.NOT
                ? List.of(bool(2, 3)) : List.of(bool(2, 1), bool(1, 3));
        return query(new Operation(kind, NoOperationAttrs.INSTANCE), inputs, List.of(bool(2, 3)));
    }

    private static OperationCapabilityQuery affine(Row row) {
        return switch (row) {
            case RESHAPE -> query(new Operation(ShapeTransformKind.RESHAPE,
                            new TargetShapeAttrs(Shape.of(2, 3))),
                    List.of(f32(6)), List.of(view(DataType.FLOAT32, Shape.of(2, 3), 0, 3, 1)));
            case EXPAND -> query(new Operation(ShapeTransformKind.EXPAND,
                            new TargetShapeAttrs(Shape.of(2, 3))),
                    List.of(f32(1, 3)), List.of(view(DataType.FLOAT32, Shape.of(2, 3), 0, 0, 1)));
            case PERMUTE -> query(new Operation(AxisTransformKind.PERMUTE,
                            new PermutationAttrs(List.of(1, 0))),
                    List.of(f32(2, 3)), List.of(view(DataType.FLOAT32, Shape.of(3, 2), 0, 1, 3)));
            case EXPAND_DIMS -> query(new Operation(AxisTransformKind.EXPAND_DIMS,
                            new AxisTransformAttrs(1)),
                    List.of(f32(2, 3)),
                    List.of(view(DataType.FLOAT32, Shape.of(2, 1, 3), 0, 3, 3, 1)));
            case SQUEEZE -> query(new Operation(AxisTransformKind.SQUEEZE,
                            new AxisTransformAttrs(1)),
                    List.of(f32(2, 1, 3)),
                    List.of(view(DataType.FLOAT32, Shape.of(2, 3), 0, 3, 1)));
            case CONTIGUOUS -> query(new Operation(ContiguousKind.CONTIGUOUS,
                            NoOperationAttrs.INSTANCE),
                    List.of(view(DataType.FLOAT32, Shape.of(2, 3), 0, 0, 1)),
                    List.of(f32(2, 3)));
            default -> throw new AssertionError(row);
        };
    }

    private static OperationCapabilityQuery indexing(Row row) {
        return switch (row) {
            case GATHER -> query(new Operation(AxisGatherKind.GATHER, new IndexAxisAttrs(1)),
                    List.of(f32(2, 3, 4), i32(5, 6)), List.of(f32(2, 5, 6, 4)));
            case ONE_HOT -> query(new Operation(OneHotKind.ONE_HOT, new OneHotAttrs(4)),
                    List.of(i32(2, 3)), List.of(bool(2, 3, 4)));
            case SCATTER_ELEMENTS -> query(new Operation(AxisScatterKind.SCATTER_ELEMENTS,
                            new ScatterElementsAttrs(1, ScatterReduction.NONE)),
                    List.of(f32(2, 3), i32(2, 2), f32(2, 2)), List.of(f32(2, 3)));
            case GATHER_ELEMENTS -> query(new Operation(AxisGatherKind.GATHER_ELEMENTS,
                            new IndexAxisAttrs(1)),
                    List.of(f32(2, 3), i32(2, 2)), List.of(f32(2, 2)));
            case SCATTER_ADD -> query(new Operation(AxisScatterKind.SCATTER_ADD,
                            new IndexAxisAttrs(0)),
                    List.of(f32(3), i32(2), f32(2)), List.of(f32(3)));
            case GATHER_ND -> query(new Operation(GatherNdKind.GATHER_ND, new GatherNdAttrs(0)),
                    List.of(f32(2, 3), i32(2, 1)), List.of(f32(2, 3)));
            case SCATTER_ND -> query(new Operation(ScatterNdKind.SCATTER_ND,
                            new ScatterNdAttrs(0, ScatterReduction.NONE)),
                    List.of(f32(3), i32(2, 1), f32(2)), List.of(f32(3)));
            default -> throw new AssertionError(row);
        };
    }

    private static OperationCapabilityQuery window(Row row) {
        Window2dAttrs unfold2d = new Window2dAttrs(2, 2, 1, 1, 0, 0, 1, 1, false);
        Window3dAttrs unfold3d = new Window3dAttrs(2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, false);
        Window2dAttrs fold2d = new Window2dAttrs(2, 2, 2, 2, 1, 1, 1, 1, true);
        Window3dAttrs fold3d = new Window3dAttrs(2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, true);
        return switch (row) {
            case UNFOLD_AXIS -> query(new Operation(WindowTransformKind.UNFOLD_AXIS,
                            new UnfoldAxisAttrs(1, 3, 2)),
                    List.of(f32(2, 6)), List.of(f32(2, 2, 3)));
            case FOLD_AXIS -> query(new Operation(WindowTransformKind.FOLD_AXIS,
                            new FoldAxisAttrs(1, 4, 2)),
                    List.of(f32(2, 2, 2)), List.of(f32(2, 4)));
            case UNFOLD2D -> query(new Operation(WindowTransformKind.UNFOLD2D, unfold2d),
                    List.of(f32(1, 1, 2, 2)), List.of(f32(1, 4, 1)));
            case FOLD2D -> query(new Operation(WindowTransformKind.FOLD2D,
                            new Fold2dAttrs(Shape.of(1, 1, 3, 3), fold2d)),
                    List.of(f32(1, 4, 9)), List.of(f32(1, 1, 3, 3)));
            case UNFOLD3D -> query(new Operation(WindowTransformKind.UNFOLD3D, unfold3d),
                    List.of(f32(1, 1, 2, 2, 2)), List.of(f32(1, 8, 1)));
            case FOLD3D -> query(new Operation(WindowTransformKind.FOLD3D,
                            new Fold3dAttrs(Shape.of(1, 1, 3, 3, 3), fold3d)),
                    List.of(f32(1, 8, 27)), List.of(f32(1, 1, 3, 3, 3)));
            default -> throw new AssertionError(row);
        };
    }

    private static OperationCapabilityQuery movement(Row row) {
        return switch (row) {
            case SELECT -> query(new Operation(SelectKind.SELECT, new SelectAttrs(0, 1)),
                    List.of(f32(2, 3)),
                    List.of(view(DataType.FLOAT32, Shape.of(3), 3, 1)));
            case PAD -> query(new Operation(PadKind.PAD,
                            new PadAttrs(List.of(1L), List.of(2L), ScalarValue.float32(-1.0f))),
                    List.of(f32(2)), List.of(f32(5)));
            case SLICE -> query(new Operation(SliceKind.SLICE,
                            new SliceAttrs(List.of(1L), List.of(2L), List.of(1), List.of(1L))),
                    List.of(f32(2, 3)),
                    List.of(view(DataType.FLOAT32, Shape.of(2, 2), 1, 3, 1)));
            case SLICE_UPDATE -> query(new Operation(SliceKind.SLICE_UPDATE,
                            new SliceAttrs(List.of(3L), List.of(2L), List.of(0), List.of(-2L))),
                    List.of(f32(4), f32(2)), List.of(f32(4)));
            case CONCAT -> query(new Operation(TensorCompositionKind.CONCAT,
                            new CompositionAxisAttrs(0)),
                    List.of(f32(2), f32(1), f32(2)), List.of(f32(5)));
            case STACK -> query(new Operation(TensorCompositionKind.STACK,
                            new CompositionAxisAttrs(1)),
                    List.of(f32(2), f32(2)), List.of(f32(2, 2)));
            case TILE -> query(new Operation(TileKind.TILE, new TileAttrs(List.of(3L))),
                    List.of(f32(2)), List.of(f32(6)));
            default -> throw new AssertionError(row);
        };
    }

    private static OperationCapabilityQuery cast(DataType source, DataType target) {
        Shape shape = Shape.of(2, 3);
        return query(new Operation(CastKind.CAST, new CastAttrs(target)),
                List.of(descriptor(source, shape, false)),
                List.of(descriptor(target, shape, false)));
    }

    private static OperationCapabilityQuery attention() {
        TensorDescriptor value = f32(1, 1, 1);
        return query(new Operation(ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION,
                        new ScaledDotProductAttentionAttrs(Optional.empty(), false)),
                List.of(value, value, value), List.of(value));
    }

    private static OperationCapabilityQuery convolution(Row row) {
        if (row == Row.CONV2D) {
            return query(new Operation(Conv2dKind.CONV2D, Conv2dAttrs.defaults()),
                    List.of(f32(1, 2, 3, 4), f32(4, 2, 1, 1), f32(4)),
                    List.of(f32(1, 4, 3, 4)));
        }
        return query(new Operation(Conv3dKind.CONV3D,
                        new Conv3dAttrs(1, 2, 1, 0, 1, 0, 1, 1, 2, 1)),
                List.of(f32(1, 2, 4, 5, 6), f32(3, 2, 2, 2, 2)),
                List.of(f32(1, 3, 3, 3, 4)));
    }

    private static OperationCapabilityQuery loss(Row row) {
        TensorDescriptor value = f32(2, 3);
        return switch (row) {
            case MEAN_SQUARED_ERROR -> query(new Operation(LossKind.MEAN_SQUARED_ERROR,
                            new MeanSquaredErrorAttrs(LossReduction.NONE)),
                    List.of(value, value), List.of(value));
            case DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> query(new Operation(
                            LossKind.DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                            new DenseCategoricalCrossEntropyWithLogitsAttrs(1, LossReduction.NONE)),
                    List.of(value, value), List.of(f32(2)));
            case INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS -> query(new Operation(
                            LossKind.INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
                            new IndexCategoricalCrossEntropyWithLogitsAttrs(
                                    1, LossReduction.NONE, Optional.empty())),
                    List.of(value, i32(2)), List.of(f32(2)));
            default -> throw new AssertionError(row);
        };
    }

    private static OperationCapabilityQuery normalization(Row row) {
        TensorDescriptor value = f32(2, 3);
        TensorDescriptor channel = f32(3);
        ScalarValue epsilon = ScalarValue.float32(1.0e-5f);
        return switch (row) {
            case BATCH_NORM_INFERENCE -> query(new Operation(BatchNormKind.BATCH_NORM_INFERENCE,
                            new BatchNormInferenceAttrs(1, epsilon)),
                    List.of(value, channel, channel, channel, channel), List.of(value));
            case BATCH_NORM_TRAINING -> query(new Operation(BatchNormKind.BATCH_NORM_TRAINING,
                            new BatchNormTrainingAttrs(1, ScalarValue.float32(0.5f), epsilon)),
                    List.of(value, channel, channel, channel, channel),
                    List.of(value, channel, channel, channel, channel));
            case LAYER_NORM -> query(new Operation(LayerNormKind.LAYER_NORM,
                            new LayerNormAttrs(Shape.of(3), epsilon)),
                    List.of(value), List.of(value));
            case RMS_NORM -> query(new Operation(RmsNormKind.RMS_NORM,
                            new RmsNormAttrs(Shape.of(3), epsilon)),
                    List.of(value), List.of(value));
            case SOFTMAX, LOG_SOFTMAX -> query(new Operation(row == Row.SOFTMAX
                            ? SoftmaxKind.SOFTMAX : SoftmaxKind.LOG_SOFTMAX,
                            new SoftmaxAttrs(1)), List.of(value), List.of(value));
            default -> throw new AssertionError(row);
        };
    }

    private static OperationCapabilityQuery ordering(Row row) {
        TensorDescriptor input = f32(2, 4, 3);
        return switch (row) {
            case SORT -> query(new Operation(OrderingKind.SORT, new SortAttrs(1, true)),
                    List.of(input), List.of(input));
            case ARGSORT -> query(new Operation(OrderingKind.ARGSORT, new SortAttrs(1, false)),
                    List.of(input), List.of(descriptor(DataType.INT64, input.shape(), false)));
            case TOP_K -> query(new Operation(TopKKind.TOP_K, new TopKAttrs(1, 2, true, false)),
                    List.of(input), List.of(f32(2, 2, 3),
                            descriptor(DataType.INT64, Shape.of(2, 2, 3), false)));
            default -> throw new AssertionError(row);
        };
    }

    private static OperationCapabilityQuery pooling(Row row) {
        return switch (row) {
            case MAX_POOL2D -> query(new Operation(Pool2dKind.MAX_POOL2D,
                            new MaxPool2dAttrs(2, 3, 2, 2, 1, 1, 1, 1, true)),
                    List.of(f32(1, 2, 3, 4)), List.of(f32(1, 2, 3, 3)));
            case AVERAGE_POOL2D -> query(new Operation(Pool2dKind.AVERAGE_POOL2D,
                            new AveragePool2dAttrs(2, 3, 2, 2, 1, 1, 1, 1, true)),
                    List.of(f32(1, 2, 3, 4)), List.of(f32(1, 2, 3, 3)));
            case MAX_POOL3D -> query(new Operation(Pool3dKind.MAX_POOL3D,
                            new MaxPool3dAttrs(2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 1, 1, false)),
                    List.of(f32(1, 1, 3, 3, 3)), List.of(f32(1, 1, 2, 2, 2)));
            case AVERAGE_POOL3D -> query(new Operation(Pool3dKind.AVERAGE_POOL3D,
                            new AveragePool3dAttrs(1, 1, 2, 1, 1, 2, 0, 0, 2, 1, 1, 1, true)),
                    List.of(f32(1, 1, 1, 1, 1)), List.of(f32(1, 1, 1, 1, 3)));
            default -> throw new AssertionError(row);
        };
    }

    private static OperationCapabilityQuery random(Row row) {
        TensorDescriptor state = descriptor(DataType.INT64, Shape.of(2), false);
        if (row == Row.INITIAL_STATE) {
            return query(new Operation(GraphRngKind.INITIAL_STATE,
                            new GraphRngStateAttrs(0x1234L, 0x5678L)),
                    List.of(), List.of(state));
        }
        return query(new Operation(DropoutKind.DROPOUT, new DropoutAttrs(0.25d)),
                List.of(f32(2, 3), state), List.of(f32(2, 3), bool(2, 3), state));
    }

    private static OperationCapabilityQuery recurrent(Row row) {
        RecurrentScanKind kind = RecurrentScanKind.valueOf(row.name());
        int gates = switch (kind) {
            case RNN_TANH -> 1;
            case GRU_RESET_AFTER -> 3;
            case LSTM -> 4;
        };
        var inputs = new ArrayList<TensorDescriptor>();
        inputs.add(f32(2, 2, 3));
        inputs.add(descriptor(DataType.INT64, Shape.of(2), false));
        inputs.add(f32(2, 4));
        if (kind == RecurrentScanKind.LSTM) inputs.add(f32(2, 4));
        inputs.add(f32(gates * 4L, 3));
        inputs.add(f32(gates * 4L, 4));
        var outputs = new ArrayList<>(List.of(f32(2, 2, 4), f32(2, 4)));
        if (kind == RecurrentScanKind.LSTM) outputs.add(f32(2, 4));
        return query(new Operation(kind, RecurrentDirection.FORWARD), inputs, outputs);
    }

    private static OperationCapabilityQuery query(Operation operation,
            List<TensorDescriptor> inputs, List<TensorDescriptor> outputs) {
        return new OperationCapabilityQuery(NumericalProfile.ACCELERATOR,
                operation, inputs, outputs);
    }

    private static TensorDescriptor f32(long... dimensions) {
        return descriptor(DataType.FLOAT32, Shape.of(dimensions), false);
    }

    private static TensorDescriptor i32(long... dimensions) {
        return descriptor(DataType.INT32, Shape.of(dimensions), false);
    }

    private static TensorDescriptor bool(long... dimensions) {
        return descriptor(DataType.BOOL, Shape.of(dimensions), false);
    }

    private static TensorDescriptor f32Scalar() {
        return descriptor(DataType.FLOAT32, Shape.scalar(), false);
    }

    private static TensorDescriptor i32Scalar() {
        return descriptor(DataType.INT32, Shape.scalar(), false);
    }

    private static TensorDescriptor boolScalar() {
        return descriptor(DataType.BOOL, Shape.scalar(), false);
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, boolean requiresGrad) {
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), requiresGrad);
    }

    private static TensorDescriptor view(DataType type, Shape shape, long offset, long... strides) {
        return new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.of(shape, strides, offset, true)), false);
    }

    private record Seed(String id, OperationCapabilityQuery query) { }

    private record Target(String bfloat16, String float16, String profile, String exclusion) { }

    /** Frozen current Model-kind basis; names follow the append-only Metal wires 1 through 115. */
    private enum Row {
        NEG, ADD, SUB, MUL, DIV, RESHAPE, EXPAND, PERMUTE, EXPAND_DIMS, SQUEEZE, CONTIGUOUS,
        ABS, SUM, MEAN, MATMUL, GATHER, ONE_HOT, SCATTER_ELEMENTS, UNFOLD_AXIS, GT, GE, LT, LE,
        EQ, NE, TENSOR_MIN, TENSOR_MAX, SCALAR_MIN, SCALAR_MAX, CLAMP, REDUCTION_MIN,
        REDUCTION_MAX, CUM_SUM, CUM_PROD, SCALED_DOT_PRODUCT_ATTENTION, CONV2D, CONV3D,
        TENSOR_POW, CAST, IS_FINITE, IS_NAN, IS_INF, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT,
        SCALAR_ADD, SCALAR_SUB, SCALAR_MUL, SCALAR_DIV, SCALAR_POW, WHERE, RECIPROCAL, LOG,
        LOG1P, EXP, EXPM1, ERF, SQRT, RSQRT, FLOOR, CEIL, SIGN, RELU, SIGMOID, TANH, GELU,
        GELU_TANH_APPROXIMATION, SILU, GATHER_ELEMENTS, SCATTER_ADD, GATHER_ND, SCATTER_ND,
        SELECT, PAD, SLICE, SLICE_UPDATE, CONCAT, STACK, TILE, FOLD_AXIS, UNFOLD2D, FOLD2D,
        UNFOLD3D, FOLD3D, MEAN_SQUARED_ERROR, DENSE_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS,
        INDEX_CATEGORICAL_CROSS_ENTROPY_WITH_LOGITS, BATCH_NORM_INFERENCE, BATCH_NORM_TRAINING,
        LAYER_NORM, RMS_NORM, SOFTMAX, LOG_SOFTMAX, SORT, ARGSORT, TOP_K, MAX_POOL2D,
        AVERAGE_POOL2D, MAX_POOL3D, AVERAGE_POOL3D, DROPOUT, INITIAL_STATE, RNN_TANH,
        GRU_RESET_AFTER, LSTM, PROD, ALL, ANY, ARG_MAX, ARG_MIN, LOG_SUM_EXP, VARIANCE,
        STANDARD_DEVIATION, L1_NORM, L2_NORM
    }
}
