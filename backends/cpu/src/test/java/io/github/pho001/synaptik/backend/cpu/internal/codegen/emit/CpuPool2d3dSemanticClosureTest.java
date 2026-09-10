package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool2dLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool3dLowering;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import org.junit.jupiter.api.Test;

/** Direct generated-entry semantic closure for every current direct Pool2d and Pool3d row. */
class CpuPool2d3dSemanticClosureTest {
    private static final String RESOURCE = "/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-inventory.tsv";
    private static final String INVENTORY_SHA256 = "1dcb69796c00fe3793d86f3f4cc3e816176062a45312ddbbaadfba9f8036cf20";
    private static final Set<String> FORMS = Set.of("AVERAGE_POOL2D", "MAX_POOL2D", "AVERAGE_POOL3D", "MAX_POOL3D");

    @Test void everyExactGeneratedPoolOwnerDefinesAndInvokesItsSpecializedEntry() throws Throwable {
        Map<String, Candidate> expected = inventoryOwners(resource());
        Set<String> invoked = new TreeSet<>();
        for (Candidate candidate : expected.values()) {
            assertTrue(invoked.add(candidate.owner()), "duplicate reconstructed candidate " + candidate.owner());
            // Invocation is intentionally in this exact-row loop: no representative form projects coverage.
            execute(candidate);
        }
        assertEquals(120, invoked.size(), "four pool forms × three types × ceil mode × five requests");
        assertEquals(expected.keySet(), invoked, "exact current inventory owner projection");
        assertEquals(30L, expected.values().stream().filter(value -> value.form().equals("AVERAGE_POOL2D")).count());
        assertEquals(30L, expected.values().stream().filter(value -> value.form().equals("MAX_POOL2D")).count());
        assertEquals(30L, expected.values().stream().filter(value -> value.form().equals("AVERAGE_POOL3D")).count());
        assertEquals(30L, expected.values().stream().filter(value -> value.form().equals("MAX_POOL3D")).count());
    }

    @Test void shaBoundProjectionFailsClosedForDuplicateOrphanStaleAndMutation() throws Exception {
        String inventory = resource();
        assertEquals(INVENTORY_SHA256, sha256(inventory), "inventory changed: refresh exact closure deliberately");
        Map<String, Candidate> owners = inventoryOwners(inventory);
        assertEquals(120, owners.size());
        String one = owners.keySet().iterator().next();
        assertThrows(AssertionError.class, () -> inventoryOwners(inventory + lineFor(inventory, one) + '\n'), "duplicate owner");
        assertThrows(AssertionError.class, () -> requireSame(owners.keySet(), Set.of("specialized:orphan-pool")), "orphan candidate");
        Set<String> stale = new TreeSet<>(owners.keySet());
        stale.remove(one);
        assertThrows(AssertionError.class, () -> requireSame(owners.keySet(), stale), "stale omission");
        assertNotEquals(INVENTORY_SHA256, sha256(inventory.replaceFirst("AVERAGE_POOL2D", "AVERAGE_POOL2D_MUTATED")),
                "semantic inventory mutation");
    }

    private static void execute(Candidate candidate) throws Throwable {
        var context = CpuSpecializedGeneratedMatrixTest.semanticConfigure(
                CpuSpecializedGeneratedMatrixTest.semanticPoolContext(candidate.form(), candidate.type(), candidate.ceilMode()),
                candidate.request());
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        assertEquals(1, plan.units().size(), candidate.owner());
        assertTrue(plan.materializations().isEmpty(), candidate.owner() + " materialization candidate stays unselected");
        var route = plan.units().getFirst().portablePlan();
        byte[] first = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
        byte[] second = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(first, second, candidate.owner() + " deterministic bytes");
        CpuGeneratedDirectEvidenceClosureTest.assertClosedSpecializedClass(first, route.specialization());
        var artifact = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), first);
        List<Storage> storage = storage(route.specialization().boundaryDataTypes(), route.specialization().carrierPattern());
        Storage input = storage.getFirst();
        Storage output = storage.getLast();
        Object beforeInput;
        Object expected;
        long start;
        long end;
        Object packed;
        if (candidate.form().endsWith("2D")) {
            var geometry = plan.units().getFirst().pool2dGeometry().orElseThrow(() -> new AssertionError(candidate.owner()));
            fill2dInput(geometry, input);
            output.fill(91.25);
            beforeInput = input.copy();
            expected = output.copy();
            start = 1;
            end = geometry.outputCount() - 1;
            assertTrue(end > start, candidate.owner() + " non-empty selected output range");
            oracle2d(geometry, input, expected, start, end);
            packed = geometry.pack(0, 0);
        } else {
            var geometry = plan.units().getFirst().pool3dGeometry().orElseThrow(() -> new AssertionError(candidate.owner()));
            fill3dInput(geometry, input);
            output.fill(91.25);
            beforeInput = input.copy();
            expected = output.copy();
            start = 1;
            end = geometry.outputCount() - 1;
            assertTrue(end > start, candidate.owner() + " non-empty selected output range");
            oracle3d(geometry, input, expected, start, end);
            packed = geometry.pack(0, 0);
        }
        Object[] arguments = storage.stream().map(Storage::argument).toArray();
        Object[] invocation = Arrays.copyOf(arguments, arguments.length + 3);
        invocation[arguments.length] = packed;
        invocation[arguments.length + 1] = start;
        invocation[arguments.length + 2] = end;
        artifact.entryPoint().invokeWithArguments(invocation);
        assertRawEquals(expected, output.heap, candidate.owner() + " output and untouched cells");
        assertRawEquals(beforeInput, input.heap, candidate.owner() + " source immutability");
    }

    /* Independent clean scalar pooling oracles; they do not call production lowering or kernels. */
    private static void oracle2d(CpuPool2dLowering.Geometry g, Storage input, Object output, long start, long end) {
        long[] ie = g.input().extents(), is = g.input().strides(), oe = g.output().extents(), os = g.output().strides();
        for (long cell = start; cell < end; cell++) {
            long rest = cell, ow = rest % oe[3]; rest /= oe[3]; long oh = rest % oe[2]; rest /= oe[2]; long c = rest % oe[1], n = rest / oe[1];
            long target = g.output().offset() + n * os[0] + c * os[1] + oh * os[2] + ow * os[3];
            Window window = new Window(g.kind().name().equals("MAX"), g.dataType(), g.divisor());
            for (long kh = 0; kh < g.kernelHeight(); kh++) for (long kw = 0; kw < g.kernelWidth(); kw++) {
                long ih = oh * g.strideHeight() - g.paddingHeight() + kh * g.dilationHeight();
                long iw = ow * g.strideWidth() - g.paddingWidth() + kw * g.dilationWidth();
                if (ih < 0 || iw < 0 || ih >= ie[2] || iw >= ie[3]) window.padding();
                else window.value(input.get(g.input().offset() + n * is[0] + c * is[1] + ih * is[2] + iw * is[3]));
            }
            put(g.dataType(), output, target, window.result());
        }
    }

    private static void oracle3d(CpuPool3dLowering.Geometry g, Storage input, Object output, long start, long end) {
        long[] ie = g.input().extents(), is = g.input().strides(), oe = g.output().extents(), os = g.output().strides();
        for (long cell = start; cell < end; cell++) {
            long rest = cell, ow = rest % oe[4]; rest /= oe[4]; long oh = rest % oe[3]; rest /= oe[3]; long od = rest % oe[2]; rest /= oe[2]; long c = rest % oe[1], n = rest / oe[1];
            long target = g.output().offset() + n * os[0] + c * os[1] + od * os[2] + oh * os[3] + ow * os[4];
            Window window = new Window(g.kind().name().equals("MAX"), g.dataType(), g.divisor());
            for (long kd = 0; kd < g.kernelDepth(); kd++) for (long kh = 0; kh < g.kernelHeight(); kh++) for (long kw = 0; kw < g.kernelWidth(); kw++) {
                long id = od * g.strideDepth() - g.paddingDepth() + kd * g.dilationDepth();
                long ih = oh * g.strideHeight() - g.paddingHeight() + kh * g.dilationHeight();
                long iw = ow * g.strideWidth() - g.paddingWidth() + kw * g.dilationWidth();
                if (id < 0 || ih < 0 || iw < 0 || id >= ie[2] || ih >= ie[3] || iw >= ie[4]) window.padding();
                else window.value(input.get(g.input().offset() + n * is[0] + c * is[1] + id * is[2] + ih * is[3] + iw * is[4]));
            }
            put(g.dataType(), output, target, window.result());
        }
    }

    private static void fill2dInput(CpuPool2dLowering.Geometry g, Storage input) {
        input.fill(-17.0); long[] e = g.input().extents(), s = g.input().strides();
        for (long n = 0; n < e[0]; n++) for (long c = 0; c < e[1]; c++) for (long h = 0; h < e[2]; h++) for (long w = 0; w < e[3]; w++)
            input.put(g.input().offset() + n * s[0] + c * s[1] + h * s[2] + w * s[3], poolValue(n + c + h + w, h == 0 && w == 0, h == 0 && w == 1));
    }

    private static void fill3dInput(CpuPool3dLowering.Geometry g, Storage input) {
        input.fill(-17.0); long[] e = g.input().extents(), s = g.input().strides();
        for (long n = 0; n < e[0]; n++) for (long c = 0; c < e[1]; c++) for (long d = 0; d < e[2]; d++) for (long h = 0; h < e[3]; h++) for (long w = 0; w < e[4]; w++)
            input.put(g.input().offset() + n * s[0] + c * s[1] + d * s[2] + h * s[3] + w * s[4], poolValue(n + c + d + h + w, d == 0 && h == 0 && w == 0, d == 0 && h == 0 && w == 1));
    }

    private static double poolValue(long sum, boolean negativeZero, boolean positiveZero) { return negativeZero ? -0.0 : positiveZero ? 0.0 : sum % 7 - 3.25; }

    private static List<Storage> storage(List<DataType> types, List<CarrierAccess> carriers) {
        List<Storage> result = new ArrayList<>();
        for (int i = 0; i < types.size(); i++) result.add(new Storage(types.get(i), carriers.get(i)));
        return result;
    }

    private static Map<String, Candidate> inventoryOwners(String text) {
        String[] lines = text.split("\\n", -1); assertTrue(text.endsWith("\n") && !text.contains("\r"));
        Map<String, Candidate> result = new LinkedHashMap<>();
        for (int index = 1; index < lines.length - 1; index++) {
            String[] row = lines[index].split("\\t", -1); assertEquals(27, row.length, "inventory line " + (index + 1));
            if (!row[0].startsWith("specialized:") || !FORMS.contains(row[2]) || !row[21].equals("GENERATED")) continue;
            Candidate candidate = Candidate.parse(row[0], row[2]);
            if (result.put(candidate.owner(), candidate) != null) throw new AssertionError("duplicate owner " + candidate.owner());
        }
        return result;
    }

    private static void requireSame(Set<String> expected, Set<String> actual) { if (!expected.equals(actual)) throw new AssertionError("orphan or stale owner projection"); }
    private static String lineFor(String inventory, String owner) { return inventory.lines().filter(line -> line.startsWith(owner + '\t')).findFirst().orElseThrow(); }
    private static String resource() throws Exception { try (var stream = CpuPool2d3dSemanticClosureTest.class.getResourceAsStream(RESOURCE)) { assertTrue(stream != null, RESOURCE); return new String(stream.readAllBytes(), StandardCharsets.UTF_8); } }
    private static String sha256(String value) throws Exception { return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8))); }
    private static boolean positive(double value) { return Double.doubleToRawLongBits(value) == 0L; }
    private static boolean negativeZero(double value) { return Double.doubleToRawLongBits(value) == Long.MIN_VALUE; }
    private static void put(DataType type, Object target, long index, double value) { switch (type) { case BFLOAT16 -> ((short[]) target)[(int) index] = ScalarValue.bfloat16((float) value).bfloat16Bits(); case FLOAT32 -> ((float[]) target)[(int) index] = (float) value; case FLOAT64 -> ((double[]) target)[(int) index] = value; default -> throw new AssertionError(type); } }
    private static void assertRawEquals(Object expected, Object actual, String message) { if (expected instanceof short[] a) assertArrayEquals(a, (short[]) actual, message); else if (expected instanceof float[] a) assertArrayEquals(bits(a), bits((float[]) actual), message); else assertArrayEquals(bits((double[]) expected), bits((double[]) actual), message); }
    private static int[] bits(float[] values) { int[] result = new int[values.length]; for (int i = 0; i < values.length; i++) result[i] = Float.floatToRawIntBits(values[i]); return result; }
    private static long[] bits(double[] values) { long[] result = new long[values.length]; for (int i = 0; i < values.length; i++) result[i] = Double.doubleToRawLongBits(values[i]); return result; }

    private record Candidate(String owner, String form, DataType type, boolean ceilMode, String request) {
        static Candidate parse(String owner, String form) {
            String[] parts = owner.substring("specialized:".length()).split("/", -1);
            if (parts.length != 4 || !parts[0].equals(form.toLowerCase().replace('_', '-'))) throw new AssertionError("unparseable pool owner " + owner);
            return new Candidate(owner, form, DataType.valueOf(parts[1]), Boolean.parseBoolean(parts[2]), parts[3]);
        }
    }

    private static final class Window {
        private final boolean max;
        private final DataType type;
        private final long divisor;
        private boolean found;
        private boolean allNegativeZero = true;
        private double best = Double.NEGATIVE_INFINITY;
        private double sum;

        Window(boolean max, DataType type, long divisor) {
            this.max = max;
            this.type = type;
            this.divisor = divisor;
        }

        void padding() {
            if (!max) allNegativeZero = false;
        }

        void value(double value) {
            if (max) {
                if (!found || Double.isNaN(value) || value > best
                        || value == 0 && best == 0 && positive(value)) {
                    best = value;
                    found = true;
                }
                return;
            }
            allNegativeZero &= negativeZero(value);
            sum = type == DataType.FLOAT64 ? sum + value : (float) sum + (float) value;
        }

        double result() {
            if (max) return found ? best : Double.NEGATIVE_INFINITY;
            double value = type == DataType.FLOAT64 ? sum / divisor : (float) sum / (float) divisor;
            return allNegativeZero ? -0.0 : value;
        }
    }

    private static final class Storage {
        final DataType type; final CarrierAccess carrier; final Object heap; final MemorySegment segment;
        Storage(DataType type, CarrierAccess carrier) {
            this.type = type;
            this.carrier = carrier;
            heap = switch (type) {
                case BFLOAT16 -> new short[4096];
                case FLOAT32 -> new float[4096];
                case FLOAT64 -> new double[4096];
                default -> throw new AssertionError(type);
            };
            segment = heap instanceof short[] x ? MemorySegment.ofArray(x)
                    : heap instanceof float[] x ? MemorySegment.ofArray(x)
                    : MemorySegment.ofArray((double[]) heap);
        }
        Object argument() { return carrier == CarrierAccess.MEMORY_SEGMENT ? segment : heap; }
        void fill(double value) { for (int i = 0; i < 4096; i++) CpuPool2d3dSemanticClosureTest.put(type, heap, i, value); }
        void put(long index, double value) { CpuPool2d3dSemanticClosureTest.put(type, heap, index, value); }
        double get(long index) { return switch (type) { case BFLOAT16 -> Float.intBitsToFloat(((((short[]) heap)[(int) index]) & 0xffff) << 16); case FLOAT32 -> ((float[]) heap)[(int) index]; case FLOAT64 -> ((double[]) heap)[(int) index]; default -> throw new AssertionError(type); }; }
        Object copy() { return heap instanceof short[] x ? x.clone() : heap instanceof float[] x ? x.clone() : ((double[]) heap).clone(); }
    }
}
