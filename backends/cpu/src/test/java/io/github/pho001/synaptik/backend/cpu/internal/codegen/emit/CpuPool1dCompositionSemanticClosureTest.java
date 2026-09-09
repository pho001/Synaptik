package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuPool2dLowering;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
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

/** Direct generated-entry semantic closure for every current generated Pool1d composition row. */
class CpuPool1dCompositionSemanticClosureTest {
    private static final String RESOURCE = "/io/github/pho001/synaptik/backend/cpu/internal/codegen/emit/generated-coverage-inventory.tsv";
    private static final String INVENTORY_SHA256 = "527f36de41b64c228dd215c6f3182138b4c70db24c915117c4d7f82743ac41cc";
    private static final List<DataType> TYPES = List.of(DataType.BFLOAT16, DataType.FLOAT32, DataType.FLOAT64);
    private static final List<String> GEOMETRIES = List.of("max-floor", "max-ceil", "average-floor", "average-ceil");
    private static final List<String> REQUESTS = List.of("HEAP_GENERAL_MATERIALIZATION", "HEAP_GENERAL_PARALLEL_SCALAR", "MIXED_GENERAL_PARALLEL_VECTOR");

    @Test void everyExactGeneratedPool1dCompositionOwnerDefinesAndInvokesItsSpecializedEntry() throws Throwable {
        Set<String> expected = inventoryOwners(resource());
        Set<String> reconstructed = new TreeSet<>();
        for (String geometry : GEOMETRIES) for (DataType type : TYPES) for (String request : REQUESTS) {
            String owner = "composition:pool1d/" + geometry + '/' + type + '/' + request;
            assertTrue(reconstructed.add(owner), "duplicate reconstructed candidate " + owner);
            execute(new Candidate(owner, geometry, type, request));
        }
        assertEquals(36, reconstructed.size(), "four Model pool mappings × three types × three admitted requests");
        assertEquals(reconstructed, expected, "exact current inventory owner projection");
    }

    @Test void shaBoundProjectionFailsClosedForDuplicateOrphanStaleAndMutation() throws Exception {
        String inventory = resource();
        assertEquals(INVENTORY_SHA256, sha256(inventory), "inventory changed: refresh exact closure deliberately");
        Set<String> owners = inventoryOwners(inventory);
        assertEquals(36, owners.size());
        String one = owners.iterator().next();
        assertThrows(AssertionError.class, () -> inventoryOwners(inventory + lineFor(inventory, one) + '\n'), "duplicate owner");
        assertThrows(AssertionError.class, () -> requireSame(owners, Set.of("composition:pool1d/orphan")), "orphan candidate");
        Set<String> stale = new TreeSet<>(owners); stale.remove(one);
        assertThrows(AssertionError.class, () -> requireSame(owners, stale), "stale omission");
        String mutated = inventory.replaceFirst("POOL1D_COMPOSITION", "POOL1D_COMPOSITION_MUTATED");
        assertNotEquals(INVENTORY_SHA256, sha256(mutated), "semantic inventory mutation");
    }

    private static void execute(Candidate candidate) throws Throwable {
        var context = CpuOneDimensionalCompositionGeneratedMatrixTest.semanticConfigure(
                CpuOneDimensionalCompositionGeneratedMatrixTest.semanticPoolContext(candidate.geometry, candidate.type), candidate.request);
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        assertEquals(1, plan.units().size(), candidate.owner);
        assertTrue(plan.materializations().isEmpty(), candidate.owner + " materialization stays unselected");
        var unit = plan.units().getFirst(); var route = unit.portablePlan();
        assertTrue(unit.pool2dGeometry().isPresent(), candidate.owner + " recognized Pool1d composition must lower to Pool2d geometry");
        assertEquals(55, route.specialization().classIdentitySchema(), candidate.owner);
        byte[] first = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
        byte[] second = new CpuClassFileKernelGenerator().generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(first, second, candidate.owner + " deterministic bytes");
        CpuGeneratedDirectEvidenceClosureTest.assertClosedSpecializedClass(first, route.specialization());
        var artifact = new CpuClassFileKernelGenerator().defineClassBytes(route.specialization(), first);
        var geometry = unit.pool2dGeometry().orElseThrow(() -> new AssertionError(candidate.owner + " Pool2d geometry"));
        List<Storage> storage = new ArrayList<>();
        for (int i = 0; i < route.specialization().boundaryDataTypes().size(); i++) storage.add(new Storage(
                route.specialization().boundaryDataTypes().get(i), route.specialization().carrierPattern().get(i)));
        fillInput(geometry, storage.getFirst());
        Storage output = storage.getLast(); output.fill(91.25);
        Object[] beforeInput = storage.subList(0, storage.size() - 1).stream().map(Storage::copy).toArray();
        Object expected = output.copy();
        long start = 1, end = geometry.outputCount() - 1;
        assertTrue(end > start, candidate.owner + " non-empty selected output range");
        oracle(geometry, storage.getFirst(), expected, start, end);
        Object[] arguments = storage.stream().map(Storage::argument).toArray();
        Object[] invocation = Arrays.copyOf(arguments, arguments.length + 3);
        invocation[arguments.length] = geometry.pack(0, 0); invocation[arguments.length + 1] = start; invocation[arguments.length + 2] = end;
        artifact.entryPoint().invokeWithArguments(invocation);
        assertRawEquals(expected, output.heap, candidate.owner + " output and untouched cells");
        for (int i = 0; i < beforeInput.length; i++) assertRawEquals(beforeInput[i], storage.get(i).heap, candidate.owner + " source immutability");
    }

    /* Independent clean scalar pooling oracle; it does not call lowering, production kernels, or metadata. */
    private static void oracle(CpuPool2dLowering.Geometry g, Storage input, Object output, long start, long end) {
        long[] ie = g.input().extents(), is = g.input().strides(), oe = g.output().extents(), os = g.output().strides();
        for (long cell = start; cell < end; cell++) {
            long rest = cell;
            long ow = rest % oe[3];
            rest /= oe[3];
            long oh = rest % oe[2];
            rest /= oe[2];
            long c = rest % oe[1];
            long n = rest / oe[1];
            long outputIndex = g.output().offset() + n * os[0] + c * os[1] + oh * os[2] + ow * os[3];
            if (g.kind().name().equals("MAX")) {
                boolean found = false; double best = Double.NEGATIVE_INFINITY;
                for (long kh = 0; kh < g.kernelHeight(); kh++) for (long kw = 0; kw < g.kernelWidth(); kw++) {
                    long ih = oh * g.strideHeight() - g.paddingHeight() + kh * g.dilationHeight();
                    long iw = ow * g.strideWidth() - g.paddingWidth() + kw * g.dilationWidth();
                    if (ih < 0 || iw < 0 || ih >= ie[2] || iw >= ie[3]) continue;
                    double value = input.get(g.input().offset() + n * is[0] + c * is[1] + ih * is[2] + iw * is[3]);
                    if (!found || Double.isNaN(value) || value > best || value == 0 && best == 0 && positive(value)) { best = value; found = true; }
                }
                put(g.dataType(), output, outputIndex, found ? best : Double.NEGATIVE_INFINITY);
            } else {
                double sum = 0; boolean allNegativeZero = true;
                for (long kh = 0; kh < g.kernelHeight(); kh++) for (long kw = 0; kw < g.kernelWidth(); kw++) {
                    long ih = oh * g.strideHeight() - g.paddingHeight() + kh * g.dilationHeight();
                    long iw = ow * g.strideWidth() - g.paddingWidth() + kw * g.dilationWidth();
                    if (ih < 0 || iw < 0 || ih >= ie[2] || iw >= ie[3]) { allNegativeZero = false; continue; }
                    double value = input.get(g.input().offset() + n * is[0] + c * is[1] + ih * is[2] + iw * is[3]);
                    allNegativeZero &= negativeZero(value); sum = g.dataType() == DataType.FLOAT64 ? sum + value : (float) sum + (float) value;
                }
                double result = g.dataType() == DataType.FLOAT64 ? sum / g.divisor() : (float) sum / (float) g.divisor();
                if (allNegativeZero) result = -0.0;
                put(g.dataType(), output, outputIndex, result);
            }
        }
    }

    private static void fillInput(CpuPool2dLowering.Geometry g, Storage input) {
        input.fill(-17.0); long[] e = g.input().extents(), s = g.input().strides();
        for (long n = 0; n < e[0]; n++) for (long c = 0; c < e[1]; c++) for (long h = 0; h < e[2]; h++) for (long w = 0; w < e[3]; w++) {
            double value = (n + c + h + w) % 7 - 3.25; if (h == 0 && w == 0) value = -0.0; if (h == 0 && w == 1) value = 0.0;
            input.put(g.input().offset() + n * s[0] + c * s[1] + h * s[2] + w * s[3], value);
        }
    }

    private static Set<String> inventoryOwners(String text) {
        String[] lines = text.split("\\n", -1); assertTrue(text.endsWith("\n") && !text.contains("\r"));
        Set<String> result = new TreeSet<>();
        for (int i = 1; i < lines.length - 1; i++) { String[] row = lines[i].split("\\t", -1); assertEquals(27, row.length, "inventory line " + (i + 1));
            if (row[0].startsWith("composition:") && row[2].equals("POOL1D_COMPOSITION") && row[21].equals("GENERATED")) assertTrue(result.add(row[0]), "duplicate owner " + row[0]); }
        return result;
    }
    private static void requireSame(Set<String> expected, Set<String> actual) { if (!expected.equals(actual)) throw new AssertionError("orphan or stale owner projection"); }
    private static String lineFor(String inventory, String owner) { return inventory.lines().filter(line -> line.startsWith(owner + '\t')).findFirst().orElseThrow(); }
    private static String resource() throws Exception { try (var stream = CpuPool1dCompositionSemanticClosureTest.class.getResourceAsStream(RESOURCE)) { assertNotNull(stream, RESOURCE); return new String(stream.readAllBytes(), StandardCharsets.UTF_8); } }
    private static String sha256(String value) throws Exception { return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8))); }
    private static boolean positive(double value) { return Double.doubleToRawLongBits(value) == 0L; }
    private static boolean negativeZero(double value) { return Double.doubleToRawLongBits(value) == Long.MIN_VALUE; }
    private static void put(DataType type, Object target, long index, double value) { switch (type) { case BFLOAT16 -> ((short[]) target)[(int) index] = ScalarValue.bfloat16((float) value).bfloat16Bits(); case FLOAT32 -> ((float[]) target)[(int) index] = (float) value; case FLOAT64 -> ((double[]) target)[(int) index] = value; default -> throw new AssertionError(type); } }
    private static void assertRawEquals(Object expected, Object actual, String message) { if (expected instanceof short[] a) assertArrayEquals(a, (short[]) actual, message); else if (expected instanceof float[] a) assertArrayEquals(bits(a), bits((float[]) actual), message); else assertArrayEquals(bits((double[]) expected), bits((double[]) actual), message); }
    private static int[] bits(float[] values) { int[] result = new int[values.length]; for (int i = 0; i < values.length; i++) result[i] = Float.floatToRawIntBits(values[i]); return result; }
    private static long[] bits(double[] values) { return Arrays.stream(values).mapToLong(Double::doubleToRawLongBits).toArray(); }

    private record Candidate(String owner, String geometry, DataType type, String request) { }
    private static final class Storage {
        final DataType type; final CarrierAccess carrier; final Object heap; final MemorySegment segment;
        Storage(DataType type, CarrierAccess carrier) { this.type = type; this.carrier = carrier; heap = switch (type) { case BFLOAT16 -> new short[512]; case FLOAT32 -> new float[512]; case FLOAT64 -> new double[512]; default -> throw new AssertionError(type); }; segment = heap instanceof short[] x ? MemorySegment.ofArray(x) : heap instanceof float[] x ? MemorySegment.ofArray(x) : MemorySegment.ofArray((double[]) heap); }
        Object argument() { return carrier == CarrierAccess.MEMORY_SEGMENT ? segment : heap; }
        void fill(double value) { for (int i = 0; i < 512; i++) CpuPool1dCompositionSemanticClosureTest.put(type, heap, i, value); }
        void put(long index, double value) { CpuPool1dCompositionSemanticClosureTest.put(type, heap, index, value); }
        double get(long index) { return switch (type) { case BFLOAT16 -> Float.intBitsToFloat(((((short[]) heap)[(int) index]) & 0xffff) << 16); case FLOAT32 -> ((float[]) heap)[(int) index]; case FLOAT64 -> ((double[]) heap)[(int) index]; default -> throw new AssertionError(type); }; }
        Object copy() { return heap instanceof short[] x ? x.clone() : heap instanceof float[] x ? x.clone() : ((double[]) heap).clone(); }
    }
}
