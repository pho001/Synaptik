package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.index.OneHotAttrs;
import io.github.pho001.synaptik.model.operation.random.DropoutAttrs;
import io.github.pho001.synaptik.model.operation.random.GraphRngStateAttrs;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.io.InputStream;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;

/** Direct generated-entry semantics for every ordinary random and one-hot inventory owner. */
class CpuRandomOneHotSemanticClosureTest {
    private static final String INVENTORY = "generated-coverage-inventory.tsv";
    private static final String INVENTORY_SHA256 =
            "1dcb69796c00fe3793d86f3f4cc3e816176062a45312ddbbaadfba9f8036cf20";
    private static final Set<String> FORMS = Set.of("DROPOUT", "ONE_HOT", "INITIAL_STATE");
    private static final Map<String, Long> FORM_COUNTS = Map.of(
            "DROPOUT", 16L, "ONE_HOT", 8L, "INITIAL_STATE", 4L);
    private static final long KEY_BIAS = 0x9e3779b97f4a7c15L;
    private static final long MIX_1 = 0xbf58476d1ce4e5b9L;
    private static final long MIX_2 = 0x94d049bb133111ebL;

    @Test void everyOrdinaryRandomAndOneHotOwnerDefinesAndExecutesItsActualEntry() throws Throwable {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.randomOrOneHotCandidates();
        assertEquals(28, candidates.size());
        assertEquals(FORM_COUNTS, counts(candidates));
        assertEquals(candidateIds(candidates), projection(inventoryBytes()).keySet());
        for (var candidate : candidates) execute(candidate);
    }

    @Test void shaBoundProjectionFailsClosedForDuplicateOrphanStaleAndMutationInputs()
            throws Exception {
        byte[] bytes = inventoryBytes();
        var owners = projection(bytes);
        assertEquals(28, owners.size());
        String text = new String(bytes, StandardCharsets.UTF_8);
        String owner = owners.keySet().iterator().next();
        String row = Arrays.stream(text.split("\\R")).filter(line -> line.startsWith(owner + "\t"))
                .findFirst().orElseThrow();
        assertThrows(AssertionError.class, () -> projectionIgnoringHash((text + '\n' + row)
                .getBytes(StandardCharsets.UTF_8)));
        String orphaned = Arrays.stream(text.split("\\R")).filter(line -> !line.startsWith(owner + "\t"))
                .reduce((left, right) -> left + '\n' + right).orElseThrow();
        assertThrows(AssertionError.class, () -> projectionIgnoringHash(orphaned.getBytes(StandardCharsets.UTF_8)));
        assertThrows(AssertionError.class, () -> projection(bytes, "0".repeat(64)));
        byte[] mutated = bytes.clone(); mutated[mutated.length - 2] ^= 1;
        assertThrows(AssertionError.class, () -> projection(mutated));
    }

    private static void execute(CpuOrdinaryNonPointwiseGeneratedMatrixTest.RandomOrOneHotCandidate candidate)
            throws Throwable {
        var context = candidate.context();
        var plan = new CpuPartitionPreparer().analyze(context).plan();
        var route = plan.units().getFirst().portablePlan();
        var generator = new CpuClassFileKernelGenerator();
        byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
        assertArrayEquals(first, generator.generateClassBytes(route.specialization(), route.kernelIr()),
                candidate.ownerId() + " deterministic bytes");
        var stores = stores(context, plan.boundaryValues());
        var inputs = snapshots(context, stores);
        long end = candidate.operationForm().equals("INITIAL_STATE") ? 0 : plan.elementCount();
        if (end > 0) assertTrue(end > 0, candidate.ownerId() + " non-empty selected range");
        oracle(context, stores, 0, end);
        var expected = snapshots(stores);
        restoreOutputs(context, stores);
        var args = new ArrayList<Object>();
        for (ValueId id : plan.boundaryValues()) args.add(carrier(stores.get(id),
                route.specialization().carrierPattern().get(args.size())));
        long[] geometry = candidate.operationForm().equals("ONE_HOT")
                ? plan.indexingGeometry().orElseThrow().pack(new long[plan.boundaryValues().size()], 0, end)
                : plan.randomGeometry().orElseThrow().pack(new long[plan.boundaryValues().size()]);
        args.add(geometry); args.add(0L); args.add(end);
        var handle = generator.defineClassBytes(route.specialization(), first).entryPoint();
        // Random entries establish their state-output write at the zero-width range boundary;
        // then the same direct entry owns the representative non-empty draw range.
        if (candidate.operationForm().equals("DROPOUT")) {
            args.set(args.size() - 1, 0L); handle.invokeWithArguments(args);
            args.set(args.size() - 1, end);
        }
        handle.invokeWithArguments(args);
        for (var entry : expected.entrySet()) assertRaw(entry.getValue(), stores.get(entry.getKey()).raw(),
                candidate.ownerId() + " output and untouched physical area");
        for (var entry : inputs.entrySet()) assertRaw(entry.getValue(), stores.get(entry.getKey()).raw(),
                candidate.ownerId() + " input/state immutability");
    }

    private static void oracle(PrepareContext<?> context, Map<ValueId, Store> stores, long start, long end) {
        var node = context.nodes().getFirst();
        switch (node.operation().kind().name()) {
            case "INITIAL_STATE" -> {
                var attrs = (GraphRngStateAttrs) node.operation().attrs();
                Store output = stores.get(node.outputs().getFirst());
                output.set(0, attrs.key()); output.set(1, attrs.counter());
            }
            case "ONE_HOT" -> {
                Store index = stores.get(node.inputs().getFirst()); Store output = stores.get(node.outputs().getFirst());
                long depth = ((OneHotAttrs) node.operation().attrs()).depth();
                for (long ordinal = start; ordinal < end; ordinal++) {
                    long source = ordinal / depth, category = ordinal % depth;
                    output.set(ordinal, index.get(source) == category ? 1 : 0);
                }
            }
            case "DROPOUT" -> {
                Store value = stores.get(node.inputs().get(0)), state = stores.get(node.inputs().get(1));
                Store output = stores.get(node.outputs().get(0)), mask = stores.get(node.outputs().get(1));
                Store next = stores.get(node.outputs().get(2));
                double probability = ((DropoutAttrs) node.operation().attrs()).probability();
                long key = state.get(0), counter = state.get(1);
                for (long ordinal = start; ordinal < end; ordinal++) {
                    boolean keep = uniform(word(key, counter, ordinal)) >= probability;
                    mask.set(ordinal, keep ? 1 : 0);
                    output.setFloating(ordinal, keep ? value.floating(ordinal) / (1.0 - probability) : 0.0);
                }
                next.set(0, key); next.set(1, counter + end - start);
            }
            default -> throw new AssertionError("unexpected form");
        }
    }

    private static Map<ValueId, Store> stores(PrepareContext<?> context, List<ValueId> boundary) {
        var result = new LinkedHashMap<ValueId, Store>();
        for (ValueId id : boundary) {
            var descriptor = descriptor(context, id); Store store = new Store(descriptor);
            store.fill(-71); result.put(id, store);
        }
        var node = context.nodes().getFirst();
        if (node.operation().kind().name().equals("DROPOUT")) {
            Store value = result.get(node.inputs().get(0));
            for (long i = 0; i < value.elements(); i++) value.setFloating(i, i + 1.25);
            result.get(node.inputs().get(1)).set(0, 0x1234); result.get(node.inputs().get(1)).set(1, 7);
        } else if (node.operation().kind().name().equals("ONE_HOT")) {
            Store index = result.get(node.inputs().getFirst());
            index.set(0, 2); index.set(1, -1); // includes a valid and an invalid category index.
        }
        return result;
    }

    private static Map<ValueId, Object> snapshots(PrepareContext<?> context, Map<ValueId, Store> stores) {
        var result = new LinkedHashMap<ValueId, Object>();
        for (ValueId id : context.nodes().getFirst().inputs()) result.put(id, stores.get(id).copy());
        return result;
    }
    private static Map<ValueId, Object> snapshots(Map<ValueId, Store> stores) {
        var result = new LinkedHashMap<ValueId, Object>(); stores.forEach((id, store) -> result.put(id, store.copy())); return result;
    }
    private static void restoreOutputs(PrepareContext<?> context, Map<ValueId, Store> stores) {
        for (ValueId id : context.nodes().getFirst().outputs()) stores.get(id).fill(-71);
    }
    private static TensorDescriptor descriptor(PrepareContext<?> c, ValueId id) {
        return c.values().stream().filter(value -> value.id().equals(id)).findFirst().orElseThrow().descriptor();
    }
    private static Object carrier(Store store, CarrierAccess access) {
        if (access != CarrierAccess.MEMORY_SEGMENT) return store.raw();
        Object raw = store.raw();
        if (raw instanceof double[] array) return MemorySegment.ofArray(array);
        if (raw instanceof float[] array) return MemorySegment.ofArray(array);
        if (raw instanceof long[] array) return MemorySegment.ofArray(array);
        if (raw instanceof int[] array) return MemorySegment.ofArray(array);
        return MemorySegment.ofArray((byte[]) raw);
    }

    private static Map<String, Long> counts(List<CpuOrdinaryNonPointwiseGeneratedMatrixTest.RandomOrOneHotCandidate> candidates) {
        var counts = new TreeMap<String, Long>(); candidates.forEach(c -> counts.merge(c.operationForm(), 1L, Long::sum)); return counts;
    }
    private static Set<String> candidateIds(List<CpuOrdinaryNonPointwiseGeneratedMatrixTest.RandomOrOneHotCandidate> candidates) {
        return candidates.stream().map(CpuOrdinaryNonPointwiseGeneratedMatrixTest.RandomOrOneHotCandidate::ownerId)
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
    }
    private static byte[] inventoryBytes() throws Exception {
        try (InputStream input = CpuRandomOneHotSemanticClosureTest.class.getResourceAsStream(INVENTORY)) {
            if (input == null) throw new AssertionError("missing checked inventory resource"); return input.readAllBytes();
        }
    }
    private static Map<String, String> projection(byte[] bytes) { return projection(bytes, INVENTORY_SHA256); }
    private static Map<String, String> projection(byte[] bytes, String hash) {
        if (!hash.equals(sha256(bytes))) throw new AssertionError("stale or mutated inventory"); return projectionIgnoringHash(bytes);
    }
    private static Map<String, String> projectionIgnoringHash(byte[] bytes) {
        var rows = new LinkedHashMap<String, String>(); String[] lines = new String(bytes, StandardCharsets.UTF_8).split("\\R");
        if (lines.length == 0 || !lines[0].startsWith("owner-id\t")) throw new AssertionError("inventory header");
        for (int line = 1; line < lines.length; line++) { if (lines[line].isEmpty()) continue; String[] f = lines[line].split("\t", -1);
            if (f.length != 27) throw new AssertionError("inventory columns at " + line);
            if (!f[0].startsWith("ordinary:") || !FORMS.contains(f[2]) || !f[21].equals("GENERATED")) continue;
            if (rows.putIfAbsent(f[0], f[2]) != null) throw new AssertionError("duplicate owner " + f[0]); }
        var actual = new TreeMap<String, Long>(); rows.values().forEach(form -> actual.merge(form, 1L, Long::sum));
        if (!FORM_COUNTS.equals(actual)) throw new AssertionError("orphan or stale owner projection: " + actual); return Map.copyOf(rows);
    }
    private static String sha256(byte[] bytes) { try { return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)); } catch (Exception exception) { throw new AssertionError(exception); } }
    private static long word(long key, long counter, long index) { return mix(counter + index + mix(key + KEY_BIAS)); }
    private static long mix(long value) { value = (value ^ value >>> 30) * MIX_1; value = (value ^ value >>> 27) * MIX_2; return value ^ value >>> 31; }
    private static double uniform(long word) { return (word >>> 11) * 0x1.0p-53; }
    private static void assertRaw(Object expected, Object actual, String message) {
        if (expected instanceof double[] x) assertArrayEquals(x, (double[]) actual, message); else if (expected instanceof float[] x) assertArrayEquals(x, (float[]) actual, message); else if (expected instanceof long[] x) assertArrayEquals(x, (long[]) actual, message); else if (expected instanceof int[] x) assertArrayEquals(x, (int[]) actual, message); else assertArrayEquals((byte[]) expected, (byte[]) actual, message);
    }

    private static final class Store {
        private final TensorDescriptor descriptor; private final Object raw;
        Store(TensorDescriptor descriptor) { this.descriptor = descriptor; this.raw = array(descriptor.dataType(), capacity(descriptor)); }
        Object raw() { return raw; } long elements() { return descriptor.shape().knownElementCount().orElseThrow(); }
        void fill(long value) { for (int i = 0; i < java.lang.reflect.Array.getLength(raw); i++) put(i, value); }
        void set(long logical, long value) { put(address(logical), value); }
        void setFloating(long logical, double value) { int at = address(logical); if (descriptor.dataType() == DataType.FLOAT64) ((double[]) raw)[at] = value; else ((float[]) raw)[at] = (float) value; }
        long get(long logical) { return getRaw(address(logical)); }
        double floating(long logical) { int at = address(logical); return descriptor.dataType() == DataType.FLOAT64 ? ((double[]) raw)[at] : ((float[]) raw)[at]; }
        Object copy() { int n = java.lang.reflect.Array.getLength(raw); Object copy = java.lang.reflect.Array.newInstance(raw.getClass().componentType(), n); System.arraycopy(raw, 0, copy, 0, n); return copy; }
        private int address(long logical) { long[] shape = descriptor.shape().toLongArray(); long[] strides = descriptor.layout().orElseThrow().strides(); long at = descriptor.layout().orElseThrow().storageOffset(); for (int axis = shape.length - 1; axis >= 0; axis--) { long coordinate = logical % shape[axis]; logical /= shape[axis]; at += coordinate * strides[axis]; } return Math.toIntExact(at); }
        private long getRaw(int at) { return switch (descriptor.dataType()) { case INT64 -> ((long[]) raw)[at]; case INT32 -> ((int[]) raw)[at]; case BOOL -> ((byte[]) raw)[at]; default -> throw new AssertionError("not an index/state store"); }; }
        private void put(int at, long value) { switch (descriptor.dataType()) { case FLOAT64 -> ((double[]) raw)[at] = value; case FLOAT32 -> ((float[]) raw)[at] = value; case INT64 -> ((long[]) raw)[at] = value; case INT32 -> ((int[]) raw)[at] = (int) value; case BOOL -> ((byte[]) raw)[at] = (byte) value; default -> throw new AssertionError("unexpected type"); } }
        private static int capacity(TensorDescriptor d) { long max = d.layout().orElseThrow().storageOffset(); long[] shape = d.shape().toLongArray(), strides = d.layout().orElseThrow().strides(); for (int i = 0; i < shape.length; i++) if (shape[i] != 0) max += (shape[i] - 1) * strides[i]; return Math.toIntExact(max + 1); }
        private static Object array(DataType type, int n) { return switch (type) { case FLOAT64 -> new double[n]; case FLOAT32 -> new float[n]; case INT64 -> new long[n]; case INT32 -> new int[n]; case BOOL -> new byte[n]; default -> throw new AssertionError("unexpected type"); }; }
    }
}
