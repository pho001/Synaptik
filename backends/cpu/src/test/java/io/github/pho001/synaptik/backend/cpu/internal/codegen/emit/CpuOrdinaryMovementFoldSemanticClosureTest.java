package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.CompositionAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.Fold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.FoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadAttrs;
import io.github.pho001.synaptik.model.operation.layout.PadKind;
import io.github.pho001.synaptik.model.operation.layout.SliceAttrs;
import io.github.pho001.synaptik.model.operation.layout.SliceKind;
import io.github.pho001.synaptik.model.operation.layout.TensorCompositionKind;
import io.github.pho001.synaptik.model.operation.layout.TileAttrs;
import io.github.pho001.synaptik.model.operation.layout.TileKind;
import io.github.pho001.synaptik.model.operation.layout.Unfold2dAttrs;
import io.github.pho001.synaptik.model.operation.layout.UnfoldAxisAttrs;
import io.github.pho001.synaptik.model.operation.layout.WindowTransformKind;
import io.github.pho001.synaptik.model.operation.layout.Window2dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import org.junit.jupiter.api.Test;

/** Direct, test-local semantic closure for the exact ordinary movement and fold owner projection. */
class CpuOrdinaryMovementFoldSemanticClosureTest {
    private static final String INVENTORY = "generated-coverage-inventory.tsv";
    private static final String INVENTORY_SHA256 =
            "1dcb69796c00fe3793d86f3f4cc3e816176062a45312ddbbaadfba9f8036cf20";
    private static final Set<String> FORMS = Set.of("CONCAT", "STACK", "TILE", "SLICE_UPDATE",
            "UNFOLD_AXIS", "FOLD_AXIS", "FOLD2D", "UNFOLD2D", "PAD");
    private static final Map<String, Long> FORM_COUNTS = Map.of("CONCAT", 24L, "STACK", 24L,
            "TILE", 24L, "SLICE_UPDATE", 24L, "UNFOLD_AXIS", 24L, "FOLD_AXIS", 20L,
            "FOLD2D", 12L, "UNFOLD2D", 4L, "PAD", 4L);

    @Test void everyExactOrdinaryMovementAndFoldOwnerExecutesAgainstIndependentOracle()
            throws Throwable {
        var candidates = CpuOrdinaryNonPointwiseGeneratedMatrixTest.movementOrFoldCandidates();
        assertEquals(160, candidates.size());
        assertEquals(FORM_COUNTS, counts(candidates));
        var inventory = projection(inventoryBytes());
        assertEquals(candidateIds(candidates), inventory.keySet(), "exact checked owner projection");

        for (var candidate : candidates) {
            var plan = new CpuPartitionPreparer().analyze(candidate.context()).plan();
            var route = plan.units().getFirst().portablePlan();
            var generator = new CpuClassFileKernelGenerator();
            byte[] first = generator.generateClassBytes(route.specialization(), route.kernelIr());
            byte[] second = generator.generateClassBytes(route.specialization(), route.kernelIr());
            assertArrayEquals(first, second, candidate.ownerId() + " deterministic bytes");

            var storages = storages(candidate.context(), plan.boundaryValues());
            var sourceSnapshots = inputSnapshots(candidate.context(), storages);
            var output = storages.get(candidate.context().nodes().getFirst().outputs().getFirst());
            long count = output.elements();
            long start = 1;
            long end = count;
            assertTrue(end > start, candidate.ownerId() + " non-empty selected range");
            oracle(candidate.context(), storages, start, end);
            Object expected = output.copy();
            output.restoreSentinel();

            var artifact = generator.defineClassBytes(route.specialization(), first);
            long[] geometry = plan.foldGeometry().isPresent()
                    ? plan.foldGeometry().orElseThrow().pack(new long[plan.boundaryValues().size()], start, end)
                    : plan.movementGeometry().orElseThrow().pack(new long[plan.boundaryValues().size()], start, end);
            var arguments = new ArrayList<Object>();
            for (ValueId id : plan.boundaryValues()) arguments.add(storages.get(id).carrier());
            arguments.add(geometry); arguments.add(start); arguments.add(end);
            artifact.entryPoint().invokeWithArguments(arguments);

            assertRawEquals(expected, output.copy(), candidate.ownerId());
            assertUntouchedOutsideRange(output, start, end, candidate.ownerId());
            for (var snapshot : sourceSnapshots.entrySet()) assertRawEquals(snapshot.getValue(),
                    storages.get(snapshot.getKey()).copy(), candidate.ownerId() + " source immutability");
        }
    }

    @Test void shaBoundProjectionFailsClosedForDuplicateOrphanStaleAndMutationInputs()
            throws Exception {
        byte[] bytes = inventoryBytes();
        var projection = projection(bytes);
        assertEquals(160, projection.size());
        String text = new String(bytes, StandardCharsets.UTF_8);
        String owner = projection.keySet().iterator().next();
        String row = Arrays.stream(text.split("\\R")).filter(line -> line.startsWith(owner + "\t"))
                .findFirst().orElseThrow();
        byte[] duplicate = (text + "\n" + row).getBytes(StandardCharsets.UTF_8);
        assertThrows(AssertionError.class, () -> projectionIgnoringHash(duplicate));
        assertThrows(AssertionError.class, () -> projection(duplicate));
        String orphaned = Arrays.stream(text.split("\\R")).filter(line -> !line.startsWith(owner + "\t"))
                .reduce((left, right) -> left + "\n" + right).orElseThrow();
        assertThrows(AssertionError.class, () -> projectionIgnoringHash(orphaned.getBytes(StandardCharsets.UTF_8)));
        assertThrows(AssertionError.class, () -> projection(orphaned.getBytes(StandardCharsets.UTF_8)));
        assertThrows(AssertionError.class, () -> projection(bytes, "0".repeat(64)));
        byte[] mutated = bytes.clone(); mutated[mutated.length - 2] ^= 1;
        assertThrows(AssertionError.class, () -> projection(mutated));
        assertThrows(AssertionError.class, () -> assertEquals(projection.keySet(), Set.of(owner)));
    }

    private static Map<String, Long> counts(List<CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate> candidates) {
        var result = new TreeMap<String, Long>();
        candidates.forEach(candidate -> result.merge(candidate.operationForm(), 1L, Long::sum));
        return result;
    }

    private static Set<String> candidateIds(List<CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate> candidates) {
        return candidates.stream().map(CpuOrdinaryNonPointwiseGeneratedMatrixTest.MovementOrFoldCandidate::ownerId)
                .collect(java.util.stream.Collectors.toUnmodifiableSet());
    }

    private static byte[] inventoryBytes() throws IOException {
        try (InputStream input = CpuOrdinaryMovementFoldSemanticClosureTest.class.getResourceAsStream(INVENTORY)) {
            if (input == null) throw new AssertionError("missing checked inventory resource");
            return input.readAllBytes();
        }
    }

    private static Map<String, String> projection(byte[] bytes) {
        return projection(bytes, INVENTORY_SHA256);
    }

    private static Map<String, String> projection(byte[] bytes, String expectedHash) {
        if (!expectedHash.equals(sha256(bytes))) throw new AssertionError("stale or mutated inventory");
        return projectionIgnoringHash(bytes);
    }

    private static Map<String, String> projectionIgnoringHash(byte[] bytes) {
        var rows = new LinkedHashMap<String, String>();
        String[] lines = new String(bytes, StandardCharsets.UTF_8).split("\\R");
        if (lines.length == 0 || !lines[0].startsWith("owner-id\t")) throw new AssertionError("inventory header");
        for (int line = 1; line < lines.length; line++) {
            if (lines[line].isEmpty()) continue;
            String[] fields = lines[line].split("\\t", -1);
            if (fields.length != 27) throw new AssertionError("inventory columns at " + line);
            if (!fields[0].startsWith("ordinary:") || !FORMS.contains(fields[2])
                    || !fields[21].equals("GENERATED")) continue;
            if (rows.putIfAbsent(fields[0], fields[2]) != null) throw new AssertionError("duplicate owner " + fields[0]);
        }
        var actual = new TreeMap<String, Long>();
        rows.values().forEach(form -> actual.merge(form, 1L, Long::sum));
        if (!FORM_COUNTS.equals(actual)) throw new AssertionError("orphan or stale owner projection: " + actual);
        return Map.copyOf(rows);
    }

    private static Map<ValueId, Storage> storages(PrepareContext<CpuPartitionAnalysisInputs> context,
            List<ValueId> boundary) {
        var result = new HashMap<ValueId, Storage>();
        for (ValueId id : boundary) {
            TensorDescriptor descriptor = descriptor(context, id);
            Storage storage = new Storage(descriptor, isSegment(context, id));
            storage.fillInput();
            result.put(id, storage);
        }
        Storage output = result.get(context.nodes().getFirst().outputs().getFirst());
        if (output == null) throw new AssertionError("output is not a boundary");
        output.restoreSentinel();
        return result;
    }

    private static boolean isSegment(PrepareContext<CpuPartitionAnalysisInputs> context, ValueId id) {
        int index = context.values().stream().map(value -> value.id()).toList().indexOf(id);
        return context.backendInputs().carrierPattern().get(index)
                == io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT;
    }

    private static Map<ValueId, Object> inputSnapshots(PrepareContext<CpuPartitionAnalysisInputs> context,
            Map<ValueId, Storage> storages) {
        var result = new HashMap<ValueId, Object>();
        for (ValueId id : context.nodes().getFirst().inputs()) result.putIfAbsent(id, storages.get(id).copy());
        return result;
    }

    private static void oracle(PrepareContext<CpuPartitionAnalysisInputs> context,
            Map<ValueId, Storage> storage, long start, long end) {
        Operation operation = context.nodes().getFirst().operation();
        ValueId outputId = context.nodes().getFirst().outputs().getFirst();
        Storage output = storage.get(outputId);
        for (long linear = start; linear < end; linear++) {
            long[] coordinate = coordinates(output.descriptor.shape(), linear);
            long raw = switch (operation.kind()) {
                case PadKind ignored -> pad((PadAttrs) operation.attrs(), storage.get(context.nodes().getFirst().inputs().getFirst()), coordinate);
                case TileKind ignored -> tile(storage.get(context.nodes().getFirst().inputs().getFirst()), coordinate);
                case TensorCompositionKind kind when kind == TensorCompositionKind.CONCAT -> concat(context, storage, coordinate);
                case TensorCompositionKind kind when kind == TensorCompositionKind.STACK -> stack(context, storage, coordinate);
                case WindowTransformKind kind when kind == WindowTransformKind.UNFOLD_AXIS -> unfoldAxis((UnfoldAxisAttrs) operation.attrs(), storage.get(context.nodes().getFirst().inputs().getFirst()), coordinate);
                case WindowTransformKind kind when kind == WindowTransformKind.UNFOLD2D -> unfold2d((Unfold2dAttrs) operation.attrs(), storage.get(context.nodes().getFirst().inputs().getFirst()), coordinate);
                case SliceKind ignored -> sliceUpdate((SliceAttrs) operation.attrs(), context, storage, coordinate);
                case WindowTransformKind kind when kind == WindowTransformKind.FOLD_AXIS -> foldAxis((FoldAxisAttrs) operation.attrs(), storage.get(context.nodes().getFirst().inputs().getFirst()), output, coordinate);
                case WindowTransformKind kind when kind == WindowTransformKind.FOLD2D -> fold2d((Fold2dAttrs) operation.attrs(), storage.get(context.nodes().getFirst().inputs().getFirst()), output, coordinate);
                default -> throw new AssertionError("unexpected closure form " + operation.kind());
            };
            output.set(coordinate, raw);
        }
    }

    private static long pad(PadAttrs attrs, Storage input, long[] out) {
        long[] in = new long[out.length];
        for (int i = 0; i < out.length; i++) { in[i] = out[i] - attrs.before().get(i); if (in[i] < 0 || in[i] >= input.extent(i)) return scalarRaw(attrs.constantValue()); }
        return input.get(in);
    }
    private static long tile(Storage input, long[] out) { long[] in = new long[out.length]; for (int i = 0; i < out.length; i++) in[i] = out[i] % input.extent(i); return input.get(in); }
    private static long concat(PrepareContext<CpuPartitionAnalysisInputs> c, Map<ValueId, Storage> s, long[] out) {
        int axis = ((CompositionAxisAttrs) c.nodes().getFirst().operation().attrs()).axis(); long at = out[axis];
        for (ValueId id : c.nodes().getFirst().inputs()) { Storage input = s.get(id); if (at < input.extent(axis)) { long[] in = out.clone(); in[axis] = at; return input.get(in); } at -= input.extent(axis); }
        throw new AssertionError("concat coordinate");
    }
    private static long stack(PrepareContext<CpuPartitionAnalysisInputs> c, Map<ValueId, Storage> s, long[] out) {
        int axis = ((CompositionAxisAttrs) c.nodes().getFirst().operation().attrs()).axis(); Storage input = s.get(c.nodes().getFirst().inputs().get(Math.toIntExact(out[axis]))); long[] in = new long[out.length - 1]; for (int i = 0, j = 0; i < out.length; i++) if (i != axis) in[j++] = out[i]; return input.get(in);
    }
    private static long unfoldAxis(UnfoldAxisAttrs attrs, Storage input, long[] out) { long[] in = new long[out.length - 1]; int axis = attrs.axis(); for (int i = 0; i < in.length; i++) in[i] = i < axis ? out[i] : i == axis ? out[i] * attrs.step() + out[i + 1] : out[i + 1]; return input.get(in); }
    private static long unfold2d(Unfold2dAttrs attrs, Storage input, long[] out) {
        Window2dAttrs w = attrs.window(); long[] shape = input.descriptor.shape().toLongArray(); long oh = windows(shape[2], w.kernelHeight(), w.strideHeight(), w.paddingHeight(), w.dilationHeight(), w.ceilMode()); long ow = windows(shape[3], w.kernelWidth(), w.strideWidth(), w.paddingWidth(), w.dilationWidth(), w.ceilMode()); long channel = out[1]; long c = channel / (w.kernelHeight() * w.kernelWidth()); long k = channel % (w.kernelHeight() * w.kernelWidth()); long y = (out[2] / ow) * w.strideHeight() - w.paddingHeight() + (k / w.kernelWidth()) * w.dilationHeight(); long x = (out[2] % ow) * w.strideWidth() - w.paddingWidth() + (k % w.kernelWidth()) * w.dilationWidth(); return y < 0 || x < 0 || y >= shape[2] || x >= shape[3] ? scalarRaw(attrs.paddingValue()) : input.get(new long[]{out[0], c, y, x});
    }
    private static long sliceUpdate(SliceAttrs attrs, PrepareContext<CpuPartitionAnalysisInputs> c, Map<ValueId, Storage> s, long[] out) { Storage base = s.get(c.nodes().getFirst().inputs().getFirst()); Storage update = s.get(c.nodes().getFirst().inputs().get(1)); long[] u = out.clone(); for (int i = 0; i < attrs.axes().size(); i++) { int axis = attrs.axes().get(i); long delta = out[axis] - attrs.starts().get(i); if (delta % attrs.steps().get(i) != 0) return base.get(out); long relative = delta / attrs.steps().get(i); if (relative < 0 || relative >= attrs.lengths().get(i)) return base.get(out); u[axis] = relative; } return update.get(u); }
    private static long foldAxis(FoldAxisAttrs attrs, Storage input, Storage output, long[] out) { int axis = attrs.axis(); long sum = output.zero(); for (long i = 0; i < input.elements(); i++) { long[] coordinate = coordinates(input.descriptor.shape(), i); long[] target = new long[out.length]; for (int a = 0, b = 0; a < coordinate.length; a++) { if (a == axis) target[b++] = coordinate[a] * attrs.step() + coordinate[a + 1]; else if (a != axis + 1) target[b++] = coordinate[a]; } if (Arrays.equals(target, out)) sum = output.add(sum, input.get(coordinate)); } return sum; }
    private static long fold2d(Fold2dAttrs attrs, Storage input, Storage output, long[] out) { Window2dAttrs w = attrs.window(); long[] inputShape = input.descriptor.shape().toLongArray(); long oh = windows(output.extent(2), w.kernelHeight(), w.strideHeight(), w.paddingHeight(), w.dilationHeight(), w.ceilMode()); long ow = windows(output.extent(3), w.kernelWidth(), w.strideWidth(), w.paddingWidth(), w.dilationWidth(), w.ceilMode()); long sum = output.zero(); for (long linear = 0; linear < input.elements(); linear++) { long[] col = coordinates(input.descriptor.shape(), linear); long channel = col[1], c = channel / (w.kernelHeight() * w.kernelWidth()), k = channel % (w.kernelHeight() * w.kernelWidth()); long y = (col[2] / ow) * w.strideHeight() - w.paddingHeight() + (k / w.kernelWidth()) * w.dilationHeight(); long x = (col[2] % ow) * w.strideWidth() - w.paddingWidth() + (k % w.kernelWidth()) * w.dilationWidth(); if (col[0] == out[0] && c == out[1] && y == out[2] && x == out[3]) sum = output.add(sum, input.get(col)); } return sum; }
    private static long windows(long size, long kernel, long stride, long pad, long dilation, boolean ceil) { long numerator = size + 2 * pad - (dilation * (kernel - 1) + 1); return (ceil ? Math.floorDiv(numerator + stride - 1, stride) : Math.floorDiv(numerator, stride)) + 1; }

    private static TensorDescriptor descriptor(PrepareContext<CpuPartitionAnalysisInputs> c, ValueId id) { return c.values().stream().filter(value -> value.id().equals(id)).findFirst().orElseThrow().descriptor(); }
    private static long scalarRaw(io.github.pho001.synaptik.model.datatype.ScalarValue value) { return switch (value.dataType()) { case FLOAT64 -> Double.doubleToRawLongBits(value.float64Value()); case FLOAT32 -> Float.floatToRawIntBits(value.float32Value()) & 0xffffffffL; case BFLOAT16 -> value.bfloat16Bits() & 0xffffL; case INT32 -> value.int32Value() & 0xffffffffL; case INT64 -> value.int64Value(); case BOOL -> value.booleanValue() ? 1L : 0L; }; }
    private static long[] coordinates(Shape shape, long linear) { long[] dims = shape.toLongArray(), result = new long[dims.length]; for (int i = dims.length - 1; i >= 0; i--) { result[i] = linear % dims[i]; linear /= dims[i]; } return result; }
    private static void assertUntouchedOutsideRange(Storage output, long start, long end, String label) { for (long i = 0; i < output.elements(); i++) if (i < start || i >= end) assertEquals(output.sentinel(), output.get(coordinates(output.descriptor.shape(), i)), label + " untouched " + i); }
    private static void assertRawEquals(Object expected, Object actual, String label) { if (expected instanceof double[] a) assertArrayEquals(a, (double[]) actual, label); else if (expected instanceof float[] a) assertArrayEquals(a, (float[]) actual, label); else if (expected instanceof short[] a) assertArrayEquals(a, (short[]) actual, label); else if (expected instanceof int[] a) assertArrayEquals(a, (int[]) actual, label); else if (expected instanceof long[] a) assertArrayEquals(a, (long[]) actual, label); else assertArrayEquals((byte[]) expected, (byte[]) actual, label); }
    private static String sha256(byte[] bytes) { try { return java.util.HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)); } catch (NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); } }

    private static final class Storage {
        private final TensorDescriptor descriptor; private final Object array; private final Object carrier;
        Storage(TensorDescriptor descriptor, boolean segment) { this.descriptor = descriptor; int length = Math.toIntExact(maxAddress(descriptor) + 1); array = switch (descriptor.dataType()) { case FLOAT64 -> new double[length]; case FLOAT32 -> new float[length]; case BFLOAT16 -> new short[length]; case INT32 -> new int[length]; case INT64 -> new long[length]; case BOOL -> new byte[length]; }; carrier = segment ? segment(array) : array; }
        long elements() { return descriptor.shape().knownElementCount().orElseThrow(); } long extent(int axis) { return descriptor.shape().toLongArray()[axis]; } long sentinel() { return switch (descriptor.dataType()) { case FLOAT64 -> Double.doubleToRawLongBits(-991.25); case FLOAT32 -> Float.floatToRawIntBits(-991.25f); case BFLOAT16 -> 0xc47d; case INT32 -> 0x55aa55aaL; case INT64 -> 0x55aa55aa55aa55aaL; case BOOL -> 0x5a; }; } long zero() { return 0; }
        void fillInput() { for (long i = 0; i < elements(); i++) set(coordinates(descriptor.shape(), i), sample(i)); }
        void restoreSentinel() { for (long i = 0; i < elements(); i++) set(coordinates(descriptor.shape(), i), sentinel()); }
        Object carrier() { return carrier; } Object copy() { if (array instanceof double[] a) return a.clone(); if (array instanceof float[] a) return a.clone(); if (array instanceof short[] a) return a.clone(); if (array instanceof int[] a) return a.clone(); if (array instanceof long[] a) return a.clone(); return ((byte[]) array).clone(); }
        long get(long[] c) { int p = Math.toIntExact(address(descriptor, c)); return switch (descriptor.dataType()) { case FLOAT64 -> Double.doubleToRawLongBits(((double[]) array)[p]); case FLOAT32 -> Float.floatToRawIntBits(((float[]) array)[p]); case BFLOAT16 -> ((short[]) array)[p] & 0xffffL; case INT32 -> ((int[]) array)[p] & 0xffffffffL; case INT64 -> ((long[]) array)[p]; case BOOL -> ((byte[]) array)[p] & 0xffL; }; }
        void set(long[] c, long raw) { int p = Math.toIntExact(address(descriptor, c)); switch (descriptor.dataType()) { case FLOAT64 -> ((double[]) array)[p] = Double.longBitsToDouble(raw); case FLOAT32 -> ((float[]) array)[p] = Float.intBitsToFloat((int) raw); case BFLOAT16 -> ((short[]) array)[p] = (short) raw; case INT32 -> ((int[]) array)[p] = (int) raw; case INT64 -> ((long[]) array)[p] = raw; case BOOL -> ((byte[]) array)[p] = (byte) raw; } }
        long sample(long i) { return switch (descriptor.dataType()) { case FLOAT64 -> Double.doubleToRawLongBits((i % 7 - 3) * 1.25); case FLOAT32 -> Float.floatToRawIntBits((float) ((i % 7 - 3) * 1.25)); case BFLOAT16 -> bfloat((float) ((i % 7 - 3) * 1.25)); case INT32 -> (i % 7 - 3) * 0x40000001L; case INT64 -> (i % 7 - 3) * 0x4000000000000001L; case BOOL -> i & 1; }; }
        long add(long left, long right) { return switch (descriptor.dataType()) { case FLOAT64 -> Double.doubleToRawLongBits(Double.longBitsToDouble(left) + Double.longBitsToDouble(right)); case FLOAT32 -> Float.floatToRawIntBits(Float.intBitsToFloat((int) left) + Float.intBitsToFloat((int) right)); case BFLOAT16 -> bfloat(Float.intBitsToFloat((int) left << 16) + Float.intBitsToFloat((int) right << 16)); case INT32 -> (int) left + (int) right; case INT64 -> left + right; case BOOL -> throw new AssertionError(); }; }
        private static long bfloat(float value) { int bits = Float.floatToRawIntBits(value); int upper = bits >>> 16, lower = bits & 0xffff; return (lower > 0x8000 || lower == 0x8000 && (upper & 1) != 0 ? upper + 1 : upper) & 0xffffL; }
    }
    private static long address(TensorDescriptor d, long[] coordinates) { long[] strides = d.layout().orElseThrow().strides(); long result = d.layout().orElseThrow().storageOffset(); for (int i = 0; i < coordinates.length; i++) result += coordinates[i] * strides[i]; return result; }
    private static long maxAddress(TensorDescriptor d) { long[] shape = d.shape().toLongArray(), strides = d.layout().orElseThrow().strides(); long result = d.layout().orElseThrow().storageOffset(); for (int i = 0; i < shape.length; i++) result += (shape[i] - 1) * strides[i]; return result; }
    private static MemorySegment segment(Object array) { return switch (array) { case double[] a -> MemorySegment.ofArray(a); case float[] a -> MemorySegment.ofArray(a); case short[] a -> MemorySegment.ofArray(a); case int[] a -> MemorySegment.ofArray(a); case long[] a -> MemorySegment.ofArray(a); case byte[] a -> MemorySegment.ofArray(a); default -> throw new AssertionError(); }; }
}
