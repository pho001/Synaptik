package io.github.pho001.synaptik.backend.metal;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.Scan;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.ScanActivityPlan;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.OptionalInt;

/**
 * Independent Java acceptance of the one native Scan grammar, including untrusted raw callers.
 * Ordinary instructions are reconstructed with their existing constructors; their semantic/type
 * validation remains with the ordinary native-ABI owner. This class decodes only new binding,
 * resource, demand and structural-presence information, never an alternate ordinary math grammar.
 */
final class MetalScanDecoding {
    private MetalScanDecoding() {}

    /**
     * Parses and independently validates the complete intrinsic body/binder image.
     * @param source exact attribute words; borrowed only for this cold validation call
     * @return immutable decoded physical metadata; its word array is not exposed or retained by execution
     * @throws IllegalArgumentException if any version, section, local reference, control or resource fact disagrees
     * @throws ArithmeticException if checked geometry overflows
     */
    static Decoded decode(long[] source) {
        if (source == null || source.length < 32 || source.length > MetalScanEncoding.MAX_WORDS) throw malformed();
        long[] w = source;
        if (w[0] != MetalScanEncoding.MAGIC || w[1] != MetalScanEncoding.VERSION || w[2] != w.length || w[29] != w.length
                || w[3] < 0 || (w[4] & ~7L) != 0 || w[5] < 0 || w[6] < 1 || w[7] < 0 || w[8] < 0 || w[9] < 0
                || w[5] + w[8] < 1 || w[10] < 0 || w[10] > w[5] || w[11] < 0
                || w[12] < 1 || w[12] > MetalScanEncoding.MAX_VALUES || w[13] < 0 || w[13] > MetalScanEncoding.MAX_UNITS
                || w[14] != Math.addExact(Math.addExact(w[5], w[6]), w[7])
                || w[15] != Math.addExact(Math.addExact(w[5], w[8]), w[9])
                || w[14] > MetalScanEncoding.MAX_VALUES || w[15] > MetalScanEncoding.MAX_VALUES
                || w[16] < 0 || w[16] > w[15] || w[17] != Math.addExact(w[19], w[20])
                || w[17] < 0 || w[17] > w[14] || w[18] < 0 || w[18] > MetalScanEncoding.MAX_VALUES + MetalScanEncoding.MAX_UNITS
                || w[19] < 0 || w[20] < 0 || w[21] < 0 || w[19] + w[21] > w[5]
                || w[22] < 0 || w[23] != 32 || w[30] < 0 || w[31] < 15
                || (w[22] & 7) != 0 || (w[30] & 7) != 0
                || Math.addExact(w[30], Math.multiplyExact(w[31], 8)) != w[22]) throw malformed();
        for (int section = 23; section < 29; section++) if (w[section] > w[section + 1] || w[section + 1] > w.length) throw malformed();
        boolean lengths = (w[4] & 2) != 0, activity = (w[4] & 4) != 0;
        if ((!lengths && (w[10] != 0 || w[11] != 0)) || (!activity && (w[17] != 0 || w[18] != 0 || w[19] != 0 || w[20] != 0 || w[21] != 0))) throw malformed();
        var values = new ArrayList<MetalScanPlan.Value>();
        int cursor = 32;
        long resourceEnd = 0;
        for (int v = 0; v < w[12]; v++) {
            if (w[24] - cursor < 13) throw malformed();
            int rank = bounded(w[cursor + 2], 16);
            int record = Math.addExact(13, Math.multiplyExact(rank, 2));
            if (w[cursor] != record || record > w[24] - cursor) throw malformed();
            DataType type = carrier(w[cursor + 1]);
            if (w[cursor + 3] < 0 || w[cursor + 3] > 1 || w[cursor + 3] != 0 && !MetalScanLowering.numericCarrier(type)
                    || w[cursor + 4] != 1 || w[cursor + 5] != 0 || w[cursor + 10] < 0 || w[cursor + 10] > 1 || w[cursor + 12] != 0) throw malformed();
            long[] dimensions = Arrays.copyOfRange(w, cursor + 13, cursor + 13 + rank);
            Shape shape = Shape.of(dimensions);
            var layout = LayoutDescriptor.contiguous(shape);
            if (!Arrays.equals(layout.strides(), Arrays.copyOfRange(w, cursor + 13 + rank, cursor + record))
                    || w[cursor + 6] != layout.referencedElementSpan()) throw malformed();
            long bytes = Math.multiplyExact(layout.referencedElementSpan(), type.byteWidth());
            if (w[cursor + 8] != bytes) throw malformed();
            int row = Math.toIntExact(w[cursor + 9]);
            if (row < -1 || row >= rank || !lengths && (row != -1 || w[cursor + 10] != 0)
                    || row >= 0 && (dimensions[row] != w[11] || w[cursor + 10] != 0)) throw malformed();
            long inner = row < 0 ? 0 : 1;
            if (row >= 0) for (int a = row + 1; a < rank; a++) inner = Math.multiplyExact(inner, dimensions[a]);
            if (w[cursor + 11] != inner) throw malformed();
            long offset = w[cursor + 7];
            if (offset != -1) {
                if (offset != align(resourceEnd)) throw malformed();
                resourceEnd = Math.addExact(offset, bytes);
                if (resourceEnd > w[30]) throw malformed();
            }
            values.add(new MetalScanPlan.Value(new TensorDescriptor(type, shape, java.util.Optional.of(layout), w[cursor + 3] != 0),
                    row, w[cursor + 10] != 0, offset, bytes, inner));
            cursor += record;
        }
        if (cursor != w[24] || Math.addExact(w[25], w[14]) != w[26] || Math.addExact(w[26], w[15]) != w[27]
                || Math.addExact(w[27], w[16]) != w[28]) throw malformed();
        List<Integer> formals = indices(w, w[25], w[14], values.size());
        List<Integer> roles = indices(w, w[26], w[15], values.size());
        List<Integer> demands = indices(w, w[27], w[16], roles.size());
        if (new HashSet<>(formals).size() != formals.size()) throw malformed();
        for (int d = 1; d < demands.size(); d++) if (demands.get(d) <= demands.get(d - 1)) throw malformed();
        Set<Integer> available = new HashSet<>(formals);
        var semantics = new Scan.ValueSemantics[values.size()];
        var shared = new Scan.ValueSemantics(OptionalInt.empty(), false, List.of());
        var batch = lengths ? new StaticDimension(w[11]) : null;
        for (int f = 0; f < formals.size(); f++) {
            var value = values.get(formals.get(f));
            if (value.terminal() || (value.byteOffset() < 0) != (f >= w[5]) || lengths && (f < w[10] && value.rowAxis() != 0
                    || f >= w[10] && f < w[5] && value.rowAxis() != -1 || f >= w[5] + w[6] && value.rowAxis() != -1)) throw malformed();
            semantics[formals.get(f)] = lengths ? Scan.declaredSemantics(value.descriptor(),
                    value.rowAxis() < 0 ? OptionalInt.empty() : OptionalInt.of(value.rowAxis()), batch) : shared;
        }
        var nextUses = new java.util.HashMap<Integer, Integer>();
        roles.subList(0, Math.toIntExact(w[5])).forEach(id -> nextUses.merge(id, 1, Integer::sum));
        for (int c = 0; c < w[5]; c++) if (available.contains(roles.get(c)) || nextUses.get(roles.get(c)) != 1) {
            resourceEnd = Math.addExact(align(resourceEnd), values.get(formals.get(c)).byteSize());
        }
        var physicalDescriptors = values.stream().map(v -> MetalMpsGraphProgram.ValueDescriptor.from(v.descriptor())).toList();
        var localDescriptors = values.stream().map(MetalScanPlan.Value::descriptor).toList();
        var units = new ArrayList<MetalScanPlan.Unit>();
        for (int u = 0; u < w[13]; u++) {
            if (w[25] - cursor < 14) throw malformed();
            var kind = kind(w[cursor + 1]);
            boolean variadic = kind == MetalMpsGraphProgram.NodeKind.CONCAT || kind == MetalMpsGraphProgram.NodeKind.STACK;
            int inputs = bounded(w[cursor + 3], variadic ? MetalScanEncoding.MAX_WORDS : 16);
            int outputs = bounded(w[cursor + 4], 5), attributes = bounded(w[cursor + 5], 65);
            int record = Math.addExact(14, Math.addExact(inputs, Math.addExact(outputs, attributes)));
            if (outputs == 0 || w[cursor] != record || record > w[25] - cursor) throw malformed();
            var attributeKind = attributeKind(w[cursor + 2]);
            int[] operandIds = indices(w, cursor + 14L, inputs, values.size()).stream().mapToInt(Integer::intValue).toArray();
            int[] outputIds = indices(w, cursor + 14L + inputs, outputs, values.size()).stream().mapToInt(Integer::intValue).toArray();
            for (int operand : operandIds)
                if (values.get(operand).descriptor().dataType() == DataType.FLOAT64) throw malformed();
            for (int output : outputIds)
                if (values.get(output).descriptor().dataType() == DataType.FLOAT64) throw malformed();
            for (int operand : operandIds) if (!available.contains(operand)) throw malformed();
            for (int output : outputIds) if (!available.add(output) || values.get(output).byteOffset() < 0) throw malformed();
            var node = MetalMpsGraphProgram.Node.genericForScanBody(kind, operandIds, outputIds, attributeKind,
                    Arrays.copyOfRange(w, cursor + 14 + inputs + outputs, cursor + record));
            // Prove ordinary metadata first, then derive coordinates from the Model owner.
            // Input facts were established at declarations or preceding SSA producers; claimed
            // output axes/terminal bits are never used as inference inputs.
            MetalNativeApi.ProgramExecutableAbi.validateScanBody(node, physicalDescriptors);
            var expectedFacts = lengths ? Scan.ordinarySemantics(
                    MetalScanOrdinaryOperation.decode(node, localDescriptors),
                    Arrays.stream(operandIds).mapToObj(localDescriptors::get).toList(),
                    Arrays.stream(operandIds).mapToObj(id -> semantics[id]).toList(),
                    Arrays.stream(outputIds).mapToObj(localDescriptors::get).toList(), batch)
                    : Arrays.stream(outputIds).mapToObj(unused -> shared).toList();
            for (int port = 0; port < outputIds.length; port++) {
                var claim = values.get(outputIds[port]);
                var expected = expectedFacts.get(port);
                if (claim.rowAxis() != expected.rowAxis().orElse(-1) || claim.terminal() != expected.terminal()) throw malformed();
                semantics[outputIds[port]] = expected;
            }
            int scope = scope(w[cursor + 6], w[18]), left = scope(w[cursor + 7], w[18]), right = scope(w[cursor + 8], w[18]);
            int row = Math.toIntExact(w[cursor + 9]), accumulator = Math.toIntExact(w[cursor + 11]);
            if (activity && scope < 0 || !activity && (scope >= 0 || left >= 0 || right >= 0 || accumulator >= 0)
                    || (left < 0) != (right < 0) || row != values.get(outputIds[0]).rowAxis()
                    || w[cursor + 10] < 0 || w[cursor + 10] > 1 || accumulator < -1 || accumulator >= w[5]) throw malformed();
            boolean contraction = row < 0 && Arrays.stream(operandIds).anyMatch(id -> semantics[id].rowAxis().isPresent());
            if ((w[cursor + 10] != 0) != contraction) throw malformed();
            if (left >= 0 && (kind != MetalMpsGraphProgram.NodeKind.ADD || inputs != 2 || outputs != 1)) throw malformed();
            long scratch = MetalProgramWorkspace.scratchBytes(node, physicalDescriptors);
            long scratchOffset = scratch == 0 ? -1 : align(resourceEnd);
            if (w[cursor + 12] != scratchOffset || w[cursor + 13] != scratch) throw malformed();
            if (scratch != 0) resourceEnd = Math.addExact(scratchOffset, scratch);
            units.add(new MetalScanPlan.Unit(node, scope, left, right, row, w[cursor + 10] != 0, accumulator, scratchOffset, scratch));
            cursor += record;
        }
        if (cursor != w[25] || align(resourceEnd) != w[30] || !available.containsAll(roles)) throw malformed();
        for (int c = 0; c < w[5]; c++) if (!sameValueShapeType(values.get(formals.get(c)), values.get(roles.get(c)))) throw malformed();
        if (lengths) {
            for (int p = 0; p < w[5] + w[8]; p++) {
                var value = values.get(roles.get(p));
                if (p < w[10] || p >= w[5]) {
                    if (value.rowAxis() != 0 || value.terminal()) throw malformed();
                } else if (value.rowAxis() != -1) throw malformed();
            }
            var dependent = new HashSet<Integer>();
            var expectedNext = new java.util.HashMap<Integer, Integer>();
            var seen = new HashSet<Integer>();
            for (int c = Math.toIntExact(w[10]); c < w[5]; c++) {
                dependent.add(formals.get(c)); expectedNext.put(formals.get(c), roles.get(c));
            }
            for (var unit : units) {
                int[] operands = unit.node().inputs();
                for (int i = 0; i < operands.length; i++) if (dependent.contains(operands[i])) {
                    Integer next = expectedNext.get(operands[i]);
                    if (next == null || unit.node().kind() != MetalMpsGraphProgram.NodeKind.ADD || i != 0
                            || unit.node().outputs().length != 1 || unit.node().outputIndex() != next) throw malformed();
                    seen.add(operands[i]); dependent.add(next);
                }
            }
            if (seen.size() != expectedNext.size()) throw malformed();
        }
        var carries = new ArrayList<ScanActivityPlan.Carry>();
        var seeds = new ArrayList<Integer>();
        var scopes = new ArrayList<List<Integer>>();
        var accumulators = new ArrayList<ScanActivityPlan.Accumulator>();
        Set<Integer> numeric = new HashSet<>(), seedSet = new HashSet<>();
        cursor = Math.toIntExact(w[28]);
        for (int c = 0; c < w[19]; c++) {
            if (w.length - cursor < 3) throw malformed();
            int position = bounded(w[cursor], Math.toIntExact(w[5]) - 1), nextScope = scope(w[cursor + 2], w[18]);
            if (nextScope < 0 || w[cursor + 1] < 0 || w[cursor + 1] > 1 || !numeric.add(position)
                    || lengths && position >= w[10]
                    || !MetalScanLowering.numericCarrier(values.get(formals.get(position)).descriptor().dataType())) throw malformed();
            carries.add(new ScanActivityPlan.Carry(position, w[cursor + 1] != 0, nextScope)); cursor += 3;
        }
        for (int s = 0; s < w[20]; s++) {
            if (cursor >= w.length) throw malformed();
            int position = Math.toIntExact(w[cursor++]);
            if (position < w[5] || position >= w[5] + w[6] || !seedSet.add(position)
                    || !MetalScanLowering.numericCarrier(values.get(formals.get(position)).descriptor().dataType())) throw malformed();
            seeds.add(position);
        }
        for (int s = 0; s < w[18]; s++) {
            if (cursor >= w.length) throw malformed();
            int count = bounded(w[cursor++], Math.toIntExact(w[17]));
            List<Integer> members = indices(w, cursor, count, Math.toIntExact(w[17]));
            for (int r = 1; r < members.size(); r++) if (members.get(r) <= members.get(r - 1)) throw malformed();
            scopes.add(members); cursor += count;
        }
        for (int a = 0; a < w[21]; a++) {
            if (w.length - cursor < 2) throw malformed();
            int position = bounded(w[cursor], Math.toIntExact(w[5]) - 1), contribution = scope(w[cursor + 1], w[18]);
            if (contribution < 0 || !numeric.add(position) || lengths && position < w[10]
                    || !MetalScanLowering.numericCarrier(values.get(formals.get(position)).descriptor().dataType())) throw malformed();
            accumulators.add(new ScanActivityPlan.Accumulator(position, contribution)); cursor += 2;
        }
        if (cursor != w.length || activity && numeric.size() != w[5]) throw malformed();
        long roots = w[17], rootWords = Math.addExact(roots, 63) / 64, rows = lengths ? w[11] : 1;
        long membershipWords = Math.multiplyExact(Math.addExact(w[18], roots), rootWords);
        long controlWords = Math.addExact(15, Math.addExact(membershipWords,
                Math.addExact(Math.multiplyExact(Math.multiplyExact(rows, rootWords), 2), lengths ? rows : 0)));
        if (membershipWords > MetalScanEncoding.MAX_WORDS || controlWords != w[31]) throw malformed();
        var decoded = new Decoded(w, List.copyOf(values), List.copyOf(units), formals, roles, demands,
                List.copyOf(carries), List.copyOf(seeds), List.copyOf(scopes), List.copyOf(accumulators));
        validateActivity(decoded);
        return decoded;
    }

    private static void validateActivity(Decoded d) {
        var memberships = new ArrayList<BitSet>();
        for (var scope : d.scopes()) { var bits = new BitSet(); scope.forEach(bits::set); memberships.add(bits); }
        int[] availability = new int[d.values().size()]; Arrays.fill(availability, -1);
        int[] producer = new int[d.values().size()]; Arrays.fill(producer, -1);
        int root = 0;
        for (var carry : d.carries()) { var bits = new BitSet(); bits.set(root++); availability[d.formals().get(carry.numericCarryPosition())] = memberships.size(); memberships.add(bits); }
        for (int seed : d.seeds()) { var bits = new BitSet(); bits.set(root++); availability[d.formals().get(seed)] = memberships.size(); memberships.add(bits); }
        for (int u = 0; u < d.units().size(); u++) {
            var unit = d.units().get(u); int[] inputs = unit.node().inputs();
            for (int i = 0; i < inputs.length; i++) {
                int required = unit.leftScope() < 0 ? unit.scope() : i == 0 ? unit.leftScope() : unit.rightScope();
                int source = availability[inputs[i]];
                if (source >= 0) {
                    if (required < 0) throw malformed();
                    var missing = (BitSet) memberships.get(required).clone(); missing.andNot(memberships.get(source));
                    if (!missing.isEmpty()) throw malformed();
                }
            }
            if (unit.leftScope() >= 0) {
                var union = (BitSet) memberships.get(unit.leftScope()).clone(); union.or(memberships.get(unit.rightScope()));
                if (!union.equals(memberships.get(unit.scope())) || !sameValueShapeType(d.values().get(inputs[0]), d.values().get(inputs[1]))
                        || !sameValueShapeType(d.values().get(inputs[0]), d.values().get(unit.node().outputIndex()))) throw malformed();
            }
            for (int output : unit.node().outputs()) { availability[output] = unit.scope(); producer[output] = u; }
        }
        for (var carry : d.carries()) {
            int next = d.roles().get(carry.numericCarryPosition()), source = availability[next];
            BitSet expected = source < 0 ? new BitSet() : memberships.get(source);
            if (!expected.equals(memberships.get(carry.nextScope()))) throw malformed();
        }
        Set<Integer> accumulatorUnits = new HashSet<>();
        for (var accumulator : d.accumulators()) {
            int position = accumulator.numericSharedCarryPosition(), u = producer[d.roles().get(position)];
            if (u < 0 || !accumulatorUnits.add(u)) throw malformed();
            var unit = d.units().get(u); int[] inputs = unit.node().inputs();
            if (unit.node().kind() != MetalMpsGraphProgram.NodeKind.ADD || inputs.length != 2 || unit.node().outputs().length != 1
                    || inputs[0] != d.formals().get(position) || unit.leftScope() >= 0 || unit.accumulator() != position
                    || !memberships.get(unit.scope()).equals(memberships.get(accumulator.contributionScope()))) throw malformed();
        }
        for (int u = 0; u < d.units().size(); u++) if ((d.units().get(u).accumulator() >= 0) != accumulatorUnits.contains(u)) throw malformed();
    }

    /**
     * Validates the exact containing occurrence against independently decoded local metadata.
     * @param node raw Scan instruction with demanded-only output references
     * @param outer complete outer physical descriptor table, with original gradient flags
     * @throws IllegalArgumentException if an actual, demanded result or ordinary body fact disagrees
     */
    static void validateOuter(MetalMpsGraphProgram.Node node, List<MetalMpsGraphProgram.ValueDescriptor> outer) {
        Decoded d = decode(node.attributeWords()); long[] w = d.words();
        int[] inputs = node.inputs(), outputs = node.outputs();
        if (inputs.length != w[14] + ((w[4] & 2) != 0 ? 1 : 0) || outputs.length != w[16]) throw malformed();
        boolean gradient = false;
        for (int f = 0; f < d.formals().size(); f++) {
            var actual = outer.get(inputs[f]); var formal = d.values().get(d.formals().get(f)).descriptor();
            long[] expected = formal.shape().toLongArray();
            if (f >= w[5] && f < w[5] + w[6]) expected = prepend(w[3], expected);
            if (actual.dataType() != formal.dataType() || actual.requiresGrad() != formal.requiresGrad()
                    || !Arrays.equals(actual.dimensions(), expected)
                    || !denseInput(actual) && !mappedInput(actual)) throw malformed();
            gradient |= actual.requiresGrad();
            if ((w[4] & 2) != 0) for (int earlier = 0; earlier < f; earlier++) if (inputs[earlier] == inputs[f]) {
                int axis = d.values().get(d.formals().get(f)).rowAxis();
                int previousAxis = d.values().get(d.formals().get(earlier)).rowAxis();
                if (axis >= 0 && f >= w[5] && f < w[5] + w[6]) axis++;
                if (previousAxis >= 0 && earlier >= w[5] && earlier < w[5] + w[6]) previousAxis++;
                if (axis != previousAxis) throw malformed();
            }
        }
        if ((w[4] & 2) != 0) {
            var lengths = outer.get(inputs[inputs.length - 1]);
            if (lengths.dataType() != DataType.INT64 || lengths.requiresGrad() || !Arrays.equals(lengths.dimensions(), new long[] {w[11]}) || !denseInput(lengths)) throw malformed();
        }
        for (int demand = 0; demand < d.demands().size(); demand++) {
            int p = d.demands().get(demand);
            var result = outer.get(outputs[demand]); var local = d.values().get(d.roles().get(p)).descriptor();
            long[] expected = local.shape().toLongArray(); if (p >= w[5]) expected = prepend(w[3], expected);
            if (result.dataType() != local.dataType() || !Arrays.equals(result.dimensions(), expected)
                    || result.requiresGrad() != (gradient && local.dataType().isDifferentiable()) || !canonical(result)) throw malformed();
        }
    }

    private static long[] prepend(long time, long[] shape) { long[] result = new long[shape.length + 1]; result[0] = time; System.arraycopy(shape, 0, result, 1, shape.length); return result; }
    /**
     * A produced affine view has canonical represented storage, authenticated by its ordinary
     * producer, despite retaining its exact logical view layout. External non-dense storage
     * cannot claim this flag; complete-program validation authenticates every source separately.
     */
    private static boolean denseInput(MetalMpsGraphProgram.ValueDescriptor d) {
        return canonical(d) || d.densePhysical() && d.layout().isPresent();
    }

    /**
     * External numeric layouts use the ordinary mapped CONTIGUOUS owner before dense binding.
     * Produced dense representations are never walked again using their logical view strides.
     * Typed controls, including lengths, retain the existing dense-only ingress contract.
     */
    static boolean mappedInput(MetalMpsGraphProgram.ValueDescriptor d) {
        if (!MetalScanLowering.numericCarrier(d.dataType()) || d.densePhysical()
                || canonical(d) || d.layout().isEmpty()) return false;
        return d.supportsStorageLayout(0) || d.elementCount() == 0 && d.layout()
                .filter(layout -> layout.isView()
                        && Arrays.stream(layout.strides()).allMatch(stride -> stride >= 0)).isPresent();
    }

    private static boolean canonical(MetalMpsGraphProgram.ValueDescriptor d) { return d.layout().filter(l -> l.equals(LayoutDescriptor.contiguous(Shape.of(d.dimensions())))).isPresent() && !d.densePhysical(); }
    private static boolean sameValueShapeType(MetalScanPlan.Value a, MetalScanPlan.Value b) { return a.descriptor().dataType() == b.descriptor().dataType() && a.descriptor().shape().equals(b.descriptor().shape()); }
    private static List<Integer> indices(long[] w, long offset, long count, int bound) {
        if (offset < 0 || count < 0 || offset > w.length || count > w.length - offset) throw malformed();
        var result = new ArrayList<Integer>(); for (long i = 0; i < count; i++) result.add(bounded(w[Math.toIntExact(offset + i)], bound - 1)); return List.copyOf(result);
    }
    private static int scope(long value, long count) { if (value == -1) return -1; if (value < 0 || value >= count) throw malformed(); return Math.toIntExact(value); }
    private static int bounded(long value, int maximum) { if (value < 0 || value > maximum) throw malformed(); return (int) value; }
    private static DataType carrier(long wire) {
        if (wire < 0 || wire > 7) throw malformed();
        return switch ((int) wire) {
            case 1 -> DataType.FLOAT32;
            case 2 -> DataType.INT32;
            case 3 -> DataType.BOOL;
            case 4 -> DataType.FLOAT64;
            case 5 -> DataType.BFLOAT16;
            case 6 -> DataType.INT64;
            case 7 -> DataType.FLOAT16;
            default -> throw malformed();
        };
    }
    private static MetalMpsGraphProgram.NodeKind kind(long wire) { if (wire >= 101 && wire <= 105) throw malformed(); for (var kind : MetalMpsGraphProgram.NodeKind.values()) if (kind.wireIdentity() == wire && wire >= 1 && wire <= 115) return kind; throw malformed(); }
    private static MetalMpsGraphProgram.AttributeKind attributeKind(long wire) { for (var kind : MetalMpsGraphProgram.AttributeKind.values()) if (kind.wireIdentity() == wire && wire != 42) return kind; throw malformed(); }

    /** Applies the same Model-pure ordinary kind boundary to direct physical-unit validation. */
    static void validateOrdinaryKind(MetalMpsGraphProgram.NodeKind candidate) { kind(candidate.wireIdentity()); }
    private static long align(long bytes) { return Math.addExact(bytes, 7) & ~7L; }
    private static IllegalArgumentException malformed() { return new IllegalArgumentException("Malformed Metal Scan body/binder image"); }

    /** Immutable cold structural metadata; no runtime graph or executable is retained. */
    record Decoded(long[] words, List<MetalScanPlan.Value> values, List<MetalScanPlan.Unit> units,
            List<Integer> formals, List<Integer> roles, List<Integer> demands,
            List<ScanActivityPlan.Carry> carries, List<Integer> seeds, List<List<Integer>> scopes,
            List<ScanActivityPlan.Accumulator> accumulators) {}
}
