package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.*;

/** Cold, storage-free legality and structural-budget proof shared by ordinary and Scan callers. */
public final class CpuRowMapReductionLowering {
    private CpuRowMapReductionLowering() { }

    /**
     * Selects complete, actually consumed external feature views for scalar staging. The
     * binding owner authenticates the two read-only origin scopes over the entire lifetime;
     * this query proves their spatial mapping and consumer coverage without reading storage.
     * Results are in first-consumer order. Equivalent affine domains of the same storage
     * origin appear once; partially overlapping domains conservatively remain checked.
     * Carries, scratch and bank bindings must not be supplied as authenticated sources.
     *
     * <p>The existing owners use conservative phase ceilings of 112 caller and 116 helper,
     * plus three slots for each additional demanded row-wrapper binding. The
     * caller also includes two slots for every demanded history, including forwarded formals:
     * their bases can remain live across publication and the complete staged lifetime.
     * non-pool slots. The structural floor below includes the existing typed producer and
     * terminal cells, carrier/base bindings, control and address state, and symbolic stride
     * facts. Staged cells, one readiness flag and the remaining CPU0075 pool share the
     * unchanged 128-slot envelope. A failed optimization never changes region admission.</p>
     *
     * @param candidate nonnull, already legality-proved immutable regional inventory
     * @param invocationInvariantOrigins nonnull disjoint read-only origins stable across time
     * @param activeTimeOrigins nonnull read-only origins bound at the consuming original time
     * @param callerBaseLiveSlots conservative non-pool caller slots, never allocated local IDs
     * @param helperBaseLiveSlots conservative non-pool helper slots; zero means no helper
     * @return immutable ordered actual consumer-view ordinals, or empty if unproved/over budget
     * @throws NullPointerException if a required reference or scope element is null
     * @throws IllegalArgumentException if scopes overlap, contain duplicates/nonroots or a
     *     negative phase bound is supplied
     */
    public static List<Integer> invariantLoadStaging(CpuRowMapReductionIr candidate,
            List<Integer> invocationInvariantOrigins, List<Integer> activeTimeOrigins,
            long callerBaseLiveSlots, long helperBaseLiveSlots) {
        Objects.requireNonNull(candidate,"candidate");
        var invariant = List.copyOf(invocationInvariantOrigins);
        var temporal = List.copyOf(activeTimeOrigins);
        int n = candidate.arithmetic().values().size();
        var allowed = new HashSet<Integer>();
        for (var scope : List.of(invariant,temporal)) for (int root : scope)
            if (root < 0 || root >= n || candidate.viewOrigins().get(root) != root || !allowed.add(root))
                throw new IllegalArgumentException("staging scopes require disjoint unique storage origins");
        if (callerBaseLiveSlots < 0 || helperBaseLiveSlots < 0)
            throw new IllegalArgumentException("negative staging phase bound");
        if (allowed.isEmpty()) return List.of();
        try {
            if (candidate.rowSites().isEmpty()) return List.of();
            int anchor = candidate.rowSites().getFirst(), rowAxis = candidate.rowAxes().get(anchor);
            if (rowAxis < 0) return List.of();
            var rowDomain = candidate.accesses().get(anchor);
            if (rowDomain.extentConstants().get(rowAxis) == 0 && rowDomain.extentBatchFactors().get(rowAxis) == 0)
                return List.of();
            // The existing range phase proof is rank-two. Wider copy/control nests keep
            // checked emission rather than extrapolating that proof to additional indices.
            if (candidate.accesses().stream().anyMatch(a -> a.plan().iterationRank() > 2)) return List.of();
            if (!views(candidate)) return List.of();
            int w = candidate.arithmetic().values().getFirst().dataType() == DataType.FLOAT64 ? 2 : 1;
            long cells = 0;
            for (int v : candidate.rowSites()) cells = Math.addExact(cells,cells(candidate,v,true));
            for (int v : candidate.terminalOutputs()) cells = Math.addExact(cells,cells(candidate,v,false));
            long boundaries = candidate.arithmetic().values().stream().filter(v -> v.kind()
                    != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.VIRTUAL).count();
            var strides = new HashSet<List<Long>>();
            for (var a : candidate.accesses()) for (int axis = 0; axis < a.plan().iterationRank(); axis++)
                if (a.strideBatchFactors().get(axis) != 0)
                    strides.add(List.of(a.strideConstants().get(axis),a.strideBatchFactors().get(axis)));
            long floor = Math.addExact(23+3*boundaries+2L*strides.size(),Math.multiplyExact(cells+6,w));
            var requests = stagingRequests(candidate);
            var readRoots = new HashSet<Integer>(); var effectiveStrides = new HashSet<List<Long>>();
            for (var request : requests) {
                int v = Math.toIntExact(request.getFirst()), root = candidate.viewOrigins().get(v); readRoots.add(root);
                var a = candidate.accesses().get(v);
                // An evolving scalar carry can switch even a constant nominal stride. Count
                // every distinct rooted recipe, not just nonzero B coefficients. This is a
                // conservative superset of the existing helper's initial/nominal fact pool.
                for (int axis = 0; axis < a.plan().iterationRank(); axis++) effectiveStrides.add(
                        List.of((long)root,a.strideConstants().get(axis),a.strideBatchFactors().get(axis)));
            }
            long writes = candidate.rowSites().stream().filter(candidate.retainedStores()::contains).count();
            // Existing helper's 24-slot header, producing destinations and rooted bindings,
            // plus the two-slot CPU0075 offset. Duplicate wrappers are supplied by the binding
            // owner in BOTH phase bases, before cuts, not guessed from the region's value count.
            long helperFloor = 26+3L*readRoots.size()+3*writes+2L*effectiveStrides.size()+cells*w+3L*w;
            floor = Math.max(floor,helperFloor-4);
            // Wrapper allowance cannot disguise an excessive ordinary binding floor: prove
            // the no-duplicate phase independently, then add the caller-authenticated wrapper
            // slots to the full staging/pool bound. Larger structural domains remain checked.
            if (floor > Math.min(112,callerBaseLiveSlots) || helperBaseLiveSlots != 0
                    && Math.max(floor+4,helperFloor) > Math.min(116,helperBaseLiveSlots)) return List.of();
            var requested = new LinkedHashMap<Integer,Set<List<Long>>>();
            for (var request : requests) {
                int v = Math.toIntExact(request.getFirst());
                requested.computeIfAbsent(v,k -> new LinkedHashSet<>()).add(request.subList(1,request.size()));
            }
            var selected = new ArrayList<Integer>(); var occupied = new HashSet<List<Long>>();
            long live = 1; // One helper first-active-consumer flag; conservative for inline emission.
            for (var entry : requested.entrySet()) {
                int v = entry.getKey(), root = candidate.viewOrigins().get(v);
                if (!allowed.contains(root) || candidate.rowAxes().get(v) >= 0
                        || candidate.rowSites().contains(root) || candidate.terminalOutputs().contains(root)
                        || candidate.retainedStores().contains(root)) continue;
                var a = candidate.accesses().get(v);
                if (a.extentBatchFactors().stream().anyMatch(f -> f != 0)) continue;
                int count = cells(candidate,v,false);
                if (count == 0) continue;
                var domain = new LinkedHashSet<List<Long>>();
                for (int cell = 0; cell < count; cell++) domain.add(stagingCell(candidate,v,stagingCoordinates(candidate,v,cell,false)));
                if (domain.stream().anyMatch(k -> k.get(3) != 0 || k.get(4) != 0)
                        || !entry.getValue().containsAll(domain)) continue;
                // Unknown or overlapping producing storage ends the full-view lifetime. The
                // caller excludes physical input/output overlap before invoking generated code.
                if (candidate.retainedStores().stream().anyMatch(out -> candidate.viewOrigins().get(out) == root)) continue;
                if (occupied.containsAll(domain)) continue;
                if (domain.stream().anyMatch(occupied::contains)) continue;
                long next = Math.addExact(live,Math.multiplyExact(domain.size(),w));
                if (Math.addExact(callerBaseLiveSlots,next) > 128
                        || helperBaseLiveSlots != 0 && Math.addExact(helperBaseLiveSlots,next) > 128) continue;
                selected.add(v); occupied.addAll(domain); live = next;
            }
            return List.copyOf(selected);
        } catch (ArithmeticException | IllegalArgumentException unproved) { return List.of(); }
    }

    /**
     * Enumerates actual external scalar requests, including represented SUM contributors.
     * @param ir complete regional inventory
     * @return view ordinal followed by its root/constant/B/row/row-B affine key per request
     */
    private static List<List<Long>> stagingRequests(CpuRowMapReductionIr ir) {
        var result = new ArrayList<List<Long>>();
        var instructions = new HashMap<Integer,io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Instruction>();
        ir.arithmetic().instructions().forEach(i -> instructions.put(i.output(),i));
        var products = new HashMap<Integer,CpuRowMapReductionIr.Contraction>();
        ir.contractions().forEach(c -> products.put(c.output(),c));
        for (int out : ir.rowSites()) for (int cell = 0; cell < cells(ir,out,true); cell++) {
            int[] logical = stagingCoordinates(ir,out,cell,true); var product = products.get(out);
            if (product != null) {
                int k = Math.toIntExact(ir.accesses().get(product.left()).extentConstants().get(1));
                for (int f = 0; f < k; f++) {
                    stagingRequest(ir,result,product.left(),new int[]{-1,f});
                    stagingRequest(ir,result,product.right(),new int[]{f,logical[1]});
                }
            } else for (int in : new LinkedHashSet<>(instructions.get(out).inputs())) stagingRequest(ir,result,in,logical);
        }
        for (int out : ir.terminalOutputs()) for (int cell = 0; cell < cells(ir,out,false); cell++) {
            var product = products.get(out);
            if (product != null) {
                int[] logical = stagingCoordinates(ir,out,cell,false);
                stagingRequest(ir,result,product.left(),new int[]{logical[0],-1});
                stagingRequest(ir,result,product.right(),new int[]{-1,logical[1]});
            } else for (var sum : ir.sums()) if (sum.output() == out) {
                int domain = cells(ir,sum.input(),true);
                for (int f = 0; f < domain; f++) stagingRequest(ir,result,sum.input(),stagingCoordinates(ir,sum.input(),f,true));
            }
        }
        return result;
    }

    /**
     * Records only external storage requests, never completed row producers.
     * @param ir complete region
     * @param result generation-only request inventory to append to
     * @param v actual operand view
     * @param logical consuming coordinates, with minus one denoting the row
     */
    private static void stagingRequest(CpuRowMapReductionIr ir,List<List<Long>> result,int v,int[] logical) {
        if (ir.rowSites().contains(ir.viewOrigins().get(v))) return;
        var request = new ArrayList<Long>(); request.add((long)v); request.addAll(stagingCell(ir,v,logical));
        result.add(List.copyOf(request));
    }

    /**
     * Normalizes an actual request coefficientwise for every admitted B, without sampled sizes.
     * @param ir authenticated access inventory
     * @param v actual operand/view ordinal
     * @param logical right-aligned consuming coordinates; minus one is the original row
     * @return immutable origin, constant, B, row and row-B coefficients
     * @throws ArithmeticException if checked coefficient arithmetic overflows
     */
    private static List<Long> stagingCell(CpuRowMapReductionIr ir,int v,int[] logical) {
        var a = ir.accesses().get(v); long c = a.baseElementOffset(), b = 0, r = 0, rb = 0;
        int shift = logical.length-a.plan().iterationRank();
        for (int axis = 0; axis < a.plan().iterationRank(); axis++) {
            int x = a.extentConstants().get(axis) == 1 && a.extentBatchFactors().get(axis) == 0 ? 0 : logical[shift+axis];
            long sc = a.strideConstants().get(axis), sf = a.strideBatchFactors().get(axis);
            if (x < 0) { r = Math.addExact(r,sc); rb = Math.addExact(rb,sf); }
            else { c = Math.addExact(c,Math.multiplyExact(x,sc)); b = Math.addExact(b,Math.multiplyExact(x,sf)); }
        }
        return List.of((long)ir.viewOrigins().get(v),c,b,r,rb);
    }

    /**
     * Decodes one fixed feature cell, preserving the authenticated row axis symbolically.
     * @param ir immutable access inventory
     * @param v value ordinal
     * @param cell fixed feature-cell index
     * @param row whether to omit the row from the fixed mixed-radix domain
     * @return new logical coordinate array with minus one at the omitted row
     */
    private static int[] stagingCoordinates(CpuRowMapReductionIr ir,int v,int cell,boolean row) {
        var a = ir.accesses().get(v); int[] result = new int[a.plan().iterationRank()];
        for (int axis = result.length-1; axis >= 0; axis--) {
            if (row && axis == ir.rowAxes().get(v)) { result[axis] = -1; continue; }
            int extent = Math.toIntExact(a.extentConstants().get(axis)); result[axis] = cell%extent; cell /= extent;
        }
        return result;
    }

    /**
     * Proves one complete candidate before cuts, resource declarations or assigned slots exist.
     * Callers supply authenticated Dimension-derived roles, composed access recipes, complete
     * uses (including views and external demand), and conservative temporal/physical order edges.
     * Outside uses of row values require retained exact stores. A path leaving the region and
     * reentering it rejects even when original ordinals are adjacent. Terminal feedback rejects.
     * A real computed row producer must reach a terminal and another retained/terminal sink;
     * independent operations sharing only a direct input or movement are not a row-map group.
     * Unsupported or over-budget candidates return empty without changing the fallback graph.
     * Actual backing overlap must still be checked by the caller's invocation binder.
     * @param candidate nonnull immutable candidate with total access/origin/role coverage
     * @param allUses nonnull complete value-to-consumer adjacency; candidate ordinals first,
     *     followed by any outside full-DAG ordinals
     * @param requiredOrder nonnull must-precede adjacency in the identical full inventory
     * @return the identical candidate iff every proof succeeds; otherwise empty, never null
     * @throws NullPointerException if an argument, adjacency list or element is null
     * @throws IllegalArgumentException if adjacency coverage or an ordinal is malformed
     */
    public static Optional<CpuRowMapReductionIr> prove(CpuRowMapReductionIr candidate,
            List<List<Integer>> allUses, List<List<Integer>> requiredOrder) {
        Objects.requireNonNull(candidate, "candidate");
        int valueCount = candidate.arithmetic().values().size();
        int n = Objects.requireNonNull(allUses).size();
        if (n < valueCount) throw new IllegalArgumentException("full uses omit candidate values");
        var uses = checked(allUses, n); var order = checked(requiredOrder, n);
        var group = new HashSet<>(candidate.rowSites()); group.addAll(candidate.terminalOutputs());
        for (int v = 0; v < valueCount; v++) if (group.contains(candidate.viewOrigins().get(v))) group.add(v);
        var terminals = new HashSet<>(candidate.terminalOutputs());
        var emissionPosition = new HashMap<Integer,Integer>();
        int position = 0;
        for (int v : candidate.rowSites()) emissionPosition.put(v,position++);
        for (int v : candidate.terminalOutputs()) emissionPosition.put(v,position++);
        for (int from = 0; from < n; from++) for (int target : order.get(from)) {
            if (from >= valueCount || target >= valueCount) continue;
            int a = candidate.viewOrigins().get(from), b = candidate.viewOrigins().get(target);
            if (a != b && emissionPosition.containsKey(a) && emissionPosition.containsKey(b)
                    && emissionPosition.get(a) >= emissionPosition.get(b)) return Optional.empty();
        }
        if (candidate.rowSites().isEmpty() || terminals.isEmpty() || candidate.rowSites().size() > 32
                || terminals.size() > 4 || candidate.arithmetic().values().stream()
                        .filter(v -> v.kind() != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.VIRTUAL)
                        .count() > 32) return Optional.empty();
        DataType type = candidate.arithmetic().values().getFirst().dataType();
        if (type != DataType.FLOAT32 && type != DataType.FLOAT64
                || candidate.arithmetic().values().stream().anyMatch(v -> v.dataType() != type)) return Optional.empty();
        var operands = new HashMap<Integer,List<Integer>>();
        candidate.arithmetic().instructions().forEach(i -> operands.put(i.output(),i.inputs()));
        candidate.contractions().forEach(c -> operands.put(c.output(),List.of(c.left(),c.right())));
        candidate.sums().forEach(s -> operands.put(s.output(),List.of(s.input())));
        for (int v = 0; v < valueCount; v++) {
            int root = candidate.viewOrigins().get(v);
            if (root != v && !uses.get(root).contains(v)) return Optional.empty();
        }
        for (var entry : operands.entrySet()) for (int input : entry.getValue())
            if (!uses.get(input).contains(entry.getKey())) return Optional.empty();
        var full = new ArrayList<Set<Integer>>(n);
        int[] indegree = new int[n];
        for (int v = 0; v < n; v++) {
            var edges = new HashSet<>(uses.get(v)); edges.addAll(order.get(v)); full.add(edges);
            for (int target : edges) indegree[target]++;
        }
        var pending = new ArrayDeque<Integer>();
        for (int v = 0; v < n; v++) if (indegree[v] == 0) pending.add(v);
        int visited = 0;
        while (!pending.isEmpty()) { int v = pending.removeFirst(); visited++;
            for (int target : full.get(v)) if (--indegree[target] == 0) pending.add(target); }
        if (visited != n) return Optional.empty();
        if (!sharedProducer(candidate,uses,group,terminals)) return Optional.empty();
        // Complete leave/reentry proof, including required physical/temporal edges.
        for (int v : group) for (int target : full.get(v)) if (!group.contains(target)) {
            var outside = new ArrayDeque<Integer>(); var seen = new HashSet<Integer>(); outside.add(target);
            while (!outside.isEmpty()) {
                int current = outside.removeLast();
                if (group.contains(current)) return Optional.empty();
                if (seen.add(current)) outside.addAll(full.get(current));
            }
            if (!terminals.contains(candidate.viewOrigins().get(v))
                    && !candidate.retainedStores().contains(candidate.viewOrigins().get(v))) return Optional.empty();
        }
        var available = new HashSet<Integer>();
        for (int v = 0; v < valueCount; v++) if (!group.contains(v)) available.add(v);
        long rowSlots = 2, terminalSlots = 0, terminalCells = 0, bytes = 1024;
        int referenceRow = candidate.rowSites().getFirst();
        int referenceAxis = candidate.rowAxes().get(referenceRow);
        if (referenceAxis < 0) return Optional.empty();
        var reference = candidate.accesses().get(referenceRow);
        long rc = reference.extentConstants().get(referenceAxis), rf = reference.extentBatchFactors().get(referenceAxis);
        if (rc > Integer.MAX_VALUE || rf > 1 || rf != 0 && rc != 0) return Optional.empty();
        try {
            if (!views(candidate)) return Optional.empty();
            for (int v : candidate.rowSites()) {
                int axis = candidate.rowAxes().get(v); var access = candidate.accesses().get(v);
                if (axis < 0 || access.extentConstants().get(axis) != rc
                        || access.extentBatchFactors().get(axis) != rf) return Optional.empty();
                for (int in : operands.get(v)) if (!available.contains(candidate.viewOrigins().get(in))
                        || terminals.contains(candidate.viewOrigins().get(in))) return Optional.empty();
                int cells = cells(candidate,v,true);
                rowSlots = Math.addExact(rowSlots,(long)cells*(type == DataType.FLOAT64 ? 2 : 1));
                bytes += 1024L*cells;
                available.add(v);
            }
            for (var instruction : candidate.arithmetic().instructions()) if (candidate.rowSites().contains(instruction.output())) {
                var out = candidate.accesses().get(instruction.output()); int outRow = candidate.rowAxes().get(instruction.output());
                for (int in : instruction.inputs()) {
                    var access = candidate.accesses().get(in); int shift = out.plan().iterationRank()-access.plan().iterationRank();
                    if (shift < 0) return Optional.empty();
                    int inRow = candidate.rowAxes().get(in);
                    if (inRow >= 0 && inRow+shift != outRow) return Optional.empty();
                    for (int a = 0; a < access.plan().iterationRank(); a++) {
                        boolean singleton = access.extentConstants().get(a) == 1 && access.extentBatchFactors().get(a) == 0;
                        if (!singleton && !sameExtent(access,a,out,a+shift)) return Optional.empty();
                        if (a+shift == outRow && inRow < 0 && !singleton
                                && (access.strideConstants().get(a) != 0 || access.strideBatchFactors().get(a) != 0)) return Optional.empty();
                    }
                }
            }
            for (int v : candidate.terminalOutputs()) {
                if (candidate.rowAxes().get(v) != -1) return Optional.empty();
                for (int in : operands.get(v)) if (!available.contains(candidate.viewOrigins().get(in))
                        || terminals.contains(candidate.viewOrigins().get(in))) return Optional.empty();
                int cells = cells(candidate,v,false); terminalCells += cells;
                terminalSlots += (long)cells*(type == DataType.FLOAT64 ? 2 : 1); bytes += 512L*cells;
            }
            for (var c : candidate.contractions()) {
                var left = candidate.accesses().get(c.left()); var right = candidate.accesses().get(c.right());
                var out = candidate.accesses().get(c.output());
                if (left.plan().iterationRank() != 2 || right.plan().iterationRank() != 2
                        || out.plan().iterationRank() != 2 || !sameExtent(left,1,right,0)
                        || !sameExtent(left,0,out,0) || !sameExtent(right,1,out,1)) return Optional.empty();
                if (terminals.contains(c.output())) {
                    if (candidate.rowAxes().get(c.left()) != 1 || candidate.rowAxes().get(c.right()) != 0
                            || left.extentConstants().get(1) != rc || left.extentBatchFactors().get(1) != rf)
                        return Optional.empty();
                } else {
                    if (candidate.rowAxes().get(c.output()) != 0 || candidate.rowAxes().get(c.left()) != 0 || candidate.rowAxes().get(c.right()) != -1
                            || left.extentBatchFactors().get(1) != 0 || left.extentConstants().get(1) > 16)
                        return Optional.empty();
                    bytes += 512L*cells(candidate,c.output(),true)*left.extentConstants().get(1);
                }
            }
            for (var sum : candidate.sums()) {
                int row = candidate.rowAxes().get(sum.input());
                if (!terminals.contains(sum.output()) || !sum.axes().contains(row)) return Optional.empty();
                var input = candidate.accesses().get(sum.input()); int expected = 1;
                for (int a = 0; a < input.plan().iterationRank(); a++) {
                    if (!sum.axes().contains(a)) expected = Math.multiplyExact(expected,Math.toIntExact(input.extentConstants().get(a)));
                    // Common increasing-R traversal must not transpose the represented contributor order.
                    else if (a < row && input.extentConstants().get(a) != 1) return Optional.empty();
                }
                if (expected != cells(candidate,sum.output(),false)) return Optional.empty();
            }
        } catch (ArithmeticException | IllegalArgumentException rejected) { return Optional.empty(); }
        long boundaries = candidate.arithmetic().values().stream().filter(v -> v.kind()
                != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.VIRTUAL).count();
        bytes += 256L*boundaries;
        long bindingSlots = 5+3*boundaries+6*(type == DataType.FLOAT64 ? 2 : 1);
        // Before cuts, reserve at most the remaining slots for the replacement's one LONG
        // affine offset and type-width-bounded last-use pool. Binders enforce this capacity,
        // not a late route decision. No capacity means checked unshared loads. The existing
        // 1024/512-byte per-cell estimates cover the extra dup/store/load instructions.
        long baseSlots = rowSlots+terminalSlots+bindingSlots;
        long replacementSlots = baseSlots+Math.max(0,128-baseSlots);
        // Ordinary candidates expose their boundary stores here. Scan authenticates its
        // formal/time scopes in rowRegion, still before shapedResources declares any cut.
        if (!candidate.arithmetic().stores().isEmpty()) invariantLoadStaging(candidate,
                candidate.arithmetic().values().stream().filter(v -> v.kind()
                    == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.INPUT
                    && candidate.viewOrigins().get(v.ordinal()) == v.ordinal()).map(v -> v.ordinal()).toList(),
                List.of(),112,116);
        return replacementSlots <= 128 && terminalCells <= 64 && bytes <= 48000
                ? Optional.of(candidate) : Optional.empty();
    }

    /**
     * Distinguishes a shared computed map from unrelated operations or input movement.
     * @param ir immutable candidate
     * @param uses authenticated complete adjacency
     * @param group candidate producers and their aliases
     * @param terminals completed terminal sinks
     * @return whether a non-movement producer reaches two required sinks, including a terminal
     */
    private static boolean sharedProducer(CpuRowMapReductionIr ir,List<List<Integer>> uses,
            Set<Integer> group,Set<Integer> terminals) {
        var computed = new HashSet<Integer>();
        ir.arithmetic().instructions().stream().filter(i -> i.opcode()
                != io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode.CAST)
                .forEach(i -> computed.add(i.output()));
        ir.contractions().forEach(c -> computed.add(c.output()));
        for (int producer : ir.rowSites()) if (computed.contains(producer)) {
            var seen = new HashSet<Integer>(); var pending = new ArrayDeque<Integer>(); pending.add(producer);
            int sinks = 0; boolean terminal = false;
            while (!pending.isEmpty()) {
                int value = pending.removeFirst();
                if (!group.contains(value) || !seen.add(value)) continue;
                if (terminals.contains(value)) { terminal = true; sinks++; }
                else if (ir.retainedStores().contains(value)) sinks++;
                pending.addAll(uses.get(value));
            }
            if (terminal && sinks >= 2) return true;
        }
        return false;
    }

    /**
     * Proves total self-rooted origin composition and coefficientwise feature-span dominance.
     * Root writes must be canonical; read broadcast/transpose views cannot invent a row or
     * expose a future producing cell. No representative B participates in the proof.
     * @param ir complete immutable candidate
     * @return whether every view preserves its authenticated root and bounded feature mapping
     * @throws ArithmeticException if checked coefficient/span arithmetic overflows
     */
    private static boolean views(CpuRowMapReductionIr ir) {
        for (int v = 0; v < ir.accesses().size(); v++) {
            var access = ir.accesses().get(v); int root = ir.viewOrigins().get(v); var origin = ir.accesses().get(root);
            int row = ir.rowAxes().get(v), rootRow = ir.rowAxes().get(root);
            cells(ir,v,row >= 0);
            if (ir.arithmetic().values().get(v).dataType() != ir.arithmetic().values().get(root).dataType()
                    || access.baseElementOffset() != 0) return false;
            if (root == v) {
                if (!canonical(access) && !(ir.arithmetic().values().get(v).kind()
                        == io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr.Value.Kind.INPUT
                        && access.strideConstants().stream().allMatch(s -> s == 0)
                        && access.strideBatchFactors().stream().allMatch(s -> s == 0))) return false;
                continue;
            }
            if (rootRow >= 0) {
                if (row < 0 || !sameExtent(access,row,origin,rootRow)
                        || !access.strideConstants().get(row).equals(origin.strideConstants().get(rootRow))
                        || !access.strideBatchFactors().get(row).equals(origin.strideBatchFactors().get(rootRow))) return false;
            } else if (row >= 0 && (access.strideConstants().get(row) != 0 || access.strideBatchFactors().get(row) != 0)) return false;
            long unitC = 1, unitF = 0;
            if (rootRow == 1) {
                if (origin.plan().iterationRank() != 2) return false;
                unitC = origin.strideConstants().get(0); unitF = origin.strideBatchFactors().get(0);
            }
            long last = 0;
            for (int a = 0; a < access.plan().iterationRank(); a++) if (a != row) {
                if (access.extentBatchFactors().get(a) != 0) return false;
                long sc = access.strideConstants().get(a), sf = access.strideBatchFactors().get(a);
                long step;
                if (unitC != 0) {
                    if (sc%unitC != 0) return false; step = sc/unitC;
                    if (Math.multiplyExact(step,unitF) != sf) return false;
                } else if (unitF != 0) {
                    if (sf%unitF != 0 || sc != 0) return false; step = sf/unitF;
                } else { if (sc != 0 || sf != 0) return false; step = 0; }
                long extent = access.extentConstants().get(a);
                if (extent > 16) return false;
                if (extent > 0) last = Math.addExact(last,Math.multiplyExact(extent-1,step));
            }
            int capacity = cells(ir,root,rootRow >= 0);
            if (capacity != 0 && last >= capacity) return false;
        }
        return true;
    }

    /**
     * Checks canonical affine strides symbolically, including structurally empty axes.
     * @param access complete fixed/constant-plus-B access
     * @return whether every stride equals the exact trailing extent product
     * @throws ArithmeticException if coefficient multiplication or addition overflows
     */
    private static boolean canonical(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.BatchBinding access) {
        long sc = 1, sf = 0;
        for (int a = access.plan().iterationRank()-1; a >= 0; a--) {
            if (access.strideConstants().get(a) != sc || access.strideBatchFactors().get(a) != sf) return false;
            long ec = access.extentConstants().get(a), ef = access.extentBatchFactors().get(a);
            if (sf != 0 && ef != 0) return false;
            sf = Math.addExact(Math.multiplyExact(sf,ec),Math.multiplyExact(sc,ef)); sc = Math.multiplyExact(sc,ec);
        }
        return true;
    }

    /**
     * Enforces the per-axis feature ceiling before local or resource declaration.
     * @param ir candidate access inventory
     * @param v value ordinal
     * @param row whether to omit its authenticated row axis
     * @return fixed feature-cell count, including zero
     * @throws IllegalArgumentException if a feature is symbolic or exceeds sixteen
     * @throws ArithmeticException if the feature product overflows
     */
    private static int cells(CpuRowMapReductionIr ir,int v,boolean row) {
        var a = ir.accesses().get(v); int cells = 1;
        for (int axis = 0; axis < a.plan().iterationRank(); axis++) {
            if (row && axis == ir.rowAxes().get(v)) continue;
            long extent = a.extentConstants().get(axis);
            if (a.extentBatchFactors().get(axis) != 0 || extent > 16)
                throw new IllegalArgumentException("unbounded region feature");
            cells = Math.toIntExact(Math.multiplyExact(cells,extent));
        }
        return cells;
    }

    /**
     * Compares named-axis coefficient facts rather than resolved or coincident magnitudes.
     * @param a first access
     * @param x first axis
     * @param b second access
     * @param y second axis
     * @return whether both affine extent coefficients match
     */
    private static boolean sameExtent(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.BatchBinding a,
            int x, io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan.BatchBinding b,int y) {
        return a.extentConstants().get(x).equals(b.extentConstants().get(y))
                && a.extentBatchFactors().get(x).equals(b.extentBatchFactors().get(y));
    }

    /**
     * Snapshots one complete graph adjacency before traversing it.
     * @param input nonnull outer list and nonnull ordinal rows
     * @param n complete candidate-plus-outside inventory size
     * @return immutable adjacency snapshot
     * @throws IllegalArgumentException if coverage or any target ordinal is invalid
     * @throws NullPointerException if a list or element is null
     */
    private static List<List<Integer>> checked(List<List<Integer>> input,int n) {
        var result = Objects.requireNonNull(input).stream().map(List::copyOf).toList();
        if (result.size() != n || result.stream().flatMap(List::stream).anyMatch(v -> v < 0 || v >= n))
            throw new IllegalArgumentException("incomplete or invalid full-DAG proof inventory");
        return result;
    }
}
