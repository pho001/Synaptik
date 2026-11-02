package io.github.pho001.synaptik.backend.cpu.internal.codegen.emit;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuGeneralScanIr;
import java.util.*;

/**
 * Generation-only proof for delaying bank-free row-carry writes behind independent consumers.
 * The candidate groups row results, shared results, then row-carry results, stably within each
 * group. It is accepted only if producing dependencies, physical final-port hazards and the
 * already assigned inclusive scratch lifetimes survive. Mixed region stage values resolve
 * pointwise, contraction and represented-SUM operands explicitly; a region is indivisible,
 * and its distinct shared finish ADDs stay after its complete row reduction. The proof never
 * changes the IR or resources.
 */
final class CpuGeneralScanStageSchedule {
    /** No runtime state is created by this proof owner. */
    private CpuGeneralScanStageSchedule() {}

    /**
     * Selects the reference-like order only when all existing bindings prove it safe.
     * Bank transitions and unproved dependencies/overlaps retain the original complete order.
     * @param ir non-null combined Scan with exact stage and assigned scratch recipes
     * @return immutable proved candidate or the unchanged declared stage list
     */
    static List<CpuGeneralScanIr.Stage> order(CpuGeneralScanIr ir) {
        var plan = ir.stages().orElseThrow(); var original = plan.order();
        if (!ir.symbolic() || !ir.lengthAware() || plan.carryOffsets().stream().anyMatch(n -> n >= 0))
            return original;
        var definitions = new ArrayList<Set<Integer>>(); var reads = new ArrayList<Set<Integer>>();
        var instructions = new HashMap<Integer,List<Integer>>();
        ir.arithmetic().instructions().forEach(i -> instructions.put(i.output(),i.inputs()));
        plan.rowMapReductionIr().ifPresent(r -> {
            r.contractions().forEach(c -> instructions.put(c.output(),List.of(c.left(),c.right())));
            r.sums().forEach(s -> instructions.put(s.output(),List.of(s.input())));
        });
        for (var stage : original) {
            var defined = new HashSet<>(stage.values()); defined.addAll(stage.epilogue());
            if (stage.output() >= 0) defined.add(stage.output());
            var consumed = new HashSet<Integer>();
            if (stage.input() >= 0) consumed.add(stage.input());
            if (stage.right() >= 0) consumed.add(stage.right());
            for (int out : stage.values()) consumed.addAll(instructions.get(out));
            for (int out : stage.epilogue()) consumed.addAll(instructions.get(out));
            consumed.removeAll(defined);
            var origins = new HashSet<Integer>();
            for (int in : consumed) origins.add(plan.views().get(in).origin());
            definitions.add(defined); reads.add(origins);
        }
        var candidate = new ArrayList<Integer>();
        for (int i = 0; i < original.size(); i++) candidate.add(i);
        candidate.sort(Comparator.comparingInt(i -> {
            for (int c = 0; c < plan.rowCarryCount(); c++)
                if (definitions.get(i).contains(ir.nextOrdinals().get(c))) return 2;
            int domain = original.get(i).output() >= 0 ? original.get(i).output()
                    : original.get(i).values().getFirst();
            return plan.rowAxes().get(domain) >= 0 ? 0 : 1;
        }));
        boolean changed = false;
        for (int i = 0; i < candidate.size(); i++) if (candidate.get(i) != i) { changed = true; break; }
        if (!changed) return original;
        int[] position = new int[original.size()+1]; position[original.size()] = original.size();
        for (int i = 0; i < candidate.size(); i++) position[candidate.get(i)] = i;
        var readPorts = reads.stream().map(r -> ports(ir,r,false)).toList();
        var writePorts = definitions.stream().map(w -> ports(ir,w,true)).toList();
        for (int a = 0; a < original.size(); a++) for (int b = 0; b < original.size(); b++) {
            if (a == b) continue;
            if (!Collections.disjoint(definitions.get(a),reads.get(b)) && position[a] >= position[b])
                return original;
            if (a < b && position[a] > position[b]) {
                var aw = writePorts.get(a); var bw = writePorts.get(b);
                if (!Collections.disjoint(aw,readPorts.get(b))
                        || !Collections.disjoint(readPorts.get(a),bw)
                        || !Collections.disjoint(aw,bw)) return original;
            }
        }
        var scratch = plan.cuts().stream().filter(c -> c.port() < 0 && c.carry() < 0).toList();
        for (int a = 0; a < scratch.size(); a++) for (int b = a+1; b < scratch.size(); b++) {
            var left = scratch.get(a); var right = scratch.get(b);
            try {
                if (before(left,right) || before(right,left)) continue;
            } catch (ArithmeticException overflow) {
                return original; // Broader cold IR probes need not inherit lowering's checked sums.
            }
            int[] l = lifetime(left,position), r = lifetime(right,position);
            if (l[0] <= r[1] && r[0] <= l[1]) return original;
        }
        return candidate.stream().map(original::get).toList();
    }

    /**
     * Maps reads to selected origins and writes to every physical destination emitted by
     * {@code Shaped.store}. A history-backed next-carry cut additionally writes its own stable
     * final port; reading that cut still consumes only the primary history, not that final port.
     * @param ir immutable combined binding facts
     * @param origins stage read or producing origins, never modified
     * @param write whether to include additional own-final-port publication destinations
     * @return generation-only final-port set; borrowed inputs have no writable port identity
     */
    private static Set<Integer> ports(CpuGeneralScanIr ir,Set<Integer> origins,boolean write) {
        var plan = ir.stages().orElseThrow(); var ports = new HashSet<Integer>();
        for (int c = 0; c < ir.carryCount(); c++)
            if (origins.contains(c) && plan.carryPorts().get(c) == c) ports.add(c);
        for (var cut : plan.cuts()) if (origins.contains(cut.value())) {
            if (cut.port() >= 0) ports.add(cut.port());
            if (write) for (int c = 0; c < ir.carryCount(); c++)
                if (plan.carryPorts().get(c) == c && ir.nextOrdinals().get(c) == cut.value()) ports.add(c);
        }
        return ports;
    }

    /**
     * Proves disjoint assigned byte intervals for every admitted nonnegative B coefficientwise.
     * Failure is conservative, never an inference from sampled geometry.
     * @param left scratch interval proposed before right
     * @param right scratch interval proposed after left
     * @return whether left's complete affine end never exceeds right's start
     * @throws ArithmeticException if an end coefficient overflows, requiring original-order fallback
     */
    private static boolean before(CpuGeneralScanIr.Cut left,CpuGeneralScanIr.Cut right) {
        return Math.addExact(left.offset(),left.bytes()) <= right.offset()
                && Math.addExact(left.offsetPerBatch(),left.bytesPerBatch()) <= right.offsetPerBatch();
    }

    /**
     * Conservatively retains every stage in the original inclusive cut lifetime, not just its
     * producing and final consuming endpoints. Final publication keeps the terminal position.
     * @param cut exact assigned scratch cut
     * @param position original-to-candidate index map including terminal publication
     * @return inclusive minimum and maximum candidate positions, in a fresh two-element array
     */
    private static int[] lifetime(CpuGeneralScanIr.Cut cut,int[] position) {
        int first = position[cut.birth()], last = first;
        for (int i = cut.birth(); i <= cut.death(); i++) {
            first = Math.min(first,position[i]); last = Math.max(last,position[i]);
        }
        return new int[]{first,last};
    }
}
