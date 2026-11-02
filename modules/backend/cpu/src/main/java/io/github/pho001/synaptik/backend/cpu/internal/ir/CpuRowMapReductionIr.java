package io.github.pho001.synaptik.backend.cpu.internal.ir;

import java.util.List;
import java.util.Objects;

/**
 * Immutable generation-only region sharing ordered row-local sites among completed terminals.
 * Ordinals belong to {@link #arithmetic()}, not outer slots. Access recipes describe geometry
 * relative to the self-rooted producing storage in {@link #viewOrigins()}; row roles are proved
 * by the caller from Dimensions and view transformations, never inferred from equal extents.
 * Construction validates representation, not dependency convexity or physical legality: callers
 * must use the cold proof before declaring resources. Ordinary exact SUM is not represented here.
 */
public final class CpuRowMapReductionIr implements CpuPortableKernelIr {
    /**
     * Ordered represented MATMUL bindings, either a row-feature projection or a terminal.
     * @param output producing value ordinal
     * @param left ordered left operand ordinal
     * @param right ordered right operand ordinal
     */
    public record Contraction(int output, int left, int right) {
        /**
         * Validates nonnegative bindings; region construction checks their inventory bounds.
         * @param output nonnegative producing ordinal
         * @param left nonnegative left ordinal
         * @param right nonnegative right ordinal
         * @throws IllegalArgumentException if an ordinal is negative
         */
        public Contraction {
            if (output < 0 || left < 0 || right < 0) throw new IllegalArgumentException("negative contraction ordinal");
        }
    }

    /**
     * Represented Scan SUM binding; singleton/no-axis movement remains distinct from addition.
     * @param output producing ordinal
     * @param input operand ordinal
     * @param axes ordered normalized reduction axes, defensively copied
     * @param sumToShape whether singleton geometry uses raw movement
     */
    public record Sum(int output, int input, List<Integer> axes, boolean sumToShape) {
        /**
         * Snapshots ordered axes without deriving singleton geometry from activity counts.
         * @param output nonnegative producing ordinal
         * @param input nonnegative input ordinal
         * @param axes nonnull increasing nonnegative axes without duplicates
         * @param sumToShape exact sum-to-shape producing-site marker
         * @throws NullPointerException if axes or an element is null
         * @throws IllegalArgumentException if ordinals or axes are invalid
         */
        public Sum {
            axes = List.copyOf(axes);
            if (output < 0 || input < 0) throw new IllegalArgumentException("negative SUM ordinal");
            int previous = -1;
            for (int axis : axes) {
                if (axis <= previous) throw new IllegalArgumentException("SUM axes must increase");
                previous = axis;
            }
        }
    }

    private final CpuKernelIr arithmetic;
    private final List<Integer> rowSites, terminalOutputs, viewOrigins, rowAxes, retainedStores;
    private final List<Contraction> contractions;
    private final List<Sum> sums;
    private final List<CpuAccessPlan.BatchBinding> accesses;

    /**
     * Snapshots a candidate without allocating invocation storage or choosing materialization.
     * @param arithmetic nonnull typed value/instruction/store inventory
     * @param rowSites nonnull topological interleaving of pointwise and feature-contraction sites
     * @param contractions nonnull ordered matrix bindings, including terminal bindings
     * @param sums nonnull represented Scan SUM bindings; empty for ordinary callers
     * @param terminalOutputs nonnull terminal producing ordinals in publication order
     * @param accesses nonnull total per-value affine recipes, with zero B factors in fixed mode
     * @param viewOrigins nonnull total self-rooted storage-producing ordinals
     * @param rowAxes nonnull total row-axis roles, minus one for shared values
     * @param retainedStores nonnull exact producing stores retained for outside/saved demand
     * @throws NullPointerException if a reference or list element is null
     * @throws IllegalArgumentException if coverage, bounds, roots, roles or unique sites disagree
     */
    public CpuRowMapReductionIr(CpuKernelIr arithmetic, List<Integer> rowSites,
            List<Contraction> contractions, List<Sum> sums, List<Integer> terminalOutputs,
            List<CpuAccessPlan.BatchBinding> accesses, List<Integer> viewOrigins,
            List<Integer> rowAxes, List<Integer> retainedStores) {
        this.arithmetic = Objects.requireNonNull(arithmetic, "arithmetic");
        this.rowSites = List.copyOf(rowSites); this.contractions = List.copyOf(contractions);
        this.sums = List.copyOf(sums); this.terminalOutputs = List.copyOf(terminalOutputs);
        this.accesses = List.copyOf(accesses); this.viewOrigins = List.copyOf(viewOrigins);
        this.rowAxes = List.copyOf(rowAxes); this.retainedStores = List.copyOf(retainedStores);
        int n = arithmetic.values().size();
        if (n == 0 || accesses.size() != n || viewOrigins.size() != n || rowAxes.size() != n)
            throw new IllegalArgumentException("region requires total value geometry/origins/roles");
        for (int v = 0; v < n; v++) {
            int root = viewOrigins.get(v), row = rowAxes.get(v);
            if (root < 0 || root >= n || viewOrigins.get(root) != root
                    || row < -1 || row >= accesses.get(v).plan().iterationRank())
                throw new IllegalArgumentException("region origin/row role is invalid");
        }
        var sites = new java.util.HashSet<Integer>();
        for (int v : rowSites) if (v < 0 || v >= n || !sites.add(v))
            throw new IllegalArgumentException("duplicate or unknown row site");
        for (int v : terminalOutputs) if (v < 0 || v >= n || !sites.add(v))
            throw new IllegalArgumentException("duplicate or unknown terminal");
        for (int v : retainedStores) if (v < 0 || v >= n || !sites.contains(v))
            throw new IllegalArgumentException("retained store is not a producing site");
        var producing = new java.util.HashSet<Integer>();
        arithmetic.instructions().forEach(i -> producing.add(i.output()));
        for (var c : contractions) if (c.output() >= n || c.left() >= n || c.right() >= n
                || !producing.add(c.output())) throw new IllegalArgumentException("ambiguous contraction site");
        for (var s : sums) if (s.output() >= n || s.input() >= n || !producing.add(s.output())
                || s.axes().stream().anyMatch(a -> a >= accesses.get(s.input()).plan().iterationRank()))
            throw new IllegalArgumentException("ambiguous SUM site");
        if (!producing.containsAll(sites)) throw new IllegalArgumentException("unresolved region site");
    }

    /**
     * Supplies the shared ordinal space and typed pointwise sites; matrix and SUM bindings
     * remain separate so they cannot be mistaken for scalar instructions.
     * @return immutable typed arithmetic inventory, never null
     */
    public CpuKernelIr arithmetic() { return arithmetic; }
    /**
     * Supplies the proved interleaving emitted once for each active row, before terminal finish.
     * @return immutable ordered row-site ordinals, never null
     */
    public List<Integer> rowSites() { return rowSites; }
    /**
     * Supplies matrix operands in their represented left/right order, including both row-feature
     * projections and terminal contractions.
     * @return immutable ordered matrix bindings, never null
     */
    public List<Contraction> contractions() { return contractions; }
    /**
     * Supplies represented Scan reduction sites without conflating raw singleton movement
     * with addition; ordinary exact SUM is excluded from this inventory.
     * @return immutable represented SUM bindings, empty for ordinary regions and never null
     */
    public List<Sum> sums() { return sums; }
    /**
     * Supplies the completion order for terminal accumulators after the shared row traversal.
     * @return immutable ordered terminal producing ordinals, never null
     */
    public List<Integer> terminalOutputs() { return terminalOutputs; }
    /**
     * Supplies one affine geometry recipe per value, retaining symbolic coefficients rather
     * than invocation-resolved dimensions or addresses.
     * @return immutable total per-value access recipes, never null
     */
    public List<CpuAccessPlan.BatchBinding> accesses() { return accesses; }
    /**
     * Identifies each value's self-rooted storage producer so views share the correct binding
     * and lifetime instead of acquiring independent storage.
     * @return immutable total per-value storage-origin ordinals, never null
     */
    public List<Integer> viewOrigins() { return viewOrigins; }
    /**
     * Supplies row-axis provenance proved from dimensions and view transformations, never
     * guessed from equal numeric extents.
     * @return immutable total per-value row-axis indices, minus one for shared values; never null
     */
    public List<Integer> rowAxes() { return rowAxes; }
    /**
     * Identifies producing stores that outside or saved-value demand requires callers to retain,
     * even when the corresponding arithmetic is shared within the region.
     * @return immutable exact producing-store obligations, never null
     */
    public List<Integer> retainedStores() { return retainedStores; }

    /**
     * Returns an instruction-free ordinary cache envelope, not a replacement semantic graph.
     * Topology and represented-site distinctions enter its family, never concrete geometry.
     * @return immutable boundary/store envelope with this region's structural key
     */
    public CpuKernelIr encodedKernelIr() {
        return new CpuKernelIr(arithmetic.values(), List.of(), arithmetic.loop(), arithmetic.stores(),
                "row-map-multi-reduction-v1:arithmetic=" + arithmetic.structuralKey() + ":rows=" + rowSites
                        + ":contractions=" + contractions + ":sums=" + sums + ":terminals=" + terminalOutputs
                        + ":origins=" + viewOrigins + ":axes=" + rowAxes + ":stores=" + retainedStores);
    }

    /**
     * Identifies canonical topology independently of the baked geometry payload.
     * @return geometry-free lowercase hexadecimal structural key, never null
     */
    @Override public String structuralKey() { return encodedKernelIr().structuralKey(); }

    /**
     * Encodes all baked geometry and code-shaping limits; symbolic B is represented by its
     * coefficients, never a resolved invocation extent, length value, address or assigned slot.
     * @return deterministic nonnull baked payload for both compatibility streams
     */
    public String bakedSpecializationIdentity() {
        return structuralKey() + ":accesses=" + accesses + ":range=full-body-1"
                + ":limits=32,4,64,16,128,32,64,48000";
    }

    /**
     * Compares immutable facts, allowing independently lowered equivalent prepared owners to
     * reuse the exact same generated artifact without depending on a cold object identity.
     * @param other any object, including null
     * @return true exactly for a region with identical topology, geometry and store obligations
     */
    @Override public boolean equals(Object other) {
        return this == other || other instanceof CpuRowMapReductionIr ir && arithmetic.equals(ir.arithmetic)
                && rowSites.equals(ir.rowSites) && contractions.equals(ir.contractions) && sums.equals(ir.sums)
                && terminalOutputs.equals(ir.terminalOutputs) && accesses.equals(ir.accesses)
                && viewOrigins.equals(ir.viewOrigins) && rowAxes.equals(ir.rowAxes) && retainedStores.equals(ir.retainedStores);
    }

    /**
     * Hashes the same immutable facts compared by value equality for cold artifact reuse.
     * @return value hash of every immutable code-shaping and legality fact
     */
    @Override public int hashCode() {
        return Objects.hash(arithmetic,rowSites,contractions,sums,terminalOutputs,accesses,viewOrigins,rowAxes,retainedStores);
    }
}
