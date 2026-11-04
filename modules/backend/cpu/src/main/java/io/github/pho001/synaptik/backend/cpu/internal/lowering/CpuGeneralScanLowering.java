package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.ir.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind;
import io.github.pho001.synaptik.model.operation.elementwise.scalar.ScalarElementwiseKind;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.operation.linalg.MatmulKind;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.operation.reduction.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.*;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;

/**
 * Whole-body and binder proof for retained homogeneous recipes and ordinary typed composition. This cold
 * boundary performs no physical allocation, class generation, AD or body rewriting. An empty time or
 * coordinate domain does not bypass validation of dead instructions or canonical residual roles.
 * Carry liveness includes the transitive next-role forwarding closure, so an instruction-free
 * old-carry source retains its next time version when another live carry reads it later.
 * Occurrences outside the exact retained recipe intersections delegate to
 * {@link CpuGeneralScanCompositionLowering}; their per-value physical facts never use a first
 * carry's type, shape or width. Empty-carry and no-node bodies remain genuine ordinary cases.
 * The shaped path proves per-value affine access and complete-stage lifetimes, not one common
 * frame size. It declares disjoint live carry versions and only necessary materialized cuts
 * before shared slot assignment; forward admission does not qualify a generated reverse body.
 * Rank-two MATMUL stages use CpuMatmulLowering's common exact Dimension equations for M/K/N,
 * then prove both ordered affine operands under Scan's unchanged whole-occurrence guards. In
 * that additive domain only, dense no-axis SUM-to-same-Shape bindings are raw aliases, avoiding
 * artificial product copies while preserving saved/public represented values. Proved same-cell
 * product tails compose at generation time; dead products never retain full-K work.
 * Shared pointwise and aggregate lowerers validate current arithmetic and selected-axis facts.
 * Scan alone retains whole-body, role, layout, demand and resource admission; shared
 * normalization does not sample B or select an exact versus sequential reduction algorithm.
 * Represented dtype determines checked four/eight-byte spans, affine coefficients, frames and
 * resources, including their aggregate byte count. FLOAT64 local budgets include two-slot JVM
 * values. Carrier availability alone never qualifies a type, body operation or generated reverse.
 * Static actual reads admit resolved zero-offset canonical strides independently of the view flag,
 * and retain all-zero-stride scalar reads. Neither admission changes the source descriptor;
 * static internal formals and outer write ports still require complete non-view canonical layouts.
 * Symbolic FLOAT32 and length-mode FLOAT64 use the separate named-coordinate proof: exact affine resource recipes,
 * provenance-preserving views and actual boundary reads are retained without selecting B.
 * Symbolic formal/result layouts remain unresolved; named-axis proof does not grant symbolic
 * all-valid FLOAT64 or arbitrary layout support.
 * Fixed length mode independently proves explicit row roles and terminal SHARED suffixes,
 * keeps INT64 control outside arithmetic/formals, and declares mixed-width resources. Fusion
 * preserves activity domains, not just equal shapes. Histories cannot back inactive state;
 * only proved direct-old same-cell transitions may own their distinct final output in-place.
 * Static non-MATMUL staged/pointwise and symbolic identity text and code remain unchanged.
 * Low floating bodies use ordinary physical composition rather than reinterpreting the legacy
 * homogeneous arithmetic locals. Byte/short carries, saved values and public ports remain raw;
 * each ordinary helper owns FLOAT32 working sites and its final represented narrowing.
 * CPU0071's approved retained static MATMUL cutovers instead advance all-valid v4 to v5 and
 * fixed-length MATMUL v1 to length-aware-matmul v2 for invocation-bound constant strides.
 */
public final class CpuGeneralScanLowering {
    /** Creates a stateless scan proof boundary. */
    public CpuGeneralScanLowering() {}

    /**
     * Proves one projected occurrence and preserves demanded output wrappers without aliasing.
     * @param context non-null single-scan CPU projection with exact demand
     * @return immutable generated lowering with typed resource facts
     * @throws NullPointerException if context is null
     * @throws IllegalArgumentException if the unit is not one qualified scan
     * @throws ArithmeticException if exact geometry overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            io.github.pho001.synaptik.prepare.analysis.PrepareContext<? extends
                io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs> context) {
        if (context.nodes().size() != 1) throw invalid();
        var node = context.nodes().getFirst();
        var descriptors = new HashMap<ValueId, TensorDescriptor>();
        context.values().forEach(v -> descriptors.put(v.id(), v.descriptor()));
        Facts facts = analyze(new OperationCapabilityQuery(node.operation(), node.inputs().stream()
                .map(descriptors::get).toList(), node.outputs().stream().map(descriptors::get).toList(),
                node.scanBody(), Optional.ofNullable(context.scanActivityPlans().get(node.id()))));
        if (facts.ir().composition().isPresent()) return new CpuGeneralScanCompositionLowering().lower(context);
        if (facts.ir().symbolic()) return lowerSymbolic(context,node,facts,CpuPartitionLowering.batchLayouts(context));
        var unique = new LinkedHashMap<ValueId, Integer>();
        var positions = new ArrayList<Integer>();
        var scalar = new ArrayList<Boolean>(); var spans = new ArrayList<Long>();
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        for (int p = 0; p < node.inputs().size(); p++) {
            ValueId id = node.inputs().get(p);
            if (!unique.containsKey(id)) {
                unique.put(id, unique.size()); scalar.add(facts.ir().scalarReads().get(p));
                spans.add(facts.spans().get(p)); bindings.add(facts.accessBindings().get(p));
            }
            if (!facts.ir().lengthAware() || p < facts.ir().formalOrdinals().size()) positions.add(unique.get(id));
        }
        int oldReads = facts.ir().scalarReads().size(), reads = unique.size();
        var boundaries = new ArrayList<>(unique.keySet());
        for (int w = 0; w < facts.ir().demandedPorts().size(); w++) {
            ValueId id = node.outputs().get(facts.ir().demandedPorts().get(w));
            if (unique.containsKey(id) || boundaries.contains(id)) throw invalid();
            boundaries.add(id); spans.add(facts.spans().get(oldReads + w));
            bindings.add(facts.accessBindings().get(oldReads + w));
        }
        var i = facts.ir();
        var locations = i.carries().stream().map(c -> switch (c.storage()) {
            case FRAME, OUTPUT -> new CpuGeneralScanIr.Carry(c.storage(), c.position() - oldReads + reads);
            default -> c;
        }).toList();
        var stageFacts = i.stages();
        if (i.lengthAware()) {
            var s = stageFacts.orElseThrow();
            var outerRoles = new HashMap<ValueId,Integer>();
            for (int f = 0; f < positions.size(); f++) {
                int role = s.rowAxes().get(i.formalOrdinals().get(f));
                if (role >= 0 && f >= i.carryCount() && f < i.carryCount()+i.sequenceCount()) role++;
                Integer previous = outerRoles.put(node.inputs().get(f),role);
                if (previous != null && previous != role) throw invalid();
            }
            if (outerRoles.containsKey(node.inputs().getLast())) throw invalid();
            stageFacts = Optional.of(new CpuGeneralScanIr.Stages(s.views(),s.order(),s.cuts(),s.carryOffsets(),s.carryPorts(),s.scratchBytes(),
                    s.carryOffsetFactors(),s.scratchBytesPerBatch(),false,s.sequenceStrides(),s.rowAxes(),s.terminal(),
                    s.rowCarryCount(),unique.get(node.inputs().getLast()),s.batch(),s.rowMapReductionIr()));
        }
        var ir = new CpuGeneralScanIr(i.arithmetic(), i.formalOrdinals(), positions, scalar,
                i.nextOrdinals(), i.portOrdinals(), i.demandedPorts(), locations,
                bindings.stream().map(CpuAccessPlan.Binding::plan).toList(), i.carryCount(),
                i.sequenceCount(), i.reverse(), i.scratchBanks(), stageFacts);
        return new CpuPartitionLowering.LoweredPartition(new Facts(ir, facts.geometry(), spans, bindings), boundaries);
    }

    /**
     * Deduplicates exact outer read identities, retaining positional formal bindings and
     * authenticated outer transpose/expand recipes rather than guessing from numeric sizes.
     * @param context exact scoped partition
     * @param node sole scan occurrence
     * @param facts independently qualified symbolic body
     * @param layouts complete partition's authenticated actual boundary mappings
     * @return symbolic lowering with only demanded write boundaries
     * @throws NullPointerException if required scoped facts or a boundary mapping are absent
     * @throws IllegalArgumentException if an actual access recipe is unsupported
     * @throws ArithmeticException if a structural byte-stride coefficient overflows
     */
    private CpuPartitionLowering.LoweredPartition lowerSymbolic(
            io.github.pho001.synaptik.prepare.analysis.PrepareContext<? extends
                io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs> context,CompiledNode node,Facts facts,
            Map<ValueId,CpuAccessPlan.BatchBinding> layouts) {
        var unique = new LinkedHashMap<ValueId,Integer>(); var actuals = new ArrayList<Integer>();
        var bindings = new ArrayList<CpuAccessPlan.BatchBinding>(); var scalar = new ArrayList<Boolean>();
        for (var id : node.inputs()) {
            if (!unique.containsKey(id)) {
                unique.put(id,unique.size()); var binding = layouts.get(id); bindings.add(binding);
                scalar.add(binding.plan().regime() == CpuAccessPlan.Regime.SCALAR_ALL_ZERO);
            }
            if (!facts.ir().lengthAware() || actuals.size() < facts.ir().formalOrdinals().size())
                actuals.add(unique.get(id));
        }
        var boundaries = new ArrayList<>(unique.keySet());
        for (int p : facts.ir().demandedPorts()) {
            boundaries.add(node.outputs().get(p));
            bindings.add(facts.batchBindings().get(node.inputs().size()+facts.ir().demandedPorts().indexOf(p)));
        }
        var i = facts.ir();
        var stages = i.stages().orElseThrow(); var views = new ArrayList<CpuGeneralScanIr.View>();
        for (var view : stages.views()) {
            var initial = view.batch().orElseThrow();
            if (view.origin() < i.formalOrdinals().size()) {
                int formal = view.origin(); var actual = layouts.get(node.inputs().get(formal));
                int shift = formal >= i.carryCount() && formal < i.carryCount()+i.sequenceCount() ? 1 : 0;
                var sc = new ArrayList<Long>(); var sf = new ArrayList<Long>();
                for (int axis : view.originAxes()) {
                    sc.add(axis < 0 ? 0L : actual.strideConstants().get(axis+shift));
                    sf.add(axis < 0 ? 0L : actual.strideBatchFactors().get(axis+shift));
                }
                initial = CpuPartitionLowering.batchAccess(CpuAccessPlan.AccessKind.READ,initial.extentConstants(),initial.extentBatchFactors(),sc,sf);
            }
            views.add(new CpuGeneralScanIr.View(view.origin(),view.shape(),view.strides(),view.batch(),view.originAxes(),Optional.of(initial)));
        }
        var mapped = new CpuGeneralScanIr.Stages(views,stages.order(),stages.cuts(),stages.carryOffsets(),stages.carryPorts(),
                stages.scratchBytes(),stages.carryOffsetFactors(),stages.scratchBytesPerBatch(),true,
                java.util.stream.IntStream.range(i.carryCount(),i.carryCount()+i.sequenceCount()).mapToObj(f -> {
                    var actual = layouts.get(node.inputs().get(f));
                    int width = i.dataType().byteWidth();
                    return new BatchByteSize(Math.multiplyExact(actual.strideConstants().getFirst(),width),Math.multiplyExact(actual.strideBatchFactors().getFirst(),width),width);
                }).toList(),stages.rowAxes(),stages.terminal(),stages.rowCarryCount(),
                i.lengthAware() ? unique.get(node.inputs().getLast()) : -1,0,stages.rowMapReductionIr());
        if (i.lengthAware()) {
            var roles = new HashMap<ValueId,Integer>();
            for (int f = 0; f < actuals.size(); f++) {
                int role = stages.rowAxes().get(i.formalOrdinals().get(f));
                if (role >= 0 && f >= i.carryCount() && f < i.carryCount()+i.sequenceCount()) role++;
                Integer previous = roles.putIfAbsent(node.inputs().get(f),role);
                if (previous != null && previous != role) throw invalid();
            }
            if (roles.containsKey(node.inputs().getLast())) throw invalid();
        }
        var ir = new CpuGeneralScanIr(i.arithmetic(),i.formalOrdinals(),actuals,scalar,i.nextOrdinals(),i.portOrdinals(),
                i.demandedPorts(),i.carries(),bindings.stream().map(CpuAccessPlan.BatchBinding::plan).toList(),
                i.carryCount(),i.sequenceCount(),i.reverse(),i.scratchBanks(),Optional.of(mapped));
        return new CpuPartitionLowering.LoweredPartition(new Facts(ir,facts.geometry(),List.of(),List.of(),bindings),boundaries);
    }

    /**
     * Lowers an isolated Scan, retaining actual mappings from its complete partition in symbolic
     * mode and delegating unchanged fixed-mode lowering otherwise.
     * The mapping parameter is a CPU-internal lowering seam, not a supported API or Runtime handoff.
     * @param context exact single-node scoped projection
     * @param layouts complete partition's immutable prepared read mappings, used in symbolic mode
     * @return non-null qualified fixed or symbolic lowering; symbolic initial-read provenance
     *     remains independent of evolving carry banks
     * @throws NullPointerException if the context or a required mapping is null
     * @throws IllegalArgumentException if this is not a qualified single Scan
     * @throws ArithmeticException if structural geometry or a byte coefficient overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            io.github.pho001.synaptik.prepare.analysis.PrepareContext<? extends
                io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs> context,
            Map<ValueId,CpuAccessPlan.BatchBinding> layouts) {
        if (context.nodes().size() != 1) throw invalid(); var node = context.nodes().getFirst();
        var descriptors = new HashMap<ValueId,TensorDescriptor>(); context.values().forEach(v -> descriptors.put(v.id(),v.descriptor()));
        var facts = analyze(new OperationCapabilityQuery(node.operation(),node.inputs().stream().map(descriptors::get).toList(),
                node.outputs().stream().map(descriptors::get).toList(),node.scanBody(),
                Optional.ofNullable(context.scanActivityPlans().get(node.id()))));
        if (facts.ir().composition().isPresent()) return new CpuGeneralScanCompositionLowering().lower(context, layouts);
        if (!facts.ir().symbolic()) return lower(context);
        return lowerSymbolic(context,node,facts,layouts);
    }

    /**
     * Exact cold invocation/resource geometry; no mutable invocation cursor is retained. Fixed
     * source time is packed at invocation, not varied within one prepared template. Staged local
     * extents and resource magnitudes are additionally keyed in the
     * existing generated specialization, while canonical identity remains geometry-free.
     * @param time non-negative original time extent
     * @param fixedElements fixed first-carry frame elements, or empty in explicitly run-bound mode
     * @param workspaceSize exact fixed or constant-plus-B bank/cut byte recipe, including zero
     */
    public record Geometry(long time, OptionalLong fixedElements,
            io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.ByteSize workspaceSize) {
        /**
         * Constructs retained fixed invocation geometry.
         * @param time fixed time steps
         * @param elements fixed first-carry frame elements
         * @param scratchBytes exact fixed workspace bytes
         * @throws IllegalArgumentException if a count is negative or scratch is not four-byte aligned
         */
        public Geometry(long time, long elements, long scratchBytes) {
            this(time,OptionalLong.of(elements),new io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan.FixedByteSize(scratchBytes));
        }
        /**
         * Checks exact non-negative runtime geometry without choosing storage.
         * @param time original time extent, in steps
         * @param fixedElements non-null optional fixed first-frame element count, never a symbolic surrogate
         * @param workspaceSize non-null exact byte recipe; fixed sizes require four-byte alignment,
         *     and containing {@link Facts} additionally enforces the exact dtype width
         * @throws NullPointerException if either reference is null
         * @throws IllegalArgumentException if a count is negative or scratch is misaligned
         */
        public Geometry {
            Objects.requireNonNull(fixedElements,"fixedElements"); Objects.requireNonNull(workspaceSize,"workspaceSize");
            if (time < 0 || fixedElements.isPresent() && fixedElements.getAsLong() < 0) throw invalid();
            workspaceSize.fixedByteSize().ifPresent(bytes -> { if (bytes % 4 != 0) throw invalid(); });
        }
        /**
         * Returns static first-frame geometry, never a symbolic placeholder.
         * @return exact fixed frame elements
         * @throws IllegalStateException if the frame requires invocation B
         */
        public long elements() { return fixedElements.orElseThrow(() -> new IllegalStateException("run-bound scan frame")); }
        /**
         * Returns static workspace geometry, never a zero/maximum surrogate.
         * @return exact fixed workspace bytes
         * @throws IllegalStateException if the declaration is run-bound, even with zero slope
         */
        public long scratchBytes() { return workspaceSize.fixedByteSize().orElseThrow(() -> new IllegalStateException("run-bound scan workspace")); }
    }

    /**
     * One fully proved body and its exact cold boundary/resource facts.
     * @param ir non-null typed generation facts with no captured graph
     * @param geometry non-null checked invocation geometry and exact total workspace
     * @param spans exact element spans of unique reads then demanded writes
     * @param accessBindings resolved accesses in that same order
     * @param batchBindings symbolic accesses in boundary order, empty for static facts;
     *     symbolic facts instead leave spans and accessBindings empty
     * @param ordinaryUnits exact ordinary lowered units aligned with composite helper declarations;
     *     empty for all qualified legacy recipes and legitimate zero-node bodies
     */
    public record Facts(CpuGeneralScanIr ir, Geometry geometry, List<Long> spans,
            List<CpuAccessPlan.Binding> accessBindings, List<CpuAccessPlan.BatchBinding> batchBindings,
            List<CpuPartitionLowering.LoweredPartition> ordinaryUnits) {
        /**
         * Retains legacy fixed/symbolic facts without an ordinary-unit inventory.
         * @param ir qualified legacy typed facts
         * @param geometry exact resource geometry
         * @param spans fixed boundary spans, empty for symbolic mode
         * @param accessBindings fixed boundary accesses, empty for symbolic mode
         * @param batchBindings symbolic boundary accesses, empty for fixed mode
         */
        public Facts(CpuGeneralScanIr ir, Geometry geometry, List<Long> spans,
                List<CpuAccessPlan.Binding> accessBindings, List<CpuAccessPlan.BatchBinding> batchBindings) {
            this(ir, geometry, spans, accessBindings, batchBindings, List.of());
        }
        /**
         * Retains fully resolved static analysis facts.
         * @param ir fixed generated topology
         * @param geometry resolved invocation facts
         * @param spans fixed boundary element spans
         * @param accessBindings fixed accesses
         * @throws NullPointerException if a component or list element is null
         * @throws IllegalArgumentException if fixed boundary/resource facts disagree
         * @throws ArithmeticException if a represented byte product or aggregate sum overflows
         */
        public Facts(CpuGeneralScanIr ir, Geometry geometry, List<Long> spans,
                List<CpuAccessPlan.Binding> accessBindings) {
            this(ir,geometry,spans,accessBindings,List.of());
        }
        /**
         * Snapshots metadata and checks boundary cardinalities. Fixed facts additionally check
         * represented-width alignment and aggregate bytes; symbolic sizes remain recipes for
         * checked binding. No list is writable runtime state.
         * @param ir non-null derived generation facts
         * @param geometry non-null checked geometry
         * @param spans exact non-null boundary spans
         * @param accessBindings non-null resolved boundary accesses
         * @param batchBindings non-null symbolic boundary recipes, mutually exclusive with resolved accesses
         * @param ordinaryUnits exact independently admitted composite ordinary lowerings
         * @throws NullPointerException if a component or list element is null
         * @throws IllegalArgumentException if boundary cardinalities, exact scratch bytes or
         *     represented-width workspace alignment disagree
         * @throws ArithmeticException if an individual resource product or aggregate byte sum overflows
         */
        public Facts {
            Objects.requireNonNull(ir, "ir");
            Objects.requireNonNull(geometry, "geometry");
            spans = List.copyOf(spans);
            accessBindings = List.copyOf(accessBindings);
            batchBindings = List.copyOf(batchBindings);
            ordinaryUnits = List.copyOf(ordinaryUnits);
            if (ir.composition().isPresent()) {
                var composition = ir.composition().orElseThrow();
                var workspace = geometry.workspaceSize();
                long constant = workspace instanceof BatchByteSize affine ? affine.constantBytes()
                        : workspace.fixedByteSize().orElseThrow();
                long factor = workspace instanceof BatchByteSize affine ? affine.bytesPerBatch() : 0;
                if (ordinaryUnits.size() != composition.units().size()
                        || constant < composition.workspaceSize().constantBytes()
                        || factor < composition.workspaceSize().bytesPerBatch()) throw invalid();
                if (ir.symbolic()) {
                    if (!spans.isEmpty() || !accessBindings.isEmpty()
                            || batchBindings.size() != ir.boundaryAccesses().size()) throw invalid();
                } else if (!batchBindings.isEmpty() || spans.size() != accessBindings.size()
                        || spans.size() != ir.boundaryAccesses().size()) throw invalid();
            } else {
                if (!ordinaryUnits.isEmpty()) throw invalid();
                if (ir.symbolic()) {
                    if (!spans.isEmpty() || !accessBindings.isEmpty()
                            || batchBindings.size() != ir.boundaryAccesses().size()) throw invalid();
                } else if (!batchBindings.isEmpty() || spans.size() != accessBindings.size() || spans.size() != ir.boundaryAccesses().size()
                        || geometry.scratchBytes() != ir.stages().map(CpuGeneralScanIr.Stages::scratchBytes)
                            .orElseGet(() -> Math.multiplyExact(Math.multiplyExact(geometry.elements(), ir.scratchBanks()), (long) ir.dataType().byteWidth())))
                    throw invalid();
                int width = ir.dataType().byteWidth();
                if (!ir.symbolic() && geometry.scratchBytes() % width != 0) throw invalid();
            }
            long total = ir.symbolic() ? 0 : geometry.scratchBytes();
            for (int p = 0; p < spans.size(); p++)
                total = Math.addExact(total, Math.multiplyExact(spans.get(p), (long) ir.boundaryType(p).byteWidth()));
        }

        /**
         * Packs checked invocation geometry into the established long-array entry ABI. Layout is
         * T, first-frame elements, then each ordinal's frame bytes, extents, nominal element
         * strides and initial-read element strides, followed by actual sequence time-byte strides,
         * cut-byte offsets and carry-bank byte offsets. Combined length mode appends explicit B,
         * independent of frame width (including zero features). Fixed entries retain only T/elements.
         * Composite entries instead retain only T and explicit bound B; their independently
         * typed helper packs are prepared once by the ordinary resource/binding owners.
         * No resolved fact changes the prepared topology.
         * @param batch admitted non-negative B within CPU integer limits; ignored for fixed facts
         * @return fresh mutable numeric entry array owned exclusively by this invocation
         * @throws IllegalArgumentException if symbolic B is negative or a prepared recipe is invalid
         * @throws ArithmeticException if any local access/frame/offset is unrepresentable
         */
        public long[] pack(long batch) {
            if (ir.composition().isPresent()) {
                if (batch < 0) throw invalid();
                Math.toIntExact(batch);
                return new long[] {geometry.time(), batch};
            }
            if (!ir.symbolic()) return new long[] {geometry.time(),geometry.elements()};
            var stages = ir.stages().orElseThrow(); int width = ir.dataType().byteWidth();
            var packed = new ArrayList<Long>(); packed.add(geometry.time()); packed.add(0L);
            for (var view : stages.views()) {
                var binding = view.batch().orElseThrow().bind(batch);
                long bytes = view.frameSize(width).byteSize(batch);
                packed.add(bytes); packed.addAll(binding.extents()); packed.addAll(binding.effectiveStrides());
                packed.addAll(view.initialRead().orElse(view.batch().orElseThrow()).bind(batch).effectiveStrides());
            }
            packed.set(1,packed.get(2)/width);
            for (var stride : stages.sequenceStrides()) packed.add(stride.byteSize(batch));
            for (var cut : stages.cuts()) packed.add(new BatchByteSize(cut.offset(),cut.offsetPerBatch(),width).byteSize(batch));
            for (int c = 0; c < stages.carryOffsets().size(); c++) packed.add(stages.carryOffsets().get(c) < 0 ? -1L
                    : new BatchByteSize(stages.carryOffsets().get(c),stages.carryOffsetFactors().get(c),width).byteSize(batch));
            stages.workspaceSize(width).byteSize(batch);
            if (ir.lengthAware()) packed.add((long)Math.toIntExact(batch));
            return packed.stream().mapToLong(Long::longValue).toArray();
        }
    }

    /**
     * Validates the actual query, including local inference, canonical saved ports and ABI limits.
     * Repeated descriptors in this descriptor-only query remain distinct conservative reads;
     * occurrence lowering may subsequently deduplicate exact outer identities. Qualified old
     * recipes are tried unchanged when activity is absent; all other known PURE occurrences use
     * exact ordinary body decomposition. Empty-carry, explicit no-ROW length and zero-node bodies
     * have no fabricated declaration, representative B or dummy numerical operation.
     * @param query non-null complete body-bearing occurrence
     * @return immutable proved facts; never null
     * @throws NullPointerException if query is null
     * @throws IllegalArgumentException if any binder, algebra, layout or budget predicate fails
     * @throws ArithmeticException if exact sizes or address arithmetic overflow
     */
    public static Facts analyze(OperationCapabilityQuery query) {
        Objects.requireNonNull(query, "query");
        if (query.scanActivityPlan().isPresent() || query.scanBody().stream()
                .flatMap(body -> body.graph().values().stream())
                .anyMatch(value -> value.descriptor().dataType().isFloating()
                        && value.descriptor().dataType().byteWidth() < Float.BYTES))
            return CpuGeneralScanCompositionLowering.analyze(query);
        try {
            if (query.operation().attrs() instanceof ScanAttrs a
                    && (a.carryCount() == 0 || a.lengthAware() && a.rowCarryCount() == 0)) throw invalid();
            if (query.operation().attrs() instanceof ScanAttrs attrs && attrs.lengthAware())
                return query.inputs().stream().anyMatch(d -> !d.shape().isFullyStatic())
                        ? analyzeSymbolic(query) : analyzeShaped(query);
            if (java.util.stream.Stream.concat(query.inputs().stream(),query.outputs().stream())
                    .anyMatch(d -> !d.shape().isFullyStatic())) return analyzeSymbolic(query);
            try { return analyzePointwise(query); }
            catch (IllegalArgumentException rejectedPointwise) { return analyzeShaped(query); }
        } catch (IllegalArgumentException rejectedLegacyRecipe) {
            return CpuGeneralScanCompositionLowering.analyze(query);
        }
    }

    /**
     * Proves typed pointwise execution while preserving the accepted FLOAT32 generated identity.
     * @param query complete occurrence under proof
     * @return proved original pointwise facts
     * @throws IllegalArgumentException if the original domain does not cover this occurrence
     * @throws ArithmeticException if typed spans, frames or aggregate resource bytes overflow
     */
    private static Facts analyzePointwise(OperationCapabilityQuery query) {
        Objects.requireNonNull(query, "query");
        if (query.operation().kind() != ScanKind.SCAN
                || !(query.operation().attrs() instanceof ScanAttrs a)) throw invalid();
        CompiledScanBody body = query.scanBody().orElseThrow(CpuGeneralScanLowering::invalid);
        var graph = body.graph();
        if (query.inputs().isEmpty()) throw invalid();
        DataType type = query.inputs().getFirst().dataType();
        long width = type.byteWidth();
        if (query.inputs().size() != a.inputCount() || query.outputs().size() != a.outputCount()
                || graph.inputs().size() != a.inputCount() || body.nextCarry().size() != a.carryCount()
                || body.emissions().size() != a.emissionCount() || body.residuals().size() != a.residualCount()
                || graph.nodes().size() > 128 || graph.values().size() > 256 || a.outputCount() > 256
                || graph.inputs().size() > 64
                || a.inputCount() + body.demandedOutputIndices().size() > 248) throw invalid();
        // Double locals occupy two JVM slots, including activation scratch. Conservative F64
        // method bounds cover entry/layout slots, all selected carriers, publications and banks.
        // F32 keeps its accepted bounded proof and byte-exact specialization unchanged.
        if (type == DataType.FLOAT64 && (1024L + 400L*graph.nodes().size()
                + 100L*(a.inputCount()+body.demandedOutputIndices().size()+a.carryCount()) > 48_000
                || 512L + 2L*graph.values().size() + 8L*(a.inputCount()+a.outputCount()) > 60_000)) throw invalid();
        var local = new LinkedHashMap<ValueId, TensorDescriptor>();
        graph.values().forEach(v -> { checkTypeShape(v.descriptor()); if (v.descriptor().dataType() != type) throw invalid(); local.put(v.id(), v.descriptor()); });
        Shape step = local.get(graph.inputs().getFirst()).shape();
        if (step.rank() > 16) throw invalid();
        long n = step.knownElementCount().orElseThrow();
        long frames = Math.multiplyExact(a.timeCount(), n);
        Math.multiplyExact(frames, width);
        var ordinals = new LinkedHashMap<ValueId, Integer>();
        var irValues = new ArrayList<CpuKernelIr.Value>();
        var instructions = new ArrayList<CpuKernelIr.Instruction>();
        var formals = new ArrayList<Integer>();
        var actuals = new ArrayList<Integer>();
        var scalars = new ArrayList<Boolean>();
        var spans = new ArrayList<Long>();
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        boolean grad = query.inputs().stream().anyMatch(TensorDescriptor::requiresGrad);
        for (int p = 0; p < graph.inputs().size(); p++) {
            TensorDescriptor actual = query.inputs().get(p), formal = local.get(graph.inputs().get(p));
            checkTypeShape(actual);
            if (actual.dataType() != type) throw invalid();
            boolean sequence = p >= a.carryCount() && p < a.carryCount() + a.sequenceCount();
            boolean scalarInvariant = p >= a.carryCount() + a.sequenceCount()
                    && formal.shape().rank() == 0 && !formal.requiresGrad();
            Shape expected = sequence ? stacked(step, a.timeCount()) : scalarInvariant ? Shape.of() : step;
            if (!actual.shape().equals(expected) || !formal.shape().equals(scalarInvariant ? Shape.of() : step)
                    || actual.requiresGrad() != formal.requiresGrad() || !canonical(formal)) throw invalid();
            boolean scalar = scalarRead(actual);
            addBoundary(actual, CpuAccessPlan.AccessKind.READ, scalar, spans, bindings);
            scalars.add(scalar); actuals.add(p);
            int ordinal = irValues.size();
            ordinals.put(graph.inputs().get(p), ordinal); formals.add(ordinal);
            irValues.add(new CpuKernelIr.Value(ordinal, type, CpuKernelIr.Value.Kind.INPUT,
                    access(formal.shape().rank(), CpuAccessPlan.AccessKind.READ, scalarInvariant)));
        }
        var derivedLayouts = new HashMap<ValueId, LayoutDescriptor>();
        graph.inputs().forEach(id -> derivedLayouts.put(id, local.get(id).layout().orElseThrow()));
        for (CompiledNode node : graph.nodes()) {
            if (node.outputs().size() != 1 || node.scanBody().isPresent()) throw invalid();
            var inputs = node.inputs().stream().map(local::get).toList();
            TensorDescriptor output = local.get(node.outputs().getFirst());
            Object kind = node.operation().kind();
            if (kind == ShapeTransformKind.EXPAND || kind == ContiguousKind.CONTIGUOUS) {
                if (inputs.size() != 1 || !output.shape().equals(step)
                        || output.requiresGrad() != inputs.getFirst().requiresGrad()) throw invalid();
                var input = inputs.getFirst();
                boolean scalar = input.shape().rank() == 0 && !input.requiresGrad();
                if (kind == ShapeTransformKind.EXPAND) {
                    if (!(node.operation().attrs() instanceof TargetShapeAttrs attrs)
                            || !attrs.targetShape().equals(step) || !scalar && !input.shape().equals(step))
                        throw invalid();
                    LayoutDescriptor expected = LayoutDescriptor.of(step, scalar ? new long[step.rank()]
                            : derivedLayouts.get(node.inputs().getFirst()).strides(), 0, true);
                    if (output.layout().isPresent() && !output.layout().orElseThrow().equals(expected)) throw invalid();
                    derivedLayouts.put(node.outputs().getFirst(), expected);
                } else {
                    if (node.operation().attrs() != NoOperationAttrs.INSTANCE
                            || !input.shape().equals(step)
                            || output.layout().isPresent() && !canonical(output)) throw invalid();
                    derivedLayouts.put(node.outputs().getFirst(), LayoutDescriptor.contiguous(step));
                }
                ordinals.put(node.outputs().getFirst(), ordinals.get(node.inputs().getFirst()));
                continue;
            }
            boolean binary = kind == BinaryArithmeticKind.ADD || kind == BinaryArithmeticKind.SUB
                    || kind == BinaryArithmeticKind.MUL || kind == BinaryArithmeticKind.DIV;
            boolean scalar = kind == ScalarElementwiseKind.ADD || kind == ScalarElementwiseKind.SUB
                    || kind == ScalarElementwiseKind.MUL || kind == ScalarElementwiseKind.DIV;
            boolean unary = kind == UnaryElementwiseKind.NEG || kind == UnaryElementwiseKind.TANH
                    || kind == UnaryElementwiseKind.SIGMOID;
            if (!binary && !scalar && !unary
                    || !output.shape().equals(step) || inputs.stream().anyMatch(d -> !d.shape().equals(step))
                    || output.requiresGrad() != inputs.stream().anyMatch(TensorDescriptor::requiresGrad)
                    || output.layout().isPresent() && !canonical(output)) throw invalid();
            CpuPartitionLowering.validatePointwise(node,inputs,output);
            int ordinal = irValues.size();
            ordinals.put(node.outputs().getFirst(), ordinal);
            derivedLayouts.put(node.outputs().getFirst(), LayoutDescriptor.contiguous(step));
            irValues.add(new CpuKernelIr.Value(ordinal, type, CpuKernelIr.Value.Kind.VIRTUAL,
                    access(step.rank(), CpuAccessPlan.AccessKind.READ, false)));
            instructions.add(new CpuKernelIr.Instruction(CpuPartitionLowering.opcode(node.operation().kind()),
                    node.inputs().stream().map(ordinals::get).toList(), ordinal,
                    CpuPartitionLowering.scalarImmediate(node)));
        }
        validateInventory(body, a.carryCount());
        var roles = new ArrayList<>(body.nextCarry()); roles.addAll(body.emissions()); roles.addAll(body.residuals());
        for (int p = 0; p < roles.size(); p++) {
            var d = local.get(roles.get(p));
            if (p < a.carryCount() + a.emissionCount() && !d.shape().equals(step)) throw invalid();
            Shape expected = p < a.carryCount() ? step : stacked(d.shape(), a.timeCount());
            TensorDescriptor outer = query.outputs().get(p);
            if (outer.dataType() != type || !outer.shape().equals(expected)
                    || outer.requiresGrad() != grad || !canonical(outer)) throw invalid();
        }
        for (int p : body.demandedOutputIndices())
            addBoundary(query.outputs().get(p), CpuAccessPlan.AccessKind.WRITE, false, spans, bindings);
        var next = body.nextCarry().stream().map(ordinals::get).toList();
        var ports = roles.stream().map(ordinals::get).toList();
        var carries = new ArrayList<CpuGeneralScanIr.Carry>();
        var readOrdinals = new HashSet<Integer>();
        instructions.forEach(i -> readOrdinals.addAll(i.inputs()));
        body.demandedOutputIndices().forEach(p -> readOrdinals.add(ports.get(p)));
        // A live carry may forward another old carry without an arithmetic instruction.
        // Close those next-role edges before allocating banks: temporal reads are simultaneous,
        // and a transitive forwarding chain must retain every source's next time version.
        boolean changed;
        do {
            changed = false;
            for (int c = 0; c < a.carryCount(); c++)
                if (readOrdinals.contains(formals.get(c)) || body.demandedOutputIndices().contains(c))
                    changed |= readOrdinals.add(next.get(c));
        } while (changed);
        var assigned = new HashMap<Integer, CpuGeneralScanIr.Carry>();
        int banks = 0;
        for (int c = 0; c < a.carryCount(); c++) {
            int value = next.get(c);
            if (!readOrdinals.contains(formals.get(c)) && !body.demandedOutputIndices().contains(c)) {
                carries.add(new CpuGeneralScanIr.Carry(CpuGeneralScanIr.Storage.UNUSED, -1));
                continue;
            }
            CpuGeneralScanIr.Carry location = null;
            int formal = formals.indexOf(value);
            if (formal >= a.carryCount() || formal >= 0 && next.get(formal) == value)
                location = new CpuGeneralScanIr.Carry(CpuGeneralScanIr.Storage.SOURCE, formal);
            for (int w = 0; location == null && w < body.demandedOutputIndices().size(); w++) {
                int p = body.demandedOutputIndices().get(w);
                // Prior-carry residuals are not next-value histories by assumption.
                if (p >= a.carryCount() && ports.get(p) == value
                        && (p < a.carryCount() + a.emissionCount() || formal < 0))
                    location = new CpuGeneralScanIr.Carry(CpuGeneralScanIr.Storage.FRAME, actuals.size() + w);
            }
            if (location == null) location = assigned.get(value);
            if (location == null) {
                int w = -1;
                for (int candidate : body.demandedOutputIndices()) {
                    if (candidate < a.carryCount() && next.get(candidate) == value) {
                        w = body.demandedOutputIndices().indexOf(candidate); break;
                    }
                }
                if (w >= 0) location = new CpuGeneralScanIr.Carry(CpuGeneralScanIr.Storage.OUTPUT, actuals.size() + w);
                else if (a.timeCount() > 1 && n > 0)
                    location = new CpuGeneralScanIr.Carry(CpuGeneralScanIr.Storage.SCRATCH, banks++);
                else location = new CpuGeneralScanIr.Carry(CpuGeneralScanIr.Storage.UNUSED, -1);
            }
            carries.add(location); assigned.putIfAbsent(value, location);
        }
        var arithmetic = new CpuKernelIr(irValues, instructions, new CpuKernelIr.Loop("start", "end"), List.of());
        var ir = new CpuGeneralScanIr(arithmetic, formals, actuals, scalars, next, ports,
                body.demandedOutputIndices(), carries, bindings.stream().map(CpuAccessPlan.Binding::plan).toList(),
                a.carryCount(), a.sequenceCount(), a.reverse(), banks);
        long scratch = Math.multiplyExact(Math.multiplyExact(n, banks), width);
        long totalBytes = scratch;
        for (long span : spans) totalBytes = Math.addExact(totalBytes, Math.multiplyExact(span, width));
        return new Facts(ir, new Geometry(a.timeCount(), n, scratch), spans, bindings);
    }

    /**
     * Proves the shape-changing extension independently of the unchanged pointwise proof.
     * All specialized local geometry, stages and resource offsets enter shaped/MATMUL identity;
     * original time remains an invocation fact. No physical storage is allocated here.
     * @param query complete scoped occurrence
     * @return exact pre-slot generation and resource facts
     * @throws IllegalArgumentException if descriptors, algebra, roles or method budgets fail
     * @throws ArithmeticException if geometry overflows
     */
    private static Facts analyzeShaped(OperationCapabilityQuery query) {
        if (query.operation().kind() != ScanKind.SCAN || !(query.operation().attrs() instanceof ScanAttrs a)) throw invalid();
        var body = query.scanBody().orElseThrow(CpuGeneralScanLowering::invalid);
        var graph = body.graph();
        if (query.inputs().isEmpty()) throw invalid();
        DataType type = query.inputs().getFirst().dataType();
        long width = type.byteWidth();
        boolean length = a.lengthAware();
        int formalCount = a.carryCount()+a.sequenceCount()+a.invariantCount();
        if (query.inputs().size() != a.inputCount() || query.outputs().size() != a.outputCount()
                || graph.inputs().size() != formalCount || graph.inputs().size() > 64
                || graph.nodes().size() > 128 || graph.values().size() > 256 || a.outputCount() > 256
                || a.inputCount() + body.demandedOutputIndices().size() > 248
                || body.nextCarry().size() != a.carryCount() || body.emissions().size() != a.emissionCount()
                || body.residuals().size() != a.residualCount()) throw invalid();
        validateInventory(body, a.carryCount());
        var descriptors = new LinkedHashMap<ValueId, TensorDescriptor>();
        for (var value : graph.values()) {
            checkTypeShape(value.descriptor());
            if (value.descriptor().dataType() != type) throw invalid();
            if (value.descriptor().shape().rank() > 16 || descriptors.put(value.id(), value.descriptor()) != null) throw invalid();
        }
        var terminal = new HashSet<ValueId>();
        Map<ValueId,Integer> rowRoles = length ? lengthRows(query,descriptors,terminal) : Map.of();
        var ids = new HashMap<ValueId, Integer>();
        var values = new ArrayList<CpuKernelIr.Value>();
        var views = new ArrayList<CpuGeneralScanIr.View>();
        var instructions = new ArrayList<CpuKernelIr.Instruction>();
        var formals = new ArrayList<Integer>(); var actuals = new ArrayList<Integer>();
        var scalars = new ArrayList<Boolean>(); var spans = new ArrayList<Long>();
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        for (int f = 0; f < graph.inputs().size(); f++) {
            var formal = descriptors.get(graph.inputs().get(f)); var actual = query.inputs().get(f);
            checkTypeShape(actual);
            if (actual.dataType() != type) throw invalid();
            boolean sequence = f >= a.carryCount() && f < a.carryCount() + a.sequenceCount();
            if (!canonical(formal) || !actual.shape().equals(sequence ? stacked(formal.shape(), a.timeCount()) : formal.shape())
                    || formal.requiresGrad() != actual.requiresGrad()) throw invalid();
            boolean scalar = scalarRead(actual);
            addBoundary(actual, CpuAccessPlan.AccessKind.READ, scalar, spans, bindings);
            scalars.add(scalar); actuals.add(f); formals.add(f);
            if (ids.put(graph.inputs().get(f), f) != null) throw invalid();
            values.add(new CpuKernelIr.Value(f, type, CpuKernelIr.Value.Kind.INPUT,
                    access(formal.shape().rank(), CpuAccessPlan.AccessKind.READ, false)));
            views.add(view(f, formal.shape(), LayoutDescriptor.contiguous(formal.shape()).strides()));
        }
        var stages = new ArrayList<CpuGeneralScanIr.Stage>();
        boolean matmulBody = graph.nodes().stream().anyMatch(n -> n.operation().kind() == MatmulKind.MATMUL);
        var derived = new HashMap<ValueId, LayoutDescriptor>();
        graph.inputs().forEach(id -> derived.put(id, descriptors.get(id).layout().orElseThrow()));
        var group = new ArrayList<Integer>(); Shape groupShape = null; int groupRow = -2;
        for (var node : graph.nodes()) {
            if (node.outputs().size() != 1 || node.scanBody().isPresent() || node.inputs().stream().anyMatch(id -> !ids.containsKey(id))) throw invalid();
            var input = node.inputs().stream().map(descriptors::get).toList();
            var output = descriptors.get(node.outputs().getFirst());
            if (output == null || ids.containsKey(node.outputs().getFirst()) || input.isEmpty()
                    || output.requiresGrad() != input.stream().anyMatch(TensorDescriptor::requiresGrad)) throw invalid();
            Object kind = node.operation().kind(); Object attrs = node.operation().attrs();
            int ordinal = values.size(); int first = ids.get(node.inputs().getFirst());
            boolean binary = kind == BinaryArithmeticKind.ADD || kind == BinaryArithmeticKind.SUB
                    || kind == BinaryArithmeticKind.MUL || kind == BinaryArithmeticKind.DIV;
            boolean scalar = kind == ScalarElementwiseKind.ADD || kind == ScalarElementwiseKind.SUB
                    || kind == ScalarElementwiseKind.MUL || kind == ScalarElementwiseKind.DIV;
            boolean unary = kind == UnaryElementwiseKind.NEG || kind == UnaryElementwiseKind.TANH || kind == UnaryElementwiseKind.SIGMOID;
            LayoutDescriptor layout = LayoutDescriptor.contiguous(output.shape());
            CpuGeneralScanIr.View result = view(ordinal, output.shape(), layout.strides());
            if (binary || scalar || unary) {
                Shape expected = CpuPartitionLowering.validatePointwise(node,input,output);
                if (output.layout().isPresent() && !canonical(output)) throw invalid();
                if (groupShape != null && (!groupShape.equals(expected) || length
                        && groupRow != rowRoles.get(node.outputs().getFirst()))) {
                    flush(stages, group); group = new ArrayList<>();
                }
                groupShape = expected; group.add(ordinal);
                if (length) groupRow = rowRoles.get(node.outputs().getFirst());
                instructions.add(new CpuKernelIr.Instruction(CpuPartitionLowering.opcode(node.operation().kind()),
                        node.inputs().stream().map(ids::get).toList(), ordinal, CpuPartitionLowering.scalarImmediate(node)));
            } else {
                flush(stages, group); group = new ArrayList<>(); groupShape = null;
                var before = input.getFirst(); var source = views.get(first);
                if (kind == MatmulKind.MATMUL) {
                    if (input.size() != 2 || attrs != NoOperationAttrs.INSTANCE
                            || output.layout().isPresent() && !canonical(output)) throw invalid();
                    CpuMatmulLowering.validateRankTwo(before.shape(),input.get(1).shape(),output.shape());
                    stages.add(new CpuGeneralScanIr.Stage(List.of(), ordinal, first, List.of(), false,
                            List.of(), ids.get(node.inputs().get(1))));
                } else if (input.size() != 1) throw invalid();
                else if (kind == AggregateReductionKind.SUM) {
                    List<Integer> axes = CpuAggregateLowering.normalizedAxes(before.shape(), output.shape(), attrs);
                    if (output.layout().isPresent() && !canonical(output)) throw invalid();
                    // A same-Shape no-axis SUM is exact raw identity, not an arithmetic site.
                    // Resolve it at the generation binding seam only in the new MATMUL domain;
                    // old shaped-v1 topology, keys and bytes are deliberately unchanged.
                    if (matmulBody && axes.isEmpty() && before.shape().equals(output.shape())
                            && source.strides().equals(longs(layout.strides()))) {
                        ids.put(node.outputs().getFirst(),first); derived.put(node.outputs().getFirst(),layout); continue;
                    }
                    stages.add(new CpuGeneralScanIr.Stage(List.of(), ordinal, first, axes, true));
                } else if (kind == ContiguousKind.CONTIGUOUS) {
                    if (attrs != NoOperationAttrs.INSTANCE || !before.shape().equals(output.shape())
                            || output.layout().isPresent() && !canonical(output)) throw invalid();
                    if (source.strides().equals(longs(layout.strides()))) result = source;
                    else {
                        // CONTIGUOUS is exact represented movement. Its same-coordinate load can
                        // share the next scalar loop; only a later reshape/reduction/cross-stage
                        // consumer makes the canonical result a materialized cut.
                        instructions.add(new CpuKernelIr.Instruction(CpuPointwiseOpcode.CAST,List.of(first),ordinal));
                        group.add(ordinal); groupShape = output.shape();
                        if (length) groupRow = rowRoles.get(node.outputs().getFirst());
                    }
                } else {
                    long[] stride = source.strides().stream().mapToLong(Long::longValue).toArray();
                    Shape expected;
                    if (kind == ShapeTransformKind.EXPAND && attrs instanceof TargetShapeAttrs target) {
                        expected = target.targetShape();
                        if (!ShapeBroadcast.broadcast(before.shape(), expected).equals(expected) || expected.rank() < before.shape().rank()) throw invalid();
                        long[] expanded = new long[expected.rank()]; int leading = expected.rank() - before.shape().rank();
                        for (int axis = leading; axis < expanded.length; axis++)
                            expanded[axis] = before.shape().dimension(axis-leading).equals(expected.dimension(axis)) ? stride[axis-leading] : 0;
                        stride = expanded;
                    } else if (kind == ShapeTransformKind.RESHAPE && attrs instanceof TargetShapeAttrs target) {
                        expected = target.targetShape();
                        if (before.shape().knownElementCount().orElseThrow() != expected.knownElementCount().orElseThrow()) throw invalid();
                        // Storage-compatible dense views and all-zero splats need no reorder.
                        if (Arrays.equals(stride, LayoutDescriptor.contiguous(before.shape()).strides())) stride = LayoutDescriptor.contiguous(expected).strides();
                        else if (Arrays.stream(stride).allMatch(n -> n == 0)) stride = new long[expected.rank()];
                        else throw invalid();
                    } else if (kind == AxisTransformKind.PERMUTE && attrs instanceof PermutationAttrs permutation
                            && before.shape().rank() == 2 && permutation.axes().equals(List.of(1, 0))) {
                        expected = Shape.of(before.shape().toLongArray()[1], before.shape().toLongArray()[0]);
                        stride = new long[] {stride[1], stride[0]};
                    } else if ((kind == AxisTransformKind.EXPAND_DIMS || kind == AxisTransformKind.SQUEEZE)
                            && attrs instanceof AxisTransformAttrs axisAttrs) {
                        int axis = axisAttrs.axis(); var shape = new ArrayList<>(before.shape().dimensions());
                        var strides = new ArrayList<>(longs(stride));
                        if (kind == AxisTransformKind.EXPAND_DIMS) {
                            if (axis < 0 || axis > shape.size()) throw invalid();
                            strides.add(axis, axis == shape.size() ? 1L : Math.multiplyExact(stride[axis], before.shape().toLongArray()[axis]));
                            shape.add(axis, new StaticDimension(1));
                        } else {
                            if (axis < 0 || axis >= shape.size() || !shape.get(axis).equals(new StaticDimension(1))) throw invalid();
                            strides.remove(axis); shape.remove(axis);
                        }
                        expected = Shape.ofDimensions(shape.toArray(Dimension[]::new));
                        stride = strides.stream().mapToLong(Long::longValue).toArray();
                    } else if (kind == io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind.CAST
                            && attrs instanceof io.github.pho001.synaptik.model.operation.elementwise.cast.CastAttrs cast
                            && cast.targetDataType() == type) {
                        expected = before.shape();
                    } else throw invalid();
                    if (!expected.equals(output.shape())) throw invalid();
                    layout = LayoutDescriptor.of(expected, stride, 0, true);
                    result = view(source.origin(), expected, stride);
                }
            }
            if (output.layout().isPresent() && !output.layout().orElseThrow().equals(layout)) throw invalid();
            derived.put(node.outputs().getFirst(), layout);
            values.add(new CpuKernelIr.Value(ordinal, type, CpuKernelIr.Value.Kind.VIRTUAL,
                    access(output.shape().rank(), CpuAccessPlan.AccessKind.READ, false)));
            views.add(result); ids.put(node.outputs().getFirst(), ordinal);
        }
        flush(stages, group);
        if (ids.size() != descriptors.size()) throw invalid();
        for (int c = 0; c < a.carryCount(); c++)
            if (!descriptors.get(graph.inputs().get(c)).shape().equals(descriptors.get(body.nextCarry().get(c)).shape())) throw invalid();
        var roles = new ArrayList<>(body.nextCarry()); roles.addAll(body.emissions()); roles.addAll(body.residuals());
        boolean grad = query.inputs().stream().anyMatch(TensorDescriptor::requiresGrad);
        for (int p = 0; p < roles.size(); p++) {
            var local = descriptors.get(roles.get(p)); var outer = query.outputs().get(p);
            if (local == null || outer.dataType() != type || outer.requiresGrad() != grad || !canonical(outer)
                    || !outer.shape().equals(p < a.carryCount() ? local.shape() : stacked(local.shape(), a.timeCount()))) throw invalid();
        }
        if (length) {
            var lengths = query.inputs().getLast();
            addBoundary(lengths,CpuAccessPlan.AccessKind.READ,false,spans,bindings); scalars.add(false);
        }
        for (int p : body.demandedOutputIndices()) addBoundary(query.outputs().get(p), CpuAccessPlan.AccessKind.WRITE, false, spans, bindings);
        var next = body.nextCarry().stream().map(ids::get).toList(); var ports = roles.stream().map(ids::get).toList();
        var arithmetic = new CpuKernelIr(values, instructions, new CpuKernelIr.Loop("start", "end"), List.of());
        var activity = new ArrayList<>(Collections.nCopies(length ? views.size() : 0,-1));
        if (length) ids.forEach((id,ordinal) -> activity.set(ordinal,rowRoles.get(id)));
        stages = fuseMatmulStages(stages, views, arithmetic,activity);
        stages = fuseReductionStages(stages, views, arithmetic,activity);
        var rowRegion = rowRegion(views,stages,arithmetic,activity,next,ports,body.demandedOutputIndices(),a.carryCount(),a.sequenceCount(),formals.size(),scalars);
        var plan = shapedResources(views, stages, arithmetic, next, ports, body.demandedOutputIndices(), a.carryCount(), a.sequenceCount(), formals.size(),length,rowRegion);
        if (length) {
            var rows = new ArrayList<>(Collections.nCopies(views.size(),-1));
            var terminals = new ArrayList<>(Collections.nCopies(views.size(),false));
            ids.forEach((id,ordinal) -> { rows.set(ordinal,rowRoles.get(id)); terminals.set(ordinal,terminal.contains(id)); });
            plan = new CpuGeneralScanIr.Stages(plan.views(),plan.order(),plan.cuts(),plan.carryOffsets(),plan.carryPorts(),plan.scratchBytes(),
                    plan.carryOffsetFactors(),plan.scratchBytesPerBatch(),false,List.of(),rows,terminals,a.rowCarryCount(),formalCount,
                    query.inputs().getFirst().shape().dimension(0).staticSize().orElseThrow(),plan.rowMapReductionIr());
        }
        // Fail closed before finalization: conservative bytecode/local bounds include every
        // affine operand, producing store, nested loop and publication, not merely node count.
        long estimate = 1024L + 32L*views.size(), localBudget = 280L + (type == DataType.FLOAT64 ? 7L : 6L)*views.size();
        for (var view : views) for (long stride : view.strides()) Math.multiplyExact(stride,width);
        for (var stage : plan.order()) {
            var outputs = new ArrayList<>(stage.values()); outputs.addAll(stage.epilogue());
            for (int out : outputs) {
                var op = instructions.stream().filter(i -> i.output() == out).findFirst().orElse(null);
                if (op == null) {
                    if (rowRegion.isEmpty()) throw invalid();
                    estimate += 1800; localBudget += 16; continue;
                }
                estimate += 180L + 40L*views.get(out).shape().size();
                for (int in : op.inputs()) estimate += 40L*views.get(in).shape().size();
                if (op.opcode() == CpuPointwiseOpcode.SIGMOID) estimate += 200;
            }
            int rank = views.get(stage.input() >= 0 ? stage.input() : stage.values().getFirst()).shape().size();
            estimate += 300L+120L*rank; localBudget += 12L+8L*rank;
            if (stage.right() >= 0) { estimate += 1600; localBudget += 40; }
        }
        for (var cut : plan.cuts()) estimate += 100L+40L*views.get(cut.value()).shape().size();
        for (int p : body.demandedOutputIndices()) { int rank = views.get(ports.get(p)).shape().size();
            estimate += 180L+80L*rank; localBudget += 4L+2L*rank; }
        for (int c = 0; c < a.carryCount(); c++) { estimate += 500L+80L*views.get(c).shape().size(); localBudget += 40; }
        if (length) {
            estimate += 1500L+100L*views.size()+200L*body.demandedOutputIndices().size()+400L*a.carryCount();
            localBudget += 100L+12L*body.demandedOutputIndices().size()+8L*views.size();
        }
        if (estimate > 48_000 || localBudget > 60_000) throw invalid();
        int scratch = plan.scratchBytes() > 0 ? 1 : 0;
        var ir = new CpuGeneralScanIr(arithmetic, formals, actuals, scalars, next, ports, body.demandedOutputIndices(),
                Collections.nCopies(a.carryCount(), new CpuGeneralScanIr.Carry(CpuGeneralScanIr.Storage.UNUSED, -1)),
                bindings.stream().map(CpuAccessPlan.Binding::plan).toList(), a.carryCount(), a.sequenceCount(), a.reverse(), scratch, Optional.of(plan));
        long total = plan.scratchBytes();
        for (int p = 0; p < spans.size(); p++) total = Math.addExact(total, Math.multiplyExact(spans.get(p),(long)ir.boundaryType(p).byteWidth()));
        return new Facts(ir, new Geometry(a.timeCount(), descriptors.get(graph.inputs().getFirst()).shape().knownElementCount().orElseThrow(), plan.scratchBytes()), spans, bindings);
    }

    /**
     * Proves the bounded symbolic algebra using dimension identity and affine coefficient
     * arithmetic only. A zero constant coefficient never stands for an evaluated B=0 shape.
     * Static lowering remains a separate path so its decisions, keys and bytes are unchanged.
     * @param query complete body-bearing symbolic occurrence
     * @return immutable topology and exact numeric recipes before slot assignment
     * @throws IllegalArgumentException if any role, named coordinate, layout or algebra is unsupported
     * @throws ArithmeticException if structural coefficients overflow
     */
    private static Facts analyzeSymbolic(OperationCapabilityQuery query) {
        if (query.operation().kind() != ScanKind.SCAN || !(query.operation().attrs() instanceof ScanAttrs a)
                || query.inputs().size() != a.inputCount()
                || query.outputs().size() != a.outputCount()) throw invalid();
        var body = query.scanBody().orElseThrow(CpuGeneralScanLowering::invalid); var graph = body.graph();
        boolean length = a.lengthAware();
        DataType type = query.inputs().getFirst().dataType(); int width = type.byteWidth();
        if (type != DataType.FLOAT32 && (!length || type != DataType.FLOAT64)) throw invalid();
        if (graph.inputs().size() != a.carryCount()+a.sequenceCount()+a.invariantCount() || graph.inputs().size() > 64
                || graph.nodes().size() > 128 || graph.values().size() > 256 || a.outputCount() > 256
                || a.inputCount()+body.demandedOutputIndices().size() > 248
                || body.nextCarry().size() != a.carryCount() || body.emissions().size() != a.emissionCount()
                || body.residuals().size() != a.residualCount()) throw invalid();
        validateInventory(body,a.carryCount());
        var descriptors = new LinkedHashMap<ValueId,TensorDescriptor>(); var symbols = new HashSet<String>();
        var all = new ArrayList<TensorDescriptor>(query.inputs()); all.addAll(query.outputs());
        graph.values().forEach(v -> all.add(v.descriptor()));
        for (var d : all) {
            if (d != (length ? query.inputs().getLast() : null) && d.dataType() != type || d.shape().rank() > 3) throw invalid();
            long fixed = 1;
            int named = 0;
            for (var dimension : d.shape().dimensions()) {
                if (dimension instanceof DynamicDimension n) { symbols.add(n.symbol()); named++; }
                else if (dimension instanceof StaticDimension s && s.size() <= Integer.MAX_VALUE) {
                    if (s.size() != 0) fixed = Math.multiplyExact(fixed,s.size());
                } else throw invalid();
            }
            if (named > 1 || !d.shape().isFullyStatic() && d.layout().isPresent()) throw invalid();
            CpuPartitionLowering.batchCanonical(d,CpuAccessPlan.AccessKind.READ);
        }
        if (symbols.size() != 1) throw invalid();
        graph.values().forEach(v -> { if (descriptors.put(v.id(),v.descriptor()) != null) throw invalid(); });
        var terminal = new HashSet<ValueId>();
        Map<ValueId,Integer> rowRoles = length ? lengthRows(query,descriptors,terminal) : Map.of();
        var ids = new LinkedHashMap<ValueId,Integer>(); var values = new ArrayList<CpuKernelIr.Value>();
        var views = new ArrayList<CpuGeneralScanIr.View>(); var instructions = new ArrayList<CpuKernelIr.Instruction>();
        var formals = new ArrayList<Integer>(); var actuals = new ArrayList<Integer>();
        var scalars = new ArrayList<Boolean>(); var boundaries = new ArrayList<CpuAccessPlan.BatchBinding>();
        for (int f = 0; f < graph.inputs().size(); f++) {
            var formal = descriptors.get(graph.inputs().get(f)); var actual = query.inputs().get(f);
            boolean sequence = f >= a.carryCount() && f < a.carryCount()+a.sequenceCount();
            if (!symbolicCanonical(formal) || !actual.shape().equals(sequence ? stacked(formal.shape(),a.timeCount()) : formal.shape())
                    || formal.requiresGrad() != actual.requiresGrad()) throw invalid();
            boolean scalar = false;
            if (actual.shape().isFullyStatic()) {
                try { scalar = scalarRead(actual); }
                catch (IllegalArgumentException notCanonical) {
                    var layout = actual.layout().orElseThrow(CpuGeneralScanLowering::invalid);
                    if (actual.shape().rank() != 2 || layout.storageOffset() != 0
                            || !Arrays.equals(layout.strides(),new long[] {1,actual.shape().toLongArray()[0]})) throw invalid();
                }
            }
            var binding = CpuPartitionLowering.batchCanonical(actual,CpuAccessPlan.AccessKind.READ);
            if (actual.layout().isPresent()) binding = CpuPartitionLowering.batchAccess(CpuAccessPlan.AccessKind.READ,binding.extentConstants(),
                    binding.extentBatchFactors(),longs(actual.layout().orElseThrow().strides()),Collections.nCopies(actual.shape().rank(),0L));
            if (scalar) binding = CpuPartitionLowering.batchAccess(CpuAccessPlan.AccessKind.READ,binding.extentConstants(),
                    binding.extentBatchFactors(),Collections.nCopies(actual.shape().rank(),0L),Collections.nCopies(actual.shape().rank(),0L));
            boundaries.add(binding); scalars.add(scalar); actuals.add(f); formals.add(f);
            var recipe = CpuPartitionLowering.batchCanonical(formal,CpuAccessPlan.AccessKind.READ);
            views.add(symbolicView(f,recipe)); ids.put(graph.inputs().get(f),f);
            values.add(new CpuKernelIr.Value(f,type,CpuKernelIr.Value.Kind.INPUT,recipe.plan()));
        }
        var stages = new ArrayList<CpuGeneralScanIr.Stage>(); var group = new ArrayList<Integer>(); Shape groupShape = null; int groupRow = -2;
        for (var node : graph.nodes()) {
            if (node.outputs().size() != 1 || node.scanBody().isPresent() || node.inputs().isEmpty()
                    || node.inputs().stream().anyMatch(id -> !ids.containsKey(id))) throw invalid();
            var input = node.inputs().stream().map(descriptors::get).toList();
            var output = descriptors.get(node.outputs().getFirst()); int ordinal = values.size();
            if (output == null || ids.containsKey(node.outputs().getFirst())
                    || output.requiresGrad() != input.stream().anyMatch(TensorDescriptor::requiresGrad)) throw invalid();
            int first = ids.get(node.inputs().getFirst()); var source = views.get(first);
            Object kind = node.operation().kind(), attrs = node.operation().attrs();
            var recipe = CpuPartitionLowering.batchCanonical(output,CpuAccessPlan.AccessKind.READ);
            var view = symbolicView(ordinal,recipe);
            boolean binary = kind == BinaryArithmeticKind.ADD || kind == BinaryArithmeticKind.SUB || kind == BinaryArithmeticKind.MUL || kind == BinaryArithmeticKind.DIV;
            boolean scalar = kind == ScalarElementwiseKind.ADD || kind == ScalarElementwiseKind.SUB || kind == ScalarElementwiseKind.MUL || kind == ScalarElementwiseKind.DIV;
            boolean unary = kind == UnaryElementwiseKind.NEG || kind == UnaryElementwiseKind.TANH || kind == UnaryElementwiseKind.SIGMOID;
            boolean copy = kind == ContiguousKind.CONTIGUOUS || kind == io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind.CAST;
            if (binary || scalar || unary || copy) {
                var expected = CpuPartitionLowering.validatePointwise(node,input,output);
                if (output.layout().isPresent() && !canonical(output)) throw invalid();
                if (groupShape != null && (!groupShape.equals(expected) || length && groupRow != rowRoles.get(node.outputs().getFirst()))) { flush(stages,group); group = new ArrayList<>(); }
                groupShape = expected; group.add(ordinal);
                if (length) groupRow = rowRoles.get(node.outputs().getFirst());
                instructions.add(new CpuKernelIr.Instruction(copy ? CpuPointwiseOpcode.CAST : CpuPartitionLowering.opcode(node.operation().kind()),
                        node.inputs().stream().map(ids::get).toList(),ordinal,copy ? null : CpuPartitionLowering.scalarImmediate(node)));
            } else {
                flush(stages,group); group = new ArrayList<>(); groupShape = null;
                var before = input.getFirst();
                if (kind == MatmulKind.MATMUL) {
                    if (input.size() != 2 || attrs != NoOperationAttrs.INSTANCE
                            || output.layout().isPresent() && !canonical(output)) throw invalid();
                    CpuMatmulLowering.validateRankTwo(before.shape(),input.get(1).shape(),output.shape());
                    stages.add(new CpuGeneralScanIr.Stage(List.of(),ordinal,first,List.of(),false,List.of(),ids.get(node.inputs().get(1))));
                } else if (kind == AggregateReductionKind.SUM && input.size() == 1) {
                    var axes = CpuAggregateLowering.normalizedAxes(before.shape(),output.shape(),attrs);
                    if (output.layout().isPresent() && !canonical(output)) throw invalid();
                    if (axes.isEmpty() && before.shape().equals(output.shape())
                            && source.batch().orElseThrow().strideConstants().equals(recipe.strideConstants())
                            && source.batch().orElseThrow().strideBatchFactors().equals(recipe.strideBatchFactors())) {
                        // Proved no-axis SUM-to-same-shape is represented identity, not a
                        // producer/copy stage. It must not split a product from its scalar tail.
                        ids.put(node.outputs().getFirst(),first); continue;
                    }
                    stages.add(new CpuGeneralScanIr.Stage(List.of(),ordinal,first,axes,true,List.of(),-1,attrs instanceof SumToShapeAttrs));
                } else if (input.size() == 1) {
                    var mapping = source.batch().orElseThrow(); var sc = new ArrayList<Long>(); var sf = new ArrayList<Long>();
                    var originAxes = new ArrayList<Integer>();
                    Shape expected;
                    if (kind == AxisTransformKind.PERMUTE && attrs instanceof PermutationAttrs p
                            && before.shape().rank() == 2 && p.axes().equals(List.of(1,0))) {
                        expected = Shape.ofDimensions(before.shape().dimension(1),before.shape().dimension(0));
                        sc.add(mapping.strideConstants().get(1)); sc.add(mapping.strideConstants().get(0));
                        sf.add(mapping.strideBatchFactors().get(1)); sf.add(mapping.strideBatchFactors().get(0));
                        originAxes.add(source.originAxes().get(1)); originAxes.add(source.originAxes().get(0));
                    } else if (kind == ShapeTransformKind.EXPAND && attrs instanceof TargetShapeAttrs target) {
                        expected = target.targetShape();
                        if (!ShapeBroadcast.broadcast(before.shape(),expected).equals(expected) || expected.rank() < before.shape().rank()) throw invalid();
                        int shift = expected.rank()-before.shape().rank();
                        for (int axis = 0; axis < expected.rank(); axis++) {
                            int from = axis-shift; boolean broadcast = from < 0 || before.shape().dimension(from).equals(new StaticDimension(1));
                            sc.add(broadcast ? 0L : mapping.strideConstants().get(from)); sf.add(broadcast ? 0L : mapping.strideBatchFactors().get(from));
                            originAxes.add(broadcast ? -1 : source.originAxes().get(from));
                        }
                    } else if ((kind == AxisTransformKind.EXPAND_DIMS || kind == AxisTransformKind.SQUEEZE) && attrs instanceof AxisTransformAttrs axisAttrs) {
                        var dims = new ArrayList<>(before.shape().dimensions()); sc.addAll(mapping.strideConstants()); sf.addAll(mapping.strideBatchFactors());
                        originAxes.addAll(source.originAxes());
                        int axis = axisAttrs.axis();
                        if (kind == AxisTransformKind.EXPAND_DIMS) {
                            if (axis < 0 || axis > dims.size()) throw invalid();
                            long c = 1, f = 0;
                            if (axis < dims.size()) {
                                long ec = mapping.extentConstants().get(axis), ef = mapping.extentBatchFactors().get(axis);
                                c = Math.multiplyExact(ec,sc.get(axis));
                                if (ef != 0 && sf.get(axis) != 0) throw invalid();
                                f = Math.addExact(Math.multiplyExact(ec,sf.get(axis)),Math.multiplyExact(ef,sc.get(axis)));
                            }
                            dims.add(axis,new StaticDimension(1)); sc.add(axis,c); sf.add(axis,f);
                            originAxes.add(axis,-1);
                        } else {
                            if (axis < 0 || axis >= dims.size() || !dims.get(axis).equals(new StaticDimension(1))) throw invalid();
                            dims.remove(axis); sc.remove(axis); sf.remove(axis);
                            originAxes.remove(axis);
                        }
                        expected = Shape.ofDimensions(dims.toArray(Dimension[]::new));
                    } else if (kind == ShapeTransformKind.RESHAPE && attrs instanceof TargetShapeAttrs target) {
                        expected = target.targetShape();
                        var oldAxes = before.shape().dimensions().stream().filter(d -> !d.equals(new StaticDimension(1))).toList();
                        var newAxes = expected.dimensions().stream().filter(d -> !d.equals(new StaticDimension(1))).toList();
                        var canonical = CpuPartitionLowering.batchCanonical(before,CpuAccessPlan.AccessKind.READ);
                        if (!oldAxes.equals(newAxes) || !mapping.strideConstants().equals(canonical.strideConstants())
                                || !mapping.strideBatchFactors().equals(canonical.strideBatchFactors())) throw invalid();
                        sc.addAll(recipe.strideConstants()); sf.addAll(recipe.strideBatchFactors());
                        int from = 0;
                        for (var dimension : expected.dimensions()) {
                            if (dimension.equals(new StaticDimension(1))) originAxes.add(-1);
                            else {
                                while (before.shape().dimension(from).equals(new StaticDimension(1))) from++;
                                originAxes.add(source.originAxes().get(from++));
                            }
                        }
                    } else throw invalid();
                    if (!expected.equals(output.shape())) throw invalid();
                    recipe = CpuPartitionLowering.batchAccess(CpuAccessPlan.AccessKind.READ,recipe.extentConstants(),recipe.extentBatchFactors(),sc,sf);
                    if (output.layout().isPresent() && !output.layout().orElseThrow().equals(
                            LayoutDescriptor.of(output.shape(),sc.stream().mapToLong(Long::longValue).toArray(),0,true))) throw invalid();
                    view = new CpuGeneralScanIr.View(source.origin(),recipe.extentConstants(),recipe.strideConstants(),Optional.of(recipe),originAxes,Optional.empty());
                } else throw invalid();
            }
            values.add(new CpuKernelIr.Value(ordinal,type,CpuKernelIr.Value.Kind.VIRTUAL,recipe.plan()));
            views.add(view); ids.put(node.outputs().getFirst(),ordinal);
        }
        flush(stages,group);
        if (ids.size() != descriptors.size()) throw invalid();
        for (int c = 0; c < a.carryCount(); c++) if (!descriptors.get(graph.inputs().get(c)).shape().equals(descriptors.get(body.nextCarry().get(c)).shape())) throw invalid();
        var roles = new ArrayList<>(body.nextCarry()); roles.addAll(body.emissions()); roles.addAll(body.residuals());
        boolean grad = query.inputs().stream().anyMatch(TensorDescriptor::requiresGrad);
        for (int p = 0; p < roles.size(); p++) {
            var local = descriptors.get(roles.get(p)); var outer = query.outputs().get(p);
            if (local == null || !symbolicCanonical(outer) || outer.requiresGrad() != grad
                    || !outer.shape().equals(p < a.carryCount() ? local.shape() : stacked(local.shape(),a.timeCount()))) throw invalid();
        }
        if (length) { boundaries.add(CpuPartitionLowering.batchCanonical(query.inputs().getLast(),CpuAccessPlan.AccessKind.READ)); scalars.add(false); }
        for (int p : body.demandedOutputIndices()) boundaries.add(CpuPartitionLowering.batchCanonical(query.outputs().get(p),CpuAccessPlan.AccessKind.WRITE));
        var next = body.nextCarry().stream().map(ids::get).toList(); var ports = roles.stream().map(ids::get).toList();
        var arithmetic = new CpuKernelIr(values,instructions,new CpuKernelIr.Loop("start","end"),List.of());
        var activity = new ArrayList<>(Collections.nCopies(length ? views.size() : 0,-1));
        if (length) ids.forEach((id,ordinal) -> activity.set(ordinal,rowRoles.get(id)));
        if (!length) stages = joinSymbolicScalarStages(stages,views,arithmetic);
        stages = fuseMatmulStages(stages,views,arithmetic,activity);
        stages = fuseReductionStages(stages,views,arithmetic,activity,length);
        var rowRegion = rowRegion(views,stages,arithmetic,activity,next,ports,body.demandedOutputIndices(),a.carryCount(),a.sequenceCount(),formals.size(),scalars);
        var resources = shapedResources(views,stages,arithmetic,next,ports,body.demandedOutputIndices(),a.carryCount(),a.sequenceCount(),formals.size(),length,rowRegion);
        var terminals = new ArrayList<>(Collections.nCopies(length ? views.size() : 0,false));
        if (length) ids.forEach((id,ordinal) -> terminals.set(ordinal,terminal.contains(id)));
        var plan = new CpuGeneralScanIr.Stages(resources.views(),resources.order(),resources.cuts(),resources.carryOffsets(),resources.carryPorts(),
                resources.scratchBytes(),resources.carryOffsetFactors(),resources.scratchBytesPerBatch(),true,
                java.util.stream.IntStream.range(a.carryCount(),a.carryCount()+a.sequenceCount()).mapToObj(f -> {
                    var actual = boundaries.get(f);
                    return new BatchByteSize(Math.multiplyExact(actual.strideConstants().getFirst(),width),Math.multiplyExact(actual.strideBatchFactors().getFirst(),width),width);
                }).toList(),activity,terminals,a.rowCarryCount(),length ? formals.size() : -1,0,resources.rowMapReductionIr());
        long estimate = 2000L + 300L*values.size()+1000L*stages.size()+300L*body.demandedOutputIndices().size();
        if (estimate > 48000) throw invalid();
        var ir = new CpuGeneralScanIr(arithmetic,formals,actuals,scalars,next,ports,body.demandedOutputIndices(),
                Collections.nCopies(a.carryCount(),new CpuGeneralScanIr.Carry(CpuGeneralScanIr.Storage.UNUSED,-1)),
                boundaries.stream().map(CpuAccessPlan.BatchBinding::plan).toList(),a.carryCount(),a.sequenceCount(),a.reverse(),1,Optional.of(plan));
        return new Facts(ir,new Geometry(a.timeCount(),OptionalLong.empty(),plan.workspaceSize(width)),List.of(),List.of(),boundaries);
    }

    /**
     * Retains the named-axis recipe alongside its constant projection, never a sampled shape.
     * @param origin ultimate storage-producing ordinal
     * @param recipe exact affine access
     * @return immutable symbolic view
     */
    private static CpuGeneralScanIr.View symbolicView(int origin,CpuAccessPlan.BatchBinding recipe) {
        return new CpuGeneralScanIr.View(origin,recipe.extentConstants(),recipe.strideConstants(),Optional.of(recipe),
                java.util.stream.IntStream.range(0,recipe.plan().iterationRank()).boxed().toList(),Optional.empty());
    }

    /**
     * Checks unresolved symbolic formals/results without fabricating numeric layouts.
     * @param descriptor boundary under proof
     * @return whether it has the required static canonical or unresolved symbolic layout
     */
    private static boolean symbolicCanonical(TensorDescriptor descriptor) {
        return descriptor.shape().isFullyStatic() ? canonical(descriptor) : descriptor.layout().isEmpty();
    }

    /**
     * Snapshots specialized affine geometry without retaining a descriptor or mutable array.
     * @param origin backing ordinal
     * @param shape logical shape
     * @param strides element strides
     * @return immutable access geometry
     */
    private static CpuGeneralScanIr.View view(int origin, Shape shape, long[] strides) {
        return new CpuGeneralScanIr.View(origin, longs(shape.toLongArray()), longs(strides));
    }

    /**
     * Snapshots primitive specialized geometry into immutable generation facts.
     * @param values primitive geometry, not mutated
     * @return immutable boxed geometry, used only cold
     */
    private static List<Long> longs(long[] values) { return Arrays.stream(values).boxed().toList(); }

    /**
     * Seals a nonempty same-domain arithmetic group in generation order.
     * @param stages cold stage accumulator, appended to
     * @param group ordered scalar output ordinals, snapshotted without mutation
     */
    private static void flush(List<CpuGeneralScanIr.Stage> stages, List<Integer> group) {
        if (!group.isEmpty()) stages.add(new CpuGeneralScanIr.Stage(group, -1, -1, List.of(), false));
    }


    /**
     * Keeps same-cell consumers in a product's output loop without crossing its represented
     * result or losing any demanded/fan-out cut. Cross-coordinate views and later products
     * retain a complete stage boundary; no operand or contraction site is recomputed.
     * @param stages original topological stages
     * @param views proved affine accesses
     * @param arithmetic sole typed scalar inventory
     * @return derived stages with only proved same-cell product epilogues fused
     */
    private static ArrayList<CpuGeneralScanIr.Stage> fuseMatmulStages(List<CpuGeneralScanIr.Stage> stages,
            List<CpuGeneralScanIr.View> views, CpuKernelIr arithmetic) {
        return fuseMatmulStages(stages,views,arithmetic,List.of());
    }

    /**
     * Retains the legacy proof and additionally requires every length-mode epilogue output to
     * share the product's activity domain. Coincident feature/batch shapes prove no such fact.
     * @param stages qualified topological stages
     * @param views ordinal affine views
     * @param arithmetic represented scalar inventory
     * @param activity explicit ordinal row axes, or empty for unchanged legacy fusion
     * @return fused stages without moving SHARED sites into a row gate
     */
    private static ArrayList<CpuGeneralScanIr.Stage> fuseMatmulStages(List<CpuGeneralScanIr.Stage> stages,
            List<CpuGeneralScanIr.View> views, CpuKernelIr arithmetic,List<Integer> activity) {
        var result = new ArrayList<>(stages); var ops = new HashMap<Integer,CpuKernelIr.Instruction>();
        arithmetic.instructions().forEach(i -> ops.put(i.output(),i));
        for (int s = 0; s+1 < result.size(); s++) {
            var product = result.get(s); var tail = result.get(s+1);
            if (product.right() < 0 || tail.output() >= 0
                    || !sameActivity(tail.values(),product.output(),activity)
                    || !sameCellEpilogue(tail,product.output(),List.of(),views,ops)) continue;
            result.set(s,new CpuGeneralScanIr.Stage(product.values(),product.output(),product.input(),product.axes(),false,tail.values(),product.right()));
            result.remove(s+1);
        }
        return result;
    }

    /**
     * Places same-domain producers directly inside the following reduction traversal. Independent
     * later scalar groups may join only when every dependency is already available. This removes
     * contribution arrays/reload loops, not producing sites: demanded and fan-out values retain
     * exact cuts at their typed instructions.
     * @param stages original topological generation groups
     * @param views exact local access proofs
     * @param arithmetic sole scalar instruction inventory
     * @return topologically valid fused generation stages
     */
    private static ArrayList<CpuGeneralScanIr.Stage> fuseReductionStages(List<CpuGeneralScanIr.Stage> stages,
            List<CpuGeneralScanIr.View> views, CpuKernelIr arithmetic) {
        return fuseReductionStages(stages,views,arithmetic,List.of());
    }

    /**
     * Fuses only activity-compatible contributors and tails in length mode. A later independent
     * SHARED group may equal a ROW shape numerically; moving it into a contributor gate would
     * skip its feature cells and corrupt saved histories or later shared consumers.
     * @param stages qualified topological stages
     * @param views complete affine views
     * @param arithmetic represented scalar inventory
     * @param activity explicit ordinal row axes, or empty for unchanged legacy fusion
     * @return stages retaining each site's original row or active-time domain
     */
    private static ArrayList<CpuGeneralScanIr.Stage> fuseReductionStages(List<CpuGeneralScanIr.Stage> stages,
            List<CpuGeneralScanIr.View> views, CpuKernelIr arithmetic,List<Integer> activity) {
        return fuseReductionStages(stages,views,arithmetic,activity,false);
    }

    /**
     * Separates combined-mode same-cell tail fusion from contributor-producer fusion. A SUM
     * whose input was already materialized may still keep its typed result in a local for one
     * distinct carry ADD. Resource liveness subsequently retains demanded, saved and fan-out
     * cuts, so eliminating an unobservable cut never removes an observable producing store.
     * @param stages qualified topological stages
     * @param views exact local affine/domain proofs
     * @param arithmetic sole represented scalar instruction inventory
     * @param activity explicit row/SHARED execution domains
     * @param independentTails whether combined symbolic-length mode alone may fuse a SUM tail
     *     without requiring a preceding contributor-producer group
     * @return valid stages preserving typed sequential SUM and its separate typed tail sites
     */
    private static ArrayList<CpuGeneralScanIr.Stage> fuseReductionStages(List<CpuGeneralScanIr.Stage> stages,
            List<CpuGeneralScanIr.View> views, CpuKernelIr arithmetic,List<Integer> activity,boolean independentTails) {
        var result = new ArrayList<>(stages);
        var ops = new HashMap<Integer,CpuKernelIr.Instruction>(); arithmetic.instructions().forEach(i -> ops.put(i.output(),i));
        for (int s = 1; s < result.size(); s++) {
            var sum = result.get(s); var producer = result.get(s-1);
            if (!sum.reduction() || sum.axes().isEmpty() || producer.output() >= 0
                    || !producer.values().contains(sum.input()) || !sameActivity(producer.values(),sum.input(),activity)) continue;
            var shape = views.get(sum.input()).shape();
            if (!sameDomain(views.get(producer.values().getFirst()),views.get(sum.input()))) continue;
            var fused = new ArrayList<>(producer.values());
            result.remove(s-1); s--;
            var unavailable = new HashSet<Integer>(); unavailable.add(sum.output());
            for (int later = s+1; later < result.size(); later++) {
                var candidate = result.get(later);
                boolean safe = candidate.output() < 0 && sameDomain(views.get(candidate.values().getFirst()),views.get(sum.input()))
                        && sameActivity(candidate.values(),sum.input(),activity);
                if (safe) for (int value : candidate.values()) for (int input : ops.get(value).inputs()) {
                    int origin = views.get(input).origin();
                    if (unavailable.contains(origin) || fused.contains(origin) && input != origin) safe = false;
                }
                if (safe) { fused.addAll(candidate.values()); result.remove(later--); }
                else {
                    if (candidate.output() >= 0) unavailable.add(candidate.output());
                    unavailable.addAll(candidate.values());
                    unavailable.addAll(candidate.epilogue());
                }
            }
            result.set(s,new CpuGeneralScanIr.Stage(fused,sum.output(),sum.input(),sum.axes(),true,List.of(),-1,sum.sumToShape()));
            if (s+1 < result.size()) {
                var tail = result.get(s+1);
                if (tail.output() < 0 && sameActivity(tail.values(),sum.output(),activity)
                        && sameCellEpilogue(tail, sum.output(), fused, views, ops)) {
                    result.set(s,new CpuGeneralScanIr.Stage(fused,sum.output(),sum.input(),sum.axes(),true,tail.values(),-1,sum.sumToShape()));
                    result.remove(s+1);
                }
            }
        }
        if (independentTails) for (int s = 0; s+1 < result.size(); s++) {
            var sum = result.get(s); var tail = result.get(s+1);
            if (sum.reduction() && !sum.axes().isEmpty() && sum.epilogue().isEmpty()
                    && tail.output() < 0 && sameActivity(tail.values(),sum.output(),activity)
                    && sameCellEpilogue(tail,sum.output(),sum.values(),views,ops)) {
                result.set(s,new CpuGeneralScanIr.Stage(sum.values(),sum.output(),sum.input(),sum.axes(),true,tail.values(),-1,sum.sumToShape()));
                result.remove(s+1);
            }
        }
        return result;
    }

    /**
     * Compares explicit execution domains without guessing a batch axis from extents.
     * @param outputs proposed scalar sites
     * @param reference enclosing stage's activity-bearing ordinal
     * @param activity length-mode ordinal roles, empty in legacy modes
     * @return whether all sites have the same row axis or SHARED active-time gate
     */
    private static boolean sameActivity(List<Integer> outputs,int reference,List<Integer> activity) {
        return activity.isEmpty() || outputs.stream().allMatch(v -> activity.get(v).equals(activity.get(reference)));
    }

    /**
     * Proves an epilogue reads only the represented SUM/product cell, not a contributor local or a
     * different cell of a partially produced result. Shape equality alone is insufficient:
     * permuted views require the complete reduction/product at a separate stage boundary.
     * Contributor-to-tail fan-out likewise keeps that boundary and its exact live cut.
     * @param tail proposed scalar epilogue
     * @param sum represented SUM or MATMUL output ordinal
     * @param contributors scalar outputs computed inside the selected-domain loop
     * @param views proved affine accesses
     * @param instructions typed scalar instruction inventory
     * @return whether every dependency on a fused site is the same output cell
     */
    private static boolean sameCellEpilogue(CpuGeneralScanIr.Stage tail, int sum, List<Integer> contributors,
            List<CpuGeneralScanIr.View> views, Map<Integer,CpuKernelIr.Instruction> instructions) {
        var shape = views.get(sum).shape();
        if (!sameDomain(views.get(tail.values().getFirst()),views.get(sum))) return false;
        var available = new HashSet<Integer>(); available.add(sum);
        for (int value : tail.values()) {
            for (int input : instructions.get(value).inputs()) {
                var view = views.get(input); int origin = view.origin();
                if (contributors.contains(origin)) return false;
                if (!available.contains(origin)) continue;
                // Availability is ordinal-local; aliased views retain an explicit stage/cut.
                if (input != origin) return false;
                if (!sameDomain(view,views.get(sum))) return false;
                if (view.batch().isPresent()) {
                    var mapping=view.batch().orElseThrow(); long constant=1,factor=0;
                    for (int axis=shape.size()-1;axis>=0;axis--) {
                        long ec=mapping.extentConstants().get(axis),ef=mapping.extentBatchFactors().get(axis);
                        if ((ec!=1 || ef!=0) && (mapping.strideConstants().get(axis)!=constant
                                || mapping.strideBatchFactors().get(axis)!=factor)) return false;
                        long nextFactor=Math.addExact(Math.multiplyExact(constant,ef),Math.multiplyExact(factor,ec));
                        if (factor!=0 && ef!=0) return false;
                        constant=Math.multiplyExact(constant,ec); factor=nextFactor;
                    }
                    continue;
                }
                long stride = 1;
                for (int axis = shape.size()-1; axis >= 0; axis--) {
                    if (shape.get(axis) > 1 && view.strides().get(axis) != stride) return false;
                    stride = Math.multiplyExact(stride, shape.get(axis));
                }
            }
            available.add(value);
        }
        return true;
    }

    /**
     * Proves coordinate-domain equality without equating a batch coefficient with fixed zero
     * or a coincident feature extent. Fixed stages retain their original equality predicate.
     * @param left first proved view
     * @param right second proved view
     * @return whether every named/static axis has identical prepared coefficients
     */
    private static boolean sameDomain(CpuGeneralScanIr.View left,CpuGeneralScanIr.View right) {
        if (left.batch().isEmpty() && right.batch().isEmpty()) return left.shape().equals(right.shape());
        return left.batch().isPresent() && right.batch().isPresent()
                && left.batch().orElseThrow().extentConstants().equals(right.batch().orElseThrow().extentConstants())
                && left.batch().orElseThrow().extentBatchFactors().equals(right.batch().orElseThrow().extentBatchFactors());
    }

    /**
     * Rejoins same-coordinate scalar groups split only by non-executable view declarations.
     * A cross-coordinate alias of a just-produced value retains its materialized stage boundary.
     * @param stages ordered symbolic stages after exact identity binding
     * @param views complete coefficient/provenance inventory
     * @param arithmetic typed producing sites
     * @return stages without artificial view-only scalar traversal boundaries
     */
    private static ArrayList<CpuGeneralScanIr.Stage> joinSymbolicScalarStages(List<CpuGeneralScanIr.Stage> stages,
            List<CpuGeneralScanIr.View> views,CpuKernelIr arithmetic) {
        var result=new ArrayList<CpuGeneralScanIr.Stage>(); var ops=new HashMap<Integer,CpuKernelIr.Instruction>();
        arithmetic.instructions().forEach(i->ops.put(i.output(),i));
        for (var stage:stages) {
            if (!result.isEmpty() && stage.output()<0 && result.getLast().output()<0) {
                var previous=result.getLast(); boolean safe=sameDomain(views.get(stage.values().getFirst()),views.get(previous.values().getFirst()));
                for (int out:stage.values()) for (int in:ops.get(out).inputs())
                    if (previous.values().contains(views.get(in).origin()) && in!=views.get(in).origin()) safe=false;
                if (safe) { var joined=new ArrayList<>(previous.values()); joined.addAll(stage.values());
                    result.set(result.size()-1,new CpuGeneralScanIr.Stage(joined,-1,-1,List.of(),false)); continue; }
            }
            result.add(stage);
        }
        return result;
    }

    /**
     * Selects a common row region before any cut or bank is declared. Complete uses include
     * both matrix operands, view origins, demanded histories and temporal next-state roots.
     * Existing explicit length roles and symbolic B recipes take precedence over fixed-shape
     * inference; a fixed candidate needs an authenticated transposed row-contraction anchor.
     * @param views total authenticated local access inventory
     * @param stages existing represented traversal sites; replaced only after proof succeeds
     * @param arithmetic existing typed scalar inventory
     * @param activity complete length roles, or empty in all-valid modes
     * @param next positional next-carry roots
     * @param ports complete outer-role roots
     * @param demand live outer roles
     * @param carries leading evolving carry formals, excluded from staging scopes
     * @param sequences following original-time formals
     * @param formals total authenticated formal count
     * @param scalars positional actual scalar-read flags before boundary deduplication
     * @return immutable selected region or empty, without declaring resources
     */
    private static Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr> rowRegion(
            List<CpuGeneralScanIr.View> views,ArrayList<CpuGeneralScanIr.Stage> stages,CpuKernelIr arithmetic,
            List<Integer> activity,List<Integer> next,List<Integer> ports,List<Integer> demand,
            int carries,int sequences,int formals,List<Boolean> scalars) {
        int n = views.size(); var rows = new ArrayList<>(Collections.nCopies(n,-1));
        if (!activity.isEmpty()) rows = new ArrayList<>(activity);
        else if (views.stream().anyMatch(v -> v.batch().isPresent())) {
            for (int v = 0; v < n; v++) for (int a = 0; a < views.get(v).shape().size(); a++)
                if (views.get(v).batch().orElseThrow().extentBatchFactors().get(a) != 0) rows.set(v,a);
        } else {
            Long count = null;
            for (var stage : stages) if (stage.right() >= 0) {
                var left = views.get(stage.input()); var root = views.get(left.origin());
                if (left.origin() != stage.input() && left.shape().size() == 2 && root.shape().size() == 2
                        && left.shape().get(1).equals(root.shape().get(0))
                        && left.strides().get(0).equals(root.strides().get(1))
                        && left.strides().get(1).equals(root.strides().get(0))) {
                    long r = left.shape().get(1);
                    if (count != null && count != r) return Optional.empty(); count = r;
                }
            }
            if (count == null) return Optional.empty();
            for (int v = 0; v < n; v++) if (views.get(v).shape().size() == 2) {
                var view = views.get(v); var root = views.get(view.origin());
                if (root.shape().size() == 2 && root.shape().get(0).equals(count)) {
                    if (view.origin() == v || view.strides().equals(root.strides())) rows.set(v,0);
                    else if (view.strides().get(0).equals(root.strides().get(1))
                            && view.strides().get(1).equals(root.strides().get(0))) rows.set(v,1);
                }
            }
        }
        var operands = new HashMap<Integer,List<Integer>>();
        arithmetic.instructions().forEach(i -> operands.put(i.output(),i.inputs()));
        var allProducts = new HashMap<Integer,io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr.Contraction>();
        var allSums = new HashMap<Integer,io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr.Sum>();
        for (var stage : stages) if (stage.output() >= 0) {
            if (stage.right() >= 0) {
                allProducts.put(stage.output(),new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr.Contraction(stage.output(),stage.input(),stage.right()));
                operands.put(stage.output(),List.of(stage.input(),stage.right()));
            } else {
                operands.put(stage.output(),List.of(stage.input()));
                if (stage.reduction()) allSums.put(stage.output(),new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr.Sum(stage.output(),stage.input(),stage.axes(),stage.sumToShape()));
            }
        }
        var needed = new HashSet<Integer>(); var pending = new ArrayDeque<Integer>();
        demand.forEach(p -> pending.add(ports.get(p)));
        while (!pending.isEmpty()) {
            int v = pending.removeLast(); if (!needed.add(v)) continue;
            if (views.get(v).origin() != v) pending.add(views.get(v).origin());
            pending.addAll(operands.getOrDefault(v,List.of()));
            if (v < next.size()) pending.add(next.get(v));
        }
        var terminal = new ArrayList<Integer>();
        for (var stage : stages) if (stage.output() >= 0 && needed.contains(stage.output())) {
            int out = stage.output();
            if (stage.right() >= 0 && rows.get(stage.input()) == 1 && rows.get(stage.right()) == 0) {
                rows.set(out,-1); terminal.add(out);
            } else if (stage.reduction() && rows.get(stage.input()) >= 0
                    && stage.axes().contains(rows.get(stage.input()))) {
                rows.set(out,-1); terminal.add(out);
            }
        }
        if (terminal.isEmpty()) return Optional.empty();
        // A feature product's ordered right operand is shared, even when F equals R.
        for (var product : allProducts.values()) if (!terminal.contains(product.output()) && rows.get(product.left()) == 0) {
            int root = views.get(product.right()).origin(); rows.set(root,-1); rows.set(product.right(),-1);
            rows.set(product.output(),0);
        }
        var rowSites = new ArrayList<Integer>();
        var scalarSites = new HashSet<Integer>(); arithmetic.instructions().forEach(i -> scalarSites.add(i.output()));
        for (int v = 0; v < n; v++) if (needed.contains(v) && rows.get(v) >= 0
                && views.get(v).origin() == v && operands.containsKey(v) && !terminal.contains(v)
                && (scalarSites.contains(v)
                    || allProducts.containsKey(v))) rowSites.add(v);
        if (rowSites.isEmpty()) return Optional.empty();
        var selected = new HashSet<>(rowSites); selected.addAll(terminal);
        var uses = new ArrayList<List<Integer>>();
        for (int v = 0; v < n; v++) uses.add(new ArrayList<>());
        operands.forEach((out,ins) -> ins.forEach(in -> uses.get(in).add(out)));
        for (int v = 0; v < n; v++) if (views.get(v).origin() != v) uses.get(views.get(v).origin()).add(v);
        // A private shared terminal's own carry ADD belongs to finish, after the complete
        // row reduction. Bind its operand directly to the terminal accumulator rather than
        // declaring a private cut merely to reload it in an immediately following traversal.
        var finish = new ArrayList<Integer>();
        var demandedTerminalRoots = new HashSet<Integer>();
        demand.forEach(p -> demandedTerminalRoots.add(views.get(ports.get(p)).origin()));
        for (var instruction : arithmetic.instructions()) if (needed.contains(instruction.output())
                && instruction.opcode() == CpuPointwiseOpcode.ADD && rows.get(instruction.output()) == -1) {
            int out = instruction.output();
            for (int terminalValue : terminal) if (instruction.inputs().contains(terminalValue)
                    && !demandedTerminalRoots.contains(terminalValue) && !next.contains(terminalValue)
                    && uses.get(terminalValue).equals(List.of(out))) {
                boolean ownSharedCarry = false;
                for (int c = 0; c < next.size(); c++) if (next.get(c) == out && rows.get(c) == -1
                        && instruction.inputs().contains(c)) ownSharedCarry = true;
                if (ownSharedCarry) finish.add(out);
            }
        }
        var retained = new ArrayList<Integer>();
        var demandedRoots = new HashSet<Integer>();
        demand.forEach(p -> demandedRoots.add(views.get(ports.get(p)).origin()));
        for (int c = 0; c < next.size(); c++) if (needed.contains(c) || demand.contains(c)) demandedRoots.add(views.get(next.get(c)).origin());
        for (int v : selected) {
            boolean outside = demandedRoots.contains(v);
            for (int alias = 0; alias < n; alias++) if (views.get(alias).origin() == v)
                for (int consumer : uses.get(alias)) if (!selected.contains(consumer) && !finish.contains(consumer)
                        && views.get(consumer).origin() != v && needed.contains(consumer)) outside = true;
            if (outside) retained.add(v);
        }
        var values = new ArrayList<CpuKernelIr.Value>(); var accesses = new ArrayList<CpuAccessPlan.BatchBinding>();
        for (int v = 0; v < n; v++) {
            var view = views.get(v);
            CpuAccessPlan.BatchBinding access = view.batch().orElseGet(() -> CpuPartitionLowering.batchAccess(
                    CpuAccessPlan.AccessKind.READ,view.shape(),Collections.nCopies(view.shape().size(),0L),
                    view.strides(),Collections.nCopies(view.shape().size(),0L)));
            accesses.add(access);
            var old = arithmetic.values().get(v);
            values.add(new CpuKernelIr.Value(v,old.dataType(),selected.contains(v) || selected.contains(view.origin())
                    ? CpuKernelIr.Value.Kind.VIRTUAL : CpuKernelIr.Value.Kind.INPUT,access.plan()));
        }
        var selectedArithmetic = new CpuKernelIr(values,arithmetic.instructions().stream()
                .filter(i -> rowSites.contains(i.output())).toList(),arithmetic.loop(),List.of());
        var candidate = new io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr(selectedArithmetic,rowSites,
                allProducts.values().stream().filter(c -> selected.contains(c.output())).sorted(java.util.Comparator.comparingInt(c -> c.output())).toList(),
                allSums.values().stream().filter(s -> terminal.contains(s.output())).sorted(java.util.Comparator.comparingInt(s -> s.output())).toList(),
                terminal,accesses,views.stream().map(CpuGeneralScanIr.View::origin).toList(),rows,retained);
        var proof = CpuRowMapReductionLowering.prove(candidate,uses,
                java.util.stream.IntStream.range(0,n).mapToObj(v -> List.<Integer>of()).toList());
        if (proof.isEmpty()) return Optional.empty();
        var invariantSources = new ArrayList<Integer>(); var temporalSources = new ArrayList<Integer>();
        for (int f = carries; f < formals; f++) if (!scalars.get(f)) {
            if (f >= carries+sequences) invariantSources.add(f);
            else if (candidate.rowAxes().get(f) < 0) temporalSources.add(f);
        }
        // Source formals are immutable borrowed input storage. Every output/cut is distinct
        // at binding, so neither saved publication nor inclusive scratch reuse can overwrite
        // these scopes. Unknown carry/scratch sources are deliberately not authenticated.
        long duplicateBindings = 0;
        for (int v : candidate.rowSites()) if (candidate.retainedStores().contains(v)) {
            long wrappers = 0;
            for (int p : demand) if (p < carries ? next.get(p) == v : candidate.viewOrigins().get(ports.get(p)) == v) wrappers++;
            duplicateBindings += Math.max(0,wrappers-1);
        }
        var collapsed = new ArrayList<CpuGeneralScanIr.Stage>();
        var definitions = new ArrayList<Integer>(rowSites); definitions.addAll(terminal);
        collapsed.add(new CpuGeneralScanIr.Stage(definitions,-1,-1,List.of(),false,finish));
        for (var stage : stages) {
            var producers = stage.values().stream().filter(v -> !selected.contains(v) && !finish.contains(v)).toList();
            var tail = stage.epilogue().stream().filter(v -> !selected.contains(v) && !finish.contains(v)).toList();
            if (stage.output() >= 0 && !selected.contains(stage.output()))
                collapsed.add(new CpuGeneralScanIr.Stage(producers,stage.output(),stage.input(),stage.axes(),stage.reduction(),tail,stage.right(),stage.sumToShape()));
            else {
                var remainder = new ArrayList<>(producers); remainder.addAll(tail);
                if (!remainder.isEmpty()) collapsed.add(new CpuGeneralScanIr.Stage(remainder,-1,-1,List.of(),false));
            }
        }
        // Contract the complete DAG stably. A failed quotient proof leaves the old stages intact.
        var births = new HashMap<Integer,Integer>();
        for (int s = 0; s < collapsed.size(); s++) {
            var stage = collapsed.get(s);
            for (int v : stage.values()) births.put(v,s);
            for (int v : stage.epilogue()) births.put(v,s);
            if (stage.output() >= 0) births.put(stage.output(),s);
        }
        var ordered = new ArrayList<CpuGeneralScanIr.Stage>(); var emitted = new HashSet<Integer>();
        while (ordered.size() < collapsed.size()) {
            boolean progress = false;
            for (int s = 0; s < collapsed.size(); s++) if (!emitted.contains(s)) {
                var stage = collapsed.get(s); var produced = new ArrayList<>(stage.values()); produced.addAll(stage.epilogue());
                if (stage.output() >= 0) produced.add(stage.output());
                boolean ready = true;
                for (int out : produced) for (int in : operands.getOrDefault(out,List.of())) {
                    Integer birth = births.get(views.get(in).origin());
                    if (birth != null && birth != s && !emitted.contains(birth)) ready = false;
                }
                if (ready) { ordered.add(stage); emitted.add(s); progress = true; }
            }
            if (!progress) return Optional.empty();
        }
        stages.clear(); stages.addAll(ordered);
        // The invocation lifetime includes publication of EVERY demanded history, including
        // forwarded inputs not produced by this region. Each independently bound LONG base
        // contributes two slots. Multi-stage lifetimes remain unqualified for staging.
        long histories = demand.stream().filter(p -> p >= carries).count();
        boolean completeLifetime = true;
        for (int v = formals; v < candidate.viewOrigins().size(); v++) if (candidate.viewOrigins().get(v) == v
                && !candidate.rowSites().contains(v) && !candidate.terminalOutputs().contains(v) && !finish.contains(v))
            completeLifetime = false;
        CpuRowMapReductionLowering.invariantLoadStaging(candidate,
                completeLifetime ? invariantSources : List.of(),completeLifetime ? temporalSources : List.of(),
                112+3*duplicateBindings+2*histories,116+3*duplicateBindings);
        return proof;
    }

    /**
     * Derives inclusive cut lifetimes before storage assignment. Only cross-stage values and
     * observed roots materialize. Demanded exact producer frames are preferred to scratch;
     * scratch cuts with disjoint lifetimes reuse aligned regions. A producing cut may directly
     * back its own next carry bank; it never aliases a live old bank or another live operand.
     * @param views proved affine accesses
     * @param stages generation order
     * @param arithmetic sole typed scalar instruction inventory
     * @param next positional next-carry roles
     * @param ports canonical role ordinals
     * @param demand demanded outer wrappers
     * @param carries positive carry count
     * @param sequences positive step-formal count
     * @param formals total formal count; remaining positions are invariant
     * @return immutable exact resource and lifetime plan
     */
    private static CpuGeneralScanIr.Stages shapedResources(List<CpuGeneralScanIr.View> views,
            List<CpuGeneralScanIr.Stage> stages, CpuKernelIr arithmetic, List<Integer> next,
            List<Integer> ports, List<Integer> demand, int carries, int sequences, int formals) {
        return shapedResources(views,stages,arithmetic,next,ports,demand,carries,sequences,formals,false);
    }

    /**
     * Declares length-mode carries independently of zero-padded history frames. Inactive state
     * cannot use its history as its next bank: histories contain +0, state preserves raw bits.
     * @param views ordinal affine bindings
     * @param stages generation order
     * @param arithmetic typed instruction sites
     * @param next positional next roles
     * @param ports canonical role bindings
     * @param demand live outer ports
     * @param carries number of carried formals
     * @param sequences number of step formals
     * @param formals number of floating formals, excluding lengths
     * @param length whether raw inactive identity requires independent banks
     * @return exact pre-slot resources and inclusive lifetimes
     */
    private static CpuGeneralScanIr.Stages shapedResources(List<CpuGeneralScanIr.View> views,
            List<CpuGeneralScanIr.Stage> stages, CpuKernelIr arithmetic, List<Integer> next,
            List<Integer> ports, List<Integer> demand, int carries, int sequences, int formals, boolean length) {
        return shapedResources(views,stages,arithmetic,next,ports,demand,carries,sequences,formals,length,Optional.empty());
    }

    /**
     * Computes exact resources after common-region selection. Mixed stage values resolve to
     * pointwise or contraction definitions explicitly. Private row/terminal locals and private
     * finish operands need no cuts; required saved/public stores and old/new hazards do.
     * @param views complete authenticated access/origin inventory
     * @param stages selected order, including a mixed region and distinct finish epilogue
     * @param arithmetic existing typed scalar sites, including shared carry ADDs
     * @param next simultaneous next-carry roots
     * @param ports complete canonical outer roots
     * @param demand exact live outer ports
     * @param carries positive carry-formal count
     * @param sequences positive step-formal count
     * @param formals all floating formals; the remainder are invariant
     * @param length whether inactive raw carry identity requires separate history handling
     * @param region nonnull already-proved optional region; no legality/topology discovery here
     * @return immutable pre-slot bank/cut declarations with inclusive lifetimes
     */
    private static CpuGeneralScanIr.Stages shapedResources(List<CpuGeneralScanIr.View> views,
            List<CpuGeneralScanIr.Stage> stages, CpuKernelIr arithmetic, List<Integer> next,
            List<Integer> ports, List<Integer> demand, int carries, int sequences, int formals, boolean length,
            Optional<io.github.pho001.synaptik.backend.cpu.internal.ir.CpuRowMapReductionIr> region) {
        long width = arithmetic.values().getFirst().dataType().byteWidth();
        boolean symbolic = views.stream().anyMatch(v -> v.batch().isPresent());
        int count = views.size(); int[] birth = new int[count], death = new int[count];
        Arrays.fill(birth, -1); Arrays.fill(death, -1);
        var instruction = new HashMap<Integer, CpuKernelIr.Instruction>();
        arithmetic.instructions().forEach(i -> instruction.put(i.output(), i));
        var specialInputs = new HashMap<Integer, List<Integer>>();
        region.ifPresent(r -> {
            r.contractions().forEach(c -> specialInputs.put(c.output(),List.of(c.left(),c.right())));
            r.sums().forEach(s -> specialInputs.put(s.output(),List.of(s.input())));
        });
        stages.stream().filter(s -> s.output() >= 0).forEach(s -> specialInputs.put(s.output(),
                s.right() >= 0 ? List.of(s.input(), s.right()) : List.of(s.input())));
        var needed = new HashSet<Integer>(); var pending = new ArrayDeque<Integer>();
        demand.forEach(p -> pending.add(ports.get(p)));
        while (!pending.isEmpty()) {
            int value = pending.removeLast();
            if (!needed.add(value)) continue;
            int origin = views.get(value).origin();
            if (origin != value) pending.add(origin);
            var op = instruction.get(value);
            if (op != null) pending.addAll(op.inputs());
            if (specialInputs.containsKey(value)) pending.addAll(specialInputs.get(value));
            // Close temporal forwarding transitively, including instruction-free carry chains.
            if (value < carries) pending.add(next.get(value));
        }
        var liveStages = new ArrayList<CpuGeneralScanIr.Stage>();
        for (var stage : stages) {
            if (stage.output() >= 0) {
                var outputs = stage.values().stream().filter(needed::contains).toList();
                var tail = stage.epilogue().stream().filter(needed::contains).toList();
                if (stage.right() >= 0 && !needed.contains(stage.output())) {
                    // Independent scalar tails must not keep a dead contraction executable.
                    if (!tail.isEmpty()) liveStages.add(new CpuGeneralScanIr.Stage(tail,-1,-1,List.of(),false));
                    continue;
                }
                if (needed.contains(stage.output()) || !tail.isEmpty()) liveStages.add(new CpuGeneralScanIr.Stage(outputs,stage.output(),stage.input(),stage.axes(),stage.reduction(),tail,stage.right(),stage.sumToShape()));
                else if (!outputs.isEmpty()) liveStages.add(new CpuGeneralScanIr.Stage(outputs,-1,-1,List.of(),false));
            }
            else {
                var outputs = stage.values().stream().filter(needed::contains).toList();
                var tail = stage.epilogue().stream().filter(needed::contains).toList();
                if (!outputs.isEmpty()) liveStages.add(new CpuGeneralScanIr.Stage(outputs, -1, -1, List.of(), false,tail));
            }
        }
        stages = liveStages;
        for (int s = 0; s < stages.size(); s++) {
            var stage = stages.get(s);
            if (stage.output() >= 0) birth[stage.output()] = s;
            for (int value : stage.values()) birth[value] = s;
            for (int value : stage.epilogue()) birth[value] = s;
        }
        for (int s = 0; s < stages.size(); s++) {
            var stage = stages.get(s); var reads = new ArrayList<Integer>();
            if (stage.input() >= 0) reads.add(stage.input());
            if (stage.right() >= 0) reads.add(stage.right());
            for (int value : stage.values()) reads.addAll(instruction.containsKey(value)
                    ? instruction.get(value).inputs() : specialInputs.get(value));
            for (int value : stage.epilogue()) reads.addAll(instruction.get(value).inputs());
            for (int value : reads) {
                int origin = views.get(value).origin();
                if (birth[origin] >= 0 && birth[origin] != s) death[origin] = Math.max(death[origin], s);
            }
        }
        for (int p : demand) death[views.get(ports.get(p)).origin()] = stages.size();
        // All next roles must survive simultaneous cross-coordinate old/new transitions.
        for (int c = 0; c < carries; c++) if (needed.contains(c) || demand.contains(c))
            death[views.get(next.get(c)).origin()] = stages.size();
        // Direct invariant products may execute once per invocation. Their complete scratch
        // frame must survive every subsequent step, so it cannot reuse a time-local cut.
        var invariantProducts = new HashSet<Integer>();
        for (var stage : stages) if (stage.right() >= 0 && stage.epilogue().isEmpty()) {
            int left = views.get(stage.input()).origin(), right = views.get(stage.right()).origin();
            if (left >= carries+sequences && left < formals && right >= carries+sequences && right < formals
                    && demand.stream().noneMatch(p -> p >= carries && ports.get(p) == stage.output())
                    && !next.contains(stage.output())) {
                invariantProducts.add(stage.output()); death[stage.output()] = stages.size();
            }
        }
        var carryOffsets = new ArrayList<Long>(); var carryFactors = new ArrayList<Long>();
        var carryPorts = new ArrayList<Integer>(); long bytes = 0, factor = 0;
        for (int c = 0; c < carries; c++) {
            if (next.get(c) == c) { carryPorts.add(-1); carryOffsets.add(-1L); carryFactors.add(0L); continue; }
            if (!needed.contains(c) && !demand.contains(c)) {
                carryPorts.add(-1); carryOffsets.add(-1L); carryFactors.add(0L); continue;
            }
            int frame = -1;
            for (int p : demand) if (!length && p >= carries && ports.get(p).equals(next.get(c))) { frame = p; break; }
            // A proved pointwise old->new transition can use its distinct final publication
            // directly. Caller storage is never written; unproved/cross-coordinate transitions
            // retain the two disjoint old/new banks.
            if ((symbolic || length) && frame < 0 && demand.contains(c)
                    && pointwiseFinalCarry(c,next,views,stages,instruction,specialInputs,birth,length)) frame=c;
            carryPorts.add(frame); carryOffsets.add(frame < 0 ? bytes : -1L);
            carryFactors.add(frame < 0 ? factor : 0L);
            if (frame < 0) {
                var size = views.get(c).frameSize(Math.toIntExact(width));
                bytes = Math.addExact(bytes,Math.multiplyExact(size.constantBytes(),2));
                factor = Math.addExact(factor,Math.multiplyExact(size.bytesPerBatch(),2));
            }
        }
        var cuts = new ArrayList<CpuGeneralScanIr.Cut>();
        for (int value = 0; value < count; value++) if (birth[value] >= 0 && death[value] >= 0) {
            int port = -1;
            for (int p : demand) if (p >= carries && ports.get(p) == value) { port = p; break; }
            if ((symbolic || length) && port < 0) for (int c=0;c<carries;c++)
                if (carryPorts.get(c)==c && next.get(c)==value) { port=c; break; }
            int carry = -1;
            if (port < 0) for (int c = 0; c < carries; c++)
                if (next.get(c) == value && carryOffsets.get(c) >= 0) { carry = c; break; }
            var frameSize = views.get(value).frameSize(Math.toIntExact(width));
            long size = frameSize.constantBytes(), sizeFactor = frameSize.bytesPerBatch();
            long offset = carry >= 0 ? carryOffsets.get(carry) : bytes;
            long offsetFactor = carry >= 0 ? carryFactors.get(carry) : factor;
            int firstStage = invariantProducts.contains(value) ? 0 : birth[value];
            if (port < 0 && carry < 0) {
                // Reuse only a complete dead cut, never an overlapping live/saved region.
                for (var candidate : cuts) if (!symbolic && candidate.port() < 0 && candidate.carry() < 0 && candidate.death() < firstStage
                        && candidate.bytes() >= size && cuts.stream().noneMatch(live -> live.port() < 0
                            && live.offset() == candidate.offset() && live.death() >= firstStage)) {
                    offset = candidate.offset(); break;
                }
                if (offset == bytes) { bytes = Math.addExact(bytes, size); factor = Math.addExact(factor,sizeFactor); }
            }
            cuts.add(new CpuGeneralScanIr.Cut(value, port, offset, size, birth[value], death[value], carry,offsetFactor,sizeFactor));
        }
        return new CpuGeneralScanIr.Stages(views, stages, cuts, carryOffsets, carryPorts, bytes,carryFactors,factor,symbolic,List.of(),
                List.of(),List.of(),0,-1,0,region);
    }

    /**
     * Proves that a final-only carry may be updated in its distinct canonical publication.
     * Every old-carry use must precede production, or read the exact same cell in that producing
     * scalar traversal. A mixed region completes its row-feature sites before its next row
     * store, and completes all rows before any shared finish ADD. Cross-coordinate old reads
     * within a producing phase, aliased reads, later uses and forwarded old versions keep
     * independent banks. This is coefficient/provenance proof,
     * independent of actual B, including zero and coincident feature extents.
     * @param carry carry-formal ordinal
     * @param next all simultaneous next roles
     * @param views complete proved view inventory
     * @param stages live fused stages
     * @param instructions scalar producing sites
     * @param specialInputs ordered contraction/represented-SUM operands, including mixed region sites
     * @param birth stage index per producing ordinal
     * @param length whether exact same-cell contributors may update distinct final state;
     *     complete direct-old-use and later-old-read checks still apply
     * @return whether one final output safely backs this carry's produced versions
     */
    private static boolean pointwiseFinalCarry(int carry,List<Integer> next,List<CpuGeneralScanIr.View> views,
            List<CpuGeneralScanIr.Stage> stages,Map<Integer,CpuKernelIr.Instruction> instructions,
            Map<Integer,List<Integer>> specialInputs,int[] birth,boolean length) {
        int value=next.get(carry), producer=birth[value];
        if (producer<0 || Collections.frequency(next,value)!=1 || views.get(value).origin()!=value || !sameDomain(views.get(carry),views.get(value))
                || next.stream().anyMatch(n->views.get(n).origin()==carry)) return false;
        for (int s=producer;s<stages.size();s++) {
            var stage=stages.get(s);
            if (stage.input()>=0 && views.get(stage.input()).origin()==carry) return false;
            if (stage.right()>=0 && views.get(stage.right()).origin()==carry) return false;
            var scalar=new ArrayList<>(stage.values()); scalar.addAll(stage.epilogue());
            boolean produced=stage.output()==value,oldLoaded=false;
            for (int site = 0; site < scalar.size(); site++) {
                // SUM contributors and their epilogue have separate scalar availability sets.
                // The epilogue retains only the reduction result; it must reload an old formal
                // even if a contributor loaded it before an in-place next-state store.
                if (length && stage.output() >= 0 && site == stage.values().size()) oldLoaded = false;
                int out = scalar.get(site);
                // A mixed region's entire row phase precedes its finish epilogue. Reads of
                // a shared old carry anywhere in that phase are complete before this ADD.
                if (stage.output() < 0 && stage.epilogue().contains(value) && site < stage.values().size()) continue;
                var instruction = instructions.get(out);
                var operands = instruction != null ? instruction.inputs() : specialInputs.get(out);
                for (int in:operands) if (views.get(in).origin()==carry) {
                    // A mixed-site contraction/SUM can read several coordinates and cannot
                    // borrow this old/new final backing within the producing traversal.
                    if (instruction == null || s!=producer || in!=carry || stage.reduction() && !length || produced && !oldLoaded
                            || !sameDomain(views.get(out),views.get(carry))) return false;
                    oldLoaded=true;
                }
                if (out==value) produced=true;
            }
        }
        return true;
    }

    /**
     * Independently proves the closed length algebra over the complete local inventory. Roles
     * come only from declarations/dataflow, never equal extents; terminal reductions cannot feed
     * row roots. Shared old state may appear only at its own distinct terminal ADD site.
     * @param query complete length occurrence including the separate control descriptor
     * @param local scoped descriptors, never outer IDs
     * @param terminal destination for row-reduced dependency identities
     * @return complete local role map (-1 SHARED, otherwise the explicit row axis)
     * @throws IllegalArgumentException if control, role algebra or shared-carry use is unproved
     * @throws ArithmeticException if feature or control geometry overflows
     */
    private static Map<ValueId,Integer> lengthRows(OperationCapabilityQuery query,
            Map<ValueId,TensorDescriptor> local, Set<ValueId> terminal) {
        var a = (ScanAttrs)query.operation().attrs(); var b = query.scanBody().orElseThrow();
        var first = query.inputs().getFirst(); var lengths = query.inputs().getLast();
        if (first.shape().rank() < 1) throw invalid();
        Dimension batch = first.shape().dimension(0);
        if (!(batch instanceof StaticDimension || batch instanceof DynamicDimension)
                || lengths.dataType() != DataType.INT64 || lengths.requiresGrad() || !lengths.shape().equals(Shape.ofDimensions(batch))
                || (lengths.shape().isFullyStatic()
                    ? !lengths.layout().map(l -> l.storageOffset() == 0 && Arrays.equals(l.strides(),new long[]{1})).orElse(false)
                    : lengths.layout().isPresent())) throw invalid();
        batch.staticSize().ifPresent(n -> Math.multiplyExact(n,8L));
        var roles = new HashMap<ValueId,Integer>(); var shared = new HashMap<ValueId,ValueId>();
        for (int f = 0; f < b.graph().inputs().size(); f++) {
            int role = f < a.carryCount() ? f < a.rowCarryCount() ? 0 : -1
                    : f < a.carryCount()+a.sequenceCount() ? a.sequenceRowAxes().get(f-a.carryCount()).orElse(-1) : -1;
            var id = b.graph().inputs().get(f); roles.put(id,role);
            if (f >= a.rowCarryCount() && f < a.carryCount()) shared.put(id,b.nextCarry().get(f));
            checkLengthRow(local.get(id).shape(),role,batch);
        }
        var dependent = new HashSet<>(shared.keySet()); var producers = new HashMap<ValueId,CompiledNode>();
        for (var node : b.graph().nodes()) {
            if (node.inputs().isEmpty() || node.outputs().size() != 1 || node.inputs().stream().anyMatch(id -> !roles.containsKey(id))) throw invalid();
            boolean reduced = node.inputs().stream().anyMatch(terminal::contains);
            boolean row = node.inputs().stream().anyMatch(id -> roles.get(id) >= 0);
            boolean state = false;
            for (int p = 0; p < node.inputs().size(); p++) if (dependent.contains(node.inputs().get(p))) {
                if (p != 0 || node.operation().kind() != BinaryArithmeticKind.ADD
                        || !node.outputs().getFirst().equals(shared.get(node.inputs().get(p)))) throw invalid();
                state = true;
            }
            if (reduced && row) throw invalid();
            var shapes = node.inputs().stream().map(id -> local.get(id).shape()).toList();
            var operandRoles = node.inputs().stream().map(roles::get).toList();
            var id = node.outputs().getFirst(); var out = local.get(id).shape();
            int role = lengthRowResult(node,shapes,operandRoles,out);
            checkLengthRow(out,role,batch); roles.put(id,role); producers.put(id,node);
            if (reduced || row && role < 0) terminal.add(id);
            if (state) dependent.add(id);
        }
        if (roles.size() != local.size()) throw invalid();
        var sites = new HashSet<NodeId>();
        for (int c = 0; c < a.carryCount(); c++) {
            var root = b.nextCarry().get(c);
            if (c < a.rowCarryCount()) { if (roles.get(root) != 0 || terminal.contains(root)) throw invalid(); }
            else {
                var site = producers.get(root);
                if (site == null || site.operation().kind() != BinaryArithmeticKind.ADD || site.inputs().size() != 2
                        || !site.inputs().getFirst().equals(b.graph().inputs().get(c)) || roles.get(root) != -1
                        || roles.get(site.inputs().get(1)) != -1 || dependent.contains(site.inputs().get(1))
                        || !local.get(root).shape().equals(local.get(site.inputs().get(1)).shape()) || !sites.add(site.id())) throw invalid();
            }
        }
        for (var id : b.emissions()) if (roles.get(id) != 0 || terminal.contains(id)) throw invalid();
        return roles;
    }

    /**
     * Checks one explicit static or named row coordinate, including empty/singleton batches.
     * @param shape local shape template
     * @param role -1 SHARED, zero ROW(0), one rank-two ROW(1)
     * @param batch exact static or named declared batch Dimension, never a numeric sample
     * @throws IllegalArgumentException if rank, extent or supported axis disagrees
     */
    private static void checkLengthRow(Shape shape,int role,Dimension batch) {
        if (role < -1 || role > 1 || role == 1 && shape.rank() != 2 || role >= 0
                && (shape.rank() <= role || !shape.dimension(role).equals(batch))) throw invalid();
        for (int axis = 0; axis < shape.rank(); axis++) if (axis != role && !shape.dimension(axis).isStatic()) throw invalid();
    }

    /**
     * Propagates length roles independently of upstream semantic proofs. Ordinary descriptor,
     * attribute and layout guards remain in shaped analysis; this is only the row-flow proof.
     * @param node one scoped ordinary operation
     * @param shapes ordered local operand shapes
     * @param roles ordered explicit operand roles
     * @param out exact result shape
     * @return explicit result role, never inferred from the numeric value of B
     * @throws IllegalArgumentException if the algebra mixes or reinterprets the row coordinate
     * @throws ArithmeticException if checked feature counts overflow
     */
    private static int lengthRowResult(CompiledNode node,List<Shape> shapes,List<Integer> roles,Shape out) {
        Object kind = node.operation().kind(), attrs = node.operation().attrs();
        var in = shapes.getFirst(); int row = roles.getFirst();
        if (kind == BinaryArithmeticKind.ADD || kind == BinaryArithmeticKind.SUB || kind == BinaryArithmeticKind.MUL || kind == BinaryArithmeticKind.DIV) {
            int result = -1;
            for (int p = 0; p < roles.size(); p++) if (roles.get(p) >= 0) {
                int axis = roles.get(p)+out.rank()-shapes.get(p).rank();
                if (result >= 0 && result != axis) throw invalid(); result = axis;
            }
            if (result >= 0) for (int p = 0; p < roles.size(); p++) if (roles.get(p) < 0) {
                int axis = result-out.rank()+shapes.get(p).rank();
                if (axis >= 0 && shapes.get(p).dimension(axis).staticSize().orElseThrow() != 1) throw invalid();
            }
            return result;
        }
        if (kind == ScalarElementwiseKind.ADD || kind == ScalarElementwiseKind.SUB || kind == ScalarElementwiseKind.MUL || kind == ScalarElementwiseKind.DIV
                || kind == UnaryElementwiseKind.NEG || kind == UnaryElementwiseKind.TANH || kind == UnaryElementwiseKind.SIGMOID
                || kind == ContiguousKind.CONTIGUOUS || kind == io.github.pho001.synaptik.model.operation.elementwise.cast.CastKind.CAST) return row;
        if (kind == MatmulKind.MATMUL && shapes.size() == 2 && in.rank() == 2 && shapes.get(1).rank() == 2) {
            int right = roles.get(1);
            if (row == -1 && right == -1 || row == 1 && right == 0) return -1;
            if (row == 0 && right == -1) return 0;
            if (row == -1 && right == 1) return 1;
            throw invalid();
        }
        if (kind == AxisTransformKind.PERMUTE && attrs instanceof PermutationAttrs p && p.axes().equals(List.of(1,0))) return row < 0 ? -1 : 1-row;
        if (kind == AggregateReductionKind.SUM) {
            if (attrs instanceof SumToShapeAttrs) {
                int removed = in.rank()-out.rank();
                return row < 0 || row < removed || !in.dimension(row).equals(out.dimension(row-removed)) ? -1 : row-removed;
            }
            var axes = CpuAggregateLowering.normalizedAxes(in,out,attrs);
            if (row < 0 || axes.contains(row)) return -1;
            if (in.rank() == out.rank()) return row;
            int result = row; for (int axis : axes) if (axis < row) result--; return result;
        }
        if (kind == ShapeTransformKind.EXPAND || kind == ShapeTransformKind.RESHAPE
                || kind == AxisTransformKind.EXPAND_DIMS || kind == AxisTransformKind.SQUEEZE) {
            if (row < 0) return -1;
            if (row != 0 || out.rank() == 0 || !in.dimension(0).equals(out.dimension(0))) throw invalid();
            if (kind == ShapeTransformKind.EXPAND && in.rank() != out.rank()
                    || attrs instanceof AxisTransformAttrs a && a.axis() == 0) throw invalid();
            if (kind == ShapeTransformKind.RESHAPE) {
                long before = 1, after = 1;
                for (int axis = 1; axis < in.rank(); axis++) before = Math.multiplyExact(before,in.dimension(axis).staticSize().orElseThrow());
                for (int axis = 1; axis < out.rank(); axis++) after = Math.multiplyExact(after,out.dimension(axis).staticSize().orElseThrow());
                if (before != after) throw invalid();
            }
            return 0;
        }
        throw invalid();
    }

    /**
     * Computes a checked per-value frame domain, independently of every other value.
     * @param view proved static logical access
     * @return exact frame elements, including scalar one and empty-domain zero
     */
    private static long elements(CpuGeneralScanIr.View view) {
        if (view.shape().contains(0L)) return 0;
        long n = 1; for (long extent : view.shape()) n = Math.multiplyExact(n, extent); return n;
    }

    /** @return a fresh failure for a rejected body, binder or resource proof */
    private static IllegalArgumentException invalid() {
        return new IllegalArgumentException("outside bounded homogeneous FLOAT32/FLOAT64 general scan body/binder/ABI domain");
    }

    /**
     * Checks the closed storage type and exact static byte count.
     * @param d non-null descriptor under proof
     * @throws IllegalArgumentException if type or static shape is unsupported
     * @throws ArithmeticException if its byte count overflows
     */
    private static void checkTypeShape(TensorDescriptor d) {
        if ((d.dataType() != DataType.FLOAT32 && d.dataType() != DataType.FLOAT64) || !d.shape().isFullyStatic()) throw invalid();
        Math.multiplyExact(d.shape().knownElementCount().orElseThrow(), (long) d.dataType().byteWidth());
    }

    /**
     * Checks resolved ordinary zero-offset, non-view canonical layout equality.
     * @param d non-null fully static descriptor
     * @return whether the complete canonical layout is present
     */
    private static boolean canonical(TensorDescriptor d) {
        return d.layout().equals(Optional.of(LayoutDescriptor.contiguous(d.shape())));
    }

    /**
     * Proves actual-read addressing without changing metadata or materializing broadcasts.
     * Dense reads require exact canonical strides on every axis, including singleton and empty
     * axes, but may be marked as views because that flag changes no address or referenced span.
     * The separate all-zero-stride case retains constant-address scalar reads, including expanded
     * constants and initial carries that become dense later. This predicate never admits formals
     * or writes; their complete non-view canonical requirement remains in {@link #canonical(TensorDescriptor)}.
     * @param d non-null static actual input descriptor
     * @return whether all logical positions use one scalar storage cell
     * @throws IllegalArgumentException if layout is absent, has a nonzero offset, or has neither
     *     exact canonical strides nor all-zero strides
     */
    private static boolean scalarRead(TensorDescriptor d) {
        LayoutDescriptor layout = d.layout().orElseThrow(CpuGeneralScanLowering::invalid);
        if (layout.storageOffset() != 0) throw invalid();
        if (Arrays.equals(layout.strides(), LayoutDescriptor.contiguous(d.shape()).strides()))
            return d.shape().rank() == 0;
        if (Arrays.stream(layout.strides()).allMatch(s -> s == 0)) return true;
        throw invalid();
    }

    /**
     * Prepends original time without imposing the cross-owner transfer rank limit.
     * @param shape non-null static local value shape
     * @param time non-negative static time extent
     * @return immutable stacked boundary shape, possibly rank seventeen
     */
    private static Shape stacked(Shape shape, long time) {
        var dimensions = new ArrayList<Dimension>(); dimensions.add(new StaticDimension(time));
        dimensions.addAll(shape.dimensions());
        return Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
    }

    /**
     * Derives the structural dense/scalar access form, not physical assignment.
     * @param rank non-negative iteration rank
     * @param kind exact read/write role
     * @param scalar whether address increments are zero
     * @return immutable normalized access proof
     */
    private static CpuAccessPlan access(int rank, CpuAccessPlan.AccessKind kind, boolean scalar) {
        return new CpuAccessPlan(kind, scalar ? CpuAccessPlan.Regime.SCALAR_ALL_ZERO
                : CpuAccessPlan.Regime.DENSE_LINEAR, rank, Collections.nCopies(rank, scalar
                    ? CpuAccessPlan.AxisRole.BROADCAST : CpuAccessPlan.AxisRole.CONTIGUOUS), scalar ? 0 : rank);
    }

    /**
     * Appends one exact referenced span and access in positional boundary order.
     * @param d non-null proved resolved boundary descriptor
     * @param kind read or distinct write role
     * @param scalar whether the boundary is constant-address
     * @param spans caller-owned cold span accumulator, mutated once
     * @param bindings caller-owned cold binding accumulator, mutated once
     * @throws ArithmeticException if exact geometry overflows
     */
    private static void addBoundary(TensorDescriptor d, CpuAccessPlan.AccessKind kind, boolean scalar,
            List<Long> spans, List<CpuAccessPlan.Binding> bindings) {
        var layout = d.layout().orElseThrow(CpuGeneralScanLowering::invalid);
        long count = d.shape().knownElementCount().orElseThrow();
        var plan = access(d.shape().rank(), kind, scalar);
        spans.add(layout.referencedElementSpan());
        bindings.add(CpuAccessPlan.Binding.create(plan, d.shape().toLongArray(), 0, layout.strides(),
                count, 0, count, layout.referencedElementSpan()));
    }

    /**
     * Re-derives canonical producer-postorder residuals, including unused carry formals.
     * @param body non-null closed captured body
     * @param count positive carry-formal count
     * @throws IllegalArgumentException if any residual or contained occurrence is missing/reordered
     */
    static void validateInventory(CompiledScanBody body, int count) {
        var producers = new HashMap<ValueId, CompiledNode>();
        body.graph().nodes().forEach(n -> n.outputs().forEach(v -> producers.put(v, n)));
        var carry = new HashSet<>(body.graph().inputs().subList(0, count));
        var seen = new HashSet<ValueId>(); var visited = new HashSet<NodeId>();
        var retained = new ArrayList<ValueId>();
        for (ValueId root : body.nextCarry()) retain(root, producers, carry, seen, visited, retained);
        for (ValueId root : body.emissions()) retain(root, producers, carry, seen, visited, retained);
        for (ValueId c : body.graph().inputs().subList(0, count)) if (seen.add(c)) retained.add(c);
        if (!retained.equals(body.residuals()) || visited.size() != body.graph().nodes().size()) throw invalid();
    }

    /**
     * Visits one scoped local root once, preserving source operand and canonical sibling order.
     * @param value non-null local root
     * @param producers scoped output-to-producer index, borrowed
     * @param carry scoped carried formals, borrowed
     * @param seen mutable unique retained-value membership
     * @param visited mutable visited-producer membership
     * @param retained mutable ordered residual inventory
     */
    private static void retain(ValueId value, Map<ValueId, CompiledNode> producers, Set<ValueId> carry,
            Set<ValueId> seen, Set<NodeId> visited, List<ValueId> retained) {
        var node = producers.get(value);
        if (node == null) { if (carry.contains(value) && seen.add(value)) retained.add(value); }
        else if (visited.add(node.id())) {
            for (ValueId input : node.inputs()) retain(input, producers, carry, seen, visited, retained);
            for (ValueId output : node.outputs()) if (seen.add(output)) retained.add(output);
        }
    }
}
