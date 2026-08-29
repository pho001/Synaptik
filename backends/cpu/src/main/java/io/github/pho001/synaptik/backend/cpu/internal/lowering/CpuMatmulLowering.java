package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuMatmulIr;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Arrays;
import java.util.Objects;

/**
 * Cold, checked normalization of static MATMUL descriptors into full-K invocation geometry.
 *
 * <p>Rank-one operands receive internal unit M or N axes. Leading batch axes are aligned from
 * the right and absent or expanded singleton coordinates receive effective stride zero. The
 * record retains no graph, operation, carrier, worker, policy, or allocation state.</p>
 */
public final class CpuMatmulLowering {
    /** Creates a stateless geometry lowerer. */
    public CpuMatmulLowering() { }

    /**
     * Lowers one canonical MATMUL and its already-proved optional bias/terminal suffix into an
     * executable scalar or bounded tiled/vector form.
     *
     * @param context non-null one- through three-occurrence CPU preparation projection
     * @return immutable lowering with exact boundary bindings and MATMUL geometry
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if the unit is not one exact bounded MATMUL chain
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        if(context.nodes().size()!=1)throw new IllegalArgumentException(
                "bare MATMUL lowering requires one occurrence");
        return lowerRecognized(context,CpuMatmulIr.Epilogue.none());
    }

    /**
     * Lowers exactly one 0008C-associated executable alternative without rediscovering suffix
     * semantics from the graph.
     * @param context exact projected member occurrences in stable order
     * @param fact non-null recognized MATMUL fact whose literal epilogue is authoritative
     * @return immutable fused MATMUL lowering
     * @throws IllegalArgumentException if membership, chain boundaries, or descriptors disagree
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context,
            io.github.pho001.synaptik.backend.cpu.internal.ir.CpuSpecializedSubgraph.MatmulEpilogue fact) {
        Objects.requireNonNull(fact,"fact");
        if(context.nodes().size()!=fact.memberNodeOrdinals().size())throw new IllegalArgumentException(
                "recognized MATMUL membership disagrees");
        var source=fact.epilogue();
        var order=switch(source.addInputOrder()) {
            case NONE->CpuMatmulIr.Epilogue.AddInputOrder.NONE;
            case PRECEDING_LEFT->CpuMatmulIr.Epilogue.AddInputOrder.MATMUL_LEFT;
            case PRECEDING_RIGHT->CpuMatmulIr.Epilogue.AddInputOrder.MATMUL_RIGHT;};
        var terminal=CpuMatmulIr.Epilogue.Terminal.valueOf(source.terminal().name());
        return lowerRecognized(context,new CpuMatmulIr.Epilogue(order,terminal,
                source.clampRange().orElse(null)));
    }

    private CpuPartitionLowering.LoweredPartition lowerRecognized(
            PrepareContext<? extends BackendAnalysisInputs> context,CpuMatmulIr.Epilogue epilogue) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().isEmpty()||context.nodes().size()>3
                || context.nodes().getFirst().operation().kind()
                != io.github.pho001.synaptik.model.operation.linalg.MatmulKind.MATMUL) {
            throw new IllegalArgumentException("MATMUL unit requires one bounded chain");
        }
        var node = context.nodes().getFirst();
        if (node.inputs().size() != 2 || node.outputs().size() != 1)
            throw new IllegalArgumentException("MATMUL boundary cardinality disagrees");
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        GraphValue left = require(values, node.inputs().get(0));
        GraphValue right = require(values, node.inputs().get(1));
        ValueId matmulOutput=node.outputs().getFirst();ValueId current=matmulOutput;
        ValueId biasId=null;
        int suffix=1;
        if(epilogue.hasBias()) {
            var add=context.nodes().get(suffix++);
            if(add.operation().kind()!=io.github.pho001.synaptik.model.operation.elementwise.binary.BinaryArithmeticKind.ADD
                    ||add.inputs().size()!=2||add.outputs().size()!=1)
                throw new IllegalArgumentException("MATMUL bias ADD cardinality disagrees");
            int preceding=epilogue.addInputOrder()==CpuMatmulIr.Epilogue.AddInputOrder.MATMUL_LEFT?0:1;
            if(!add.inputs().get(preceding).equals(current))throw new IllegalArgumentException(
                    "recognized MATMUL ADD order disagrees");
            biasId=add.inputs().get(1-preceding);
            current=add.outputs().getFirst();
        }
        if(epilogue.hasTerminal()) {
            var terminalNode=context.nodes().get(suffix++);
            if(terminalNode.inputs().size()!=1||terminalNode.outputs().size()!=1
                    ||!terminalNode.inputs().getFirst().equals(current))throw new IllegalArgumentException(
                            "MATMUL terminal does not consume the suffix");
            current=terminalNode.outputs().getFirst();
        }
        if(suffix!=context.nodes().size())throw new IllegalArgumentException("MATMUL suffix is not linear");
        GraphValue result = require(values,current);
        Geometry geometry = lower(left.descriptor(), right.descriptor(), result.descriptor());
        if(biasId!=null){GraphValue bias=require(values,biasId);long[] be=bias.descriptor().shape().toLongArray();
            if(be.length!=1||be[0]!=geometry.n()||bias.descriptor().dataType()!=geometry.resultType())
                throw new IllegalArgumentException("MATMUL bias descriptor disagrees");}
        var mutableIds=new ArrayList<ValueId>();mutableIds.add(node.inputs().get(0));
        mutableIds.add(node.inputs().get(1));if(biasId!=null)mutableIds.add(biasId);mutableIds.add(current);
        List<ValueId> ids=List.copyOf(mutableIds);
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        for (int i = 0; i < ids.size(); i++) bindings.add(binding(require(values, ids.get(i)),
                i == ids.size() - 1 ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ));
        int bits = preferredBits(geometry.resultType());
        int lanes = bits == 0 ? 1 : bits / geometry.resultType().bitWidth();
        var selection = new CpuMatmulCandidateSelector().select(
                new CpuMatmulCandidateSelector.Facts(geometry.leftType(), geometry.rightType(),
                        geometry.resultType(), geometry.batchCount(), geometry.m(), geometry.k(),
                        geometry.n(), geometry.rightNStride(), geometry.resultNStride(), lanes,
                        epilogue.hasTerminal()));
        boolean vector = selection.selected() == CpuMatmulIr.Realization.DIRECT_N_VECTOR
                || selection.selected() == CpuMatmulIr.Realization.TILED_N_VECTOR_2X2;
        var ir = new CpuMatmulIr(geometry.leftType(), geometry.rightType(), geometry.resultType(),
                selection.selected(), epilogue, vector ? bits : 0,
                CpuMatmulIr.NumericalForm.SEQUENTIAL,
                bindings.subList(0,bindings.size()-1).stream().map(CpuAccessPlan.Binding::plan).toList(),
                bindings.getLast().plan());
        return new CpuPartitionLowering.LoweredPartition(ir.encodedKernelIr(), ids, bindings,
                ids.stream().map(id -> require(values, id).descriptor().layout().orElseThrow()
                        .referencedElementSpan()).toList(),
                ids.stream().map(id->require(values,id).descriptor().dataType()).toList(), List.of(),
                new long[] {geometry.outputCount()}, geometry.outputCount(),
                "legal: bounded portable MATMUL " + selection.selected(), new long[0],
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(),
                Optional.empty(), Optional.empty(), Optional.of(
                        new CpuPartitionLowering.LoweredPartition.MatmulFacts(ir,geometry)));
    }

    private static int preferredBits(DataType type) {
        return switch (type) {
            case FLOAT64 -> jdk.incubator.vector.DoubleVector.SPECIES_PREFERRED.vectorBitSize();
            case FLOAT32 -> jdk.incubator.vector.FloatVector.SPECIES_PREFERRED.vectorBitSize();
            case INT32 -> jdk.incubator.vector.IntVector.SPECIES_PREFERRED.vectorBitSize();
            case INT64 -> jdk.incubator.vector.LongVector.SPECIES_PREFERRED.vectorBitSize();
            default -> 0;
        };
    }
    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("MATMUL value is not projected: " + id);
        return value;
    }
    private static CpuAccessPlan.Binding binding(GraphValue value, CpuAccessPlan.AccessKind kind) {
        long[] extents = value.descriptor().shape().toLongArray();
        LayoutDescriptor layout = value.descriptor().layout().orElseThrow();
        long[] strides = layout.strides();
        var roles = new ArrayList<CpuAccessPlan.AxisRole>();
        long expected = 1; int suffix = 0;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            if (strides[axis] != expected) break;
            suffix++; expected = Math.multiplyExact(expected, Math.max(1, extents[axis]));
        }
        for (int axis = 0; axis < extents.length; axis++) roles.add(strides[axis] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST : axis >= extents.length - suffix
                ? CpuAccessPlan.AxisRole.CONTIGUOUS : CpuAccessPlan.AxisRole.STRIDED);
        var plan = new CpuAccessPlan(kind, suffix == extents.length
                ? CpuAccessPlan.Regime.DENSE_LINEAR : CpuAccessPlan.Regime.GENERAL_ODOMETER,
                extents.length, roles, suffix);
        long elements = count(extents);
        return CpuAccessPlan.Binding.create(plan, extents, layout.storageOffset(), strides,
                elements, 0, elements, layout.referencedElementSpan());
    }

    /**
     * Revalidates and normalizes one exact static descriptor triple.
     *
     * @param left non-null left input descriptor with a resolved non-negative layout
     * @param right non-null right input descriptor with a resolved non-negative layout
     * @param result non-null exact promoted output descriptor with an injective layout
     * @return immutable checked geometry
     * @throws NullPointerException if a descriptor is {@code null}
     * @throws IllegalArgumentException if semantic or layout facts disagree
     * @throws ArithmeticException if a count or address calculation overflows
     */
    public Geometry lower(TensorDescriptor left, TensorDescriptor right, TensorDescriptor result) {
        Objects.requireNonNull(left, "left"); Objects.requireNonNull(right, "right");
        Objects.requireNonNull(result, "result");
        long[] a = extents(left), b = extents(right), c = extents(result);
        if (a.length < 1 || b.length < 1) throw new IllegalArgumentException(
                "MATMUL operand rank must be at least one");
        DataType promoted = DataTypePromotion.promoteNumeric(left.dataType(), right.dataType());
        if (promoted != result.dataType()) throw new IllegalArgumentException(
                "MATMUL result type disagrees with numeric promotion");
        long k = a[a.length - 1], rightK = b[b.length == 1 ? 0 : b.length - 2];
        if (k != rightK) throw new IllegalArgumentException("MATMUL K extents disagree");
        boolean removeM = a.length == 1, removeN = b.length == 1;
        long m = removeM ? 1 : a[a.length - 2], n = removeN ? 1 : b[b.length - 1];
        int aBatchRank = Math.max(0, a.length - 2), bBatchRank = Math.max(0, b.length - 2);
        int batchRank = Math.max(aBatchRank, bBatchRank);
        int expectedRank = batchRank + (removeM ? 0 : 1) + (removeN ? 0 : 1);
        if (c.length != expectedRank) throw new IllegalArgumentException("MATMUL result rank disagrees");
        long[] batchExtents = new long[batchRank];
        long[] leftBatchStrides = new long[batchRank], rightBatchStrides = new long[batchRank];
        long[] resultBatchStrides = new long[batchRank];
        long[] as = strides(left), bs = strides(right), cs = strides(result);
        int aShift = batchRank - aBatchRank, bShift = batchRank - bBatchRank;
        for (int axis = 0; axis < batchRank; axis++) {
            long ae = axis < aShift ? 1 : a[axis - aShift];
            long be = axis < bShift ? 1 : b[axis - bShift];
            if (ae != be && ae != 1 && be != 1) throw new IllegalArgumentException(
                    "MATMUL batch extents do not right-broadcast");
            long extent = ae == be ? ae : ae == 1 ? be : ae;
            if (c[axis] != extent) throw new IllegalArgumentException("MATMUL result batch disagrees");
            batchExtents[axis] = extent;
            leftBatchStrides[axis] = axis < aShift || ae == 1 && extent != 1
                    ? 0 : as[axis - aShift];
            rightBatchStrides[axis] = axis < bShift || be == 1 && extent != 1
                    ? 0 : bs[axis - bShift];
            resultBatchStrides[axis] = cs[axis];
        }
        int resultAxis = batchRank;
        long resultMStride = 0, resultNStride = 0;
        if (!removeM) {
            if (c[resultAxis] != m) throw new IllegalArgumentException("MATMUL result M disagrees");
            resultMStride = cs[resultAxis++];
        }
        if (!removeN) {
            if (c[resultAxis] != n) throw new IllegalArgumentException("MATMUL result N disagrees");
            resultNStride = cs[resultAxis];
        }
        if (!injective(c, cs)) throw new IllegalArgumentException("MATMUL output is not injective");
        long batchCount = count(batchExtents), outputCount = Math.multiplyExact(
                Math.multiplyExact(batchCount, m), n);
        long leftSpan = span(left, a, as), rightSpan = span(right, b, bs);
        long resultSpan = span(result, c, cs);
        return new Geometry(left.dataType(), right.dataType(), result.dataType(), batchExtents,
                leftBatchStrides, rightBatchStrides, resultBatchStrides, batchCount, m, k, n,
                removeM, removeN, left.layout().orElseThrow().storageOffset(),
                right.layout().orElseThrow().storageOffset(), result.layout().orElseThrow().storageOffset(),
                removeM ? 0 : as[a.length - 2], as[a.length - 1],
                bs[b.length == 1 ? 0 : b.length - 2], removeN ? 0 : bs[b.length - 1],
                resultMStride, resultNStride, outputCount, leftSpan, rightSpan, resultSpan);
    }

    private static long[] extents(TensorDescriptor descriptor) {
        if (!descriptor.shape().isFullyStatic() || descriptor.layout().isEmpty())
            throw new IllegalArgumentException("MATMUL requires static resolved descriptors");
        return descriptor.shape().toLongArray();
    }
    private static long[] strides(TensorDescriptor descriptor) {
        LayoutDescriptor layout = descriptor.layout().orElseThrow();
        long[] strides = layout.strides();
        if (layout.storageOffset() < 0 || Arrays.stream(strides).anyMatch(value -> value < 0))
            throw new IllegalArgumentException("MATMUL requires non-negative layouts");
        return strides;
    }
    private static long count(long[] extents) {
        for (long extent : extents) if (extent == 0) return 0;
        long count = 1;
        for (long extent : extents) count = Math.multiplyExact(count, extent);
        return count;
    }
    private static long span(TensorDescriptor descriptor, long[] extents, long[] strides) {
        if (count(extents) == 0) return 0;
        long maximum = descriptor.layout().orElseThrow().storageOffset();
        for (int axis = 0; axis < extents.length; axis++) maximum = Math.addExact(maximum,
                Math.multiplyExact(extents[axis] - 1, strides[axis]));
        return Math.addExact(maximum, 1);
    }
    private static boolean injective(long[] extents, long[] strides) {
        if (count(extents) <= 1) return true;
        Integer[] axes = new Integer[extents.length];
        for (int i = 0; i < axes.length; i++) axes[i] = i;
        Arrays.sort(axes, java.util.Comparator.comparingLong(axis -> strides[axis]));
        long covered = 1;
        for (int axis : axes) if (extents[axis] > 1) {
            if (strides[axis] < covered) return false;
            covered = Math.addExact(covered, Math.multiplyExact(extents[axis] - 1, strides[axis]));
        }
        return true;
    }

    /**
     * Complete normalized logical geometry. Array components are defensively copied.
     *
     * @param leftType exact left representation type
     * @param rightType exact right representation type
     * @param resultType exact promoted result type
     * @param batchExtents normalized right-aligned batch extents
     * @param leftBatchStrides effective left batch strides, including broadcast zeroes
     * @param rightBatchStrides effective right batch strides, including broadcast zeroes
     * @param resultBatchStrides injective result batch strides
     * @param batchCount checked batch product
     * @param m normalized M extent
     * @param k normalized K extent
     * @param n normalized N extent
     * @param removedM whether rank-one left promotion removes M from the result
     * @param removedN whether rank-one right promotion removes N from the result
     * @param leftOffset left layout origin in elements
     * @param rightOffset right layout origin in elements
     * @param resultOffset result layout origin in elements
     * @param leftMStride effective left M stride
     * @param leftKStride effective left K stride
     * @param rightKStride effective right K stride
     * @param rightNStride effective right N stride
     * @param resultMStride effective result M stride
     * @param resultNStride effective result N stride
     * @param outputCount checked logical result element count
     * @param leftSpan checked referenced left span in elements
     * @param rightSpan checked referenced right span in elements
     * @param resultSpan checked referenced result span in elements
     */
    public record Geometry(DataType leftType, DataType rightType, DataType resultType,
            long[] batchExtents, long[] leftBatchStrides, long[] rightBatchStrides,
            long[] resultBatchStrides, long batchCount, long m, long k, long n,
            boolean removedM, boolean removedN, long leftOffset, long rightOffset, long resultOffset,
            long leftMStride, long leftKStride, long rightKStride, long rightNStride,
            long resultMStride, long resultNStride, long outputCount,
            long leftSpan, long rightSpan, long resultSpan) {
        /** Snapshots and validates complete normalized geometry. */
        public Geometry {
            Objects.requireNonNull(leftType, "leftType"); Objects.requireNonNull(rightType, "rightType");
            Objects.requireNonNull(resultType, "resultType");
            batchExtents = batchExtents.clone(); leftBatchStrides = leftBatchStrides.clone();
            rightBatchStrides = rightBatchStrides.clone(); resultBatchStrides = resultBatchStrides.clone();
            int rank = batchExtents.length;
            if (leftBatchStrides.length != rank || rightBatchStrides.length != rank
                    || resultBatchStrides.length != rank || batchCount < 0 || m < 0 || k < 0
                    || n < 0 || outputCount < 0 || leftOffset < 0 || rightOffset < 0
                    || resultOffset < 0 || leftMStride < 0 || leftKStride < 0
                    || rightKStride < 0 || rightNStride < 0 || resultMStride < 0
                    || resultNStride < 0 || leftSpan < 0 || rightSpan < 0 || resultSpan < 0)
                throw new IllegalArgumentException("MATMUL geometry is invalid");
        }
        /**
         * Returns the normalized batch extents without exposing retained state.
         *
         * @return a fresh non-null array
         */
        @Override public long[] batchExtents() { return batchExtents.clone(); }
        /**
         * Returns the effective left batch strides without exposing retained state.
         *
         * @return a fresh non-null array
         */
        @Override public long[] leftBatchStrides() { return leftBatchStrides.clone(); }
        /**
         * Returns the effective right batch strides without exposing retained state.
         *
         * @return a fresh non-null array
         */
        @Override public long[] rightBatchStrides() { return rightBatchStrides.clone(); }
        /**
         * Returns the injective result batch strides without exposing retained state.
         *
         * @return a fresh non-null array
         */
        @Override public long[] resultBatchStrides() { return resultBatchStrides.clone(); }

        /**
         * Packs bases, batch arrays, M/K/N strides, and counts for a generated invocation.
         * @param leftBase carrier-relative left element base
         * @param rightBase carrier-relative right element base
         * @param resultBase carrier-relative output element base
         * @return a new packed primitive geometry array
         */
        public long[] pack(long leftBase, long rightBase, long resultBase) {
            return pack(leftBase,rightBase,resultBase,0,0);
        }

        /**
         * Packs one fused rank-one bias origin and N stride with the ordinary MATMUL geometry.
         * @param leftBase carrier-relative left element base
         * @param rightBase carrier-relative right element base
         * @param resultBase carrier-relative output element base
         * @param biasBase checked carrier-relative bias origin including its layout offset
         * @param biasNStride checked non-negative logical bias stride
         * @return a new packed primitive geometry array
         */
        public long[] pack(long leftBase,long rightBase,long resultBase,long biasBase,
                long biasNStride) {
            int rank = batchExtents.length;
            long[] packed = new long[18 + 4 * rank];
            packed[0] = Math.addExact(leftBase, leftOffset); packed[1] = Math.addExact(rightBase, rightOffset);
            packed[2] = Math.addExact(resultBase, resultOffset); packed[3] = batchCount;
            packed[4] = m; packed[5] = k; packed[6] = n; packed[7] = leftMStride;
            packed[8] = leftKStride; packed[9] = rightKStride; packed[10] = rightNStride;
            packed[11] = resultMStride; packed[12] = resultNStride; packed[13] = rank;
            packed[14] = biasBase; packed[15] = biasNStride;
            packed[16] = outputCount; packed[17] = 1;
            int cursor = 18;
            System.arraycopy(batchExtents, 0, packed, cursor, rank); cursor += rank;
            System.arraycopy(leftBatchStrides, 0, packed, cursor, rank); cursor += rank;
            System.arraycopy(rightBatchStrides, 0, packed, cursor, rank); cursor += rank;
            System.arraycopy(resultBatchStrides, 0, packed, cursor, rank);
            return packed;
        }
    }
}

/** Applies the fixed CPU 0008F realization eligibility and threshold policy on the cold path. */
final class CpuMatmulCandidateSelector {
    /** Creates a stateless bounded selector. */
    CpuMatmulCandidateSelector() { }

    /**
     * Constructs the eligible bounded set and selects its deterministic production member.
     *
     * @param facts checked non-null occurrence facts
     * @return immutable candidates in scalar, vector, scalar-tiled, vector-tiled order and selection
     * @throws NullPointerException if {@code facts} is {@code null}
     * @throws ArithmeticException if the multiply-add count overflows
     */
    Selection select(Facts facts) {
        Objects.requireNonNull(facts, "facts");
        long operations = Math.multiplyExact(Math.multiplyExact(
                Math.multiplyExact(facts.batchCount(), facts.m()), facts.k()), facts.n());
        boolean vectorEligible = facts.k() > 0 && facts.rightNStride() == 1
                && facts.resultNStride() == 1 && !facts.terminal()
                && facts.leftType() == facts.rightType() && facts.rightType() == facts.resultType()
                && (facts.resultType() == DataType.FLOAT32 || facts.resultType() == DataType.FLOAT64
                    || facts.resultType() == DataType.INT32 || facts.resultType() == DataType.INT64)
                && facts.n() >= facts.preferredLanes();
        var candidates = new ArrayList<CpuMatmulIr.Realization>();
        candidates.add(CpuMatmulIr.Realization.DIRECT_SCALAR);
        if (vectorEligible) candidates.add(CpuMatmulIr.Realization.DIRECT_N_VECTOR);
        if (facts.m() >= 2 && facts.n() >= 2 && operations >= 16_384)
            candidates.add(CpuMatmulIr.Realization.TILED_SCALAR_2X2);
        if (vectorEligible && facts.m() >= 2 && facts.n() >= 2L * facts.preferredLanes()
                && operations >= 65_536)
            candidates.add(CpuMatmulIr.Realization.TILED_N_VECTOR_2X2);
        CpuMatmulIr.Realization selected;
        if (operations == 0 || facts.k() == 0 || facts.m() == 1 || facts.n() == 1
                || operations < 4_096) selected = CpuMatmulIr.Realization.DIRECT_SCALAR;
        else if (operations >= 65_536 && candidates.contains(
                CpuMatmulIr.Realization.TILED_N_VECTOR_2X2))
            selected = CpuMatmulIr.Realization.TILED_N_VECTOR_2X2;
        else if (operations >= 16_384 && !vectorEligible)
            selected = CpuMatmulIr.Realization.TILED_SCALAR_2X2;
        else if (vectorEligible) selected = CpuMatmulIr.Realization.DIRECT_N_VECTOR;
        else selected = CpuMatmulIr.Realization.DIRECT_SCALAR;
        return new Selection(candidates, selected, operations);
    }

    /** Cold facts used only for eligibility and selection.
     * @param leftType exact left type
     * @param rightType exact right type
     * @param resultType promoted result and accumulator type
     * @param batchCount checked batch product
     * @param m normalized row count
     * @param k normalized contraction count
     * @param n normalized column count
     * @param rightNStride right logical-N element stride
     * @param resultNStride result logical-N element stride
     * @param preferredLanes exact preferred species lane count
     * @param terminal whether a fused terminal is present
     */
    record Facts(DataType leftType, DataType rightType, DataType resultType,
            long batchCount, long m, long k, long n, long rightNStride, long resultNStride,
            int preferredLanes, boolean terminal) {
        /** Validates non-negative geometry and a positive preferred lane count. */
        Facts {
            Objects.requireNonNull(leftType, "leftType"); Objects.requireNonNull(rightType, "rightType");
            Objects.requireNonNull(resultType, "resultType");
            if (batchCount < 0 || m < 0 || k < 0 || n < 0 || rightNStride < 0
                    || resultNStride < 0 || preferredLanes <= 0)
                throw new IllegalArgumentException("MATMUL candidate facts must be non-negative");
        }
    }

    /** Immutable eligible-set and selected-form result.
     * @param candidates ordered bounded eligible realization set
     * @param selected deterministic production member of {@code candidates}
     * @param multiplyAdds checked occurrence operation count
     */
    record Selection(List<CpuMatmulIr.Realization> candidates,
            CpuMatmulIr.Realization selected, long multiplyAdds) {
        /** Snapshots and validates one selection. */
        Selection {
            candidates = List.copyOf(candidates); Objects.requireNonNull(selected, "selected");
            if (candidates.isEmpty() || candidates.getFirst() != CpuMatmulIr.Realization.DIRECT_SCALAR
                    || candidates.size() > 4 || !candidates.contains(selected) || multiplyAdds < 0)
                throw new IllegalArgumentException("MATMUL selection disagrees");
        }
    }
}
