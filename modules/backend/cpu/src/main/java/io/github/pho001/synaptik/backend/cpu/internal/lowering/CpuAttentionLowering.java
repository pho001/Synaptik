package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAttentionIr;
import io.github.pho001.synaptik.model.datatype.BFloat16Bits;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.datatype.DataTypePromotion;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionAttrs;
import io.github.pho001.synaptik.model.operation.attention.ScaledDotProductAttentionKind;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/**
 * Cold fail-closed lowering for atomic static or one-named-batch scaled-dot-product attention.
 *
 * <p>This CPU-private owner converts an already selected occurrence into direct generated-route
 * identity and numeric invocation geometry. Symbolic recipes preserve typed roles, broadcast
 * layouts and optional weights without retaining descriptors or sampling B. It does not choose
 * backend ownership or execute attention.
 */
public final class CpuAttentionLowering {
  private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

  /** Creates a stateless attention lowerer. */
  public CpuAttentionLowering() {}

  /**
   * Checks original symbolic batch broadcasting and preserves every typed role and output sibling.
   * @param context complete one-node attention projection with fixed query/key/embedding axes
   * @return unresolved numeric attention geometry without a representative batch
   * @throws IllegalArgumentException if original semantics, types or dimension equations disagree
   * @throws ArithmeticException if fixed coefficient or scratch arithmetic overflows
   */
  public CpuPartitionLowering.LoweredPartition lowerBatch(PrepareContext<? extends BackendAnalysisInputs> context) {
      return lowerBatch(context,CpuPartitionLowering.batchLayouts(context));
  }

  /**
   * Lowers original descriptors using the complete local graph's proved physical mappings.
   * The map is consumed only during lowering; the recipe retains numeric boundary facts only.
   * @param context complete original ordinary unit projection
   * @param layouts non-null full-local value mappings, preserving view strides and offsets
   * @return unresolved numeric family geometry with B-independent generated facts
   * @throws NullPointerException if a required mapping or input is absent
   * @throws IllegalArgumentException if original semantics or numeric mappings disagree
   * @throws ArithmeticException if fixed geometry arithmetic overflows
   */
  public CpuPartitionLowering.LoweredPartition lowerBatch(PrepareContext<? extends BackendAnalysisInputs> context,
          Map<ValueId,CpuAccessPlan.BatchBinding> layouts) {
    var values = batchValues(context);
    if (context.nodes().size() != 1) throw new IllegalArgumentException("attention requires one node");
    var node = context.nodes().getFirst();
    if (node.operation().kind() != ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION
        || !(node.operation().attrs() instanceof ScaledDotProductAttentionAttrs attrs)
        || node.inputs().size() < 3 || node.inputs().size() > 4 || node.outputs().isEmpty() || node.outputs().size() > 2
        || node.outputs().stream().anyMatch(node.inputs()::contains)
        || new java.util.HashSet<>(node.outputs()).size() != node.outputs().size())
      throw new IllegalArgumentException("attention occurrence or output alias disagrees");
    var q = require(values,node.inputs().get(0)); var k = require(values,node.inputs().get(1)); var v = require(values,node.inputs().get(2));
    int qr = q.descriptor().shape().rank(), kr = k.descriptor().shape().rank(), vr = v.descriptor().shape().rank();
    if (qr < 2 || kr < 2 || vr < 2) throw new IllegalArgumentException("attention rank is too small");
    long l = fixed(q,qr-2), s = fixed(k,kr-2), e = fixed(q,qr-1), ev = fixed(v,vr-1);
    if (e <= 0 || fixed(k,kr-1) != e || fixed(v,vr-2) != s) throw new IllegalArgumentException("attention feature equations disagree");
    var result = DataTypePromotion.promoteFloating(DataTypePromotion.promoteFloating(q.descriptor().dataType(),k.descriptor().dataType()),v.descriptor().dataType());
    if (attrs.scale().isPresent() && attrs.scale().orElseThrow().dataType() != result)
      throw new IllegalArgumentException("attention scale representation disagrees");
    double scale = attrs.scale().isPresent() ? result == DataType.FLOAT64 ? attrs.scale().orElseThrow().float64Value()
        : attrs.scale().orElseThrow().float32Value() : result == DataType.FLOAT64 ? 1.0d/StrictMath.sqrt((double)e) : (float)(1.0d/StrictMath.sqrt((double)e));
    if (!Double.isFinite(scale) || scale <= 0) throw new IllegalArgumentException("attention scale is invalid");
    int batchRank = Math.max(qr-2,Math.max(kr-2,vr-2));
    var prefix = new ArrayList<io.github.pho001.synaptik.model.shape.Dimension>();
    for (int axis = 0; axis < batchRank; axis++) {
      io.github.pho001.synaptik.model.shape.Dimension selected = new io.github.pho001.synaptik.model.shape.StaticDimension(1);
      for (var value : List.of(q,k,v)) {
        int from = axis - (batchRank-(value.descriptor().shape().rank()-2));
        if (from < 0) continue;
        var dimension = value.descriptor().shape().dimensions().get(from);
        if (dimension instanceof io.github.pho001.synaptik.model.shape.StaticDimension d && d.size() == 1) continue;
        if (!(selected instanceof io.github.pho001.synaptik.model.shape.StaticDimension d && d.size() == 1) && !selected.equals(dimension))
          throw new IllegalArgumentException("attention symbolic prefix does not broadcast");
        selected = dimension;
      }
      prefix.add(selected);
    }
    var outShape = new ArrayList<>(prefix); outShape.add(new io.github.pho001.synaptik.model.shape.StaticDimension(l)); outShape.add(new io.github.pho001.synaptik.model.shape.StaticDimension(ev));
    var scoreShape = new ArrayList<>(prefix); scoreShape.add(new io.github.pho001.synaptik.model.shape.StaticDimension(l)); scoreShape.add(new io.github.pho001.synaptik.model.shape.StaticDimension(s));
    for (int i = 0; i < node.outputs().size(); i++) {
      var output = require(values,node.outputs().get(i));
      boolean grad = q.descriptor().requiresGrad() || k.descriptor().requiresGrad() || i == 0 && v.descriptor().requiresGrad();
      if (output.descriptor().dataType() != result || output.descriptor().requiresGrad() != grad
          || !output.descriptor().shape().dimensions().equals(i == 0 ? outShape : scoreShape))
        throw new IllegalArgumentException("attention output sibling shape/type/gradient disagrees");
    }
    if (node.inputs().size() == 4) {
      var mask = require(values,node.inputs().get(3)); int rank = mask.descriptor().shape().rank();
      if (mask.descriptor().dataType() != DataType.BOOL || mask.descriptor().requiresGrad() || rank > scoreShape.size())
        throw new IllegalArgumentException("attention mask type/rank disagrees");
      for (int axis = 0; axis < rank; axis++) {
        var d = mask.descriptor().shape().dimensions().get(axis);
        if (!(d instanceof io.github.pho001.synaptik.model.shape.StaticDimension singleton && singleton.size() == 1)
            && !d.equals(scoreShape.get(axis+scoreShape.size()-rank))) throw new IllegalArgumentException("attention mask does not broadcast");
      }
    }
    var ids = new ArrayList<ValueId>(); var roles = new ArrayList<Integer>();
    for (var id : node.inputs()) { int position = ids.indexOf(id); if (position < 0) { position=ids.size(); ids.add(id); } roles.add(position); }
    int uniqueInputs = ids.size(); ids.addAll(node.outputs());
    var bindings = new ArrayList<CpuAccessPlan.BatchBinding>(); var plans = new ArrayList<CpuAccessPlan>();
    for (int i = 0; i < ids.size(); i++) {
      var binding = batchBinding(require(values,ids.get(i)),layouts);
      if (i >= uniqueInputs) binding = writeRecipe(binding);
      bindings.add(binding);
      plans.add(new CpuAccessPlan(i < uniqueInputs ? CpuAccessPlan.AccessKind.READ : CpuAccessPlan.AccessKind.WRITE,CpuAccessPlan.Regime.SCALAR_ALL_ZERO,0,List.of(),0));
    }
    var types = ids.stream().map(id -> require(values,id).descriptor().dataType()).toList();
    var ir = new CpuAttentionIr(q.descriptor().dataType(),k.descriptor().dataType(),v.descriptor().dataType(),result,node.inputs().size()==4,attrs.causal(),node.outputs().size(),roles,types,plans);
    long[] parameters = new long[8+2*batchRank];
    parameters[0]=batchRank; parameters[1]=l; parameters[2]=s; parameters[3]=e; parameters[4]=ev;
    parameters[5]=Double.doubleToRawLongBits(scale); parameters[6]=uniqueInputs; parameters[7]=node.outputs().size();
    for (int axis=0; axis<batchRank; axis++) {
      var d=prefix.get(axis);
      if (d instanceof io.github.pho001.synaptik.model.shape.StaticDimension fixed) parameters[8+axis]=fixed.size();
      else parameters[8+batchRank+axis]=1;
    }
    return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir,ids,types,bindings,List.of(),parameters));
  }

  private static NormalizedLayout normalized(CpuAccessPlan.Binding binding, long[] target) {
    long[] strides = new long[target.length]; int shift = target.length-binding.extents().size();
    for (int axis=0; axis<binding.extents().size(); axis++)
      strides[axis+shift] = binding.extents().get(axis)==1 && target[axis+shift]!=1 ? 0 : binding.effectiveStrides().get(axis);
    return new NormalizedLayout(binding.baseElementOffset(),strides);
  }

    private static Map<ValueId, GraphValue> batchValues(PrepareContext<?> context) {
        CpuPartitionLowering.requireBatchContext(context);
        var values = new LinkedHashMap<ValueId, GraphValue>();
        context.values().forEach(value -> values.put(value.id(),value));
        return values;
    }

    private static long fixed(GraphValue value, int axis) {
        var dimension = value.descriptor().shape().dimensions().get(axis);
        if (!(dimension instanceof io.github.pho001.synaptik.model.shape.StaticDimension fixed))
            throw new IllegalArgumentException("window and feature dimensions must be fixed");
        return fixed.size();
    }

    private static void sameAxis(GraphValue left, int a, GraphValue right, int b) {
        if (!left.descriptor().shape().dimensions().get(a).equals(
                right.descriptor().shape().dimensions().get(b)))
            throw new IllegalArgumentException("symbolic dimension equation disagrees");
    }

    private static CpuAccessPlan.BatchBinding batchBinding(GraphValue value,
            Map<ValueId,CpuAccessPlan.BatchBinding> layouts) {
        var mapping = Objects.requireNonNull(layouts.get(value.id()), "missing full-local layout");
        var canonical = CpuPartitionLowering.batchCanonical(value.descriptor(),CpuAccessPlan.AccessKind.READ);
        if (!mapping.extentConstants().equals(canonical.extentConstants())
                || !mapping.extentBatchFactors().equals(canonical.extentBatchFactors()))
            throw new IllegalArgumentException("full-local mapping disagrees with original symbolic shape");
        var p = mapping.plan();
        var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,p.regime(),p.iterationRank(),p.axisRoles(),p.contiguousSuffix());
        return new CpuAccessPlan.BatchBinding(read,mapping.extentConstants(),mapping.extentBatchFactors(),
                mapping.baseElementOffset(),mapping.strideConstants(),mapping.strideBatchFactors());
    }

    private static CpuAccessPlan writePlan(CpuAccessPlan.BatchBinding binding) {
        var p = binding.plan();
        return new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE, p.regime(), p.iterationRank(),
                p.axisRoles(), p.contiguousSuffix());
    }

    private static CpuAccessPlan.BatchBinding writeRecipe(CpuAccessPlan.BatchBinding binding) {
        return new CpuAccessPlan.BatchBinding(writePlan(binding),binding.extentConstants(),
                binding.extentBatchFactors(),binding.baseElementOffset(),binding.strideConstants(),
                binding.strideBatchFactors());
    }

    private static String batchIdentity(io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPortableKernelIr ir,
            List<CpuAccessPlan.BatchBinding> bindings, long[] parameters) {
        var key = new StringBuilder(ir.structuralKey());
        for (var b : bindings) key.append('|').append(b.extentConstants()).append(b.extentBatchFactors())
                .append(b.baseElementOffset()).append(b.strideConstants()).append(b.strideBatchFactors());
        return key.append('|').append(Arrays.toString(parameters)).toString();
    }

  /**
   * Lowers one exact supported attention occurrence to complete broadcast-batch/query rows.
   *
   * @param context non-null single-node CPU preparation projection
   * @return immutable direct scalar lowering with exact unique boundaries and cold geometry
   * @throws NullPointerException if {@code context} is null
   * @throws IllegalArgumentException if semantic, descriptor, layout, or output facts disagree
   * @throws ArithmeticException if count, address, or workspace arithmetic overflows
   */
  public CpuPartitionLowering.LoweredPartition lower(
      PrepareContext<? extends BackendAnalysisInputs> context) {
    Objects.requireNonNull(context, "context");
    if (context.nodes().size() != 1)
      throw new IllegalArgumentException("CPU attention requires exactly one occurrence");
    var node = context.nodes().getFirst();
    Map<ValueId, GraphValue> values = new LinkedHashMap<>();
    context.values().forEach(value -> values.put(value.id(), value));
    var query =
        new OperationCapabilityQuery(node.operation(), node.inputs().stream().map(id -> require(values, id).descriptor()).toList(), node.outputs().stream().map(id -> require(values, id).descriptor()).toList());
    if (!capabilities.supports(query)
        || node.operation().kind() != ScaledDotProductAttentionKind.SCALED_DOT_PRODUCT_ATTENTION
        || !(node.operation().attrs() instanceof ScaledDotProductAttentionAttrs attrs))
      throw new IllegalArgumentException("unsupported CPU attention occurrence");

    var semanticInputs = node.inputs();
    var unique = new ArrayList<ValueId>();
    var rolePositions = new ArrayList<Integer>();
    for (ValueId id : semanticInputs) {
      int position = unique.indexOf(id);
      if (position < 0) {
        position = unique.size();
        unique.add(id);
      }
      rolePositions.add(position);
    }
    var boundaries = new ArrayList<ValueId>(unique);
    boundaries.addAll(node.outputs());
    var types = boundaries.stream().map(id -> require(values, id).descriptor().dataType()).toList();
    var bindings = new ArrayList<CpuAccessPlan.Binding>();
    var spans = new ArrayList<Long>();
    var plans = new ArrayList<CpuAccessPlan>();
    for (int i = 0; i < boundaries.size(); i++) {
      GraphValue value = require(values, boundaries.get(i));
      boolean write = i >= unique.size();
      CpuAccessPlan.Binding binding =
          binding(value, write ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ);
      bindings.add(binding);
      plans.add(
          new CpuAccessPlan(
              write ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ,
              CpuAccessPlan.Regime.SCALAR_ALL_ZERO,
              0,
              List.of(),
              0));
      spans.add(value.descriptor().layout().orElseThrow().referencedElementSpan());
    }

    GraphValue q = require(values, semanticInputs.get(0));
    GraphValue k = require(values, semanticInputs.get(1));
    GraphValue v = require(values, semanticInputs.get(2));
    DataType result =
        DataTypePromotion.promoteFloating(
            DataTypePromotion.promoteFloating(q.descriptor().dataType(), k.descriptor().dataType()),
            v.descriptor().dataType());
    long[] qe = q.descriptor().shape().toLongArray();
    long[] ke = k.descriptor().shape().toLongArray();
    long[] ve = v.descriptor().shape().toLongArray();
    int batchRank = Math.max(qe.length - 2, Math.max(ke.length - 2, ve.length - 2));
    long[] batch = new long[batchRank];
    for (int axis = 0; axis < batchRank; axis++) {
      long a = aligned(qe, batchRank, axis),
          b = aligned(ke, batchRank, axis),
          c = aligned(ve, batchRank, axis);
      batch[axis] = a != 1 ? a : b != 1 ? b : c;
    }
    long l = qe[qe.length - 2],
        s = ke[ke.length - 2],
        e = qe[qe.length - 1],
        ev = ve[ve.length - 1];
    long batchCount = product(batch), rowCount = Math.multiplyExact(batchCount, l);
    if (ev == 0 && node.outputs().size() == 1) rowCount = 0;
    long slice =
        s == 0 || rowCount == 0
            ? 0
            : align8(Math.multiplyExact(s, result == DataType.FLOAT64 ? 8L : 4L));
    double scale =
        attrs.scale().isPresent()
            ? switch (result) {
                      case FLOAT64 -> attrs.scale().orElseThrow().float64Value();
                      case FLOAT32 -> attrs.scale().orElseThrow().float32Value();
                      case FLOAT16, FLOAT8_E4M3, FLOAT8_E5M2 -> attrs.scale().orElseThrow().float32Value();
                      case BFLOAT16 -> BFloat16Bits.toFloat(attrs.scale().orElseThrow().bfloat16Bits());
                      default -> throw new AssertionError();
                    }
            : result == DataType.FLOAT64
                ? 1.0d / StrictMath.sqrt((double) e)
                : (float) (1.0d / StrictMath.sqrt((double) e));
    var geometry =
        new Geometry(
            batch,
            l,
            s,
            e,
            ev,
            rowCount,
            slice,
            q.descriptor().dataType(),
            k.descriptor().dataType(),
            v.descriptor().dataType(),
            result,
            scale,
            rolePositions,
            unique.size(),
            node.outputs().size(),
            normalized(q, batchRank, l, e),
            normalized(k, batchRank, s, e),
            normalized(v, batchRank, s, ev),
            semanticInputs.size() == 4
                ? normalized(require(values, semanticInputs.get(3)), batchRank, l, s)
                : Optional.empty(),
            normalized(require(values, node.outputs().get(0)), batchRank, l, ev).orElseThrow(),
            node.outputs().size() == 2
                ? normalized(require(values, node.outputs().get(1)), batchRank, l, s)
                : Optional.empty());
    var ir =
        new CpuAttentionIr(
            q.descriptor().dataType(),
            k.descriptor().dataType(),
            v.descriptor().dataType(),
            result,
            semanticInputs.size() == 4,
            attrs.causal(),
            node.outputs().size(),
            rolePositions,
            types,
            plans);
    return new CpuPartitionLowering.LoweredPartition(
        ir,
        boundaries,
        bindings,
        spans,
        types,
        List.of(),
        new long[] {rowCount},
        rowCount,
        "legal: atomic direct static scaled-dot-product attention",
        new long[0],
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.empty(),
        Optional.of(geometry));
  }

  private static Optional<NormalizedLayout> normalized(
      GraphValue value, int batchRank, long penultimate, long last) {
    var d = value.descriptor();
    long[] shape = d.shape().toLongArray(), source = d.layout().orElseThrow().strides();
    int resultRank = batchRank + 2;
    long[] strides = new long[resultRank];
    int shift = resultRank - shape.length;
    for (int axis = 0; axis < shape.length; axis++) {
      int target = axis + shift;
      long targetExtent = target < batchRank ? -1 : target == batchRank ? penultimate : last;
      strides[target] = shape[axis] == 1 && targetExtent != 1 ? 0 : source[axis];
    }
    return Optional.of(new NormalizedLayout(d.layout().orElseThrow().storageOffset(), strides));
  }

  private static long aligned(long[] shape, int batchRank, int axis) {
    int batch = shape.length - 2, shift = batchRank - batch;
    return axis < shift ? 1 : shape[axis - shift];
  }

  private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
    GraphValue value = values.get(id);
    if (value == null)
      throw new IllegalArgumentException("attention value is not projected: " + id);
    return value;
  }

  private static CpuAccessPlan.Binding binding(GraphValue value, CpuAccessPlan.AccessKind kind) {
    LayoutDescriptor layout = value.descriptor().layout().orElseThrow();
    long[] extents = value.descriptor().shape().toLongArray(), strides = layout.strides();
    int suffix = 0;
    long expected = 1;
    for (int axis = extents.length - 1; axis >= 0; axis--) {
      if (strides[axis] != expected) break;
      suffix++;
      expected = Math.multiplyExact(expected, Math.max(1, extents[axis]));
    }
    var roles = new ArrayList<CpuAccessPlan.AxisRole>();
    for (int axis = 0; axis < extents.length; axis++)
      roles.add(
          strides[axis] == 0
              ? CpuAccessPlan.AxisRole.BROADCAST
              : axis >= extents.length - suffix
                  ? CpuAccessPlan.AxisRole.CONTIGUOUS
                  : CpuAccessPlan.AxisRole.STRIDED);
    var plan =
        new CpuAccessPlan(
            kind,
            suffix == extents.length
                ? CpuAccessPlan.Regime.DENSE_LINEAR
                : CpuAccessPlan.Regime.GENERAL_ODOMETER,
            extents.length,
            roles,
            suffix);
    long count = product(extents);
    return CpuAccessPlan.Binding.create(
        plan,
        extents,
        layout.storageOffset(),
        strides,
        count,
        0,
        count,
        layout.referencedElementSpan());
  }

  private static long product(long[] extents) {
    for (long x : extents) if (x == 0) return 0;
    long result = 1;
    for (long x : extents) result = Math.multiplyExact(result, x);
    return result;
  }

  private static long align8(long x) {
    return Math.addExact(x, 7) & -8L;
  }

  /**
   * One normalized carrier-relative layout over broadcast-batch plus two trailing axes.
   *
   * @param offset non-negative carrier-relative element offset
   * @param strides non-null non-negative normalized element strides; defensively copied
   */
  public record NormalizedLayout(long offset, long[] strides) {
    /**
     * Validates and snapshots one normalized non-negative layout.
     *
     * @throws NullPointerException if {@code strides} is null
     * @throws IllegalArgumentException if {@code offset} or any stride is negative
     */
    public NormalizedLayout {
      strides = strides.clone();
      if (offset < 0 || Arrays.stream(strides).anyMatch(x -> x < 0))
        throw new IllegalArgumentException("invalid attention layout");
    }

    /**
     * Returns a defensive stride snapshot.
     *
     * @return a new non-null array of normalized non-negative element strides
     */
    @Override
    public long[] strides() {
      return strides.clone();
    }
  }

  /**
   * Complete cold primitive attention invocation geometry for one lowered occurrence.
   *
   * <p>It carries rank/layout/address facts that bind the current generated body; none of these
   * fields are Model semantics or Runtime route-selection state.
   *
   * @param batchExtents non-null broadcast-batch extents; defensively copied
   * @param queryLength non-negative query-row count per broadcast batch
   * @param keyLength non-negative key positions per row
   * @param embedding positive query/key embedding extent
   * @param valueEmbedding non-negative value/output embedding extent
   * @param rowCount non-negative complete query-row work count
   * @param scratchSliceBytes non-negative bytes per selected range
   * @param queryType exact query data type
   * @param keyType exact key data type
   * @param valueType exact value data type
   * @param resultType promoted output and optional-weights data type
   * @param scale finite positive resolved primitive scale
   * @param roleBoundaryPositions non-null semantic-role positions in the unique-input prefix
   * @param uniqueInputCount positive distinct input-boundary count
   * @param outputCount one or two ordered output boundaries
   * @param query non-null normalized query layout
   * @param key non-null normalized key layout
   * @param value non-null normalized value layout
   * @param mask non-null optional normalized canonical-BOOL mask layout
   * @param output non-null normalized output layout
   * @param weights non-null optional normalized weights layout for output count two
   */
  public record Geometry(
      long[] batchExtents,
      long queryLength,
      long keyLength,
      long embedding,
      long valueEmbedding,
      long rowCount,
      long scratchSliceBytes,
      DataType queryType,
      DataType keyType,
      DataType valueType,
      DataType resultType,
      double scale,
      List<Integer> roleBoundaryPositions,
      int uniqueInputCount,
      int outputCount,
      Optional<NormalizedLayout> query,
      Optional<NormalizedLayout> key,
      Optional<NormalizedLayout> value,
      Optional<NormalizedLayout> mask,
      NormalizedLayout output,
      Optional<NormalizedLayout> weights) {

        /**
         * Unresolved numeric invocation recipe. No representative shape, graph or invocation is retained.
         * @param ir exact B-independent generated facts
         * @param boundaries ordered physical boundaries
         * @param boundaryTypes exact boundary storage types
         * @param bindings affine numeric mappings, including the output mapping
         * @param virtualValues private removed intermediate values
         * @param parameters fixed family geometry, defensively copied

         */
        public record BatchGeometry(CpuAttentionIr ir, List<ValueId> boundaries,
                List<io.github.pho001.synaptik.model.datatype.DataType> boundaryTypes,
                List<CpuAccessPlan.BatchBinding> bindings, List<ValueId> virtualValues,
                long[] parameters) implements CpuPartitionLowering.BatchUnitGeometry {
            /**
             * Snapshots immutable compile facts and checked numeric coefficients.
             * @param ir B-independent generated facts
             * @param boundaries ordered physical boundary identities
             * @param boundaryTypes physical storage types in boundary order
             * @param bindings affine layout coefficients in boundary order
             * @param virtualValues private removed intermediates
             * @param parameters fixed family-specific numeric geometry
             */
            public BatchGeometry {
                Objects.requireNonNull(ir, "ir");
                boundaries = List.copyOf(boundaries); boundaryTypes = List.copyOf(boundaryTypes);
                bindings = List.copyOf(bindings); virtualValues = List.copyOf(virtualValues);
                parameters = parameters.clone();
            }
            /**
             * Returns isolated attention geometry cells used by numeric batch binding.
             * Prepared constants and coefficients do not contain an evaluated actual B.
             * Mutating the result cannot change this graph-free recipe or generated identity.
             * @return a new non-null caller-owned copy of the numeric geometry cells
             */
            @Override public long[] parameters() { return parameters.clone(); }
            /** @return checked affine complete-unit work count, not bytes */
            @Override public io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize iterationCount() { long c=parameters[1], f=0; int rank=Math.toIntExact(parameters[0]);
                if (parameters[4]==0 && parameters[7]==1) return new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(0,0,1);
                for (int i=0;i<rank;i++) { long ec=parameters[8+i], ef=parameters[8+rank+i];
                    if (f!=0 && ef!=0) throw new IllegalArgumentException("non-affine attention rows");
                    f=Math.addExact(Math.multiplyExact(f,ec),Math.multiplyExact(c,ef)); c=Math.multiplyExact(c,ec); }
                return new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(c,f,1); }
            /**
             * Reserves one aligned key-row slice for structurally present work. A variable-B
             * reservation is a constant capacity; bind(0) gives the ordinary zero-work geometry
             * a zero active slice without changing generated identity or allocating per step.
             * @return affine direct-scalar scratch capacity in bytes
             */
            @Override public io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize scratchBytes() { var rows=iterationCount(); long slice=rows.constantBytes()==0 && rows.bytesPerBatch()==0 || parameters[2]==0 ? 0 : align8(Math.multiplyExact(parameters[2],ir.resultType()==DataType.FLOAT64 ? 8L:4L));
                return new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(slice,0,1); }
            /** @return structural identity independent of actual B and value identities */
            @Override public String identity() { return batchIdentity(ir, bindings, parameters); }
            /** @return no mutable packed traversal cursors for this ordinary family */
            @Override public List<io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization.PackReset> resets() { return List.of(); }
            /**
             * Resolves only numeric geometry on the cold invocation boundary.
             * @param batch non-negative int-representable actual B
             * @return ordinary resolved family geometry with identical generated facts
             * @throws IllegalArgumentException if batch is negative
             * @throws ArithmeticException if geometry exceeds checked host limits
             */
            @Override public CpuPartitionLowering.LoweredPartition bind(long batch) {
                var resolved = new ArrayList<CpuAccessPlan.Binding>();
                for (var binding : bindings) resolved.add(binding.bind(batch));
                int rank=Math.toIntExact(parameters[0]), unique=Math.toIntExact(parameters[6]);
                long[] batchExtents=new long[rank];
                for (int i=0;i<rank;i++) batchExtents[i]=Math.addExact(parameters[8+i],Math.multiplyExact(parameters[8+rank+i],batch));
                long rows=iterationCount().byteSize(batch);
                long slice=rows==0 ? 0 : scratchBytes().constantBytes();
                long[] target=new long[rank+2]; System.arraycopy(batchExtents,0,target,0,rank);
                target[rank]=parameters[1]; target[rank+1]=parameters[3];
                var q=normalized(resolved.get(ir.roleBoundaryPositions().get(0)),target);
                target[rank]=parameters[2]; var k=normalized(resolved.get(ir.roleBoundaryPositions().get(1)),target);
                target[rank+1]=parameters[4]; var v=normalized(resolved.get(ir.roleBoundaryPositions().get(2)),target);
                target[rank]=parameters[1]; var output=normalized(resolved.get(unique),target);
                target[rank+1]=parameters[2];
                var mask=ir.masked() ? Optional.of(normalized(resolved.get(ir.roleBoundaryPositions().get(3)),target)) : Optional.<NormalizedLayout>empty();
                var weights=ir.outputCount()==2 ? Optional.of(normalized(resolved.get(unique+1),target)) : Optional.<NormalizedLayout>empty();
                var geometry=new Geometry(batchExtents,parameters[1],parameters[2],parameters[3],parameters[4],rows,slice,
                        ir.queryType(),ir.keyType(),ir.valueType(),ir.resultType(),Double.longBitsToDouble(parameters[5]),ir.roleBoundaryPositions(),unique,ir.outputCount(),
                        Optional.of(q),Optional.of(k),Optional.of(v),mask,output,weights);
                return new CpuPartitionLowering.LoweredPartition(ir,boundaries,resolved,
                        resolved.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),boundaryTypes,virtualValues,
                        new long[]{rows},rows,"legal: symbolic direct attention",new long[0],
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry));
            }
        }

    /**
     * One fixed generated assignment of a boundary's element base plus a layout delta.
     * @param cell non-negative destination cell in this family's packed geometry
     * @param boundary boundary position in the pack's carrier-base order
     * @param elementDelta immutable element offset added with checked arithmetic
     */
    public record BaseCell(int cell, int boundary, long elementDelta) { }

    /**
     * One original cursor seed restored before every temporal helper call, including retry.
     * @param cell non-negative destination cell in this family's packed geometry
     * @param value original range-start cursor value
     */
    public record ResetCell(int cell, long value) { }

    /**
     * Cold, immutable facts for emitting fixed numeric assignments, never a runtime program.
     * Scratch range offsets and invariant geometry remain bound once per invocation.
     * @param bases non-null boundary-base assignments, defensively snapshotted
     * @param resets non-null mutable-cursor assignments, defensively snapshotted
     */
    public record TemporalRebind(List<BaseCell> bases, List<ResetCell> resets) {
        /**
         * Snapshots the generation-time assignments; retains no invocation storage.
         * @param bases non-null base facts without null entries
         * @param resets non-null reset facts without null entries
         * @throws NullPointerException if either list or an entry is null
         */
        public TemporalRebind {
            bases = List.copyOf(bases);
            resets = List.copyOf(resets);
        }
    }

    /**
     * Describes temporal base rebinding and restoration of this family's original cursors.
     * Construct these facts only during generation or cold numeric binding; generated code
     * emits the assignments directly, without calling pack, allocating, or iterating facts.
     * Each boundary base is normalized using that boundary's own element width beforehand.
     * @param start non-negative inclusive ordinary helper range start
     * @return immutable owner-defined base locations and original cursor seeds
     * @throws IllegalArgumentException if start is negative
     * @throws ArithmeticException if checked layout or cursor arithmetic overflows
     */
    public TemporalRebind temporalRebind(long start) {
        if (start < 0) throw new IllegalArgumentException("negative temporal range start");
        int rank = batchExtents.length + 2, header = 14 + batchExtents.length;
        var bases = new ArrayList<BaseCell>(6);
        bases.add(new BaseCell(header, roleBoundaryPositions.get(0), query.orElseThrow().offset()));
        bases.add(new BaseCell(header + rank + 1, roleBoundaryPositions.get(1), key.orElseThrow().offset()));
        bases.add(new BaseCell(header + 2 * (rank + 1), roleBoundaryPositions.get(2), value.orElseThrow().offset()));
        if (mask.isPresent()) bases.add(new BaseCell(header + 3 * (rank + 1),
                roleBoundaryPositions.get(3), mask.orElseThrow().offset()));
        bases.add(new BaseCell(header + 4 * (rank + 1), uniqueInputCount, output.offset()));
        if (weights.isPresent()) bases.add(new BaseCell(header + 5 * (rank + 1),
                uniqueInputCount + 1, weights.orElseThrow().offset()));
        return new TemporalRebind(bases, List.of());
    }

    /**
     * Validates and snapshots exact geometry.
     *
     * @throws NullPointerException if a required reference component is null
     * @throws IllegalArgumentException if cardinality, layout-presence, extent, scale, or output
     *     facts do not describe one supported lowered attention occurrence
     */
    public Geometry {
      batchExtents = batchExtents.clone();
      roleBoundaryPositions = List.copyOf(roleBoundaryPositions);
      Objects.requireNonNull(queryType, "queryType");
      Objects.requireNonNull(keyType, "keyType");
      Objects.requireNonNull(valueType, "valueType");
      Objects.requireNonNull(resultType, "resultType");
      Objects.requireNonNull(query, "query");
      Objects.requireNonNull(key, "key");
      Objects.requireNonNull(value, "value");
      Objects.requireNonNull(mask, "mask");
      Objects.requireNonNull(output, "output");
      Objects.requireNonNull(weights, "weights");
      if (queryLength < 0
          || keyLength < 0
          || embedding <= 0
          || valueEmbedding < 0
          || rowCount < 0
          || scratchSliceBytes < 0
          || uniqueInputCount < 1
          || outputCount < 1
          || outputCount > 2
          || roleBoundaryPositions.size() != 3 + (mask.isPresent() ? 1 : 0)
          || weights.isPresent() != (outputCount == 2)
          || !Double.isFinite(scale)
          || scale <= 0) throw new IllegalArgumentException("attention geometry disagrees");
    }

    /**
     * Returns a defensive batch-extent snapshot.
     *
     * @return a new non-null array of non-negative broadcast-batch extents
     */
    @Override
    public long[] batchExtents() {
      return batchExtents.clone();
    }

    /**
     * Returns exact total workspace bytes for selected simultaneous ranges.
     *
     * @param ranges non-negative selected range count
     * @return the exact scratch-slice product in bytes
     * @throws ArithmeticException if the exact product overflows {@code long}
     */
    public long workspaceBytes(int ranges) {
      return Math.multiplyExact(scratchSliceBytes, ranges);
    }

    /**
     * Packs carrier bases and every normalized stride for the direct generated entry.
     *
     * @param bases non-null carrier-relative bases in unique-input then output-boundary order
     * @return a new primitive geometry array for the generated entry; never {@code null}
     * @throws NullPointerException if {@code bases} is null
     * @throws IllegalArgumentException if its count disagrees with this geometry's boundaries
     * @throws ArithmeticException if a base plus normalized offset overflows {@code long}
     */
    public long[] pack(long[] bases) {
      if (bases.length != uniqueInputCount + outputCount)
        throw new IllegalArgumentException("attention boundary bases disagree");
      int rank = batchExtents.length + 2, header = 14 + batchExtents.length;
      long[] g = new long[header + 6 * (rank + 1)];
      g[0] = roleBoundaryPositions.get(0);
      g[1] = roleBoundaryPositions.get(1);
      g[2] = roleBoundaryPositions.get(2);
      g[3] = mask.isPresent() ? roleBoundaryPositions.get(3) : -1;
      g[4] = uniqueInputCount;
      g[5] = outputCount == 2 ? uniqueInputCount + 1 : -1;
      g[6] = rank;
      g[7] = queryLength;
      g[8] = keyLength;
      g[9] = embedding;
      g[10] = valueEmbedding;
      g[11] = rowCount;
      g[12] = Double.doubleToRawLongBits(scale);
      g[13] = scratchSliceBytes;
      System.arraycopy(batchExtents, 0, g, 14, batchExtents.length);
      put(g, header, rank, bases[roleBoundaryPositions.get(0)], query.orElseThrow());
      put(g, header + rank + 1, rank, bases[roleBoundaryPositions.get(1)], key.orElseThrow());
      put(
          g,
          header + 2 * (rank + 1),
          rank,
          bases[roleBoundaryPositions.get(2)],
          value.orElseThrow());
      if (mask.isPresent())
        put(
            g,
            header + 3 * (rank + 1),
            rank,
            bases[roleBoundaryPositions.get(3)],
            mask.orElseThrow());
      put(g, header + 4 * (rank + 1), rank, bases[uniqueInputCount], output);
      if (weights.isPresent())
        put(g, header + 5 * (rank + 1), rank, bases[uniqueInputCount + 1], weights.orElseThrow());
      return g;
    }

    private static void put(long[] g, int p, int rank, long base, NormalizedLayout l) {
      g[p] = Math.addExact(base, l.offset());
      System.arraycopy(l.strides(), 0, g, p + 1, rank);
    }
  }
}
