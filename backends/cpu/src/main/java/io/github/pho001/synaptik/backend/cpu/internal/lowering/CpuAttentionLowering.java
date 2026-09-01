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
 * Cold fail-closed lowering for one atomic static scaled-dot-product-attention occurrence.
 *
 * <p>This CPU-private owner converts an already selected occurrence into direct generated-route
 * identity and invocation geometry. It does not choose backend ownership or execute attention.
 */
public final class CpuAttentionLowering {
  private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

  /** Creates a stateless attention lowerer. */
  public CpuAttentionLowering() {}

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
        new OperationCapabilityQuery(
            node.operation(),
            node.inputs().stream().map(id -> require(values, id).descriptor()).toList(),
            node.outputs().stream().map(id -> require(values, id).descriptor()).toList());
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
      batch[axis] = Math.max(a, Math.max(b, c));
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
   * <p>It carries rank/layout/address facts that bind a schema-57 generated body; none of these
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
