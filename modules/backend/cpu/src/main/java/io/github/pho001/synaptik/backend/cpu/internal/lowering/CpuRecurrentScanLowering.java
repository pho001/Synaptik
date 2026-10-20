package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentDirection;
import io.github.pho001.synaptik.model.operation.recurrent.RecurrentScanKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Cold descriptor-only admission and direct lowering for one fixed FLOAT32 recurrent occurrence. */
public final class CpuRecurrentScanLowering {
    /** Creates a stateless lowerer; no runtime values or prototype storage are retained. */
    public CpuRecurrentScanLowering() { }

    /**
     * Validates the complete implemented descriptor domain, without inspecting storage or lengths.
     * @param query non-null exact ordered occurrence
     * @return immutable checked geometry, never null
     * @throws IllegalArgumentException if any type, flag, shape or canonical layout disagrees
     * @throws ArithmeticException if dimensions, boundary counts or scratch arithmetic overflow
     */
    public static Geometry geometry(OperationCapabilityQuery query) {
        if (!(query.operation().kind() instanceof RecurrentScanKind kind)
                || !(query.operation().attrs() instanceof RecurrentDirection direction))
            throw new IllegalArgumentException("not a fixed recurrent scan");
        boolean cell = kind == RecurrentScanKind.LSTM;
        int base = cell ? 6 : 5, gates = cell ? 4 : kind == RecurrentScanKind.GRU_RESET_AFTER ? 3 : 1;
        var in = query.inputs();
        var out = query.outputs();
        if ((in.size() != base && in.size() != base + 1) || out.size() != (cell ? 3 : 2))
            throw new IllegalArgumentException("recurrent port counts disagree");
        for (var d : java.util.stream.Stream.concat(in.stream(), out.stream()).toList()) {
            if (!d.shape().isFullyStatic() || !d.layout().equals(
                    java.util.Optional.of(LayoutDescriptor.contiguous(d.shape()))))
                throw new IllegalArgumentException("scan requires canonical static layout");
            for (long dim : d.shape().toLongArray()) Math.toIntExact(dim);
            Math.toIntExact(d.layout().orElseThrow().referencedElementSpan());
            Math.multiplyExact(d.layout().orElseThrow().referencedElementSpan(), d.dataType().byteWidth());
        }
        long[] x = in.get(0).shape().toLongArray(), h = in.get(2).shape().toLongArray();
        if (x.length != 3 || h.length != 2 || x[0] < 0 || x[1] < 0 || x[2] <= 0
                || h[0] != x[1] || h[1] <= 0)
            throw new IllegalArgumentException("scan input/state geometry disagrees");
        int t = Math.toIntExact(x[0]), b = Math.toIntExact(x[1]), i = Math.toIntExact(x[2]);
        int hidden = Math.toIntExact(h[1]), packed = Math.multiplyExact(gates, hidden);
        boolean grad = false;
        for (int role = 0; role < in.size(); role++) {
            var d = in.get(role);
            if (role == 1) {
                if (d.dataType() != DataType.INT64 || d.requiresGrad() || !d.shape().equals(Shape.of(b)))
                    throw new IllegalArgumentException("scan lengths must be non-gradient INT64[B]");
            } else {
                if (d.dataType() != DataType.FLOAT32) throw new IllegalArgumentException("scan requires FLOAT32");
                grad |= d.requiresGrad();
            }
        }
        int w = cell ? 4 : 3;
        if (cell && !in.get(3).shape().equals(Shape.of(b, hidden))
                || !in.get(w).shape().equals(Shape.of(packed, i))
                || !in.get(w + 1).shape().equals(Shape.of(packed, hidden))
                || in.size() > base && !in.get(base).shape().equals(Shape.of(packed)))
            throw new IllegalArgumentException("scan packed parameters disagree");
        for (int port = 0; port < out.size(); port++) {
            var d = out.get(port);
            if (d.dataType() != DataType.FLOAT32 || d.requiresGrad() != grad
                    || !d.shape().equals(port == 0 ? Shape.of(t, b, hidden) : Shape.of(b, hidden)))
                throw new IllegalArgumentException("scan output descriptors disagree");
        }
        long scratch = t == 0 || b == 0 ? 0 : Math.multiplyExact(Math.multiplyExact(2, packed), 4L);
        return new Geometry(kind, direction == RecurrentDirection.REVERSE, in.size() > base,
                t, b, i, hidden, gates, scratch);
    }

    /**
     * Lowers a single occurrence, deduplicating reads while preserving semantic roles and siblings.
     * @param context non-null single-node CPU projection
     * @return immutable direct facts, with no generated IR
     * @throws IllegalArgumentException if the occurrence or its projected values are unsupported
     * @throws ArithmeticException if checked geometry overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(PrepareContext<? extends BackendAnalysisInputs> context) {
        if (context.nodes().size() != 1) throw new IllegalArgumentException("one scan occurrence required");
        var node = context.nodes().getFirst();
        var byId = new java.util.LinkedHashMap<ValueId, TensorDescriptor>();
        context.values().forEach(v -> byId.put(v.id(), v.descriptor()));
        var g = geometry(new OperationCapabilityQuery(node.operation(),
                node.inputs().stream().map(byId::get).toList(), node.outputs().stream().map(byId::get).toList()));
        var boundaries = new ArrayList<ValueId>();
        var roles = new ArrayList<Integer>();
        for (var id : node.inputs()) {
            int position = boundaries.indexOf(id);
            if (position < 0) { position = boundaries.size(); boundaries.add(id); }
            roles.add(position);
        }
        int reads = boundaries.size();
        for (var id : node.outputs()) {
            if (boundaries.contains(id)) throw new IllegalArgumentException("scan outputs must be distinct");
            boundaries.add(id);
        }
        var bindings = new ArrayList<CpuAccessPlan.Binding>();
        for (int p = 0; p < boundaries.size(); p++) {
            var d = byId.get(boundaries.get(p));
            var extents = d.shape().toLongArray();
            var layout = d.layout().orElseThrow();
            var plan = new CpuAccessPlan(p < reads ? CpuAccessPlan.AccessKind.READ : CpuAccessPlan.AccessKind.WRITE,
                    CpuAccessPlan.Regime.DENSE_LINEAR, extents.length,
                    java.util.Collections.nCopies(extents.length, CpuAccessPlan.AxisRole.CONTIGUOUS), extents.length);
            long count = layout.referencedElementSpan();
            bindings.add(CpuAccessPlan.Binding.create(plan, extents, 0, layout.strides(), count, 0, count, count));
        }
        return new CpuPartitionLowering.LoweredPartition(new Facts(g, boundaries, roles, bindings,
                boundaries.stream().map(id -> byId.get(id).dataType()).toList(),
                boundaries.stream().map(id -> byId.get(id).layout().orElseThrow().referencedElementSpan()).toList(), reads));
    }

    /**
     * Checked cold geometry. Scratch contains two packed projection rows, reused across batch rows.
     * @param kind exact fixed family, interpreted only during cold binding
     * @param reverse whether the valid prefix is traversed backwards
     * @param bias whether one input-side packed bias is present
     * @param time non-negative static time extent
     * @param batch non-negative static batch extent
     * @param inputSize positive static feature extent
     * @param hiddenSize positive static hidden extent
     * @param gates family gate count
     * @param scratchBytes exact projection workspace bytes; zero for an empty time/batch domain
     */
    public record Geometry(RecurrentScanKind kind, boolean reverse, boolean bias, int time,
            int batch, int inputSize, int hiddenSize, int gates, long scratchBytes) {
        /**
         * Snapshots and validates exact scan geometry; retains no runtime values.
         * @param kind exact fixed family, interpreted only during cold binding
         * @param reverse whether the valid prefix is traversed backwards
         * @param bias whether one input-side packed bias is present
         * @param time non-negative static time extent
         * @param batch non-negative static batch extent
         * @param inputSize positive static feature extent
         * @param hiddenSize positive static hidden extent
         * @param gates family gate count
         * @param scratchBytes exact projection workspace bytes; zero for an empty time/batch domain
         * @throws NullPointerException if a required component is null
         * @throws IllegalArgumentException if the fixed family or canonical geometry disagrees
         * @throws ArithmeticException if checked counts or workspace arithmetic overflow
         */
        public Geometry {
            Objects.requireNonNull(kind, "kind");
            int expectedGates = switch (kind) { case RNN_TANH -> 1; case GRU_RESET_AFTER -> 3; case LSTM -> 4; };
            if (time < 0 || batch < 0 || inputSize <= 0 || hiddenSize <= 0 || gates != expectedGates
                    || scratchBytes != (time == 0 || batch == 0 ? 0
                        : Math.multiplyExact(Math.multiplyExact(2, Math.multiplyExact(gates, hiddenSize)), 4L)))
                throw new IllegalArgumentException("invalid recurrent geometry");
            Math.toIntExact(Math.multiplyExact(Math.multiplyExact((long) time, batch), inputSize));
            Math.toIntExact(Math.multiplyExact(Math.multiplyExact((long) time, batch), hiddenSize));
            Math.toIntExact(Math.multiplyExact((long) batch, hiddenSize));
            Math.toIntExact(Math.multiplyExact(Math.multiplyExact((long) gates, hiddenSize), inputSize));
            Math.toIntExact(Math.multiplyExact(Math.multiplyExact((long) gates, hiddenSize), hiddenSize));
        }
    }

    /**
     * Exact unique boundary and ordered-role mapping for a direct occurrence; contains no storage.
     * @param geometry non-null checked geometry
     * @param boundaryValues unique reads followed by all canonical outputs
     * @param rolePositions input semantic roles as unique-read positions
     * @param accessBindings canonical occurrence bindings
     * @param boundaryTypes exact boundary dtypes
     * @param spans exact boundary element counts
     * @param readCount number of distinct read boundaries
     */
    public record Facts(Geometry geometry, List<ValueId> boundaryValues, List<Integer> rolePositions,
            List<CpuAccessPlan.Binding> accessBindings, List<DataType> boundaryTypes, List<Long> spans, int readCount) {
        /**
         * Snapshots and validates exact scan facts; retains no runtime values.
         * @param geometry non-null checked geometry
         * @param boundaryValues unique reads followed by all canonical outputs
         * @param rolePositions input semantic roles as unique-read positions
         * @param accessBindings canonical occurrence bindings
         * @param boundaryTypes exact boundary dtypes
         * @param spans exact boundary element counts
         * @param readCount number of distinct read boundaries
         * @throws NullPointerException if a required component is null
         * @throws IllegalArgumentException if the fixed family or canonical geometry disagrees
         * @throws ArithmeticException if checked counts or workspace arithmetic overflow
         */
        public Facts {
            Objects.requireNonNull(geometry, "geometry");
            boundaryValues = List.copyOf(boundaryValues); rolePositions = List.copyOf(rolePositions);
            accessBindings = List.copyOf(accessBindings); boundaryTypes = List.copyOf(boundaryTypes); spans = List.copyOf(spans);
            int outputs = geometry.kind() == RecurrentScanKind.LSTM ? 3 : 2;
            int roles = (outputs == 3 ? 6 : 5) + (geometry.bias() ? 1 : 0);
            if (readCount < 1 || boundaryValues.size() != readCount + outputs
                    || boundaryValues.stream().distinct().count() != boundaryValues.size()
                    || rolePositions.size() != roles || rolePositions.stream().anyMatch(p -> p < 0 || p >= readCount)
                    || accessBindings.size() != boundaryValues.size() || boundaryTypes.size() != boundaryValues.size()
                    || spans.size() != boundaryValues.size()) throw new IllegalArgumentException("invalid scan boundaries");
            if (rolePositions.stream().distinct().count() != readCount)
                throw new IllegalArgumentException("scan has an unused read boundary");
            int t = geometry.time(), b = geometry.batch(), i = geometry.inputSize(), h = geometry.hiddenSize();
            int packed = Math.multiplyExact(geometry.gates(), h);
            var shapes = new ArrayList<>(List.of(Shape.of(t, b, i), Shape.of(b), Shape.of(b, h)));
            if (outputs == 3) shapes.add(Shape.of(b, h));
            shapes.add(Shape.of(packed, i)); shapes.add(Shape.of(packed, h));
            if (geometry.bias()) shapes.add(Shape.of(packed));
            for (int role = 0; role < shapes.size(); role++) checkBoundary(rolePositions.get(role), shapes.get(role),
                    role == 1 ? DataType.INT64 : DataType.FLOAT32, false, accessBindings, boundaryTypes, spans);
            for (int port = 0; port < outputs; port++) checkBoundary(readCount + port,
                    port == 0 ? Shape.of(t, b, h) : Shape.of(b, h), DataType.FLOAT32, true, accessBindings, boundaryTypes, spans);
        }
    }

    /**
     * Authenticates a semantic role against one canonical boundary without retaining a descriptor.
     * @param p boundary position to check
     * @param shape exact expected static role shape
     * @param type exact expected role dtype
     * @param write whether the role is a sibling output rather than an input
     * @param bindings immutable canonical access bindings
     * @param types immutable boundary dtypes
     * @param spans immutable boundary element counts
     * @throws IllegalArgumentException if a mapping, count, type or access plan disagrees
     */
    private static void checkBoundary(int p, Shape shape, DataType type, boolean write,
            List<CpuAccessPlan.Binding> bindings, List<DataType> types, List<Long> spans) {
        var layout = LayoutDescriptor.contiguous(shape);
        long count = layout.referencedElementSpan();
        int rank = shape.rank();
        var plan = new CpuAccessPlan(write ? CpuAccessPlan.AccessKind.WRITE : CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.DENSE_LINEAR, rank,
                java.util.Collections.nCopies(rank, CpuAccessPlan.AxisRole.CONTIGUOUS), rank);
        if (types.get(p) != type || spans.get(p) != count || !bindings.get(p).equals(
                CpuAccessPlan.Binding.create(plan, shape.toLongArray(), 0, layout.strides(), count, 0, count, count)))
            throw new IllegalArgumentException("scan boundary does not match semantic role: " + p);
    }
}
