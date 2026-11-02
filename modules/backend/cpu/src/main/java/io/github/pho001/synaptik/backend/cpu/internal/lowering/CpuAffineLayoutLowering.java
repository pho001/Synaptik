package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAffineCopyIr;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.index.SelectAttrs;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.*;

/**
 * Validates and composes bounded straight-line static and symbolic affine-view families.
 *
 * <p>Lowering retains graph and logical-memory values while marking eligible unit-private
 * intermediates as CPU-private virtual values. It derives exactly one source boundary, one final
 * result boundary, and validates intermediate and final-result uses from the exact consumer
 * occurrences in the shared partition DAG. Address composition, virtual-value eligibility, and
 * represented-bit copy lowering remain CPU-owned. Zero-stride result layouts use a
 * distinct-address write domain only when every repeated logical coordinate selects the same
 * represented source value.</p>
 *
 * <p>Signed-step slices reverse-map normalized logical coordinates into the original non-negative
 * physical layout. Negative steps materialize canonical storage; they never become storage strides.
 * The same coordinate recipe is used by ordinary copies and qualified Scan body copies, including
 * the Compiler's existing slice-update pullbacks.</p>
 *
 * <p>Symbolic Scan helpers retain only numeric extent/stride coefficients and coordinate
 * recipes. Invocation binding composes the ordinary address-pair copy once for the actual batch,
 * including noncontiguous reshape materialization and compatible repeated output addresses.
 * Neither descriptors nor a graph lowerer are retained in that recipe.</p>
 */
public final class CpuAffineLayoutLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();

    /** Creates one stateless affine-layout lowerer. */
    public CpuAffineLayoutLowering() {
    }

    /**
     * Lowers one exact affine chain to one represented-bit boundary copy.
     *
     * @param context non-null complete projected partition context whose shared partition DAG
     *     contains the exact affine-chain nodes and consumer occurrences, with static resolved
     *     layouts and matching logical-memory facts
     * @return one immutable lowered partition with two boundaries and zero or more virtual
     *     intermediates; never {@code null}
     * @throws NullPointerException if required projected state is absent or {@code context} is
     *     {@code null}
     * @throws IllegalArgumentException if the partition is not an eligible one-through-eight
     *     affine chain, an intermediate has an external obligation, or the final write domain is
     *     ambiguous
     * @throws ArithmeticException if exact coordinate, address, span, or count arithmetic
     *     overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        return lower(context, false);
    }

    /**
     * Lowers a chain with original logical ordinals retained for controlled distinct writes.
     * The ordinary two-address pack and its generated identity are unchanged when uncontrolled.
     *
     * @param context exact ordinary affine-chain projection
     * @param controlled whether distinct writes require original-row provenance
     * @return exact physical copy facts with two-address or controlled three-cell entries
     * @throws IllegalArgumentException if repeated destinations select different source values
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context, boolean controlled) {
        List<CompiledNode> nodes = context.partitionDag().nodes();
        Map<ValueId, GraphValue> values = new LinkedHashMap<>();
        context.values().forEach(value -> values.put(value.id(), value));
        Map<ValueId, LogicalMemoryRequirement> memory = new LinkedHashMap<>();
        context.memoryRequirements().forEach(value -> memory.put(value.valueId(), value));
        ValueId source = null;
        ValueId previous = null;
        List<ValueId> virtual = new ArrayList<>();
        List<CpuAffineCopyIr.MappingStep> steps = new ArrayList<>();
        for (int i = 0; i < nodes.size(); i++) {
            CompiledNode node = nodes.get(i);
            if (node.inputs().size() != 1 || node.outputs().size() != 1
                    || i > 0 && !node.inputs().getFirst().equals(previous)) {
                throw new IllegalArgumentException("affine partition must be one connected one-input chain");
            }
            GraphValue input = require(values, node.inputs().getFirst());
            GraphValue output = require(values, node.outputs().getFirst());
            if (!capabilities.supports(new OperationCapabilityQuery(node.operation(), List.of(input.descriptor()), List.of(output.descriptor())))) {
                throw new IllegalArgumentException("partition contains an unsupported CPU affine occurrence");
            }
            if (i == 0) source = node.inputs().getFirst();
            if (i + 1 < nodes.size()) {
                requireVirtual(memory.get(node.outputs().getFirst()), context);
                virtual.add(node.outputs().getFirst());
            }
            steps.add(mappingStep(node, input.descriptor().shape().rank(),
                    output.descriptor().shape().rank()));
            previous = node.outputs().getFirst();
        }
        ValueId result = Objects.requireNonNull(previous);
        validateUses(context, result);
        GraphValue sourceValue = require(values, source);
        GraphValue resultValue = require(values, result);
        LayoutDescriptor sourceLayout = sourceValue.descriptor().layout().orElseThrow();
        LayoutDescriptor resultLayout = resultValue.descriptor().layout().orElseThrow();

        long[] finalExtents = resultValue.descriptor().shape().toLongArray();
        long logicalCount = product(finalExtents);
        var writes = new LinkedHashMap<Long, AffineWrite>();
        long[] finalCoordinates = new long[finalExtents.length];
        for (long logical = 0; logical < logicalCount; logical++) {
            long resultAddress = address(resultLayout, finalCoordinates);
            long[] sourceCoordinates = finalCoordinates.clone();
            for (int nodeIndex = nodes.size() - 1; nodeIndex >= 0; nodeIndex--) {
                CompiledNode node = nodes.get(nodeIndex);
                sourceCoordinates = reverse(node, require(values, node.inputs().getFirst()),
                        require(values, node.outputs().getFirst()), sourceCoordinates);
            }
            long sourceAddress = address(sourceLayout, sourceCoordinates);
            AffineWrite prior = writes.get(resultAddress);
            if (prior == null) writes.put(resultAddress, new AffineWrite(sourceAddress, logical));
            if (prior != null && prior.source() != sourceAddress) {
                throw new IllegalArgumentException(
                        "non-injective output coordinates do not select one source value");
            }
            advance(finalCoordinates, finalExtents);
        }
        long elementCount = writes.size();
        CpuAffineCopyIr.WriteDomain domain = elementCount == logicalCount
                ? CpuAffineCopyIr.WriteDomain.LOGICAL_ELEMENTS
                : CpuAffineCopyIr.WriteDomain.DISTINCT_ADDRESSES;
        boolean originalOrdinals = controlled && domain == CpuAffineCopyIr.WriteDomain.DISTINCT_ADDRESSES;
        long[] addressPairs = addressPack(writes, originalOrdinals);
        int entryWidth = originalOrdinals ? 3 : 2;
        CpuAccessPlan sourcePlan = tablePlan(CpuAccessPlan.AccessKind.READ, addressPairs, 0, entryWidth);
        CpuAccessPlan resultPlan = tablePlan(CpuAccessPlan.AccessKind.WRITE, addressPairs, 1, entryWidth);
        DataType type = resultValue.descriptor().dataType();
        var affine = new CpuAffineCopyIr(type, sourcePlan, resultPlan, steps, domain);

        CpuAccessPlan.Binding sourceBinding = conservativeBinding(sourceValue.descriptor().shape()
                .toLongArray(), sourceLayout, CpuAccessPlan.AccessKind.READ);
        CpuAccessPlan.Binding resultBinding = conservativeBinding(resultValue.descriptor().shape()
                .toLongArray(), resultLayout, CpuAccessPlan.AccessKind.WRITE);
        return new CpuPartitionLowering.LoweredPartition(affine,
                List.of(source, result), List.of(sourceBinding, resultBinding),
                List.of(sourceLayout.referencedElementSpan(), resultLayout.referencedElementSpan()),
                List.of(type, type), virtual, new long[]{elementCount}, elementCount,
                "legal: bounded connected static affine chain", addressPairs);
    }

    /**
     * Lowers an original one-named-B affine chain without substituting a representative shape.
     *
     * @param context exact non-null projected ordinary chain
     * @return unresolved represented-bit copy with immutable numeric invocation geometry
     * @throws IllegalArgumentException if ordinary semantics, layout, or chain ownership is invalid
     * @throws ArithmeticException if structural coefficient arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        return lowerBatch(context, CpuPartitionLowering.batchLayouts(context));
    }

    /**
     * Lowers using authenticated full-local mappings, preserving views produced by earlier units.
     *
     * @param context exact non-null projected ordinary chain
     * @param layouts non-null full-local numeric mappings, including every chain input and output
     * @return unresolved ordinary affine copy; no original graph or descriptors are retained
     * @throws IllegalArgumentException if semantics, mappings, or chain ownership is invalid
     * @throws ArithmeticException if structural coefficient arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lowerBatch(
            PrepareContext<? extends BackendAnalysisInputs> context,
            Map<ValueId, CpuAccessPlan.BatchBinding> layouts) {
        var batch = CpuPartitionLowering.requireBatchContext(context);
        Objects.requireNonNull(layouts, "layouts");
        var nodes = context.partitionDag().nodes();
        if (nodes.isEmpty() || nodes.size() > 8)
            throw new IllegalArgumentException("affine partition requires one through eight nodes");
        var values = new LinkedHashMap<ValueId, GraphValue>();
        context.values().forEach(value -> values.put(value.id(), value));
        var memory = new LinkedHashMap<ValueId, LogicalMemoryRequirement>();
        context.memoryRequirements().forEach(value -> memory.put(value.valueId(), value));
        var shapes = new ArrayList<CpuAccessPlan.BatchBinding>();
        var mappings = new ArrayList<BatchMapping>();
        var virtual = new ArrayList<ValueId>();
        ValueId source = nodes.getFirst().inputs().getFirst();
        ValueId previous = source;
        shapes.add(requireBatchLayout(layouts, require(values, source)));
        DataType type = require(values, source).descriptor().dataType();
        for (int n = 0; n < nodes.size(); n++) {
            var node = nodes.get(n);
            if (node.inputs().size() != 1 || node.outputs().size() != 1
                    || !node.inputs().getFirst().equals(previous))
                throw new IllegalArgumentException("affine partition must be one connected one-input chain");
            var input = require(values, previous);
            var output = require(values, node.outputs().getFirst());
            if (input.descriptor().dataType() != type || output.descriptor().dataType() != type)
                throw new IllegalArgumentException("affine copy must preserve represented type");
            if (input.descriptor().shape().isFullyStatic() && output.descriptor().shape().isFullyStatic()
                    && !capabilities.supports(new OperationCapabilityQuery(node.operation(),
                            List.of(input.descriptor()), List.of(output.descriptor()))))
                throw new IllegalArgumentException("unsupported fixed affine layout in symbolic chain");
            var target = requireBatchLayout(layouts, output);
            shapes.add(target);
            mappings.add(BatchMapping.capture(node, input.descriptor().shape().rank(),
                    output.descriptor().shape().rank(), batch));
            if (n + 1 < nodes.size()) {
                requireVirtual(memory.get(output.id()), context);
                virtual.add(output.id());
            }
            previous = output.id();
        }
        validateUses(context, previous);
        var read = shapes.getFirst();
        var target = shapes.getLast();
        var writePlan = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE, target.plan().regime(),
                target.plan().iterationRank(), target.plan().axisRoles(),
                target.plan().contiguousSuffix());
        var write = new CpuAccessPlan.BatchBinding(writePlan, target.extentConstants(),
                target.extentBatchFactors(), target.baseElementOffset(), target.strideConstants(),
                target.strideBatchFactors(), true);
        // Address-pair traversal is deliberately structural: no actual B selects an IR regime.
        var sourcePlan = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.GENERAL_ODOMETER, 1,
                List.of(CpuAccessPlan.AxisRole.STRIDED), 0);
        var resultPlan = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.GENERAL_ODOMETER, 1,
                List.of(CpuAccessPlan.AxisRole.STRIDED), 0);
        var ir = new CpuAffineCopyIr(type, sourcePlan, resultPlan,
                mappings.stream().map(BatchMapping::step).toList(),
                CpuAffineCopyIr.WriteDomain.DISTINCT_ADDRESSES);
        return new CpuPartitionLowering.LoweredPartition(new BatchGeometry(ir,
                List.of(source, previous), List.of(type, type), List.of(read, write),
                shapes, mappings, virtual, CpuPartitionLowering.batchElements(target)));
    }

    private static CpuAccessPlan.BatchBinding requireBatchLayout(
            Map<ValueId, CpuAccessPlan.BatchBinding> layouts, GraphValue value) {
        var layout = layouts.get(value.id());
        if (layout == null) throw new IllegalArgumentException("missing affine physical mapping");
        var canonical = CpuPartitionLowering.batchCanonical(value.descriptor(), CpuAccessPlan.AccessKind.READ);
        if (!layout.extentConstants().equals(canonical.extentConstants())
                || !layout.extentBatchFactors().equals(canonical.extentBatchFactors()))
            throw new IllegalArgumentException("affine mapping disagrees with original logical extents");
        if (value.descriptor().layout().isPresent()) {
            var original = value.descriptor().layout().orElseThrow();
            if (layout.baseElementOffset() != original.storageOffset()
                    || layout.strideBatchFactors().stream().anyMatch(factor -> factor != 0)
                    || !layout.strideConstants().equals(Arrays.stream(original.strides()).boxed().toList()))
                throw new IllegalArgumentException("affine mapping discards an original physical layout");
        }
        return layout;
    }

    /** Numeric-only inverse coordinate recipe; no Operation, Shape, or descriptor survives. */
    private record BatchMapping(CpuAffineCopyIr.MappingStep step, List<Long> origins,
            List<Long> originFactors, List<Long> scales) {
        private BatchMapping {
            origins = List.copyOf(origins);
            originFactors = List.copyOf(originFactors);
            scales = List.copyOf(scales);
        }

        private static BatchMapping capture(CompiledNode node, int inputRank, int outputRank,
                io.github.pho001.synaptik.model.shape.DynamicDimension batch) {
            Object kind = node.operation().kind();
            if (kind != ContiguousKind.CONTIGUOUS && kind != ShapeTransformKind.RESHAPE
                    && kind != ShapeTransformKind.EXPAND && !(kind instanceof AxisTransformKind)
                    && kind != io.github.pho001.synaptik.model.operation.index.SelectKind.SELECT
                    && kind != SliceKind.SLICE)
                throw new IllegalArgumentException("unsupported affine mapping step");
            var step = mappingStep(node, inputRank, outputRank);
            Object attrs = node.operation().attrs();
            if (attrs instanceof SelectAttrs select)
                return new BatchMapping(step, List.of(select.index()), List.of(0L), List.of());
            if (attrs instanceof SliceAttrs slice)
                return new BatchMapping(step, slice.starts(),
                        Collections.nCopies(slice.starts().size(), 0L), slice.steps());
            if (attrs instanceof CropToShapeAttrs crop) {
                var constants = new ArrayList<Long>();
                var factors = new ArrayList<Long>();
                for (var dimension : crop.prefixShape().dimensions()) {
                    if (dimension instanceof io.github.pho001.synaptik.model.shape.StaticDimension fixed) {
                        constants.add(fixed.size()); factors.add(0L);
                    } else if (dimension.equals(batch)) {
                        constants.add(0L); factors.add(1L);
                    } else throw new IllegalArgumentException("unproved affine crop origin");
                }
                return new BatchMapping(step, constants, factors, List.of());
            }
            return new BatchMapping(step, List.of(), List.of(), List.of());
        }

        private long[] bindOrigins(long batch) {
            long[] result = new long[origins.size()];
            for (int i = 0; i < result.length; i++) result[i] = Math.addExact(origins.get(i),
                    Math.multiplyExact(originFactors.get(i), batch));
            return result;
        }

        private void validate(long[] input, long[] output, long[] starts) {
            if (step.kind() == CpuAffineCopyIr.MappingKind.SELECT) {
                if (starts[0] >= input[step.axes().getFirst()])
                    throw new IllegalArgumentException("affine SELECT index exceeds actual extent");
            } else if (step.kind() == CpuAffineCopyIr.MappingKind.SLICE) {
                for (int i = 0; i < step.axes().size(); i++) {
                    int axis = step.axes().get(i);
                    long last = output[axis] == 0 ? 0 : Math.addExact(starts[i],
                            Math.multiplyExact(output[axis] - 1, scales.get(i)));
                    if (output[axis] != 0 && (starts[i] >= input[axis]
                            || last < 0 || last >= input[axis]))
                        throw new IllegalArgumentException("affine SLICE exceeds actual extent");
                }
            } else if (step.kind() == CpuAffineCopyIr.MappingKind.CROP_TO_SHAPE) {
                for (int axis = 0; axis < input.length; axis++)
                    if (Math.addExact(starts[axis], output[axis]) > input[axis])
                        throw new IllegalArgumentException("affine crop exceeds actual extent");
            } else if (step.kind() == CpuAffineCopyIr.MappingKind.RESHAPE
                    && product(input) != product(output)) {
                throw new IllegalArgumentException("affine reshape actual element counts disagree");
            }
        }

        private void reverse(long[] input, long[] output, long[] coordinates,
                long[] result, long[] starts) {
            switch (step.kind()) {
                case CONTIGUOUS -> System.arraycopy(coordinates, 0, result, 0, result.length);
                case RESHAPE -> {
                    long ordinal = linear(coordinates, output);
                    for (int axis = input.length - 1; axis >= 0; axis--) {
                        result[axis] = ordinal % input[axis];
                        ordinal /= input[axis];
                    }
                }
                case EXPAND -> {
                    int shift = output.length - input.length;
                    for (int axis = 0; axis < input.length; axis++)
                        result[axis] = input[axis] == 1 ? 0 : coordinates[axis + shift];
                }
                case PERMUTE -> {
                    for (int axis = 0; axis < input.length; axis++)
                        result[step.axes().get(axis)] = coordinates[axis];
                }
                case EXPAND_DIMS -> {
                    int removed = step.axes().getFirst();
                    for (int axis = 0; axis < input.length; axis++)
                        result[axis] = coordinates[axis < removed ? axis : axis + 1];
                }
                case SQUEEZE, SELECT -> {
                    int inserted = step.axes().getFirst();
                    for (int axis = 0; axis < input.length; axis++)
                        result[axis] = axis == inserted
                                ? step.kind() == CpuAffineCopyIr.MappingKind.SELECT ? starts[0] : 0
                                : coordinates[axis < inserted ? axis : axis - 1];
                }
                case SLICE -> {
                    System.arraycopy(coordinates, 0, result, 0, result.length);
                    for (int i = 0; i < step.axes().size(); i++) {
                        int axis = step.axes().get(i);
                        result[axis] = Math.addExact(starts[i],
                                Math.multiplyExact(coordinates[axis], scales.get(i)));
                    }
                }
                case CROP_TO_SHAPE -> {
                    for (int axis = 0; axis < input.length; axis++)
                        result[axis] = Math.addExact(coordinates[axis], starts[axis]);
                }
            }
        }
    }

    /** Immutable numeric copy geometry, bound once before the generated Scan time loop. */
    private record BatchGeometry(CpuAffineCopyIr ir, List<ValueId> boundaries,
            List<DataType> boundaryTypes, List<CpuAccessPlan.BatchBinding> bindings,
            List<CpuAccessPlan.BatchBinding> shapes, List<BatchMapping> mappings,
            List<ValueId> virtual, BatchByteSize iterationCount)
            implements CpuPartitionLowering.BatchUnitGeometry {
        private BatchGeometry {
            boundaries = List.copyOf(boundaries);
            boundaryTypes = List.copyOf(boundaryTypes);
            bindings = List.copyOf(bindings);
            shapes = List.copyOf(shapes);
            mappings = List.copyOf(mappings);
            virtual = List.copyOf(virtual);
        }

        @Override public BatchByteSize scratchBytes() { return new BatchByteSize(0, 0, 1); }
        @Override public List<CpuKernelSpecialization.PackReset> resets() { return List.of(); }
        @Override public List<ValueId> virtualValues() { return virtual; }

        @Override public CpuPartitionLowering.LoweredPartition bind(long batch) {
            var source = bindings.getFirst().bind(batch);
            var result = bindings.getLast().bind(batch);
            long[][] extents = new long[shapes.size()][];
            long[][] coordinates = new long[shapes.size()][];
            long[][] starts = new long[mappings.size()][];
            for (int i = 0; i < shapes.size(); i++) {
                var shape = shapes.get(i);
                extents[i] = new long[shape.extentConstants().size()];
                coordinates[i] = new long[extents[i].length];
                for (int axis = 0; axis < extents[i].length; axis++)
                    extents[i][axis] = Math.toIntExact(Math.addExact(shape.extentConstants().get(axis),
                            Math.multiplyExact(shape.extentBatchFactors().get(axis), batch)));
            }
            for (int i = 0; i < mappings.size(); i++) {
                starts[i] = mappings.get(i).bindOrigins(batch);
                mappings.get(i).validate(extents[i], extents[i + 1], starts[i]);
            }
            var writes = new LinkedHashMap<Long, AffineWrite>();
            int last = extents.length - 1;
            long capacity = iterationCount.byteSize(batch);
            for (long logical = 0; logical < capacity; logical++) {
                long destination = address(result, coordinates[last]);
                for (int i = mappings.size() - 1; i >= 0; i--)
                    mappings.get(i).reverse(extents[i], extents[i + 1], coordinates[i + 1],
                            coordinates[i], starts[i]);
                long origin = address(source, coordinates[0]);
                AffineWrite previous = writes.get(destination);
                if (previous == null) writes.put(destination, new AffineWrite(origin, logical));
                if (previous != null && previous.source() != origin)
                    throw new IllegalArgumentException(
                            "non-injective output coordinates do not select one source value");
                advance(coordinates[last], extents[last]);
            }
            long[] pairs = addressPack(writes, true);
            return new CpuPartitionLowering.LoweredPartition(ir, boundaries, List.of(source, result),
                    List.of(source.referencedElementSpan(), result.referencedElementSpan()),
                    boundaryTypes, virtual, new long[]{writes.size()}, writes.size(),
                    "legal: bounded connected symbolic affine chain", pairs);
        }

        @Override public String identity() {
            return "affine-batch:" + ir.structuralKey() + ':' + bindings + ':' + shapes + ':' + mappings;
        }

        private static long address(CpuAccessPlan.Binding binding, long[] coordinates) {
            long result = binding.baseElementOffset();
            for (int axis = 0; axis < coordinates.length; axis++) {
                if (coordinates[axis] < 0 || coordinates[axis] >= binding.extents().get(axis))
                    throw new IllegalArgumentException("affine coordinate exceeds actual extent");
                result = Math.addExact(result,
                        Math.multiplyExact(coordinates[axis], binding.effectiveStrides().get(axis)));
            }
            return result;
        }
    }

    /**
     * The original occurrence selected by the existing distinct-destination proof.
     * @param source exact source element address
     * @param logical first original logical result ordinal for this destination
     */
    private record AffineWrite(long source, long logical) {}

    /**
     * Packs addresses and, only for controlled distinct writes, their original logical ordinal.
     * @param writes insertion-ordered proved destination/source occurrences
     * @param originalOrdinals whether each entry includes original-row provenance
     * @return exact ordinary pairs or controlled triples without a full logical-domain side array
     */
    private static long[] addressPack(Map<Long, AffineWrite> writes, boolean originalOrdinals) {
        int width = originalOrdinals ? 3 : 2, cell = 0;
        long[] packed = new long[Math.multiplyExact(writes.size(), width)];
        for (var write : writes.entrySet()) {
            packed[cell++] = write.getValue().source();
            packed[cell++] = write.getKey();
            if (originalOrdinals) packed[cell++] = write.getValue().logical();
        }
        return packed;
    }

    private static long[] reverse(CompiledNode node, GraphValue input, GraphValue output,
            long[] coordinates) {
        Object kind = node.operation().kind();
        Object attrs = node.operation().attrs();
        long[] inputExtents = input.descriptor().shape().toLongArray();
        if (kind == ContiguousKind.CONTIGUOUS) return coordinates;
        if (kind == ShapeTransformKind.RESHAPE) {
            long linear = linear(coordinates, output.descriptor().shape().toLongArray());
            return coordinates(linear, inputExtents);
        }
        if (kind == ShapeTransformKind.EXPAND) {
            long[] result = new long[inputExtents.length];
            int offset = coordinates.length - inputExtents.length;
            for (int axis = 0; axis < result.length; axis++) result[axis] = inputExtents[axis] == 1
                    ? 0 : coordinates[axis + offset];
            return result;
        }
        if (kind == AxisTransformKind.PERMUTE) {
            long[] result = new long[coordinates.length];
            List<Integer> axes = ((PermutationAttrs) attrs).axes();
            for (int axis = 0; axis < result.length; axis++) result[axes.get(axis)] = coordinates[axis];
            return result;
        }
        if (kind == AxisTransformKind.EXPAND_DIMS) return remove(coordinates,
                ((AxisTransformAttrs) attrs).axis());
        if (kind == AxisTransformKind.SQUEEZE) return insert(coordinates,
                ((AxisTransformAttrs) attrs).axis(), 0);
        if (kind == io.github.pho001.synaptik.model.operation.index.SelectKind.SELECT) {
            SelectAttrs select = (SelectAttrs) attrs;
            return insert(coordinates, select.axis(), select.index());
        }
        if (kind == SliceKind.SLICE && attrs instanceof SliceAttrs slice) {
            long[] result = coordinates.clone();
            for (int i = 0; i < slice.axes().size(); i++) {
                int axis = slice.axes().get(i);
                result[axis] = Math.addExact(slice.starts().get(i),
                        Math.multiplyExact(result[axis], slice.steps().get(i)));
            }
            return result;
        }
        if (kind == SliceKind.SLICE && attrs instanceof CropToShapeAttrs crop) {
            long[] result = coordinates.clone();
            long[] prefix = crop.prefixShape().toLongArray();
            for (int axis = 0; axis < result.length; axis++) result[axis] = Math.addExact(
                    result[axis], prefix[axis]);
            return result;
        }
        throw new IllegalArgumentException("unsupported affine mapping step");
    }

    private static CpuAffineCopyIr.MappingStep mappingStep(CompiledNode node, int inputRank,
            int outputRank) {
        Object kind = node.operation().kind();
        Object attrs = node.operation().attrs();
        CpuAffineCopyIr.MappingKind mappingKind;
        List<Integer> axes = List.of();
        if (kind == ContiguousKind.CONTIGUOUS) mappingKind = CpuAffineCopyIr.MappingKind.CONTIGUOUS;
        else if (kind == ShapeTransformKind.RESHAPE) mappingKind = CpuAffineCopyIr.MappingKind.RESHAPE;
        else if (kind == ShapeTransformKind.EXPAND) mappingKind = CpuAffineCopyIr.MappingKind.EXPAND;
        else if (kind == AxisTransformKind.PERMUTE) {
            mappingKind = CpuAffineCopyIr.MappingKind.PERMUTE;
            axes = ((PermutationAttrs) attrs).axes();
        } else if (kind == AxisTransformKind.EXPAND_DIMS) {
            mappingKind = CpuAffineCopyIr.MappingKind.EXPAND_DIMS;
            axes = List.of(((AxisTransformAttrs) attrs).axis());
        } else if (kind == AxisTransformKind.SQUEEZE) {
            mappingKind = CpuAffineCopyIr.MappingKind.SQUEEZE;
            axes = List.of(((AxisTransformAttrs) attrs).axis());
        } else if (kind == io.github.pho001.synaptik.model.operation.index.SelectKind.SELECT) {
            mappingKind = CpuAffineCopyIr.MappingKind.SELECT;
            axes = List.of(((SelectAttrs) attrs).axis());
        } else if (attrs instanceof SliceAttrs slice) {
            mappingKind = CpuAffineCopyIr.MappingKind.SLICE;
            axes = slice.axes();
        } else if (attrs instanceof CropToShapeAttrs) {
            mappingKind = CpuAffineCopyIr.MappingKind.CROP_TO_SHAPE;
        } else throw new IllegalArgumentException("unsupported affine mapping step");
        return new CpuAffineCopyIr.MappingStep(mappingKind, inputRank, outputRank, axes);
    }

    private static CpuAccessPlan tablePlan(CpuAccessPlan.AccessKind kind, long[] pairs,
            int parity, int entryWidth) {
        boolean constant = true, dense = true;
        for (int i = parity + entryWidth; i < pairs.length; i += entryWidth) {
            constant &= pairs[i] == pairs[parity];
            dense &= pairs[i] == pairs[i - entryWidth] + 1;
        }
        CpuAccessPlan.Regime regime = constant && kind == CpuAccessPlan.AccessKind.READ
                ? CpuAccessPlan.Regime.SCALAR_ALL_ZERO
                : dense ? CpuAccessPlan.Regime.DENSE_LINEAR : CpuAccessPlan.Regime.GENERAL_ODOMETER;
        CpuAccessPlan.AxisRole role = constant ? CpuAccessPlan.AxisRole.BROADCAST
                : dense ? CpuAccessPlan.AxisRole.CONTIGUOUS : CpuAccessPlan.AxisRole.STRIDED;
        return new CpuAccessPlan(kind, regime, 1, List.of(role), dense ? 1 : 0);
    }

    private static CpuAccessPlan.Binding conservativeBinding(long[] extents,
            LayoutDescriptor layout, CpuAccessPlan.AccessKind kind) {
        long count = product(extents);
        CpuAccessPlan plan = normalizedPlan(extents, layout.strides(), kind);
        return CpuAccessPlan.Binding.create(plan, extents, layout.storageOffset(), layout.strides(),
                count, 0, count, layout.referencedElementSpan());
    }

    private static CpuAccessPlan normalizedPlan(long[] extents, long[] strides,
            CpuAccessPlan.AccessKind kind) {
        int suffix = 0; long expected = 1;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            if (strides[axis] != expected) break;
            suffix++; expected = Math.multiplyExact(expected, Math.max(1, extents[axis]));
        }
        List<CpuAccessPlan.AxisRole> roles = new ArrayList<>();
        for (int axis = 0; axis < extents.length; axis++) roles.add(strides[axis] == 0
                ? CpuAccessPlan.AxisRole.BROADCAST
                : axis >= extents.length - suffix ? CpuAccessPlan.AxisRole.CONTIGUOUS
                : CpuAccessPlan.AxisRole.STRIDED);
        boolean allZero = Arrays.stream(strides).allMatch(value -> value == 0);
        CpuAccessPlan.Regime regime = allZero && kind == CpuAccessPlan.AccessKind.READ
                ? CpuAccessPlan.Regime.SCALAR_ALL_ZERO
                : suffix == extents.length ? CpuAccessPlan.Regime.DENSE_LINEAR
                : suffix > 0 ? CpuAccessPlan.Regime.BLOCK_OUTER
                : CpuAccessPlan.Regime.GENERAL_ODOMETER;
        return new CpuAccessPlan(kind, regime, extents.length, roles, suffix);
    }

    private static void validateUses(PrepareContext<?> context, ValueId result) {
        if (!context.partitionDag().consumers(result).isEmpty())
            throw new IllegalArgumentException("final affine output must not feed the unit");
        List<CompiledNode> nodes = context.partitionDag().nodes();
        for (int i = 0; i + 1 < nodes.size(); i++) {
            ValueId intermediate = nodes.get(i).outputs().getFirst();
            if (context.partitionDag().consumers(intermediate).size() != 1) {
                throw new IllegalArgumentException("affine intermediate must have one use");
            }
        }
    }
    private static void requireVirtual(LogicalMemoryRequirement requirement, PrepareContext<?> context) {
        if (requirement == null || requirement.graphOutput() || requirement.producerPartition().isEmpty()
                || !requirement.producerPartition().orElseThrow().equals(context.partition())
                || !requirement.consumerPartitions().equals(List.of(context.partition())))
            throw new IllegalArgumentException("internal affine result must remain unit-private");
    }
    private static long address(LayoutDescriptor layout, long[] coordinates) { long result = layout.storageOffset();
        for (int i = 0; i < coordinates.length; i++) result = Math.addExact(result,
                Math.multiplyExact(coordinates[i], layout.stride(i))); return result; }
    private static long linear(long[] coordinates, long[] extents) { long result = 0;
        for (int i = 0; i < extents.length; i++) result = Math.addExact(
                Math.multiplyExact(result, extents[i]), coordinates[i]); return result; }
    private static long[] coordinates(long linear, long[] extents) { long[] result = new long[extents.length];
        for (int i = extents.length - 1; i >= 0; i--) if (extents[i] != 0) {
            result[i] = linear % extents[i]; linear /= extents[i]; } return result; }
    private static long[] remove(long[] source, int removed) { long[] result = new long[source.length - 1];
        for (int i = 0, j = 0; i < source.length; i++) if (i != removed) result[j++] = source[i]; return result; }
    private static long[] insert(long[] source, int inserted, long value) { long[] result = new long[source.length + 1];
        for (int i = 0, j = 0; i < result.length; i++) result[i] = i == inserted ? value : source[j++]; return result; }
    private static void advance(long[] coordinates, long[] extents) {
        for (int i = coordinates.length - 1; i >= 0; i--) { coordinates[i]++;
            if (coordinates[i] < extents[i]) return; coordinates[i] = 0; }
    }
    private static long product(long[] extents) { long result = 1;
        for (long extent : extents) result = Math.multiplyExact(result, extent); return result; }
    private static GraphValue require(Map<ValueId, GraphValue> values, ValueId id) {
        GraphValue value = values.get(id);
        if (value == null) throw new IllegalArgumentException("partition value is not projected: " + id);
        return value;
    }
}
