package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuConv3dIr;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dKind;
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
 * Lowers exact static or one-named-batch grouped NCDHW Conv3d to complete-output-cell work.
 *
 * <p>The lowerer declares only the ordered input, weight, optional intrinsic-bias, and output
 * boundaries. It creates no workspace or materialization and does not admit external epilogues
 * or multi-node Conv3d-led partitions.</p>
 */
public final class CpuConv3dLowering {
    private final CpuCapabilityProvider capabilities = new CpuCapabilityProvider();
    /** Creates a stateless rank-specific lowerer. */
    public CpuConv3dLowering() { }

    /**
     * Checks original symbolic NCDHW geometry without representative batch descriptors.
     * @param context complete one-node grouped convolution projection
     * @return unresolved checked numeric Conv3d recipe
     * @throws IllegalArgumentException if original semantics or dimension equations disagree
     * @throws ArithmeticException if fixed coefficient arithmetic overflows
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
        if (context.nodes().size() != 1) throw new IllegalArgumentException("Conv3d requires one node");
        var node = context.nodes().getFirst();
        if (node.operation().kind() != Conv3dKind.CONV3D || !(node.operation().attrs() instanceof Conv3dAttrs attrs)
                || node.inputs().size() < 2 || node.inputs().size() > 3 || node.outputs().size() != 1
                || node.inputs().contains(node.outputs().getFirst())) throw new IllegalArgumentException("Conv3d occurrence disagrees");

        var input = require(values,node.inputs().get(0)); var weight = require(values,node.inputs().get(1));
        var convOutput = require(values,node.outputs().getFirst());
        if (input.descriptor().shape().rank() != 5 || weight.descriptor().shape().rank() != 5
                || convOutput.descriptor().shape().rank() != 5)
            throw new IllegalArgumentException("convolution ranks disagree");
        long channels = fixed(input,1), filters = fixed(weight,0);
        if (channels % attrs.groups() != 0 || filters % attrs.groups() != 0
                || Math.multiplyExact(fixed(weight,1),attrs.groups()) != channels)
            throw new IllegalArgumentException("convolution grouped channel equation disagrees");
        var promoted = io.github.pho001.synaptik.model.datatype.DataTypePromotion.promoteFloating(input.descriptor().dataType(),weight.descriptor().dataType());
        boolean gradient = input.descriptor().requiresGrad() || weight.descriptor().requiresGrad();
        if (node.inputs().size() == 3) {
            var bias = require(values,node.inputs().get(2));
            if (bias.descriptor().shape().rank() != 1 || fixed(bias,0) != filters)
                throw new IllegalArgumentException("intrinsic bias shape disagrees");
            promoted = io.github.pho001.synaptik.model.datatype.DataTypePromotion.promoteFloating(promoted,bias.descriptor().dataType());
            gradient |= bias.descriptor().requiresGrad();
        }
        if (convOutput.descriptor().dataType() != promoted || convOutput.descriptor().requiresGrad() != gradient)
            throw new IllegalArgumentException("convolution promoted type or gradient disagrees");
        sameAxis(input,0,convOutput,0);
        if (fixed(convOutput,1) != filters) throw new IllegalArgumentException("convolution output channels disagree");
        long[] strides = new long[]{attrs.strideDepth(),attrs.strideHeight(),attrs.strideWidth()}, padding = new long[]{attrs.paddingDepth(),attrs.paddingHeight(),attrs.paddingWidth()}, dilation = new long[]{attrs.dilationDepth(),attrs.dilationHeight(),attrs.dilationWidth()};
        for (int axis = 2; axis < 5; axis++) {
            long kernel = fixed(weight,axis);
            if (kernel <= 0) throw new IllegalArgumentException("convolution kernel must be positive");
            long effective = Math.addExact(Math.multiplyExact(kernel-1,dilation[axis-2]),1);
            long padded = Math.addExact(fixed(input,axis),Math.multiplyExact(2,padding[axis-2]));
            if (padded < effective || fixed(convOutput,axis) != Math.addExact((padded-effective)/strides[axis-2],1))
                throw new IllegalArgumentException("convolution original spatial equation disagrees");
        }

        var ids = new ArrayList<ValueId>(node.inputs()); ids.add(convOutput.id());
        var bindings = new ArrayList<>(ids.stream().map(id -> batchBinding(require(values,id),layouts)).toList());
        bindings.set(bindings.size()-1,writeRecipe(bindings.getLast()));
        var types = ids.stream().map(id -> require(values,id).descriptor().dataType()).toList();
        var ir = new CpuConv3dIr(types.subList(0,types.size()-1),types.getLast(),attrs.strideDepth(),attrs.strideHeight(),attrs.strideWidth(),
                attrs.paddingDepth(),attrs.paddingHeight(),attrs.paddingWidth(),attrs.dilationDepth(),attrs.dilationHeight(),attrs.dilationWidth(),attrs.groups(),1,
                node.inputs().size()==3,bindings.subList(0,bindings.size()-1).stream().map(CpuAccessPlan.BatchBinding::plan).toList(),writePlan(bindings.getLast()));
        return new CpuPartitionLowering.LoweredPartition(new Geometry.BatchGeometry(ir,ids,types,bindings,List.of(),new long[0]));
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
     * Lowers one supported direct Conv3d occurrence with optional intrinsic bias.
     *
     * @param context complete non-null one-node CPU preparation projection
     * @return immutable zero-workspace lowering
     * @throws NullPointerException if {@code context} or a required fact is {@code null}
     * @throws IllegalArgumentException if topology, descriptors, layouts, or geometry disagree
     * @throws ArithmeticException if checked count, span, or address arithmetic overflows
     */
    public CpuPartitionLowering.LoweredPartition lower(
            PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context, "context");
        if (context.nodes().size()!=1) throw new IllegalArgumentException("CPU Conv3d requires exactly one node");
        var node=context.nodes().getFirst();
        Map<ValueId,GraphValue> values=new LinkedHashMap<>(); context.values().forEach(v->values.put(v.id(),v));
        var query=new OperationCapabilityQuery(node.operation(), node.inputs().stream().map(id->require(values,id).descriptor()).toList(), node.outputs().stream().map(id->require(values,id).descriptor()).toList());
        if(!capabilities.supports(query)||node.operation().kind()!=Conv3dKind.CONV3D||!(node.operation().attrs() instanceof Conv3dAttrs attrs))throw new IllegalArgumentException("partition contains an unsupported CPU Conv3d occurrence");
        ValueId outputId=node.outputs().getFirst(); if(node.inputs().contains(outputId))throw new IllegalArgumentException("Conv3d output must be distinct from every input");
        var boundaryIds=new ArrayList<>(node.inputs());boundaryIds.add(outputId);
        var bindings=new ArrayList<CpuAccessPlan.Binding>();var spans=new ArrayList<Long>();var types=new ArrayList<DataType>();var layouts=new ArrayList<Layout>();
        for(int i=0;i<boundaryIds.size();i++){
            GraphValue value=require(values,boundaryIds.get(i));Layout layout=layout(value);layouts.add(layout);
            bindings.add(binding(layout,i==boundaryIds.size()-1?CpuAccessPlan.AccessKind.WRITE:CpuAccessPlan.AccessKind.READ));
            spans.add(value.descriptor().layout().orElseThrow().referencedElementSpan());types.add(value.descriptor().dataType());
        }
        List<DataType> inputTypes=types.subList(0,types.size()-1);
        var ir=new CpuConv3dIr(inputTypes,types.getLast(),attrs.strideDepth(),attrs.strideHeight(),attrs.strideWidth(),attrs.paddingDepth(),attrs.paddingHeight(),attrs.paddingWidth(),attrs.dilationDepth(),attrs.dilationHeight(),attrs.dilationWidth(),attrs.groups(),1,node.inputs().size()==3,bindings.subList(0,bindings.size()-1).stream().map(CpuAccessPlan.Binding::plan).toList(),bindings.getLast().plan());
        long outputCount=count(layouts.getLast().extents());
        return new CpuPartitionLowering.LoweredPartition(ir,boundaryIds,bindings,spans,types,List.of(),layouts.getLast().extents(),outputCount,"legal: one direct static grouped NCDHW Conv3d",new long[0],Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.empty(),Optional.of(new Geometry(layouts)));
    }

    private static GraphValue require(Map<ValueId,GraphValue> values,ValueId id){GraphValue result=values.get(id);if(result==null)throw new IllegalArgumentException("partition value is not projected: "+id);return result;}
    private static Layout layout(GraphValue value){LayoutDescriptor descriptor=value.descriptor().layout().orElseThrow();if(descriptor.storageOffset()<0||Arrays.stream(descriptor.strides()).anyMatch(s->s<0))throw new IllegalArgumentException("Conv3d requires non-negative resolved layouts");return new Layout(value.descriptor().shape().toLongArray(),descriptor.storageOffset(),descriptor.strides());}
    private static CpuAccessPlan.Binding binding(Layout layout,CpuAccessPlan.AccessKind kind){int suffix=0;long expected=1;for(int axis=layout.extents().length-1;axis>=0;axis--){if(layout.strides()[axis]!=expected)break;suffix++;expected=Math.multiplyExact(expected,Math.max(1,layout.extents()[axis]));}var roles=new ArrayList<CpuAccessPlan.AxisRole>();for(int axis=0;axis<layout.extents().length;axis++)roles.add(layout.strides()[axis]==0?CpuAccessPlan.AxisRole.BROADCAST:axis>=layout.extents().length-suffix?CpuAccessPlan.AxisRole.CONTIGUOUS:CpuAccessPlan.AxisRole.STRIDED);var plan=new CpuAccessPlan(kind,suffix==layout.extents().length?CpuAccessPlan.Regime.DENSE_LINEAR:CpuAccessPlan.Regime.GENERAL_ODOMETER,layout.extents().length,roles,suffix);long count=count(layout.extents());return CpuAccessPlan.Binding.create(plan,layout.extents(),layout.offset(),layout.strides(),count,0,count,span(layout));}
    private static long count(long[] extents){for(long extent:extents)if(extent==0)return 0;long result=1;for(long extent:extents)result=Math.multiplyExact(result,extent);return result;}
    private static long span(Layout layout){if(count(layout.extents())==0)return 0;long maximum=layout.offset();for(int axis=0;axis<layout.extents().length;axis++)maximum=Math.addExact(maximum,Math.multiplyExact(layout.extents()[axis]-1,layout.strides()[axis]));return Math.addExact(maximum,1);}

    /**
     * Immutable cold rank-five boundary layout geometry used during generated-entry binding.
     *
     * @param boundaries ordered input, weight, optional rank-one intrinsic-bias, and output
     *     layouts; copied defensively
     */
    public record Geometry(List<Layout> boundaries){

        /**
         * Unresolved numeric invocation recipe. No representative shape, graph or invocation is retained.
         * @param ir exact B-independent generated facts
         * @param boundaries ordered physical boundaries
         * @param boundaryTypes exact boundary storage types
         * @param bindings affine numeric mappings, including the output mapping
         * @param virtualValues private removed intermediate values
         * @param parameters fixed family geometry, defensively copied

         */
        public record BatchGeometry(CpuConv3dIr ir, List<ValueId> boundaries,
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
             * Returns the fixed Conv3d geometry cells retained for numeric invocation binding.
             * Actual B is resolved separately through the affine boundary mappings.
             * Mutating the result cannot alter this graph-free recipe.
             * @return a new non-null caller-owned copy of the fixed numeric geometry
             */
            @Override public long[] parameters() { return parameters.clone(); }
            /** @return checked affine complete-unit work count, not bytes */
            @Override public io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize iterationCount() { return CpuPartitionLowering.batchElements(bindings.getLast()); }
            /** @return simultaneous direct-scalar workspace bytes */
            @Override public io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize scratchBytes() { return new io.github.pho001.synaptik.backend.cpu.internal.memory.CpuRepresentationRecipes.BatchByteSize(0, 0, 1); }
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
                var out = resolved.getLast();
                var geometry = new Geometry(resolved.stream().map(b -> new Layout(b.extents().stream().mapToLong(Long::longValue).toArray(), b.baseElementOffset(), b.effectiveStrides().stream().mapToLong(Long::longValue).toArray())).toList());
                return new CpuPartitionLowering.LoweredPartition(ir, boundaries, resolved,
                        resolved.stream().map(CpuAccessPlan.Binding::referencedElementSpan).toList(),
                        boundaryTypes, virtualValues, out.extents().stream().mapToLong(Long::longValue).toArray(), out.elementCount(),
                        "legal: checked symbolic ordinary geometry", new long[0],
                        Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.empty(), Optional.of(geometry));
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
            var bases = new ArrayList<BaseCell>(boundaries.size());
            for (int i = 0; i < boundaries.size(); i++)
                bases.add(new BaseCell(i, i, boundaries.get(i).offset()));
            return new TemporalRebind(bases, List.of());
        }

        /**
         * Validates and snapshots the ordered layouts.
         *
         * @throws NullPointerException if the list or an element is {@code null}
         * @throws IllegalArgumentException if the boundary count or required ranks disagree
         */
        public Geometry{boundaries=List.copyOf(boundaries);if(boundaries.size()<3||boundaries.size()>4||boundaries.get(0).extents().length!=5||boundaries.get(1).extents().length!=5||boundaries.getLast().extents().length!=5||boundaries.subList(2,boundaries.size()-1).stream().anyMatch(l->l.extents().length!=1))throw new IllegalArgumentException("Conv3d boundary geometry disagrees");}
        /**
         * Packs carrier-relative bases followed by every boundary's extents and strides.
         *
         * @param carrierBases non-null element bases in boundary order; read but not retained
         * @return a new packed geometry array owned by the caller
         * @throws NullPointerException if {@code carrierBases} is {@code null}
         * @throws IllegalArgumentException if its length differs from the boundary count
         * @throws ArithmeticException if adding a carrier base and layout offset overflows
         */
        public long[] pack(long[] carrierBases){if(carrierBases.length!=boundaries.size())throw new IllegalArgumentException("Conv3d carrier base count disagrees");int length=boundaries.size();for(Layout layout:boundaries)length+=2*layout.extents().length;long[] packed=new long[length];int cursor=boundaries.size();for(int i=0;i<boundaries.size();i++){Layout layout=boundaries.get(i);packed[i]=Math.addExact(carrierBases[i],layout.offset());System.arraycopy(layout.extents(),0,packed,cursor,layout.extents().length);cursor+=layout.extents().length;System.arraycopy(layout.strides(),0,packed,cursor,layout.strides().length);cursor+=layout.strides().length;}return packed;}
    }
    /**
     * Immutable logical Shape and non-negative element layout.
     *
     * @param extents non-null non-negative logical axis extents; copied defensively
     * @param offset non-negative element offset from the carrier base
     * @param strides non-null non-negative element strides with one value per extent; copied
     *     defensively
     */
    public record Layout(long[] extents,long offset,long[] strides){
        /**
         * Validates and snapshots one layout.
         *
         * @throws NullPointerException if an array is {@code null}
         * @throws IllegalArgumentException if rank differs or an extent, offset, or stride is
         *     negative
         */
        public Layout{extents=extents.clone();strides=strides.clone();if(extents.length!=strides.length||offset<0||Arrays.stream(extents).anyMatch(v->v<0)||Arrays.stream(strides).anyMatch(v->v<0))throw new IllegalArgumentException("Conv3d layout is invalid");}
        /**
         * Returns a defensive copy of the logical extents.
         *
         * @return a new caller-owned array
         */
        @Override public long[] extents(){return extents.clone();}
        /**
         * Returns a defensive copy of the element strides.
         *
         * @return a new caller-owned array
         */
        @Override public long[] strides(){return strides.clone();}
    }
}
