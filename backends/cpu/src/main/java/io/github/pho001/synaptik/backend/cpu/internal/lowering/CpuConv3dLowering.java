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
 * Lowers one exact static grouped NCDHW Conv3d occurrence to complete-output-cell work.
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
        var query=new OperationCapabilityQuery(node.operation(),node.inputs().stream().map(id->require(values,id).descriptor()).toList(),node.outputs().stream().map(id->require(values,id).descriptor()).toList());
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
