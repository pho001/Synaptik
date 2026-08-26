package io.github.pho001.synaptik.backend.cpu.internal.lowering;

import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dKind;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformAttrs;
import io.github.pho001.synaptik.model.operation.layout.AxisTransformKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import io.github.pho001.synaptik.prepare.analysis.BackendAnalysisInputs;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

/** Recognizes only the visible four-node NCW Conv1d composition and folds singleton views. */
public final class CpuConv1dCompositionLowering {
    private final CpuConv2dLowering conv2d = new CpuConv2dLowering();
    /** Creates a stateless exact-topology recognizer. */
    public CpuConv1dCompositionLowering() { }

    /**
     * Validates and lowers the exact two-expansion, Conv2d, squeeze composition.
     *
     * @param context complete non-null four-node CPU projection
     * @return one Conv2d generated unit whose three intermediate values remain virtual
     * @throws NullPointerException if {@code context} is {@code null}
     * @throws IllegalArgumentException if any topology, axis, mapping, descriptor, or memory fact differs
     */
    public CpuPartitionLowering.LoweredPartition lower(PrepareContext<? extends BackendAnalysisInputs> context) {
        Objects.requireNonNull(context,"context");
        if(context.nodes().size()!=4)throw new IllegalArgumentException("CPU Conv1d composition requires exactly four nodes");
        Map<ValueId,GraphValue> values=new LinkedHashMap<>();context.values().forEach(v->values.put(v.id(),v));
        Map<ValueId,LogicalMemoryRequirement> memory=new LinkedHashMap<>();context.memoryRequirements().forEach(v->memory.put(v.valueId(),v));
        List<CompiledNode> expansions=context.nodes().subList(0,2);CompiledNode conv=context.nodes().get(2),squeeze=context.nodes().get(3);
        if(expansions.stream().anyMatch(n->n.operation().kind()!=AxisTransformKind.EXPAND_DIMS||!(n.operation().attrs() instanceof AxisTransformAttrs a)||a.axis()!=2||n.inputs().size()!=1||n.outputs().size()!=1)
                ||conv.operation().kind()!=Conv2dKind.CONV2D||!(conv.operation().attrs() instanceof Conv2dAttrs attrs)
                ||conv.outputs().size()!=1||squeeze.operation().kind()!=AxisTransformKind.SQUEEZE
                ||!(squeeze.operation().attrs() instanceof AxisTransformAttrs a)||a.axis()!=2
                ||squeeze.inputs().size()!=1||squeeze.outputs().size()!=1
                ||!squeeze.inputs().getFirst().equals(conv.outputs().getFirst()))throw new IllegalArgumentException("partition is not the exact visible Conv1d composition");
        ValueId expandedInput=conv.inputs().get(0),expandedWeight=conv.inputs().get(1);
        CompiledNode inputExpand=expansions.stream().filter(n->n.outputs().getFirst().equals(expandedInput)).findFirst().orElseThrow(()->new IllegalArgumentException("Conv1d expanded input edge is missing"));
        CompiledNode weightExpand=expansions.stream().filter(n->n.outputs().getFirst().equals(expandedWeight)).findFirst().orElseThrow(()->new IllegalArgumentException("Conv1d expanded weight edge is missing"));
        if(inputExpand==weightExpand||conv.inputs().size()<2||conv.inputs().size()>3)throw new IllegalArgumentException("Conv1d expansion branches disagree");
        ValueId inputId=inputExpand.inputs().getFirst(),weightId=weightExpand.inputs().getFirst(),outputId=squeeze.outputs().getFirst();
        List<ValueId> expected=new ArrayList<>(List.of(expandedInput,expandedWeight));if(conv.inputs().size()==3)expected.add(conv.inputs().get(2));
        if(!conv.inputs().equals(expected))throw new IllegalArgumentException("Conv1d Conv2d inputs disagree");
        requirePrivate(memory.get(expandedInput),context);requirePrivate(memory.get(expandedWeight),context);requirePrivate(memory.get(conv.outputs().getFirst()),context);
        GraphValue input=require(values,inputId),weight=require(values,weightId),output=require(values,outputId);
        requireShape(input,3);requireShape(weight,3);requireShape(output,3);
        requireSingletonView(input,require(values,expandedInput));
        requireSingletonView(weight,require(values,expandedWeight));
        requireSingletonView(output,require(values,conv.outputs().getFirst()));
        if(attrs.strideHeight()!=1||attrs.paddingHeight()!=0||attrs.dilationHeight()!=1)throw new IllegalArgumentException("Conv1d singleton geometry or mapped attributes disagree");
        var syntheticInputs=new ArrayList<ValueId>(List.of(inputId,weightId));if(conv.inputs().size()==3)syntheticInputs.add(conv.inputs().get(2));
        var syntheticNode=new CompiledNode(conv.id(),new Operation(Conv2dKind.CONV2D,attrs),syntheticInputs,List.of(outputId));
        var syntheticPartition=new PlannedPartition(CpuCapabilityProvider.CPU_BACKEND_ID,List.of(conv.id()));
        var syntheticValues=new ArrayList<GraphValue>();syntheticValues.add(expanded(inputId,input));syntheticValues.add(expanded(weightId,weight));
        if(conv.inputs().size()==3)syntheticValues.add(require(values,conv.inputs().get(2)));syntheticValues.add(expanded(outputId,output));
        var syntheticMemory=new ArrayList<LogicalMemoryRequirement>();for(GraphValue v:syntheticValues){boolean produced=v.id().equals(outputId);syntheticMemory.add(new LogicalMemoryRequirement(v.id(),v.descriptor(),produced?Optional.of(syntheticPartition):Optional.empty(),produced?List.of():List.of(syntheticPartition),produced));}
        var synthetic=new PrepareContext<>(syntheticPartition,List.of(syntheticNode),syntheticValues,syntheticMemory,Map.of(),context.backendInputs());
        var lowered=conv2d.lower(synthetic);
        return new CpuPartitionLowering.LoweredPartition(lowered.portableKernelIr(),lowered.boundaryValues(),lowered.accessBindings(),lowered.referencedElementSpans(),lowered.boundaryDataTypes(),List.of(expandedInput,expandedWeight,conv.outputs().getFirst()),lowered.extents(),lowered.elementCount(),"legal: exact visible NCW Conv1d composition with virtual singleton views",lowered.affineAddressPairs(),lowered.movementGeometry(),lowered.indexingGeometry(),lowered.scatterGeometry(),lowered.foldGeometry(),lowered.orderingGeometry(),lowered.randomGeometry(),lowered.scanGeometry(),lowered.aggregateGeometry(),lowered.argExtremaGeometry(),lowered.maskedReductionGeometry(),lowered.advancedReductionGeometry(),lowered.softmaxGeometry(),lowered.trailingNormalizationGeometry(),lowered.batchNormInferenceGeometry(),lowered.batchNormTrainingGeometry(),lowered.conv2dGeometry(),Optional.empty());
    }

    private static void requirePrivate(LogicalMemoryRequirement r,PrepareContext<?> c){if(r==null||r.graphOutput()||r.producerPartition().isEmpty()||!r.producerPartition().orElseThrow().equals(c.partition())||!r.consumerPartitions().equals(List.of(c.partition())))throw new IllegalArgumentException("Conv1d intermediate must be private and single-use");}
    private static GraphValue require(Map<ValueId,GraphValue> values,ValueId id){GraphValue value=values.get(id);if(value==null)throw new IllegalArgumentException("partition value is not projected: "+id);return value;}
    private static void requireShape(GraphValue value,int rank){if(value.descriptor().shape().rank()!=rank||value.descriptor().layout().isEmpty())throw new IllegalArgumentException("Conv1d external boundary must be resolved rank three");}
    private static void requireSingletonView(GraphValue rankThree,GraphValue rankFour){var source=rankThree.descriptor();var view=rankFour.descriptor();long[] sourceShape=source.shape().toLongArray(),viewShape=view.shape().toLongArray();if(!Arrays.equals(insert(sourceShape),viewShape)||view.dataType()!=source.dataType()||view.requiresGrad()!=source.requiresGrad()||view.layout().isEmpty())throw new IllegalArgumentException("Conv1d singleton descriptor does not preserve its source");var sourceLayout=source.layout().orElseThrow();var viewLayout=view.layout().orElseThrow();long[] sourceStrides=sourceLayout.strides(),viewStrides=viewLayout.strides();if(sourceLayout.storageOffset()!=viewLayout.storageOffset()||sourceLayout.referencedElementSpan()!=viewLayout.referencedElementSpan()||viewStrides.length!=4||viewStrides[0]!=sourceStrides[0]||viewStrides[1]!=sourceStrides[1]||viewStrides[3]!=sourceStrides[2])throw new IllegalArgumentException("Conv1d singleton layout is not an address-preserving view");}
    private static long[] insert(long[] source){return new long[]{source[0],source[1],1,source[2]};}
    private static GraphValue expanded(ValueId id,GraphValue source){long[] shape=insert(source.descriptor().shape().toLongArray());LayoutDescriptor layout=source.descriptor().layout().orElseThrow();long[] old=layout.strides();long[] strides={old[0],old[1],0,old[2]};Shape expanded=Shape.of(shape);return new GraphValue(id,new TensorDescriptor(source.descriptor().dataType(),expanded,Optional.of(LayoutDescriptor.of(expanded,strides,layout.storageOffset(),true)),source.descriptor().requiresGrad()));}
}
