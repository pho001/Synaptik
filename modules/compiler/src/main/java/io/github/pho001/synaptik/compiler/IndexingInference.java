package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.index.*;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/** Derives descriptors and validates operand roles for scalar and tensor indexing families. */
final class IndexingInference {
    private IndexingInference() {}

    /**
     * Verifies one supported select, gather, scatter, or one-hot occurrence.
     *
     * @param op non-null typed operation whose kind belongs to this family
     * @param in non-null ordered input descriptors supplied by the captured node
     * @return immutable derived descriptors and ordered candidate constraints; never {@code null}
     * @throws IllegalArgumentException if the kind is unsupported here or the occurrence violates
     *     its index, operand, Shape, or attribute contract
     */
    static CapturedGraphInference.InferenceResult infer(Operation op, List<TensorDescriptor> in) {
        if (op.kind() instanceof SelectKind) return select((SelectAttrs) op.attrs(), in);
        if (op.kind() instanceof AxisGatherKind k) return gather(k, (IndexAxisAttrs) op.attrs(), in);
        if (op.kind() instanceof OneHotKind) return oneHot((OneHotAttrs) op.attrs(), in);
        if (op.kind() instanceof GatherNdKind) return gatherNd((GatherNdAttrs) op.attrs(), in);
        if (op.kind() instanceof AxisScatterKind k) return axisScatter(k, op.attrs(), in);
        if (op.kind() instanceof ScatterNdKind) return scatterNd((ScatterNdAttrs) op.attrs(), in);
        throw new IllegalArgumentException("unsupported indexing kind");
    }

    private static CapturedGraphInference.InferenceResult select(SelectAttrs attrs, List<TensorDescriptor> in) {
        TensorDescriptor input = in.get(0); axis(input.shape(), attrs.axis());
        Dimension selected = input.shape().dimension(attrs.axis());
        List<Dimension> dimensions = new ArrayList<>(input.shape().dimensions()); dimensions.remove(attrs.axis());
        Shape shape = Shape.ofDimensions(dimensions.toArray(Dimension[]::new));
        Optional<LayoutDescriptor> layout = Optional.empty();
        if (input.layout().isPresent() && shape.knownElementCount().orElseThrow() != 0) {
            LayoutDescriptor source = input.layout().orElseThrow(); long[] sourceStrides = source.strides(); long[] strides = new long[sourceStrides.length-1];
            for(int i=0,j=0;i<sourceStrides.length;i++) if(i!=attrs.axis()) strides[j++]=sourceStrides[i];
            long offset=Math.addExact(source.storageOffset(),Math.multiplyExact(attrs.index(),sourceStrides[attrs.axis()]));
            layout=Optional.of(LayoutDescriptor.of(shape,strides,offset,true));
        }
        TensorDescriptor output = new TensorDescriptor(input.dataType(), shape, layout, input.requiresGrad());
        return CapturedGraphInference.InferenceResult.constrained(List.of(output),
                new CapturedGraphInference.ConstraintRequest("select index", FitsWithin.dimension(attrs.index(),new StaticDimension(1),selected)));
    }

    private static CapturedGraphInference.InferenceResult gather(AxisGatherKind kind, IndexAxisAttrs attrs, List<TensorDescriptor> in) {
        TensorDescriptor data=in.get(0), indices=in.get(1); ElementwiseInference.requireIndex(indices,"indices"); axis(data.shape(),attrs.axis());
        Shape result;
        if(kind==AxisGatherKind.GATHER){
            List<Dimension> dims=new ArrayList<>(); dims.addAll(data.shape().dimensions().subList(0,attrs.axis())); dims.addAll(indices.shape().dimensions()); dims.addAll(data.shape().dimensions().subList(attrs.axis()+1,data.shape().rank())); result=Shape.ofDimensions(dims.toArray(Dimension[]::new));
        } else {
            if(indices.shape().rank()!=data.shape().rank()) throw new IllegalArgumentException("gather-elements rank mismatch");
            for(int i=0;i<data.shape().rank();i++) if(i!=attrs.axis()&&!data.shape().dimension(i).equals(indices.shape().dimension(i))) throw new IllegalArgumentException("gather-elements non-axis dimension mismatch");
            result=indices.shape();
        }
        return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(data.dataType(),result,data.requiresGrad()));
    }

    private static CapturedGraphInference.InferenceResult oneHot(OneHotAttrs attrs,List<TensorDescriptor> in){
        TensorDescriptor indices=in.get(0); ElementwiseInference.requireIndex(indices,"indices"); List<Dimension> dims=new ArrayList<>(indices.shape().dimensions()); dims.add(new StaticDimension(attrs.depth()));
        return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(DataType.BOOL,Shape.ofDimensions(dims.toArray(Dimension[]::new)),false));
    }

    private static CapturedGraphInference.InferenceResult gatherNd(GatherNdAttrs attrs,List<TensorDescriptor> in){
        TensorDescriptor data=in.get(0),indices=in.get(1); ElementwiseInference.requireIndex(indices,"indices"); Shape result=ndShape(data.shape(),indices.shape(),attrs.batchDimensions());
        return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(data.dataType(),result,data.requiresGrad()));
    }

    private static CapturedGraphInference.InferenceResult axisScatter(AxisScatterKind kind,Object raw,List<TensorDescriptor> in){
        TensorDescriptor data=in.get(0),indices=in.get(1),updates=in.get(2); ElementwiseInference.requireIndex(indices,"indices"); sameType(data,updates);
        int selected; ScatterReduction reduction;
        if(kind==AxisScatterKind.SCATTER_ADD){selected=((IndexAxisAttrs)raw).axis(); reduction=ScatterReduction.ADD; ElementwiseInference.requireNumeric(data,"data");
            Shape expected=gatherResult(data.shape(),indices.shape(),selected); requireShape(updates.shape(),expected,"updates");
        } else { ScatterElementsAttrs attrs=(ScatterElementsAttrs)raw; selected=attrs.axis(); reduction=attrs.reduction(); axis(data.shape(),selected);
            if(data.dataType()==DataType.BOOL&&reduction!=ScatterReduction.NONE) throw new IllegalArgumentException("BOOL scatter reduction must be NONE");
            if(indices.shape().rank()!=data.shape().rank()||updates.shape().rank()!=indices.shape().rank()) throw new IllegalArgumentException("scatter-elements rank mismatch");
            requireShape(updates.shape(),indices.shape(),"updates"); for(int i=0;i<data.shape().rank();i++) if(i!=selected&&!data.shape().dimension(i).equals(indices.shape().dimension(i))) throw new IllegalArgumentException("scatter-elements non-axis dimension mismatch");
        }
        axis(data.shape(),selected); return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(data.dataType(),data.shape(),data.requiresGrad()||updates.requiresGrad()));
    }

    private static CapturedGraphInference.InferenceResult scatterNd(ScatterNdAttrs attrs,List<TensorDescriptor> in){
        TensorDescriptor data=in.get(0),indices=in.get(1),updates=in.get(2); ElementwiseInference.requireIndex(indices,"indices"); sameType(data,updates);
        if(data.dataType()==DataType.BOOL&&attrs.reduction()!=ScatterReduction.NONE) throw new IllegalArgumentException("BOOL scatter reduction must be NONE");
        Shape expected=ndShape(data.shape(),indices.shape(),attrs.batchDimensions()); requireShape(updates.shape(),expected,"updates");
        return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(data.dataType(),data.shape(),data.requiresGrad()||updates.requiresGrad()));
    }

    private static Shape ndShape(Shape data,Shape indices,int batch){
        if(indices.rank()==0) throw new IllegalArgumentException("indices rank must be at least one");
        if(batch<0||batch>=indices.rank()||batch>=data.rank()) throw new IllegalArgumentException("invalid batchDimensions");
        for(int i=0;i<batch;i++) if(!data.dimension(i).equals(indices.dimension(i))) throw new IllegalArgumentException("batch prefix mismatch");
        if(!(indices.dimension(indices.rank()-1) instanceof StaticDimension tuple)) throw new IllegalArgumentException("tuple depth must be static");
        long depth=tuple.size(); if(depth<1||depth>data.rank()-batch) throw new IllegalArgumentException("invalid tuple depth");
        List<Dimension> dims=new ArrayList<>(indices.dimensions().subList(0,indices.rank()-1)); dims.addAll(data.dimensions().subList(Math.toIntExact(batch+depth),data.rank())); return Shape.ofDimensions(dims.toArray(Dimension[]::new));
    }
    private static Shape gatherResult(Shape data,Shape indices,int axis){ axis(data,axis); List<Dimension> d=new ArrayList<>(data.dimensions().subList(0,axis)); d.addAll(indices.dimensions()); d.addAll(data.dimensions().subList(axis+1,data.rank())); return Shape.ofDimensions(d.toArray(Dimension[]::new)); }
    private static void axis(Shape s,int axis){if(axis<0||axis>=s.rank())throw new IllegalArgumentException("axis is not normalized for input rank");}
    private static void sameType(TensorDescriptor data,TensorDescriptor updates){if(data.dataType()!=updates.dataType())throw new IllegalArgumentException("updates data type must match data");}
    private static void requireShape(Shape actual,Shape expected,String role){if(!actual.equals(expected))throw new IllegalArgumentException(role+" shape mismatch");}
}
