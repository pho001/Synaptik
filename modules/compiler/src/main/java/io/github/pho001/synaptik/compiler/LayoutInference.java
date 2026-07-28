package io.github.pho001.synaptik.compiler;

import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.layout.*;
import io.github.pho001.synaptik.model.shape.Dimension;
import io.github.pho001.synaptik.model.shape.DimensionExpressions;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.shape.StaticDimension;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Derives descriptors and occurrence-local constraints for layout, Shape, composition, and
 * window transforms.
 *
 * <p>Binding-dependent {@code EXPAND} keeps the exact requested target Shape, emits one ordered
 * source-one-or-source-equal predicate for each unresolved aligned pair, and leaves layout
 * unresolved. Structural equality, a static source singleton, and fully resolved zero-stride
 * view geometry retain their existing behavior. This owner records obligations only; it does not
 * bind dimensions, select materialization, or authorize execution.</p>
 */
final class LayoutInference {
    private LayoutInference() {}

    /**
     * Verifies one supported layout or Shape-transform occurrence.
     *
     * @param op non-null typed operation whose kind belongs to this family
     * @param in non-null ordered input descriptors supplied by the captured node
     * @return immutable derived descriptors and ordered occurrence-local candidate constraints;
     *     never {@code null}
     * @throws IllegalArgumentException if the kind is unsupported here or the occurrence violates
     *     its rank, Shape, layout, or attribute contract, including a fully static incompatible
     *     expansion pair
     */
    static CapturedGraphInference.InferenceResult infer(Operation op,List<TensorDescriptor> in){
        if(op.kind() instanceof ContiguousKind)return contiguous(in);
        if(op.kind() instanceof ShapeTransformKind k)return shapeTransform(k,(TargetShapeAttrs)op.attrs(),in);
        if(op.kind() instanceof AxisTransformKind k)return axisTransform(k,op.attrs(),in);
        if(op.kind() instanceof SliceKind k)return slice(k,op.attrs(),in);
        if(op.kind() instanceof PadKind)return pad((PadAttrs)op.attrs(),in);
        if(op.kind() instanceof TileKind)return tile((TileAttrs)op.attrs(),in);
        if(op.kind() instanceof TensorCompositionKind k)return composition(k,(CompositionAxisAttrs)op.attrs(),in);
        if(op.kind() instanceof WindowTransformKind k)return window(k,op.attrs(),in);
        throw new IllegalArgumentException("unsupported layout kind");
    }
    private static CapturedGraphInference.InferenceResult contiguous(List<TensorDescriptor> in){TensorDescriptor x=in.get(0);Optional<LayoutDescriptor> l=x.shape().isFullyStatic()?Optional.of(LayoutDescriptor.contiguous(x.shape())):Optional.empty();return CapturedGraphInference.InferenceResult.of(new TensorDescriptor(x.dataType(),x.shape(),l,x.requiresGrad()));}
    private static CapturedGraphInference.InferenceResult shapeTransform(ShapeTransformKind kind,TargetShapeAttrs attrs,List<TensorDescriptor> in){
        TensorDescriptor x=in.get(0);Shape target=attrs.targetShape();Optional<LayoutDescriptor> layout=Optional.empty();List<CapturedGraphInference.ConstraintRequest> cs=new ArrayList<>();
        if(kind==ShapeTransformKind.RESHAPE){cs.add(new CapturedGraphInference.ConstraintRequest("reshape element count",new ShapeElementCountEqual(x.shape(),target)));if(target.isFullyStatic()&&x.layout().isPresent()&&x.layout().orElseThrow().isContiguous()){LayoutDescriptor c=LayoutDescriptor.contiguous(target);layout=Optional.of(LayoutDescriptor.of(target,c.strides(),x.layout().orElseThrow().storageOffset(),true));}}
        else {
            if(target.rank()<x.shape().rank())throw new IllegalArgumentException("expand target rank below input rank");
            int off=target.rank()-x.shape().rank();
            boolean bindingDependent=false;
            for(int i=0;i<x.shape().rank();i++){
                Dimension source=x.shape().dimension(i),targetDimension=target.dimension(off+i);
                if(source.equals(targetDimension)
                        || source instanceof StaticDimension singleton&&singleton.size()==1){
                    continue;
                }
                if(source instanceof StaticDimension&&targetDimension instanceof StaticDimension){
                    throw new IllegalArgumentException("incompatible expand dimension");
                }
                bindingDependent=true;
                cs.add(new CapturedGraphInference.ConstraintRequest(
                        "expand target axis "+(off+i),
                        new AnyOf(List.of(
                                new DimensionEqual(source,new StaticDimension(1)),
                                new DimensionEqual(source,targetDimension)))));
            }
            if(!bindingDependent&&target.isFullyStatic()&&x.layout().isPresent()){
                long[] strides=new long[target.rank()];
                for(int i=0;i<x.shape().rank();i++){
                    int j=off+i;
                    Dimension source=x.shape().dimension(i),targetDimension=target.dimension(j);
                    strides[j]=source instanceof StaticDimension singleton
                                    &&singleton.size()==1&&!source.equals(targetDimension)
                            ?0:x.layout().orElseThrow().stride(i);
                }
                layout=Optional.of(LayoutDescriptor.of(
                        target,strides,x.layout().orElseThrow().storageOffset(),true));
            }
        }
        return new CapturedGraphInference.InferenceResult(List.of(new TensorDescriptor(x.dataType(),target,layout,x.requiresGrad())),cs);
    }
    private static CapturedGraphInference.InferenceResult axisTransform(AxisTransformKind kind,Object raw,List<TensorDescriptor> in){
        TensorDescriptor x=in.get(0);Shape shape;Optional<LayoutDescriptor> layout=Optional.empty();
        if(kind==AxisTransformKind.PERMUTE){PermutationAttrs a=(PermutationAttrs)raw;if(a.axes().size()!=x.shape().rank())throw new IllegalArgumentException("permutation rank mismatch");Dimension[] d=new Dimension[a.axes().size()];long[] strides=new long[d.length];for(int i=0;i<d.length;i++){d[i]=x.shape().dimension(a.axes().get(i));if(x.layout().isPresent())strides[i]=x.layout().orElseThrow().stride(a.axes().get(i));}shape=Shape.ofDimensions(d);if(x.layout().isPresent())layout=Optional.of(LayoutDescriptor.of(shape,strides,x.layout().orElseThrow().storageOffset(),true));}
        else {AxisTransformAttrs a=(AxisTransformAttrs)raw;List<Dimension>d=new ArrayList<>(x.shape().dimensions());if(kind==AxisTransformKind.EXPAND_DIMS){if(a.axis()<0||a.axis()>d.size())throw new IllegalArgumentException("insertion axis out of range");d.add(a.axis(),new StaticDimension(1));shape=Shape.ofDimensions(d.toArray(Dimension[]::new));if(x.layout().isPresent()){long[] s=new long[shape.rank()];for(int i=0;i<a.axis();i++)s[i]=x.layout().orElseThrow().stride(i);s[a.axis()]=a.axis()==x.shape().rank()?1:Math.multiplyExact(x.layout().orElseThrow().stride(a.axis()),((StaticDimension)x.shape().dimension(a.axis())).size());for(int i=a.axis();i<x.shape().rank();i++)s[i+1]=x.layout().orElseThrow().stride(i);layout=Optional.of(LayoutDescriptor.of(shape,s,x.layout().orElseThrow().storageOffset(),true));}}
            else {axis(x.shape(),a.axis());if(!(x.shape().dimension(a.axis()) instanceof StaticDimension singleton&&singleton.size()==1))throw new IllegalArgumentException("squeeze axis must be singleton");d.remove(a.axis());shape=Shape.ofDimensions(d.toArray(Dimension[]::new));if(x.layout().isPresent()){long[]resultStrides=new long[shape.rank()];for(int i=0,j=0;i<x.shape().rank();i++)if(i!=a.axis())resultStrides[j++]=x.layout().orElseThrow().stride(i);layout=Optional.of(LayoutDescriptor.of(shape,resultStrides,x.layout().orElseThrow().storageOffset(),true));}}}
        return CapturedGraphInference.InferenceResult.of(new TensorDescriptor(x.dataType(),shape,layout,x.requiresGrad()));
    }
    private static CapturedGraphInference.InferenceResult slice(SliceKind kind,Object raw,List<TensorDescriptor> in){
        TensorDescriptor x=in.get(0);if(raw instanceof CropToShapeAttrs c){if(c.targetShape().rank()!=x.shape().rank()||c.prefixShape().rank()!=x.shape().rank())throw new IllegalArgumentException("crop rank mismatch");List<CapturedGraphInference.ConstraintRequest>cs=new ArrayList<>();for(int i=0;i<x.shape().rank();i++){Dimension region=DimensionExpressions.add(c.prefixShape().dimension(i),c.targetShape().dimension(i));cs.add(new CapturedGraphInference.ConstraintRequest("crop axis "+i,FitsWithin.dimension(0,region,x.shape().dimension(i))));}return new CapturedGraphInference.InferenceResult(List.of(ElementwiseInference.descriptor(x.dataType(),c.targetShape(),x.requiresGrad())),cs);}
        SliceAttrs a=(SliceAttrs)raw;List<Dimension>d=new ArrayList<>(x.shape().dimensions());List<CapturedGraphInference.ConstraintRequest>cs=new ArrayList<>();long offset=x.layout().map(LayoutDescriptor::storageOffset).orElse(0L);long[]strides=x.layout().map(LayoutDescriptor::strides).orElse(null);
        for(int i=0;i<a.axes().size();i++){int ax=a.axes().get(i);axis(x.shape(),ax);d.set(ax,new StaticDimension(a.lengths().get(i)));long step=a.steps().get(i);long extent=a.lengths().get(i)==0?0:step>0?Math.addExact(Math.multiplyExact(a.lengths().get(i)-1,step),1):1;cs.add(new CapturedGraphInference.ConstraintRequest("slice axis "+ax,FitsWithin.dimension(a.starts().get(i),new StaticDimension(extent),x.shape().dimension(ax))));if(strides!=null&&step>0){offset=Math.addExact(offset,Math.multiplyExact(a.starts().get(i),strides[ax]));strides[ax]=Math.multiplyExact(strides[ax],step);}else if(step<0)strides=null;}
        Shape result=Shape.ofDimensions(d.toArray(Dimension[]::new));if(kind==SliceKind.SLICE_UPDATE){TensorDescriptor update=in.get(1);if(update.dataType()!=x.dataType()||!update.shape().equals(result))throw new IllegalArgumentException("slice update descriptor mismatch");return new CapturedGraphInference.InferenceResult(List.of(ElementwiseInference.descriptor(x.dataType(),x.shape(),x.requiresGrad()||update.requiresGrad())),cs);}Optional<LayoutDescriptor>l=strides==null||result.knownElementCount().orElseThrow()==0?Optional.empty():Optional.of(LayoutDescriptor.of(result,strides,offset,true));return new CapturedGraphInference.InferenceResult(List.of(new TensorDescriptor(x.dataType(),result,l,x.requiresGrad())),cs);
    }
    private static CapturedGraphInference.InferenceResult pad(PadAttrs a,List<TensorDescriptor>in){TensorDescriptor x=in.get(0);if(a.before().size()!=x.shape().rank()||a.constantValue().dataType()!=x.dataType())throw new IllegalArgumentException("pad attributes mismatch");Dimension[]d=new Dimension[x.shape().rank()];for(int i=0;i<d.length;i++)d[i]=DimensionExpressions.addConstant(x.shape().dimension(i),Math.addExact(a.before().get(i),a.after().get(i)));return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(x.dataType(),Shape.ofDimensions(d),x.requiresGrad()));}
    private static CapturedGraphInference.InferenceResult tile(TileAttrs a,List<TensorDescriptor>in){TensorDescriptor x=in.get(0);if(a.repeats().size()!=x.shape().rank())throw new IllegalArgumentException("tile repeat rank mismatch");Dimension[]d=new Dimension[x.shape().rank()];for(int i=0;i<d.length;i++)d[i]=DimensionExpressions.multiply(x.shape().dimension(i),a.repeats().get(i));return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(x.dataType(),Shape.ofDimensions(d),x.requiresGrad()));}
    private static CapturedGraphInference.InferenceResult composition(TensorCompositionKind kind,CompositionAxisAttrs a,List<TensorDescriptor>in){TensorDescriptor first=in.get(0);boolean grad=false;Shape result;if(kind==TensorCompositionKind.STACK){if(a.axis()>first.shape().rank())throw new IllegalArgumentException("stack axis out of range");for(TensorDescriptor x:in){sameType(first,x);if(!x.shape().equals(first.shape()))throw new IllegalArgumentException("stack shape mismatch");grad|=x.requiresGrad();}List<Dimension>d=new ArrayList<>(first.shape().dimensions());d.add(a.axis(),new StaticDimension(in.size()));result=Shape.ofDimensions(d.toArray(Dimension[]::new));}else{axis(first.shape(),a.axis());Dimension sum=new StaticDimension(0);for(TensorDescriptor x:in){sameType(first,x);if(x.shape().rank()!=first.shape().rank())throw new IllegalArgumentException("concat rank mismatch");for(int i=0;i<first.shape().rank();i++)if(i!=a.axis()&&!x.shape().dimension(i).equals(first.shape().dimension(i)))throw new IllegalArgumentException("concat dimension mismatch");sum=DimensionExpressions.add(sum,x.shape().dimension(a.axis()));grad|=x.requiresGrad();}List<Dimension>d=new ArrayList<>(first.shape().dimensions());d.set(a.axis(),sum);result=Shape.ofDimensions(d.toArray(Dimension[]::new));}return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(first.dataType(),result,grad));}
    private static CapturedGraphInference.InferenceResult window(WindowTransformKind kind,Object raw,List<TensorDescriptor>in){TensorDescriptor x=in.get(0);Shape result;if(kind==WindowTransformKind.UNFOLD_AXIS){UnfoldAxisAttrs a=(UnfoldAxisAttrs)raw;axis(x.shape(),a.axis());if(!(x.shape().dimension(a.axis())instanceof StaticDimension s))throw new IllegalArgumentException("unfold axis extent must be static");long count=Math.addExact((s.size()-a.size())/a.step(),1);if(s.size()<a.size())throw new IllegalArgumentException("window does not fit");List<Dimension>d=new ArrayList<>(x.shape().dimensions());d.set(a.axis(),new StaticDimension(count));d.add(new StaticDimension(a.size()));result=Shape.ofDimensions(d.toArray(Dimension[]::new));}
        else if(kind==WindowTransformKind.FOLD_AXIS){FoldAxisAttrs a=(FoldAxisAttrs)raw;if(x.shape().rank()<2)throw new IllegalArgumentException("fold input rank too small");axis(Shape.ofDimensions(x.shape().dimensions().subList(0,x.shape().rank()-1).toArray(Dimension[]::new)),a.axis());List<Dimension>d=new ArrayList<>(x.shape().dimensions());Dimension window=d.removeLast();long expected=((StaticDimension)window).size();long count=((StaticDimension)d.get(a.axis())).size();if(Math.addExact(Math.multiplyExact(Math.max(0,count-1),a.step()),expected)!=a.outputSize())throw new IllegalArgumentException("fold geometry mismatch");d.set(a.axis(),new StaticDimension(a.outputSize()));result=Shape.ofDimensions(d.toArray(Dimension[]::new));}
        else {Window2dAttrs w=raw instanceof Unfold2dAttrs u?u.window():raw instanceof Fold2dAttrs f?f.window():(Window2dAttrs)raw;if(kind==WindowTransformKind.UNFOLD2D){if(x.shape().rank()!=4)throw new IllegalArgumentException("unfold2d rank must be four");if(raw instanceof Unfold2dAttrs u&&u.paddingValue().dataType()!=x.dataType())throw new IllegalArgumentException("padding value type mismatch");Dimension oh=windowDimension(x.shape().dimension(2),w.kernelHeight(),w.paddingHeight(),w.strideHeight(),w.dilationHeight(),w.ceilMode());Dimension ow=windowDimension(x.shape().dimension(3),w.kernelWidth(),w.paddingWidth(),w.strideWidth(),w.dilationWidth(),w.ceilMode());result=Shape.ofDimensions(x.shape().dimension(0),DimensionExpressions.multiply(DimensionExpressions.multiply(x.shape().dimension(1),w.kernelHeight()),w.kernelWidth()),DimensionExpressions.multiply(oh,ow));}else{Fold2dAttrs f=(Fold2dAttrs)raw;if(x.shape().rank()!=3||f.outputShape().rank()!=4)throw new IllegalArgumentException("fold2d rank mismatch");Shape expected=unfoldShape(f.outputShape(),w);if(!x.shape().equals(expected))throw new IllegalArgumentException("fold2d columns mismatch");result=f.outputShape();}}
        return CapturedGraphInference.InferenceResult.of(ElementwiseInference.descriptor(x.dataType(),result,x.requiresGrad()));}
    private static Shape unfoldShape(Shape s,Window2dAttrs w){Dimension oh=windowDimension(s.dimension(2),w.kernelHeight(),w.paddingHeight(),w.strideHeight(),w.dilationHeight(),w.ceilMode());Dimension ow=windowDimension(s.dimension(3),w.kernelWidth(),w.paddingWidth(),w.strideWidth(),w.dilationWidth(),w.ceilMode());return Shape.ofDimensions(s.dimension(0),DimensionExpressions.multiply(DimensionExpressions.multiply(s.dimension(1),w.kernelHeight()),w.kernelWidth()),DimensionExpressions.multiply(oh,ow));}
    private static Dimension windowDimension(Dimension d,long kernel,long pad,long stride,long dilation,boolean ceil){long effective=Math.addExact(Math.multiplyExact(dilation,kernel-1),1);if(d instanceof StaticDimension s){long n=Math.subtractExact(Math.addExact(s.size(),Math.multiplyExact(2,pad)),effective);if(n<0)throw new IllegalArgumentException("window does not fit");return new StaticDimension(Math.addExact(n/stride+(ceil&&n%stride!=0?1:0),1));}Dimension n=DimensionExpressions.addConstant(d,Math.subtractExact(Math.multiplyExact(2,pad),effective));return DimensionExpressions.addConstant(ceil?DimensionExpressions.ceilingDivide(n,stride):DimensionExpressions.floorDivide(n,stride),1);}
    private static void axis(Shape s,int a){if(a<0||a>=s.rank())throw new IllegalArgumentException("axis out of range");}
    private static void sameType(TensorDescriptor a,TensorDescriptor b){if(a.dataType()!=b.dataType())throw new IllegalArgumentException("data type mismatch");}
}
