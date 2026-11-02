package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.engine.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Public, explicitly seeded scalar-loss workflow; no package-private Engine fixture. */
class EngineCpuOrdinaryDynamicBatchIntegrationTest {
    /**
     * Runs the CPU0073 actual dense AD topology through one public prepared owner, including
     * empty B and concurrent mixed-B invocations. The matching CPU fixture asserts projection
     * 72 selection and the separate exact db boundary; this test owns the public lifecycle join.
     * @throws Exception if a concurrent invocation fails
     */
    @Test void denseTwoParameterGradientsReuseOnePreparedOwnerAcrossMixedBatch() throws Exception {
        var b = new DynamicDimension("B");
        Tensor x = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,Shape.ofDimensions(b,new StaticDimension(2)),Optional.empty(),true));
        Tensor z = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,Shape.ofDimensions(b,new StaticDimension(1)),Optional.empty(),true));
        Tensor w = TensorFactory.create(descriptor(Shape.of(2,1),true)),u = TensorFactory.create(descriptor(Shape.of(1,1),true));
        Tensor bias = TensorFactory.create(descriptor(Shape.of(1),true)),seed = value(Shape.scalar(),false,new float[]{1});
        Tensor y = x.matmul(w).add(z.matmul(u)).add(bias),loss = y.mul(y).sum();
        try (var engine = Engine.standard()) {
            var compiled = engine.compile(List.of(loss),List.of(seed),List.of(x,z,w,u,bias));
            try (var prepared = engine.prepare(compiled)) {
                java.util.function.IntConsumer verify = batch -> {
                    float[] xv = new float[2*batch],zv = new float[batch];
                    for (int row = 0; row < batch; row++) { xv[2*row]=2*row+1; xv[2*row+1]=2*row+2; zv[row]=row+1; }
                    var actual = Map.of(x.id(),value(Shape.of(batch,2),true,xv),z.id(),value(Shape.of(batch,1),true,zv),
                            w.id(),value(Shape.of(2,1),true,new float[]{1,2}),u.id(),value(Shape.of(1,1),true,new float[]{1}),
                            bias.id(),value(Shape.of(1),true,new float[]{1}),seed.id(),seed);
                    var inputs = compiled.inputs().stream().map(input -> actual.get(input.tensorId())).toList();
                    try (var result = engine.runValues(prepared,inputs)) {
                        assertEquals(6,result.resultCount());
                        float[] dx = new float[2*batch],dz = new float[batch]; float dw0=0,dw1=0,du=0,db=0;
                        for (int row = 0; row < batch; row++) {
                            float d = 14*(row+1);dx[2*row]=d;dx[2*row+1]=2*d;dz[row]=d;
                            dw0+=xv[2*row]*d;dw1+=xv[2*row+1]*d;du+=zv[row]*d;db+=d;
                        }
                        var expected = List.of(dx,dz,new float[]{dw0,dw1},new float[]{du},new float[]{db});
                        for (int target = 0; target < 5; target++) {
                            var bytes = result.materialize(result.publications().get(target+1),1000000).bytes();
                            assertEquals(expected.get(target).length*4,bytes.remaining());
                            for (float cell : expected.get(target)) assertEquals(Float.floatToRawIntBits(cell),Float.floatToRawIntBits(bytes.getFloat()));
                        }
                    }
                };
                for (int batch : new int[]{64,40,1,64,0}) verify.accept(batch);
                try (var pool = java.util.concurrent.Executors.newFixedThreadPool(4)) {
                    var calls = new ArrayList<java.util.concurrent.Callable<Void>>();
                    for (int batch : new int[]{64,1,40,0,40,64}) calls.add(() -> { verify.accept(batch); return null; });
                    for (var result : pool.invokeAll(calls)) result.get();
                }
            }
        }
    }
    @Test void publicCompilePrepareAndRunValuesExecuteActualGradientsAtChangingBatch() {
        Tensor x = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("B"), new StaticDimension(4)), Optional.empty(), true));
        Tensor w = TensorFactory.create(descriptor(Shape.of(4, 3), true));
        Tensor seed = value(Shape.scalar(), false, new float[] {1});
        try (Engine engine = Engine.standard()) {
            var compiled = engine.compile(List.of(x.matmul(w).mean()), List.of(seed), List.of(x, w));
            try (var prepared = engine.prepare(compiled)) {
                for (int b : new int[] {64, 40, 1, 64, 0}) {
                    float[] xv = new float[b * 4], wv = new float[12];
                    for (int i = 0; i < xv.length; i++) xv[i] = (i % 7 - 3) * .125f;
                    for (int i = 0; i < wv.length; i++) wv[i] = (i % 5 - 2) * .25f;
                    Tensor actualX = value(Shape.of(b, 4), true, xv), actualW = value(Shape.of(4, 3), true, wv);
                    var inputs = compiled.inputs().stream().map(i -> i.tensorId().equals(x.id()) ? actualX
                            : i.tensorId().equals(w.id()) ? actualW : seed).toList();
                    try (var result = engine.runValues(prepared, inputs)) {
                        assertEquals(3, result.resultCount());
                        var p = result.publications();
                        assertEquals(Shape.of(b, 4), p.get(1).descriptor().shape());
                        assertEquals(Shape.of(4, 3), p.get(2).descriptor().shape());
                        var objective = result.materialize(p.get(0), 1000000).bytes();
                        var dx = result.materialize(p.get(1), 1000000).bytes();
                        var dw = result.materialize(p.get(2), 1000000).bytes();
                        double sum = 0;
                        for (int row = 0; row < b; row++) for (int h = 0; h < 3; h++) {
                            float y = 0; for (int f = 0; f < 4; f++) y += xv[row * 4 + f] * wv[f * 3 + h];
                            sum += y;
                        }
                        if (b == 0) assertTrue(Float.isNaN(objective.getFloat()));
                        else assertEquals((float) (sum / (b * 3)), objective.getFloat());
                        float g = 1f / (b * 3);
                        for (int row = 0; row < b; row++) for (int f = 0; f < 4; f++) {
                            float expected = 0; for (int h = 0; h < 3; h++) expected += g * wv[f * 3 + h];
                            assertEquals(expected, dx.getFloat(), 0);
                        }
                        for (int f = 0; f < 4; f++) for (int h = 0; h < 3; h++) {
                            float expected = 0; for (int row = 0; row < b; row++) expected += xv[row * 4 + f] * g;
                            assertEquals(Float.floatToRawIntBits(expected), Float.floatToRawIntBits(dw.getFloat()));
                        }
                    }
                }
            }
        }
    }

    private static TensorDescriptor descriptor(Shape shape, boolean grad) {
        return new TensorDescriptor(DataType.FLOAT32, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }
    private static Tensor value(Shape shape, boolean grad, float[] data) {
        return TensorFactory.fromFlatArray(descriptor(shape, grad), Optional.empty(), data);
    }
}
