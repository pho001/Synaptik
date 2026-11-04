package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.loss.LossReduction;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Public CPU composition regressions without changing shared Engine or Compiler fixtures. */
final class EngineCpuComposedValidationIntegrationTest {
    /** Scale AD contains rounded gradient*normalized followed by an ordinary prefix-axis SUM. */
    @Test void layerNormScaleGradientOrdinarySumReusesPreparationWithFreshInputs() {
        try (var arena = Arena.ofShared(); var engine = Engine.builder()
                .takeOwnership(io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.open()).build()) {
            var input = floats(arena,Shape.of(2,3),false,1,2,3,4,5,6);
            var scaleBase = floats(arena,Shape.of(3),false,1,1,1);
            var scale = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,Shape.of(3),scaleBase.descriptor().layout(),true),
                    Optional.empty(),scaleBase.hostStorage());
            var bias = floats(arena,Shape.of(3),false,0,0,0); var upstream = floats(arena,Shape.of(2,3),false,1,2,3,4,5,6);
            var seed = floats(arena,Shape.scalar(),false,1);
            var objective = input.layerNorm(Shape.of(3),scale,bias,io.github.pho001.synaptik.model.datatype.ScalarValue.float32(.01f)).mul(upstream).sum();
            var compiled = engine.compile(List.of(objective),List.of(seed),List.of(scale));
            var pool = List.of(input,scale,bias,upstream,seed);
            var ordered = compiled.inputs().stream().map(i -> pool.stream().filter(t -> t.id().equals(i.tensorId())).findFirst().orElseThrow()).toList();
            try (var session = engine.prepare(compiled)) {
                for (int run = 1; run <= 2; run++) {
                    var storage = ((MemorySegmentStorage)input.hostStorage().orElseThrow()).segment();
                    for (int i = 0; i < 6; i++) storage.setAtIndex(ValueLayout.JAVA_FLOAT,i,run*(i+1));
                    double normalized = run/Math.sqrt(2d*run*run/3+.01);
                    try (var result = engine.run(session,ordered)) {
                        assertEquals(4*normalized,result.materialize(result.publications().getFirst(),4).bytes().getFloat(),2e-5);
                        var gradient = result.materialize(result.publications().get(1),12).bytes();
                        assertEquals(-5*normalized,gradient.getFloat(),2e-5); assertEquals(0,gradient.getFloat()); assertEquals(9*normalized,gradient.getFloat(),2e-5);
                    }
                }
            }
        }
    }
    /** Actual Compiler ordinary broadcast gradients reuse preparation with fresh input values. */
    @Test void ordinaryProducerReductionAndCompilerGradientReuseFreshInputs() {
        try (var arena = Arena.ofShared(); var engine = Engine.builder()
                .takeOwnership(io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.open()).build()) {
            var input = floats(arena,Shape.of(2,3),false,1,2,3,4,5,6);
            var weightStorage = arena.allocate(3L*4,4);
            for (int i = 0; i < 3; i++) weightStorage.setAtIndex(ValueLayout.JAVA_FLOAT,i,2);
            var weights = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,Shape.of(3),
                    Optional.of(LayoutDescriptor.contiguous(Shape.of(3))),true),Optional.empty(),
                    Optional.of(new MemorySegmentStorage(DataType.FLOAT32,3,weightStorage)));
            var seed = floats(arena,Shape.scalar(),false,1);
            var objective = input.mul(weights).sum(1).sum();
            var compiled = engine.compile(List.of(objective),List.of(seed),List.of(weights));
            var pool = List.of(input,weights,seed);
            var ordered = compiled.inputs().stream().map(i -> pool.stream().filter(t -> t.id().equals(i.tensorId())).findFirst().orElseThrow()).toList();
            try (var session = engine.prepare(compiled)) {
                for (int run = 1; run <= 2; run++) {
                    var storage = ((MemorySegmentStorage)input.hostStorage().orElseThrow()).segment();
                    for (int i = 0; i < 6; i++) storage.setAtIndex(ValueLayout.JAVA_FLOAT,i,run*(i+1));
                    try (var result = engine.run(session,ordered)) {
                        assertEquals(42f*run,result.materialize(result.publications().getFirst(),4).bytes().getFloat());
                        var gradient = result.materialize(result.publications().get(1),12).bytes();
                        for (int i = 0; i < 3; i++) assertEquals((5f+2*i)*run,gradient.getFloat());
                    }
                }
            }
        }
    }

    /** Public producer/reduction stores survive an expanding ADD tail and repeated execution. */
    @Test void publishedProducerAndAggregateSurviveExpandingTail() {
        try (var arena = Arena.ofShared(); var engine = Engine.builder()
                .takeOwnership(io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.open()).build()) {
            var input = floats(arena,Shape.of(2,3),false,1,2,3,4,5,6);
            var factors = floats(arena,Shape.of(2,3),false,2,2,2,2,2,2);
            var bias = floats(arena,Shape.of(2),false,1,2);
            var product = input.mul(factors); var aggregate = product.sum(); var expanded = aggregate.add(bias);
            var compiled = engine.compile(List.of(product,aggregate,expanded));
            var pool = List.of(input,factors,bias);
            var ordered = compiled.inputs().stream().map(i -> pool.stream().filter(t -> t.id().equals(i.tensorId())).findFirst().orElseThrow()).toList();
            try (var session = engine.prepare(compiled)) {
                for (int run = 1; run <= 2; run++) {
                    var storage = ((MemorySegmentStorage)input.hostStorage().orElseThrow()).segment();
                    for (int i = 0; i < 6; i++) storage.setAtIndex(ValueLayout.JAVA_FLOAT,i,run*(i+1));
                    try (var result = engine.run(session,ordered)) {
                        var products = result.materialize(result.publications().get(0),24).bytes();
                        for (int i = 0; i < 6; i++) assertEquals(2f*run*(i+1),products.getFloat());
                        assertEquals(42f*run,result.materialize(result.publications().get(1),4).bytes().getFloat());
                        var output = result.materialize(result.publications().get(2),8).bytes();
                        assertEquals(42f*run+1,output.getFloat()); assertEquals(42f*run+2,output.getFloat());
                    }
                }
            }
        }
    }

    /** Checks actual Compiler mean-loss gradients through one preparation and fresh caller values. */
    @Test void compilerMeanLossCountFoldsWhileReusableSessionReadsFreshCallerValues() {
        try(var arena=Arena.ofShared();var engine=Engine.builder()
                .takeOwnership(io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.open()).build()) {
            var storage=arena.allocate(6L*Float.BYTES,Float.BYTES);
            var input=TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,Shape.of(2,3),
                    Optional.of(LayoutDescriptor.contiguous(Shape.of(2,3))),true),Optional.empty(),
                    Optional.of(new MemorySegmentStorage(DataType.FLOAT32,6,storage)));
            var target=floats(arena,Shape.of(2,3),false,0,0,0,0,0,0);
            var seed=tensor(DataType.FLOAT32,Shape.scalar(),arena.allocate(4,4));
            ((MemorySegmentStorage)seed.hostStorage().orElseThrow()).segment().setAtIndex(ValueLayout.JAVA_FLOAT,0,1);
            var objective=input.meanSquaredError(target,LossReduction.MEAN);
            var compiled=engine.compile(List.of(objective),List.of(seed),List.of(input));
            var pool=List.of(input,target,seed);
            var ordered=compiled.inputs().stream().map(i->pool.stream().filter(t->t.id().equals(i.tensorId())).findFirst().orElseThrow()).toList();
            try(var session=engine.prepare(compiled)) {
                for(int run=1;run<=2;run++) {
                    for(int i=0;i<6;i++) storage.setAtIndex(ValueLayout.JAVA_FLOAT,i,run*(i+1));
                    try(var result=engine.run(session,ordered)) {
                        assertEquals(2,result.publications().size());
                        assertEquals(run*run*91f/6,result.materialize(result.publications().getFirst(),4).bytes().getFloat(),1e-5f);
                        var gradient=result.materialize(result.publications().get(1),24).bytes();
                        for(int i=0;i<6;i++) assertEquals(run*(i+1)/3f,gradient.getFloat(),1e-6f);
                    }
                }
            }
        }
    }

    /** Checks actual normalization count plumbing through a running-mean objective and gradient. */
    @Test void compilerNormalizationCountPreservesStatisticsAndInputGradient() {
        try(var arena=Arena.ofShared();var engine=Engine.builder()
                .takeOwnership(io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration.open()).build()) {
            var base=floats(arena,Shape.of(4,2),false,1,2,3,4,5,6,7,8);
            var input=TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,Shape.of(4,2),
                    base.descriptor().layout(),true),Optional.empty(),base.hostStorage());
            var scale=floats(arena,Shape.of(2),false,1,1);
            var bias=floats(arena,Shape.of(2),false,0,0);
            var mean=floats(arena,Shape.of(2),false,0,0);
            var variance=floats(arena,Shape.of(2),false,1,1);
            var objective=input.batchNormTraining(1,scale,bias,mean,variance,
                    io.github.pho001.synaptik.model.datatype.ScalarValue.float32(.5f),
                    io.github.pho001.synaptik.model.datatype.ScalarValue.float32(.01f)).nextRunningMean().sum();
            var backward=engine.backward(objective,List.of(input),36);
            assertEquals(4.5f,backward.objective().bytes().getFloat());
            var gradient=backward.gradients().getFirst().bytes();
            for(int i=0;i<8;i++) assertEquals(.125f,gradient.getFloat());
        }
    }

    @Test void nonScalarLossesComposeWithArithmeticAndStridedCallerBoundaries() {
        try (var arena = Arena.ofShared(); var engine = Engine.standard()) {
            for (boolean strided : List.of(false, true)) {
                var prediction = floats(arena, Shape.of(2, 3), strided, -1, -2, -3, -3, -2, -1);
                var target = floats(arena, Shape.of(2, 3), strided, 0, 0, 1, 1, 0, 0);
                assertArrayEquals(new float[] {-1, -4, -4, -4, -4, -1}, run(engine,
                        prediction.neg().meanSquaredError(target, LossReduction.NONE).neg(),
                        List.of(prediction, target), 6), 0f);
                float loss = (float) Math.log(1 + Math.exp(-1) + Math.exp(-2));
                assertArrayEquals(new float[] {-loss, -loss}, run(engine,
                        prediction.neg().categoricalCrossEntropyWithLogits(target, 1,
                                LossReduction.NONE).neg(), List.of(prediction, target), 2), 1e-6f);
            }
        }
    }

    @Test void computedIndexTargetsAreRevalidatedOnEachPublicRun() {
        try (var arena = Arena.ofShared(); var engine = Engine.standard()) {
            Tensor logits = floats(arena, Shape.of(2, 3), false, 0, 0, 0, 0, 0, 0);
            var targetStorage = arena.allocate(8, Float.BYTES);
            Tensor source = tensor(DataType.FLOAT32, Shape.of(2), targetStorage);
            for (DataType type : List.of(DataType.INT32, DataType.INT64)) {
                Tensor output = logits.categoricalCrossEntropyWithLogits(source.cast(type),
                        1, LossReduction.NONE);
                try (var session = engine.prepare(engine.compile(List.of(output)))) {
                    for (int iteration = 0; iteration < 2; iteration++) {
                        targetStorage.setAtIndex(ValueLayout.JAVA_FLOAT, 0, 1);
                        targetStorage.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 3);
                        assertThrows(IndexOutOfBoundsException.class,
                                () -> engine.run(session, List.of(logits, source)));
                        targetStorage.setAtIndex(ValueLayout.JAVA_FLOAT, 1, 2);
                        try (var result = engine.run(session, List.of(logits, source))) {
                            var bytes = result.materialize(result.publications().getFirst(), 8).bytes();
                            assertEquals((float) Math.log(3), bytes.getFloat(), 1e-6f);
                            assertEquals((float) Math.log(3), bytes.getFloat(), 1e-6f);
                        }
                    }
                }
            }
        }
    }

    @Test void computedBooleanMaskRunsBeforeItsReductionConsumer() {
        try (var arena = Arena.ofShared(); var engine = Engine.standard()) {
            Tensor input = floats(arena, Shape.of(2, 2), false, 10, 20, 30, 40);
            var maskStorage = arena.allocate(4, 1);
            MemorySegment.copy(MemorySegment.ofArray(new byte[] {0, 1, 1, 0}), 0, maskStorage, 0, 4);
            Tensor mask = tensor(DataType.BOOL, Shape.of(2, 2), maskStorage);
            assertArrayEquals(new float[] {10, 40}, run(engine, input.sum(1, mask.logicalNot()),
                    List.of(input, mask), 2));
        }
    }

    private static float[] run(Engine engine, Tensor output, List<Tensor> inputs, int count) {
        try (var session = engine.prepare(engine.compile(List.of(output)));
                var result = engine.run(session, inputs)) {
            var bytes = result.materialize(result.publications().getFirst(),
                    count * (long) Float.BYTES).bytes();
            float[] values = new float[count];
            for (int i = 0; i < count; i++) values[i] = bytes.getFloat();
            return values;
        }
    }

    private static Tensor floats(Arena arena, Shape shape, boolean strided, float... values) {
        var layout = strided ? LayoutDescriptor.of(shape, new long[] {8, 2}, 1, true)
                : LayoutDescriptor.contiguous(shape);
        long count = layout.referencedElementSpan();
        var segment = arena.allocate(count * Float.BYTES, Float.BYTES);
        for (int i = 0; i < values.length; i++) {
            long address = strided ? 1 + i / 3 * 8L + i % 3 * 2L : i;
            segment.setAtIndex(ValueLayout.JAVA_FLOAT, address, values[i]);
        }
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(layout), false), Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, count, segment)));
    }

    private static Tensor tensor(DataType type, Shape shape, MemorySegment storage) {
        return TensorFactory.create(new TensorDescriptor(type, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false), Optional.empty(),
                Optional.of(new MemorySegmentStorage(type, shape.knownElementCount().orElseThrow(),
                        storage)));
    }
}
