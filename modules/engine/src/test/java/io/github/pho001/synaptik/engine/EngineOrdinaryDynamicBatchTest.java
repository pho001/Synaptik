package io.github.pho001.synaptik.engine;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.backend.contract.*;
import io.github.pho001.synaptik.backend.cpu.CpuCapabilityProvider;
import io.github.pho001.synaptik.compiler.*;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import java.util.*;
import org.junit.jupiter.api.Test;

class EngineOrdinaryDynamicBatchTest {
    @Test void intermediateScalarBackedBroadcastCountRejectsBeforeAnyInputStorageInspection() {
        var b=new DynamicDimension("B");
        Tensor anchor=TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,
                Shape.ofDimensions(b,new StaticDimension(0)),Optional.empty(),false));
        Tensor scalar=TensorFactory.create(descriptor(Shape.scalar(),false));
        Tensor weight=TensorFactory.create(descriptor(Shape.of(65536,1),false));
        Tensor expanded=scalar.expand(Shape.ofDimensions(b,new StaticDimension(65536)));
        try(Engine engine=Engine.standard()) {
            var compiled=engine.compile(List.of(anchor.sum(),expanded.matmul(weight).sum()));
            try(var prepared=engine.prepare(compiled)) {
                Tensor actualAnchor=TensorFactory.create(descriptor(Shape.of(65536,0),false));
                var inputs=compiled.inputs().stream().map(i->i.tensorId().equals(anchor.id())?actualAnchor:
                        i.tensorId().equals(scalar.id())?scalar:weight).toList();
                assertThrows(ArithmeticException.class,()->engine.runValues(prepared,inputs),
                        "intermediate 65536*65536 count fails before absent storage, despite scalar backing bytes");
            }
        }
    }

    @Test void constantByteSymbolicModeAndPreborrowNumericAdmissionRemainExplicit() {
        Tensor x=TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("B"),new StaticDimension(0)),Optional.empty(),false));
        try(Engine engine=Engine.standard()) {
            var compiled=engine.compile(List.of(x.sum()));
            try(var prepared=engine.prepare(compiled)) {
                assertTrue(prepared.execution().memoryPlan().requiresBatchSize());
                assertThrows(IllegalArgumentException.class,()->new io.github.pho001.synaptik.runtime.run.PreparedExecutionRunner()
                        .run(prepared.execution(),List.of()));
                for(int b:new int[]{64,0,1})try(var result=engine.runValues(prepared,List.of(ones(Shape.of(b,0),false)))) {
                    assertEquals(0,result.materialize(result.publications().getFirst(),4).bytes().getFloat());
                }
                var huge=TensorFactory.create(descriptor(Shape.of((long)Integer.MAX_VALUE+1,0),false));
                assertThrows(ArithmeticException.class,()->engine.runValues(prepared,List.of(huge)),"numeric failure precedes missing storage");
            }
        }
    }

    @Test void invalidDescriptorsStorageAndDifferentBatchRunsAreIsolatedAndRetryable() throws Exception {
        Tensor x=TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("B"),new StaticDimension(4)),Optional.empty(),true));
        Tensor w=TensorFactory.create(descriptor(Shape.of(4,3),true));
        try(Engine engine=Engine.standard()) {
            var compiled=engine.compile(List.of(x.matmul(w)));
            var prepared=engine.prepare(compiled);
            java.util.function.Function<Tensor,List<Tensor>> inputs=value->compiled.inputs().stream()
                    .map(i->i.tensorId().equals(x.id())?value:ones(Shape.of(4,3),true)).toList();
            assertThrows(IllegalArgumentException.class,()->engine.runValues(prepared,inputs.apply(ones(Shape.of(2,5),true))));
            assertThrows(IllegalArgumentException.class,()->engine.runValues(prepared,inputs.apply(ones(Shape.of(2,4),false))));
            assertThrows(IllegalStateException.class,()->engine.runValues(prepared,inputs.apply(TensorFactory.create(descriptor(Shape.of(2,4),true)))));
            Tensor missing=TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,Shape.of(2,4),Optional.empty(),true));
            assertThrows(IllegalArgumentException.class,()->engine.runValues(prepared,inputs.apply(missing)));
            Tensor offset=TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,Shape.of(2,4),
                    Optional.of(LayoutDescriptor.of(Shape.of(2,4),new long[]{4,1},1,true)),true));
            assertThrows(IllegalArgumentException.class,()->engine.runValues(prepared,inputs.apply(offset)));
            var denseDescriptor=new TensorDescriptor(DataType.FLOAT32,Shape.of(2,4),
                    Optional.of(LayoutDescriptor.of(Shape.of(2,4),new long[]{4,1},0,true)),true);
            Tensor dense=TensorFactory.fromFlatArray(denseDescriptor,Optional.empty(),new float[]{1,1,1,1,1,1,1,1});
            try(var first=engine.runValues(prepared,inputs.apply(dense));var executor=java.util.concurrent.Executors.newFixedThreadPool(2)) {
                var a=executor.submit(()->{try(var r=engine.runValues(prepared,inputs.apply(ones(Shape.of(40,4),true)))) {
                    return r.materialize(r.publications().getFirst(),10000).byteSize(); }});
                var b=executor.submit(()->{try(var r=engine.runValues(prepared,inputs.apply(ones(Shape.of(1,4),true)))) {
                    return r.materialize(r.publications().getFirst(),10000).byteSize(); }});
                assertEquals(480,a.get(10,java.util.concurrent.TimeUnit.SECONDS));
                assertEquals(12,b.get(10,java.util.concurrent.TimeUnit.SECONDS));
                prepared.close();
                assertEquals(4,first.materialize(first.publications().getFirst(),10000).bytes().getFloat());
                assertThrows(IllegalStateException.class,()->engine.runValues(prepared,List.of()));
            } finally { prepared.close(); }
        }
    }

    /**
     * Verifies that ordinary matrix multiplication, mean loss and first-order gradients reuse one
     * preparation across changing and empty batches. Compile optimizations may change the number
     * of generated artifacts, but the nonempty retained artifact set must keep the same size and
     * object identities across every run of that preparation.
     */
    @Test void actualForwardAndFirstOrderClosureReusesOnePreparationAtChangingAndEmptyBatch() {
        Tensor x = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32,
                Shape.ofDimensions(new DynamicDimension("B"), new StaticDimension(4)), Optional.empty(), true));
        Tensor w = TensorFactory.create(descriptor(Shape.of(4, 3), true));
        Tensor y = x.matmul(w), loss = y.mean();
        var stage = new FunctionalGradientRequest.Stage(
                List.of(new FunctionalGradientRequest.ForwardTensorReference(loss)), List.of(Optional.empty()),
                List.of(x, w), false, FunctionalGradientRequest.DisconnectedPolicy.ERROR);
        var cpu = new CpuCapabilityProvider(); var backend = CpuCapabilityProvider.CPU_BACKEND_ID;
        var providerCalls=new java.util.concurrent.atomic.AtomicInteger();
        var countedProvider=new io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider() {
            public BackendId backendId(){return backend;}
            public boolean supports(io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery query){providerCalls.incrementAndGet();return cpu.supports(query);}
        };
        var artifacts = GraphCompilationPort.compile(CompileMode.FORWARD_AND_BACKWARD, List.of(y, loss),
                Optional.of(new FunctionalGradientRequest(List.of(stage))), GraphOptimizationConfig.standard(),
                BackendIntent.unconstrained(), PartitionScoringConfig.neutral(), List.of(countedProvider),
                List.of(new BackendAvailabilitySnapshot(backend, Map.of(new BackendDeviceId(backend, "cpu"), DeviceClass.CPU))));
        System.out.println("ordinary closure=" + artifacts.graph().nodes().stream().map(n -> n.operation().toString()).toList());
        try (Engine engine = Engine.standard()) {
            // The existing low-level Compiler request selects L only while retaining both forward
            // publications. No new public Engine compile overload or publication facade is added.
            var compiled = new CompiledGraph(engine, artifacts);
            try (var prepared = engine.prepare(compiled)) {
                var plan = prepared.execution().memoryPlan(); var schedule = prepared.execution().schedule();
                int compileQueries=providerCalls.get();
                var artifactsBefore=generatedArtifacts(prepared);
                assertFalse(artifactsBefore.isEmpty());
                System.out.println("compile=1 prepare=1 compile-provider-queries="+compileQueries+" retained-generated-units="+artifactsBefore.size()
                        +" buffers="+plan.buffers().size()+" workspaces="+plan.workspaces().size());
                assertTrue(plan.requiresBatchSize());
                for (int batch : new int[] {64, 40, 1, 64, 0}) {
                    Tensor actualX = ones(Shape.of(batch, 4), true), actualW = ones(Shape.of(4, 3), true);
                    var supplied = compiled.inputs().stream().map(input -> input.tensorId().equals(x.id()) ? actualX : actualW).toList();
                    try (var result = engine.runValues(prepared, supplied)) {
                        assertEquals(4, result.resultCount());
                        var publications = result.publications();
                        assertEquals(Shape.of(batch, 3), publications.get(0).descriptor().shape());
                        assertEquals(Shape.of(batch, 4), publications.get(2).descriptor().shape());
                        assertEquals(Shape.of(4, 3), publications.get(3).descriptor().shape());
                        for(int target=0;target<2;target++) {
                            var generated=artifacts.publication().gradientBindings().get(target).valueId();
                            boolean flag=artifacts.graph().values().stream().filter(v->v.id().equals(generated))
                                    .findFirst().orElseThrow().descriptor().requiresGrad();
                            assertEquals(flag,publications.get(target+2).descriptor().requiresGrad());
                        }
                        var output = result.materialize(publications.get(0), 1_000_000);
                        var objective = result.materialize(publications.get(1), 1_000_000);
                        var dx = result.materialize(publications.get(2), 1_000_000);
                        var dw = result.materialize(publications.get(3), 1_000_000);
                        assertEquals(12L * batch, output.byteSize()); assertEquals(16L * batch, dx.byteSize());
                        assertEquals(48, dw.byteSize()); assertEquals(4, objective.byteSize());
                        if (batch == 0) assertTrue(Float.isNaN(objective.bytes().getFloat()));
                        else assertEquals(4, objective.bytes().getFloat());
                        for (int i = 0; i < batch * 3; i++) assertEquals(4, output.bytes().getFloat(4 * i));
                        for (int i = 0; i < batch * 4; i++) assertEquals(1f / batch, dx.bytes().getFloat(4 * i), 2e-7f);
                        for (int i = 0; i < 12; i++) assertEquals(batch == 0 ? 0 : 1f / 3, dw.bytes().getFloat(4 * i), 2e-7f);
                    }
                    assertSame(plan, prepared.execution().memoryPlan()); assertSame(schedule, prepared.execution().schedule());
                    assertEquals(compileQueries,providerCalls.get());
                    var after=generatedArtifacts(prepared);
                    assertEquals(artifactsBefore.size(),after.size());
                    for(int i=0;i<after.size();i++)assertSame(artifactsBefore.get(i),after.get(i));
                }
            }
        }
    }

    private static List<io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel> generatedArtifacts(PreparedExecution prepared) {
        var result=new ArrayList<io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel>();
        for(var step:prepared.execution().schedule().steps())if(step instanceof io.github.pho001.synaptik.runtime.schedule.PreparedSchedule.ExecutionStep execution) {
            var partition=(io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedPartitionExecutable)execution.executable();
            for(var child:partition.children())result.add(((io.github.pho001.synaptik.backend.cpu.internal.executable.CpuPreparedExecutable)child).artifact());
        }
        return List.copyOf(result);
    }

    static TensorDescriptor descriptor(Shape shape, boolean grad) {
        return new TensorDescriptor(DataType.FLOAT32, shape, Optional.of(LayoutDescriptor.contiguous(shape)), grad);
    }
    static Tensor ones(Shape shape, boolean grad) {
        float[] values = new float[Math.toIntExact(shape.knownElementCount().orElseThrow())]; Arrays.fill(values, 1);
        return TensorFactory.fromFlatArray(descriptor(shape, grad), Optional.empty(), values);
    }
}
