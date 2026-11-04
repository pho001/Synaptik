package io.github.pho001.synaptik.backend.cpu;

import static org.junit.jupiter.api.Assertions.*;
import static io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLoweringTest.*;
import io.github.pho001.synaptik.backend.contract.*;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanLowering;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuGeneralScanCompositionLowering;
import io.github.pho001.synaptik.backend.cpu.internal.executable.CpuTypedEmptyScanConsumerTest.Consumer;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionAnalysisInputs;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparer;
import io.github.pho001.synaptik.compiler.*;
import io.github.pho001.synaptik.config.compile.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.*;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.operation.layout.ShapeTransformKind;
import io.github.pho001.synaptik.model.operation.layout.TargetShapeAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.cast.*;
import io.github.pho001.synaptik.model.operation.scan.*;
import io.github.pho001.synaptik.model.shape.*;
import io.github.pho001.synaptik.model.tensor.*;
import io.github.pho001.synaptik.planning.capability.*;
import io.github.pho001.synaptik.planning.memory.LogicalMemoryRequirement;
import io.github.pho001.synaptik.prepare.GraphPreparation;
import io.github.pho001.synaptik.prepare.analysis.PrepareContext;
import java.lang.foreign.ValueLayout;
import java.util.*;
import org.junit.jupiter.api.Test;

/** Actual provider qualification of both original and Compiler-produced sibling bodies. */
public class CpuGeneralScanCapabilityTest {
    @Test void providerQualifiesRawSymbolicLengthQueriesIncludingZeroTimeIndependently() {
        var provider = new CpuCapabilityProvider();
        for (var type : List.of(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32,
                io.github.pho001.synaptik.model.datatype.DataType.FLOAT64))
            for (long time : new long[] {0, 3}) for (boolean reverse : List.of(false, true)) {
                assertTrue(provider.supports(symbolicLengthQuery(type, time, reverse)));
            }
    }

    @Test void directCompositeQueriesProveNamedReshapesBeforeCapabilityAdmission() {
        var batch = new DynamicDimension("Customers");
        Shape source = Shape.ofDimensions(batch,new StaticDimension(2));
        Shape invalid = Shape.ofDimensions(batch,new StaticDimension(3));
        Shape foreign = Shape.ofDimensions(new DynamicDimension("OtherCustomers"),new StaticDimension(2));
        var provider = new CpuCapabilityProvider();
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64)) for (long time : new long[] {0,3}) {
            for (Shape target : List.of(invalid,foreign)) {
                var query = directReshapeQuery(type,time,source,target);
                assertFalse(provider.supports(query),"invalid declared named relation at T="+time);
                assertThrows(IllegalArgumentException.class,() -> CpuGeneralScanCompositionLowering.analyze(query));
            }
            for (Shape target : List.of(Shape.ofDimensions(new StaticDimension(2),batch),
                    Shape.ofDimensions(new StaticDimension(1),batch,new StaticDimension(2)))) {
                assertTrue(provider.supports(directReshapeQuery(type,time,source,target)),
                        "all-valid reshape has no length-only ROW locality rule");
            }
            var unbound = directReshapeQuery(type,time,Shape.of(0),
                    Shape.ofDimensions(batch,new StaticDimension(0)));
            assertFalse(provider.supports(unbound),"a zero-sized result cannot bind its own batch coordinate");
            assertThrows(IllegalArgumentException.class,() -> CpuGeneralScanCompositionLowering.analyze(unbound));
        }
    }

    @Test void lengthsOnlyBatchExecutesGenuineSharedAccumulatorWithoutEmissions() {
        var batch = new DynamicDimension("Customers");
        for (DataType type : List.of(DataType.FLOAT32,DataType.FLOAT64))
            for (int time : new int[] {0,3}) for (boolean reverse : List.of(false,true)) {
                Tensor h = tensor(type,false,2), x = tensor(type,false,2);
                var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(h.add(x)),List.of());
                Tensor initial = tensor(type,false,2), sequence = tensor(type,false,time,2);
                Tensor lengths = TensorFactory.create(queryDescriptor(DataType.INT64, Shape.ofDimensions(batch)));
                var scan = Scan.apply(List.of(initial),List.of(sequence),List.of(),lengths,0,
                        List.of(OptionalInt.empty()),body,reverse);
                var artifact = compile(scan.finalCarry(),Optional.empty(),false,new CpuCapabilityProvider());
                for (var query : queries(artifact)) {
                    assertTrue(new CpuCapabilityProvider().supports(query));
                    assertDoesNotThrow(() -> CpuGeneralScanCompositionLowering.analyze(query));
                }
                double[] initialValues = {2.5,-3.5}, values = new double[time*2];
                for (int t = 0; t < time; t++) { values[t*2] = (t+1)*.5; values[t*2+1] = -(t+2)*.25; }
                for (int size : new int[] {0,1,3}) try (var consumer = new Consumer(artifact,OptionalLong.of(size))) {
                    long[] controls = new long[size]; long maximum = 0;
                    for (int row = 0; row < size; row++) { controls[row] = Math.min(time,row+1); maximum = Math.max(maximum,controls[row]); }
                    double[] expected = initialValues.clone();
                    for (int step = 0; step < time; step++) {
                        int t = reverse ? time-1-step : step;
                        if (t < maximum) { expected[0] += values[t*2]; expected[1] += values[t*2+1]; }
                    }
                    consumer.numbers(initial,initialValues); consumer.numbers(sequence,values); consumer.raw(lengths,controls);
                    consumer.bind(); consumer.execute();
                    assertArrayEquals(expected,consumer.forward(0),0d,"SHARED state updates once per active time");
                }
            }
    }

    /**
     * Supplies a direct compiled declaration to exercise capability independently of Prepare.
     * @param type represented floating storage type
     * @param time fixed time extent, including zero
     * @param source local sequence-step shape
     * @param target declared ordinary reshape result and attribute
     * @return complete scoped occurrence with a demanded emission and canonical produced residual
     */
    private static OperationCapabilityQuery directReshapeQuery(DataType type,long time,Shape source,Shape target) {
        var input = queryDescriptor(type,source); var output = queryDescriptor(type,target);
        var reshape = new CompiledNode(new NodeId(0),new Operation(
                ShapeTransformKind.RESHAPE,new TargetShapeAttrs(target)),List.of(new ValueId(0)),List.of(new ValueId(1)));
        var graph = new CompiledGraphModel(List.of(new GraphValue(new ValueId(0),input),
                new GraphValue(new ValueId(1),output)),List.of(reshape),List.of(new ValueId(0)),
                List.of(new ValueId(1)),Map.of(reshape.id(),GraphPhase.FORWARD));
        var body = new CompiledScanBody(graph,List.of(),List.of(new ValueId(1)),List.of(new ValueId(1)),List.of(0));
        var attrs = new ScanAttrs(time,0,1,0,1,1,false);
        var result = stackedQueryDescriptor(type,time,target);
        return new OperationCapabilityQuery(new Operation(ScanKind.SCAN,attrs),
                List.of(stackedQueryDescriptor(type,time,source)),List.of(result,result),Optional.of(body),Optional.empty());
    }

    /**
     * Creates canonical static or unresolved named fixture storage without a Tensor/ID surrogate.
     * @param type declared storage type
     * @param shape declared shape
     * @return exact descriptor for that fixture value
     */
    private static TensorDescriptor queryDescriptor(DataType type,Shape shape) {
        return new TensorDescriptor(type,shape,shape.isFullyStatic()
                ? Optional.of(LayoutDescriptor.contiguous(shape)) : Optional.empty(),false);
    }

    /**
     * Adds the actual fixed temporal axis without binding a named extent.
     * @param type declared storage type
     * @param time actual fixed time extent
     * @param step local step shape
     * @return canonical or unresolved stacked actual descriptor
     */
    private static TensorDescriptor stackedQueryDescriptor(DataType type,long time,Shape step) {
        var dimensions = new ArrayList<Dimension>();
        dimensions.add(new StaticDimension(time)); dimensions.addAll(step.dimensions());
        return queryDescriptor(type,Shape.ofDimensions(dimensions.toArray(Dimension[]::new)));
    }

    @Test void lengthModeQualifiesThroughSeparateMixedTypedStagedPath() {
        for (var type : List.of(io.github.pho001.synaptik.model.datatype.DataType.FLOAT32,
                io.github.pho001.synaptik.model.datatype.DataType.FLOAT64)) {
            for (boolean shaped : List.of(false, true)) for (long time : List.of(0L, 3L)) {
                Tensor h = tensor(type,false,2,2), x = tensor(type,false,2,2);
                Tensor w = tensor(type,false, shaped ? new long[] {2} : new long[] {2,2});
                Tensor next = (shaped ? h.add(x.mul(w)) : h.add(x)).tanh();
                var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(h));
                Tensor initial = tensor(type,false,2,2), seq = tensor(type,false,time,2,2);
                Tensor weight = tensor(type,false,w.descriptor().shape().toLongArray());
                var old = Scan.apply(List.of(initial),List.of(seq),List.of(weight),body,false);
                var q = query(old.finalCarry());
                assertTrue(new CpuCapabilityProvider().supports(q));
                assertEquals(shaped, CpuGeneralScanLowering.analyze(q).ir().stages().isPresent());
                var length = Scan.apply(List.of(initial),List.of(seq),List.of(weight),
                        tensor(io.github.pho001.synaptik.model.datatype.DataType.INT64,false,2),1,List.of(OptionalInt.of(0)),body,false);
                var p = length.finalCarry().getFirst().provenance().orElseThrow().producer();
                var rejected = new OperationCapabilityQuery(p.operation(), p.inputs().stream().map(Tensor::descriptor).toList(), p.outputDescriptors(), q.scanBody(), q.scanActivityPlan());
                assertTrue(new CpuCapabilityProvider().supports(rejected));
                assertTrue(CpuGeneralScanLowering.analyze(rejected).ir().lengthAware());
            }
        }
    }
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=io.github.pho001.synaptik.model.datatype.DataType.class,names={"FLOAT32","FLOAT64"})
    void realConv1dDenseViewReachesForwardAndActualReversePrimalReads(io.github.pho001.synaptik.model.datatype.DataType type) {
        Tensor input = tensor(type,true,3,2,5), kernel = tensor(type,true,2,2,2), prefixBias = tensor(type,true,2);
        Tensor sequence = input.conv1d(kernel,prefixBias,io.github.pho001.synaptik.model.operation.convolution.Conv1dAttrs.defaults());
        Tensor h = tensor(type,false,2,3), x = tensor(type,true,2,4), w = tensor(type,true,3,4), u = tensor(type,true,3,3), b = tensor(type,true,3);
        Tensor next = x.linear(w,b).add(h.linear(u)).tanh();
        var body = new ScanBody(List.of(h),List.of(x),List.of(w,u,b),List.of(next),List.of(next));
        Tensor weight = tensor(type,true,3,4), recurrent = tensor(type,true,3,3), bias = tensor(type,true,3);
        var scan = Scan.apply(List.of(tensor(type,false,2,3)),List.of(sequence),List.of(weight,recurrent,bias),body,false);
        var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
        var artifact = compile(roots,Optional.of(request(roots,List.of(weight,recurrent,bias,kernel,prefixBias,input))),true,new CpuCapabilityProvider());
        for (var q : queries(artifact)) {
            assertTrue(new CpuCapabilityProvider().supports(withActualReadViews(q,false)));
            assertTrue(new CpuCapabilityProvider().supports(withActualReadViews(q,true)));
        }
        assertPreparedDenseReadEquivalence(artifact);
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=io.github.pho001.synaptik.model.datatype.DataType.class,names={"FLOAT32","FLOAT64"})
    void actualDenseReadRolesQualifyIndependentlyAcrossFamiliesAndGeneratedReverse(io.github.pho001.synaptik.model.datatype.DataType type) {
        var cpu = new CpuCapabilityProvider();
        for (int family = 0; family < 3; family++) {
            var q = denseReadFixture(type,family,3,false); var views = withActualReadViews(q,true);
            for (int p = 0; p < q.inputs().size(); p++) {
                var inputs = new ArrayList<>(q.inputs()); inputs.set(p,views.inputs().get(p));
                assertTrue(cpu.supports(new OperationCapabilityQuery(q.operation(), inputs, q.outputs(), q.scanBody(), q.scanActivityPlan())), "actual role="+p+" family="+family);
            }
            var mixed = new ArrayList<>(views.inputs()); var d = mixed.getFirst();
            var other = type == io.github.pho001.synaptik.model.datatype.DataType.FLOAT32
                    ? io.github.pho001.synaptik.model.datatype.DataType.FLOAT64 : io.github.pho001.synaptik.model.datatype.DataType.FLOAT32;
            mixed.set(0,new TensorDescriptor(other,d.shape(),d.layout(),d.requiresGrad()));
            assertFalse(cpu.supports(new OperationCapabilityQuery(q.operation(), mixed, q.outputs(), q.scanBody(), q.scanActivityPlan())),"view flag cannot grant mixed-type execution");
        }
        for (int family = 0; family < 3; family++) {
            var artifact = denseReadAutomaticArtifacts(type,family);
            var qs = queries(artifact);
            for (var q : qs) {
                var dense = withActualReadViews(q,false); var views = withActualReadViews(q,true);
                assertTrue(cpu.supports(dense));
                assertTrue(cpu.supports(views));
            }
            assertPreparedDenseReadEquivalence(artifact);
        }
    }

    /**
     * Captures real first-order siblings, including a singleton-CONCAT-produced reverse seed.
     * @param type homogeneous represented dtype
     * @param family zero for pointwise, one for shaped/SUM, two for MATMUL
     * @return full containing artifact retaining the actual siblings and their scoped activity
     */
    private static CompileArtifacts denseReadAutomaticArtifacts(io.github.pho001.synaptik.model.datatype.DataType type, int family) {
        Tensor h = tensor(type,false,2,3), x = tensor(type,true,2,family == 2 ? 2 : 3);
        Tensor w = family == 0 ? tensor(type,true,2,3) : family == 1 ? tensor(type,true,3) : tensor(type,true,2,3);
        Tensor next = h.add(family == 2 ? x.matmul(w) : x.mul(w)).tanh();
        var body = new ScanBody(List.of(h),List.of(x),List.of(w),List.of(next),List.of(family == 1 ? next.sum(0,false) : next));
        Tensor seq = tensor(type,true,3,2,family == 2 ? 2 : 3), weight = tensor(type,true,w.descriptor().shape().toLongArray());
        var scan = Scan.apply(List.of(tensor(type,false,2,3)),List.of(seq),List.of(weight),body,false);
        var roots = List.of(Tensor.concat(0,scan.outputs().getFirst()));
        return compile(roots,Optional.of(request(roots,List.of(seq,weight))),true,new CpuCapabilityProvider());
    }
    @Test void float64RejectsMismatchedActualTypesAtEveryTimeAndExecutesLargeTime() {
        var type = DataType.FLOAT64;
        for (int t : List.of(0,3,100_003)) {
            Tensor h = tensor(type,true,3), x = tensor(type,true,3), next = h.add(x).tanh();
            var body = new ScanBody(List.of(h),List.of(x),List.of(),List.of(next),List.of(next));
            Tensor initial = tensor(type,true,3), seq = tensor(type,true,t,3);
            var scan = Scan.apply(List.of(initial),List.of(seq),List.of(),body,false);
            var roots = List.of(scan.finalCarry().getFirst(),scan.outputs().getFirst());
            Tensor finalSeed = tensor(type,false,3), emissionSeed = tensor(type,false,t,3);
            var stage = new FunctionalGradientRequest.Stage(roots.stream()
                    .map(root -> (FunctionalGradientRequest.OutputReference)new FunctionalGradientRequest.ForwardTensorReference(root)).toList(),
                    List.of(Optional.of(finalSeed),Optional.of(emissionSeed)),List.of(initial,seq),false,
                    t == 0 ? FunctionalGradientRequest.DisconnectedPolicy.ZERO : FunctionalGradientRequest.DisconnectedPolicy.ERROR);
            var artifact = compile(roots,Optional.of(new FunctionalGradientRequest(List.of(stage))),false,new CpuCapabilityProvider());
            for (var q : queries(artifact)) {
                var inputs = new ArrayList<>(q.inputs()); var d = inputs.getFirst();
                inputs.set(0,new TensorDescriptor(DataType.FLOAT32,d.shape(),d.layout(),d.requiresGrad()));
                assertFalse(new CpuCapabilityProvider().supports(new OperationCapabilityQuery(q.operation(), inputs, q.outputs(), q.scanBody(), q.scanActivityPlan())),"mixed dtype rejects at T="+t);
            }
            double[] initialValues = {.11,-.23,.31}, finalSeeds = {.3,-.7,.2};
            double[] xs = new double[3*t], seeds = new double[3*t], history = new double[3*t], state = initialValues.clone();
            for (int i = 0; i < xs.length; i++) { xs[i] = ((i*7)%11-5)*.025; seeds[i] = (i%5-2)*.01; }
            for (int time = 0; time < t; time++) for (int feature = 0; feature < 3; feature++) {
                int cell = 3*time+feature;
                state[feature] = Math.tanh(state[feature]+xs[cell]); history[cell] = state[feature];
            }
            double[] dh = finalSeeds.clone(), dx = new double[3*t];
            for (int time = t-1; time >= 0; time--) for (int feature = 0; feature < 3; feature++) {
                int cell = 3*time+feature;
                double y = history[cell];
                dh[feature] = (dh[feature]+seeds[cell])*(1-y*y); dx[cell] = dh[feature];
            }
            try (var consumer = new Consumer(artifact,OptionalLong.empty())) {
                consumer.numbers(initial,initialValues); consumer.numbers(seq,xs);
                consumer.numbers(finalSeed,finalSeeds); consumer.numbers(emissionSeed,seeds);
                consumer.bind(); consumer.execute();
                assertArrayEquals(state,consumer.forward(0),1e-12);
                assertArrayEquals(history,consumer.forward(1),1e-12);
                assertArrayEquals(dh,consumer.gradient(0),1e-12);
                assertArrayEquals(dx,consumer.gradient(1),1e-12);
            }
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.EnumSource(value=io.github.pho001.synaptik.model.datatype.DataType.class, names={"FLOAT32","FLOAT64"})
    void actualAutomaticReverseQualifiesWithMissingAndJointSeedClasses(io.github.pho001.synaptik.model.datatype.DataType type) {
        for (boolean reverse : List.of(false, true)) for (boolean optimized : List.of(false, true)) {
            Tensor h = tensor(type,false, 2), x = tensor(type,true, 2), w = tensor(type,true, 2);
            Tensor next = h.mul(w).add(x).tanh();
            var body = new ScanBody(List.of(h), List.of(x), List.of(w), List.of(next), List.of(h.sigmoid()));
            Tensor seq = tensor(type,true, 3, 2), weight = tensor(type,true, 2);
            var scan = Scan.apply(List.of(tensor(type,false, 2)), List.of(seq), List.of(weight), body, reverse);
            for (var roots : List.of(scan.finalCarry(), scan.outputs(),
                    List.of(scan.finalCarry().getFirst(), scan.outputs().getFirst()))) {
                var request = request(roots, List.of(seq, weight));
                var artifact = compile(roots, Optional.of(request), optimized, new CpuCapabilityProvider());
                for (var q : queries(artifact))
                    assertTrue(new CpuCapabilityProvider().supports(q));
                assertPreparedDenseReadEquivalence(artifact);
            }
        }
    }


    /**
     * Executes independently prepared descriptor twins with identical live numerical inputs.
     * Constants, scoped activity, assignment, helper inventory and binding remain production-owned.
     * @param artifact actual compiled floating forward/AD fixture
     */
    private static void assertPreparedDenseReadEquivalence(CompileArtifacts artifact) {
        var preparer = new CpuPartitionPreparer();
        try (var dense = new Consumer(artifact,OptionalLong.empty(),Map.of(),CpuPartitionAnalysisInputs.DEFAULT,
                    preparer.analyze(readViewContext(artifact,false)));
                var views = new Consumer(artifact,OptionalLong.empty(),Map.of(),CpuPartitionAnalysisInputs.DEFAULT,
                    preparer.analyze(readViewContext(artifact,true)))) {
            var descriptors = new HashMap<ValueId,TensorDescriptor>();
            artifact.graph().values().forEach(value -> descriptors.put(value.id(),value.descriptor()));
            for (int pass = 0; pass < 2; pass++) {
                int input = 0;
                for (var binding : artifact.constants().bindableInputBindings()) {
                    var type = descriptors.get(binding.valueId()).dataType();
                    var left = dense.valueStorage(binding.valueId());
                    var right = views.valueStorage(binding.valueId());
                    for (long cell = 0; cell < left.byteSize()/type.byteWidth(); cell++) {
                        double value = (input+1)*.013+(cell%7-3)*.007+pass*.003;
                        if (type == DataType.FLOAT64) {
                            left.setAtIndex(ValueLayout.JAVA_DOUBLE,cell,value);
                            right.setAtIndex(ValueLayout.JAVA_DOUBLE,cell,value);
                        } else {
                            left.setAtIndex(ValueLayout.JAVA_FLOAT,cell,(float)value);
                            right.setAtIndex(ValueLayout.JAVA_FLOAT,cell,(float)value);
                        }
                    }
                    input++;
                }
                if (pass == 0) { dense.bind(); views.bind(); }
                dense.execute(); views.execute();
                for (int port = 0; port < artifact.publication().forwardBindings().size(); port++)
                    assertArrayEquals(dense.forward(port),views.forward(port),0d,"forward read equivalence");
                for (int port = 0; port < artifact.publication().gradientBindings().size(); port++)
                    assertArrayEquals(dense.gradient(port),views.gradient(port),0d,"gradient read equivalence");
            }
        }
    }

    /**
     * Changes only producerless dense actual-read flags, retaining exact addresses and activity.
     * Produced values keep their Compiler-inferred layouts: changing an affine result's view
     * flag independently would invalidate its producing operation. Queries cover every read role.
     * @param artifact actual containing artifact with canonical static fixture geometry
     * @param view desired dense-read marker
     * @return complete independently validated projection for real backend analysis
     */
    private static PrepareContext<CpuPartitionAnalysisInputs> readViewContext(CompileArtifacts artifact,boolean view) {
        var base = GraphPreparation.project(artifact,artifact.partitions().getFirst(),CpuPartitionAnalysisInputs.DEFAULT);
        var reads = new HashSet<ValueId>(); var writes = new HashSet<ValueId>();
        for (var node : artifact.graph().nodes()) {
            writes.addAll(node.outputs());
            if (node.scanBody().isPresent()) reads.addAll(node.inputs());
        }
        var descriptors = new LinkedHashMap<ValueId,TensorDescriptor>();
        for (var value : base.values()) {
            var descriptor = value.descriptor();
            if (reads.contains(value.id()) && !writes.contains(value.id()) && descriptor.shape().isFullyStatic()
                    && descriptor.layout().isPresent()) {
                var layout = descriptor.layout().orElseThrow();
                if (layout.storageOffset() == 0 && Arrays.equals(layout.strides(),LayoutDescriptor.contiguous(descriptor.shape()).strides()))
                    descriptor = new TensorDescriptor(descriptor.dataType(),descriptor.shape(),
                            Optional.of(LayoutDescriptor.of(descriptor.shape(),layout.strides(),0,view)),descriptor.requiresGrad());
            }
            descriptors.put(value.id(),descriptor);
        }
        var values = base.values().stream().map(value -> new GraphValue(value.id(),descriptors.get(value.id()))).toList();
        var requirements = base.memoryRequirements().stream().map(requirement -> new LogicalMemoryRequirement(
                requirement.valueId(),descriptors.get(requirement.valueId()),requirement.producerPartition(),
                requirement.consumerPartitions(),requirement.graphOutput())).toList();
        return new PrepareContext<>(base.partitionDag(),values,requirements,base.constants(),base.backendInputs(),base.scanActivityPlans());
    }

    public static FunctionalGradientRequest request(List<Tensor> roots, List<Tensor> targets) {
        return new FunctionalGradientRequest(List.of(new FunctionalGradientRequest.Stage(roots.stream()
                .map(t -> (FunctionalGradientRequest.OutputReference) new FunctionalGradientRequest.ForwardTensorReference(t)).toList(),
                roots.stream().map(t -> Optional.of(tensor(t.descriptor().dataType(), false, t.descriptor().shape().toLongArray()))).toList(),
                targets, false, FunctionalGradientRequest.DisconnectedPolicy.ZERO)));
    }

    public static CompileArtifacts compile(List<Tensor> roots, Optional<FunctionalGradientRequest> request,
            boolean optimized, BackendCapabilityProvider provider) {
        var backend = provider.backendId();
        var available = new BackendAvailabilitySnapshot(backend, Map.of(new BackendDeviceId(backend, "0"), DeviceClass.CPU));
        return GraphCompilationPort.compile(request.isPresent() ? CompileMode.FORWARD_AND_BACKWARD : CompileMode.FORWARD_ONLY,
                roots, request, optimized ? GraphOptimizationConfig.standard() : GraphOptimizationConfig.disabled(),
                BackendIntent.unconstrained(), PartitionScoringConfig.neutral(), List.of(provider), List.of(available));
    }
}
