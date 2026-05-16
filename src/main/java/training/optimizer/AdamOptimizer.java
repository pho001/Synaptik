package training.optimizer;

import backend.cpu.kernels.CpuDTypeOps;
import backend.cpu.nativecpu.NativeCpuAllocator;
import backend.cpu.nativecpu.NativeCpuMaterializer;
import backend.cpu.nativecpu.NativeCpuStorageFactory;
import backend.memory.CpuMaterializationReason;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferAllocator;
import backend.metal.buffer.MetalBufferBinding;
import config.runtime.CpuStorageProfile;
import graph.CompiledNode;
import tensor.DataType;
import tensor.NativeFloat32Storage;
import tensor.NativeTensorStorage;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.Collection;
import java.util.IdentityHashMap;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

/**
 * Adam optimizer for trainable parameters.
 */
public final class AdamOptimizer extends AbstractTrainableOptimizer {
    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private int step;
    private final NativeCpuStorageFactory nativeStorageFactory;
    private final IdentityHashMap<Tensor, CpuAdamState> cpuStates = new IdentityHashMap<>();
    private final IdentityHashMap<Tensor, MetalAdamState> metalStates = new IdentityHashMap<>();
    private final IdentityHashMap<Tensor, NativeAdamState> nativeStates = new IdentityHashMap<>();

    public AdamOptimizer(float learningRate) {
        this(null, learningRate, 0.9f, 0.999f, 1.0e-8f);
    }

    public AdamOptimizer(Collection<Tensor> parameters, float learningRate) {
        this(parameters, learningRate, 0.9f, 0.999f, 1.0e-8f);
    }

    public AdamOptimizer(Collection<Tensor> parameters, float learningRate, float beta1, float beta2, float epsilon) {
        super(parameters);
        if (!(learningRate > 0.0f) || !Float.isFinite(learningRate)) {
            throw new IllegalArgumentException("learningRate must be a finite positive value.");
        }
        if (!(beta1 >= 0.0f && beta1 < 1.0f) || !(beta2 >= 0.0f && beta2 < 1.0f)) {
            throw new IllegalArgumentException("Adam beta values must be in [0, 1).");
        }
        if (!(epsilon > 0.0f) || !Float.isFinite(epsilon)) {
            throw new IllegalArgumentException("epsilon must be a finite positive value.");
        }
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.nativeStorageFactory = new NativeCpuStorageFactory(new NativeCpuAllocator());
    }

    @Override
    public void beforeExecute(OptimizerStepContext context) {
        super.beforeExecute(context);
        if (context.runtimeConfig().cpuStorageProfile() != CpuStorageProfile.CPU_NATIVE) {
            return;
        }
        for (TrainableParameterRef ref : selectedParameters(context)) {
            if (ref.parameterNode().dataType() != DataType.FLOAT32) {
                continue;
            }
            NativeAdamState state = nativeStateFor(ref.parameterNode());
            context.executionContext().attachNativeStorage(
                    ref.parameterNode().id(),
                    state.parameterView(),
                    "optimizer-owned native F32 Adam parameter"
            );
        }
    }

    @Override
    protected void beforeStep(OptimizerStepContext context) {
        step++;
    }

    @Override
    protected boolean metalStep(
            OptimizerStepContext context,
            TrainableParameterRef ref,
            MetalMpsBridgeContext bridgeContext,
            MetalBufferAllocator allocator,
            MetalBufferBinding parameter,
            MetalBufferBinding gradient,
            MetalBufferBinding output
    ) {
        MetalOptimizerBridge bridge = MetalOptimizerBridge.get();
        if (!bridge.available()) {
            return false;
        }
        Tensor source = ref.parameterNode().sourceTensor();
        MetalAdamState state = metalStates.get(source);
        if (state == null) {
            state = createMetalState(allocator, ref.parameterNode());
            metalStates.put(source, state);
        }
        bridge.adamF32(
                bridgeContext,
                parameter,
                gradient,
                state.firstMoment(),
                state.secondMoment(),
                output,
                learningRate,
                beta1,
                beta2,
                epsilon,
                step
        );
        return true;
    }

    @Override
    protected void cpuStep(OptimizerStepContext context, TrainableParameterRef ref, int gradientNodeId) {
        syncMetalStateToCpu(ref);
        syncNativeStateToCpu(ref);
        requireCpuReadable(context, ref.parameterNode().id());
        requireCpuReadable(context, gradientNodeId);
        Tensor parameter = context.executionContext().runtimeTensorForNodeId(ref.parameterNode().id());
        Tensor gradient = context.executionContext().runtimeTensorForNodeId(gradientNodeId);
        updateCpu(parameter, gradient, ref.parameterNode());
        parameter.markStorageModified();
        ref.parameterNode().sourceTensor().markStorageModified();
        context.executionContext().markCpuCurrent(ref.parameterNode().id(), "optimizer CPU Adam update");
        clearNativeState(ref.parameterNode().sourceTensor());
    }

    @Override
    protected boolean nativeCpuStep(OptimizerStepContext context, TrainableParameterRef ref, int gradientNodeId) {
        if (!nativeEligible(context, ref, gradientNodeId)) {
            return false;
        }
        try {
            NativeTensorStorage parameter = context.executionContext().nativeStorageForNodeId(ref.parameterNode().id());
            NativeTensorStorage gradient = context.executionContext().requireNativeReadable(
                    gradientNodeId,
                    CpuMaterializationReason.OPTIMIZER_STEP
            );
            NativeAdamState state = nativeStates.get(ref.parameterNode().sourceTensor());
            if (!(parameter instanceof NativeFloat32Storage parameterF32)
                    || !(gradient instanceof NativeFloat32Storage gradientF32)
                    || state == null
                    || parameterF32.getSize() != gradientF32.getSize()
                    || state.firstMoment().getSize() != parameterF32.getSize()
                    || state.secondMoment().getSize() != parameterF32.getSize()) {
                return false;
            }
            updateNativeF32(parameterF32, gradientF32, state.firstMoment(), state.secondMoment());
            parameterF32.markModified();
            state.firstMoment().markModified();
            state.secondMoment().markModified();
            context.executionContext().attachNativeStorage(
                    ref.parameterNode().id(),
                    parameterF32,
                    "optimizer native CPU Adam update"
            );
            cpuStates.remove(ref.parameterNode().sourceTensor());
            return true;
        } catch (RuntimeException ex) {
            return false;
        }
    }

    @Override
    protected String nativeCpuFallbackReason(OptimizerStepContext context, TrainableParameterRef ref, int gradientNodeId) {
        if (context.runtimeConfig().cpuStorageProfile() != CpuStorageProfile.CPU_NATIVE) {
            return "native-adam-ineligible:cpu-storage-profile-" + context.runtimeConfig().cpuStorageProfile().name();
        }
        if (ref.parameterNode().dataType() != DataType.FLOAT32) {
            return "native-adam-ineligible:dtype-" + ref.parameterNode().dataType().name();
        }
        Tensor gradient = context.executionContext().runtimeTensorForNodeId(gradientNodeId);
        if (gradient.getDataType() != DataType.FLOAT32) {
            return "native-adam-ineligible:gradient-dtype-" + gradient.getDataType().name();
        }
        if (!java.util.Arrays.equals(ref.parameterNode().shape(), gradient.getShapeUnsafe())) {
            return "native-adam-ineligible:shape";
        }
        if (context.executionContext().nativeStorageForNodeId(ref.parameterNode().id()) == null) {
            return "native-adam-ineligible:parameter-storage";
        }
        return "native-adam-ineligible:storage";
    }

    @Override
    protected String optimizerStateStorage(String route) {
        return switch (route) {
            case "CPU_NATIVE" -> "CPU_NATIVE";
            case "GPU_METAL" -> "DEVICE_OWNED";
            case "CPU_ARRAY" -> "CPU_ARRAY";
            default -> "NONE";
        };
    }

    @Override
    public void syncParametersToCpu() {
        super.syncParametersToCpu();
        for (NativeAdamState state : List.copyOf(nativeStates.values())) {
            NativeCpuMaterializer.nativeToArray(state.parameter(), state.source());
        }
    }

    @Override
    public void close() {
        for (MetalAdamState state : List.copyOf(metalStates.values())) {
            state.allocator().destroy(state.firstMoment().handle());
            state.allocator().destroy(state.secondMoment().handle());
        }
        metalStates.clear();
        for (NativeAdamState state : List.copyOf(nativeStates.values())) {
            state.close();
        }
        nativeStates.clear();
        super.close();
    }

    private MetalAdamState createMetalState(MetalBufferAllocator allocator, CompiledNode node) {
        Tensor zeros = new Tensor(node.shape(), null, "adam_state", DataType.FLOAT32);
        MetalBufferBinding m = allocator.createInputBinding(node.id(), zeros);
        MetalBufferBinding v = allocator.createInputBinding(node.id(), zeros);
        return new MetalAdamState(
                allocator,
                bindingForNode(node.id(), m, MetalBufferAccess.READ_WRITE),
                bindingForNode(node.id(), v, MetalBufferAccess.READ_WRITE)
        );
    }

    private void syncMetalStateToCpu(TrainableParameterRef ref) {
        Tensor source = ref.parameterNode().sourceTensor();
        MetalAdamState metal = metalStates.remove(source);
        if (metal == null) {
            return;
        }
        Tensor m = new Tensor(ref.parameterNode().shape(), null, "adam_m_sync", DataType.FLOAT32);
        Tensor v = new Tensor(ref.parameterNode().shape(), null, "adam_v_sync", DataType.FLOAT32);
        metal.allocator().readToCpu(metal.firstMoment(), m, backend.memory.CpuMaterializationReason.OPTIMIZER_STEP);
        metal.allocator().readToCpu(metal.secondMoment(), v, backend.memory.CpuMaterializationReason.OPTIMIZER_STEP);
        cpuStates.put(source, CpuAdamState.fromFloat32(m.getFloat32Data(), v.getFloat32Data()));
        metal.allocator().destroy(metal.firstMoment().handle());
        metal.allocator().destroy(metal.secondMoment().handle());
    }

    private void syncNativeStateToCpu(TrainableParameterRef ref) {
        Tensor source = ref.parameterNode().sourceTensor();
        NativeAdamState state = nativeStates.get(source);
        if (state == null) {
            return;
        }
        cpuStates.put(source, CpuAdamState.fromFloat32(
                nativeToArray(state.firstMoment()),
                nativeToArray(state.secondMoment())
        ));
    }

    private void updateCpu(Tensor parameter, Tensor gradient, CompiledNode node) {
        CpuAdamState state = cpuStates.computeIfAbsent(node.sourceTensor(), ignored -> CpuAdamState.zeros(parameter.getFlatDataSize()));
        switch (parameter.getDataType()) {
            case FLOAT32 -> {
                float[] p = parameter.getFloat32Data();
                float[] g = gradient.getFloat32Data();
                updateFloatArray(p, g, state);
            }
            case FLOAT64 -> {
                double[] p = parameter.getFloat64Data();
                double[] g = gradient.getFloat64Data();
                updateDoubleArray(p, g, state);
            }
            case BFLOAT16 -> {
                short[] p = parameter.getBFloat16Data();
                short[] g = gradient.getBFloat16Data();
                updateBFloat16Array(p, g, state);
            }
            case INT32, BOOL -> throw new UnsupportedOperationException(
                    "Adam supports floating trainable parameters only; got " + parameter.getDataType()
            );
        }
    }

    private void updateFloatArray(float[] p, float[] g, CpuAdamState state) {
        float bias1 = 1.0f - (float) Math.pow(beta1, step);
        float bias2 = 1.0f - (float) Math.pow(beta2, step);
        for (int i = 0; i < p.length; i++) {
            p[i] = adamValue(p[i], g[i], state, i, bias1, bias2);
        }
    }

    private void updateDoubleArray(double[] p, double[] g, CpuAdamState state) {
        float bias1 = 1.0f - (float) Math.pow(beta1, step);
        float bias2 = 1.0f - (float) Math.pow(beta2, step);
        for (int i = 0; i < p.length; i++) {
            p[i] = adamValue((float) p[i], (float) g[i], state, i, bias1, bias2);
        }
    }

    private void updateBFloat16Array(short[] p, short[] g, CpuAdamState state) {
        float bias1 = 1.0f - (float) Math.pow(beta1, step);
        float bias2 = 1.0f - (float) Math.pow(beta2, step);
        for (int i = 0; i < p.length; i++) {
            float updated = adamValue(
                    CpuDTypeOps.fromBFloat16Bits(p[i]),
                    CpuDTypeOps.fromBFloat16Bits(g[i]),
                    state,
                    i,
                    bias1,
                    bias2
            );
            p[i] = CpuDTypeOps.toBFloat16Bits(updated);
        }
    }

    private float adamValue(float parameter, float gradient, CpuAdamState state, int index, float bias1, float bias2) {
        float m = beta1 * state.firstMoment[index] + (1.0f - beta1) * gradient;
        float v = beta2 * state.secondMoment[index] + (1.0f - beta2) * gradient * gradient;
        state.firstMoment[index] = m;
        state.secondMoment[index] = v;
        float mHat = m / bias1;
        float vHat = v / bias2;
        return parameter - learningRate * mHat / ((float) Math.sqrt(vHat) + epsilon);
    }

    private boolean nativeEligible(OptimizerStepContext context, TrainableParameterRef ref, int gradientNodeId) {
        if (context.runtimeConfig().cpuStorageProfile() != CpuStorageProfile.CPU_NATIVE
                || ref.parameterNode().dataType() != DataType.FLOAT32) {
            return false;
        }
        Tensor gradient = context.executionContext().runtimeTensorForNodeId(gradientNodeId);
        return gradient.getDataType() == DataType.FLOAT32
                && java.util.Arrays.equals(ref.parameterNode().shape(), gradient.getShapeUnsafe());
    }

    private NativeAdamState nativeStateFor(CompiledNode node) {
        Tensor source = node.sourceTensor();
        NativeAdamState state = nativeStates.get(source);
        if (state == null || state.closed() || state.parameter().getSize() != source.getFlatDataSize()) {
            if (state != null) {
                state.close();
            }
            NativeFloat32Storage parameter = allocateNativeF32(source.getFlatDataSize(), "optimizer-adam-f32-param:" + source.getLabel());
            NativeFloat32Storage firstMoment = allocateNativeF32(source.getFlatDataSize(), "optimizer-adam-f32-m:" + source.getLabel());
            NativeFloat32Storage secondMoment = allocateNativeF32(source.getFlatDataSize(), "optimizer-adam-f32-v:" + source.getLabel());
            NativeCpuMaterializer.arrayToNative(source, parameter);
            CpuAdamState cpuState = cpuStates.get(source);
            if (cpuState == null || cpuState.firstMoment().length != source.getFlatDataSize()
                    || cpuState.secondMoment().length != source.getFlatDataSize()) {
                firstMoment.segment().fill((byte) 0);
                secondMoment.segment().fill((byte) 0);
            } else {
                arrayToNative(cpuState.firstMoment(), firstMoment);
                arrayToNative(cpuState.secondMoment(), secondMoment);
            }
            state = new NativeAdamState(source, parameter, firstMoment, secondMoment);
            nativeStates.put(source, state);
        }
        return state;
    }

    private NativeFloat32Storage allocateNativeF32(int elements, String label) {
        return (NativeFloat32Storage) nativeStorageFactory.allocate(DataType.FLOAT32, elements, label);
    }

    private void updateNativeF32(
            NativeFloat32Storage parameter,
            NativeFloat32Storage gradient,
            NativeFloat32Storage firstMoment,
            NativeFloat32Storage secondMoment
    ) {
        float bias1 = 1.0f - (float) Math.pow(beta1, step);
        float bias2 = 1.0f - (float) Math.pow(beta2, step);
        MemorySegment p = parameter.segment();
        MemorySegment g = gradient.segment();
        MemorySegment mSegment = firstMoment.segment();
        MemorySegment vSegment = secondMoment.segment();
        for (int i = 0; i < parameter.getSize(); i++) {
            long offset = (long) i * Float.BYTES;
            float gradientValue = g.get(JAVA_FLOAT, offset);
            float m = beta1 * mSegment.get(JAVA_FLOAT, offset) + (1.0f - beta1) * gradientValue;
            float v = beta2 * vSegment.get(JAVA_FLOAT, offset) + (1.0f - beta2) * gradientValue * gradientValue;
            mSegment.set(JAVA_FLOAT, offset, m);
            vSegment.set(JAVA_FLOAT, offset, v);
            float mHat = m / bias1;
            float vHat = v / bias2;
            p.set(
                    JAVA_FLOAT,
                    offset,
                    p.get(JAVA_FLOAT, offset) - learningRate * mHat / ((float) Math.sqrt(vHat) + epsilon)
            );
        }
    }

    private void clearNativeState(Tensor source) {
        NativeAdamState state = nativeStates.remove(source);
        if (state != null) {
            state.close();
        }
    }

    private static void arrayToNative(float[] source, NativeFloat32Storage target) {
        MemorySegment.copy(source, 0, target.segment(), JAVA_FLOAT, 0L, target.getSize());
        target.markModified();
    }

    private static float[] nativeToArray(NativeFloat32Storage source) {
        float[] result = new float[source.getSize()];
        MemorySegment.copy(source.segment(), JAVA_FLOAT, 0L, result, 0, result.length);
        return result;
    }

    private record MetalAdamState(
            MetalBufferAllocator allocator,
            MetalBufferBinding firstMoment,
            MetalBufferBinding secondMoment
    ) {
    }

    private record CpuAdamState(float[] firstMoment, float[] secondMoment) {
        static CpuAdamState zeros(int size) {
            return new CpuAdamState(new float[size], new float[size]);
        }

        static CpuAdamState fromFloat32(float[] firstMoment, float[] secondMoment) {
            return new CpuAdamState(firstMoment.clone(), secondMoment.clone());
        }
    }

    private record NativeAdamState(
            Tensor source,
            NativeFloat32Storage parameter,
            NativeFloat32Storage firstMoment,
            NativeFloat32Storage secondMoment
    ) {
        private boolean closed() {
            return parameter.closed() || firstMoment.closed() || secondMoment.closed();
        }

        private NativeFloat32Storage parameterView() {
            return new NativeFloat32Storage(parameter.getSize(), parameter.allocation(), parameter.byteOffset(), false);
        }

        private void close() {
            parameter.close();
            firstMoment.close();
            secondMoment.close();
        }
    }
}
