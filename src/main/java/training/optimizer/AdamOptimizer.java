package training.optimizer;

import backend.cpu.kernels.CpuDTypeOps;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.buffer.MetalBufferAccess;
import backend.metal.buffer.MetalBufferAllocator;
import backend.metal.buffer.MetalBufferBinding;
import graph.CompiledNode;
import tensor.DataType;
import tensor.Tensor;

import java.util.Collection;
import java.util.IdentityHashMap;
import java.util.List;

/**
 * Adam optimizer for trainable parameters.
 */
public final class AdamOptimizer extends AbstractTrainableOptimizer {
    private final float learningRate;
    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private int step;
    private final IdentityHashMap<Tensor, CpuAdamState> cpuStates = new IdentityHashMap<>();
    private final IdentityHashMap<Tensor, MetalAdamState> metalStates = new IdentityHashMap<>();

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
        requireCpuReadable(context, ref.parameterNode().id());
        requireCpuReadable(context, gradientNodeId);
        Tensor parameter = context.executionContext().runtimeTensorForNodeId(ref.parameterNode().id());
        Tensor gradient = context.executionContext().runtimeTensorForNodeId(gradientNodeId);
        updateCpu(parameter, gradient, ref.parameterNode());
        parameter.markStorageModified();
        ref.parameterNode().sourceTensor().markStorageModified();
        context.executionContext().markCpuCurrent(ref.parameterNode().id(), "optimizer CPU Adam update");
    }

    @Override
    public void close() {
        for (MetalAdamState state : List.copyOf(metalStates.values())) {
            state.allocator().destroy(state.firstMoment().handle());
            state.allocator().destroy(state.secondMoment().handle());
        }
        metalStates.clear();
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
}
