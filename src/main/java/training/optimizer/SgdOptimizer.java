package training.optimizer;

import tensor.TensorInternalAccess;

import backend.cpu.nativecpu.NativeCpuAllocator;
import backend.cpu.nativecpu.NativeCpuMaterializer;
import backend.cpu.nativecpu.NativeCpuStorageFactory;
import backend.memory.CpuMaterializationReason;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.buffer.MetalBufferAllocator;
import backend.metal.buffer.MetalBufferBinding;
import config.runtime.BFloat16TrainingPolicy;
import config.runtime.CpuStorageProfile;
import graph.CompiledNode;
import backend.cpu.kernels.CpuDTypeOps;
import tensor.DataType;
import tensor.storage.NativeBFloat16Storage;
import tensor.storage.NativeFloat32Storage;
import tensor.storage.NativeTensorStorage;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.Collection;
import java.util.IdentityHashMap;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

/**
 * Stochastic gradient descent optimizer for trainable parameters.
 */
public final class SgdOptimizer extends AbstractTrainableOptimizer {
    private final float learningRate;
    private final NativeCpuStorageFactory nativeStorageFactory;
    private final IdentityHashMap<Tensor, OwnedNativeParameter> nativeParameters = new IdentityHashMap<>();

    public SgdOptimizer(float learningRate) {
        this(null, learningRate);
    }

    public SgdOptimizer(Collection<Tensor> parameters, float learningRate) {
        super(parameters);
        if (!(learningRate > 0.0f) || !Float.isFinite(learningRate)) {
            throw new IllegalArgumentException("learningRate must be a finite positive value.");
        }
        this.learningRate = learningRate;
        this.nativeStorageFactory = new NativeCpuStorageFactory(new NativeCpuAllocator());
    }

    public float learningRate() {
        return learningRate;
    }

    @Override
    public void beforeExecute(OptimizerStepContext context) {
        super.beforeExecute(context);
        if (context.runtimeConfig().cpuStorageProfile() != CpuStorageProfile.CPU_NATIVE) {
            return;
        }
        for (TrainableParameterRef ref : selectedParameters(context)) {
            if (ref.parameterNode().dataType() != DataType.FLOAT32
                    && !nativeBf16Experimental(context, ref)) {
                continue;
            }
            OwnedNativeParameter owned = nativeParameterFor(ref);
            context.executionContext().attachNativeStorage(
                    ref.parameterNode().id(),
                    owned.view(),
                    "optimizer-owned native SGD parameter"
            );
        }
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
        bridge.sgdF32(bridgeContext, parameter, gradient, output, learningRate);
        return true;
    }

    @Override
    protected void cpuStep(OptimizerStepContext context, TrainableParameterRef ref, int gradientNodeId) {
        requireCpuReadable(context, ref.parameterNode().id());
        requireCpuReadable(context, gradientNodeId);
        Tensor parameter = context.executionContext().runtimeTensorForNodeId(ref.parameterNode().id());
        Tensor gradient = context.executionContext().runtimeTensorForNodeId(gradientNodeId);
        updateCpu(parameter, gradient, learningRate, ref.parameterNode());
        TensorInternalAccess.markStorageModified(parameter);
        TensorInternalAccess.markStorageModified(ref.parameterTensor());
        context.executionContext().markCpuCurrent(ref.parameterNode().id(), "optimizer CPU SGD update");
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
            if (parameter instanceof NativeFloat32Storage parameterF32
                    && gradient instanceof NativeFloat32Storage gradientF32
                    && parameterF32.getSize() == gradientF32.getSize()) {
                updateNativeF32(parameterF32, gradientF32, learningRate);
                parameterF32.markModified();
                context.executionContext().attachNativeStorage(
                        ref.parameterNode().id(),
                        parameterF32,
                        "optimizer native CPU SGD update"
                );
                return true;
            }
            if (nativeBf16Experimental(context, ref)
                    && parameter instanceof NativeBFloat16Storage parameterBF16
                    && gradient instanceof NativeBFloat16Storage gradientBF16
                    && parameterBF16.getSize() == gradientBF16.getSize()) {
                updateNativeBF16(parameterBF16, gradientBF16, learningRate);
                parameterBF16.markModified();
                context.executionContext().attachNativeStorage(
                        ref.parameterNode().id(),
                        parameterBF16,
                        "optimizer native CPU experimental BF16 SGD update"
                );
                return true;
            }
            return false;
        } catch (RuntimeException ex) {
            return false;
        }
    }

    @Override
    protected String nativeCpuFallbackReason(OptimizerStepContext context, TrainableParameterRef ref, int gradientNodeId) {
        if (context.runtimeConfig().cpuStorageProfile() != CpuStorageProfile.CPU_NATIVE) {
            return "native-sgd-ineligible:cpu-storage-profile-" + context.runtimeConfig().cpuStorageProfile().name();
        }
        if (ref.parameterNode().dataType() == DataType.BFLOAT16
                && context.runtimeConfig().bfloat16TrainingPolicy() == BFloat16TrainingPolicy.ACTIVATIONS_ONLY) {
            return "native-sgd-ineligible:bf16-policy-ACTIVATIONS_ONLY";
        }
        if (ref.parameterNode().dataType() == DataType.BFLOAT16
                && context.runtimeConfig().bfloat16TrainingPolicy() == BFloat16TrainingPolicy.PARAMS_WITH_F32_MASTER) {
            return "native-sgd-ineligible:bf16-master-not-implemented";
        }
        if (ref.parameterNode().dataType() != DataType.FLOAT32
                && !nativeBf16Experimental(context, ref)) {
            return "native-sgd-ineligible:dtype-" + ref.parameterNode().dataType().name();
        }
        Tensor gradient = context.executionContext().runtimeTensorForNodeId(gradientNodeId);
        if (gradient.getDataType() != ref.parameterNode().dataType()) {
            return "native-sgd-ineligible:gradient-dtype-" + gradient.getDataType().name();
        }
        if (!java.util.Arrays.equals(ref.parameterNode().shape(), gradient.getShapeUnsafe())) {
            return "native-sgd-ineligible:shape";
        }
        if (context.executionContext().nativeStorageForNodeId(ref.parameterNode().id()) == null) {
            return "native-sgd-ineligible:parameter-storage";
        }
        return "native-sgd-ineligible:storage";
    }

    @Override
    public void syncParametersToCpu() {
        super.syncParametersToCpu();
        for (OwnedNativeParameter owned : List.copyOf(nativeParameters.values())) {
            NativeCpuMaterializer.nativeToArray(owned.storage(), owned.source());
        }
    }

    @Override
    public void close() {
        for (OwnedNativeParameter owned : List.copyOf(nativeParameters.values())) {
            owned.storage().close();
        }
        nativeParameters.clear();
        super.close();
    }

    private static void updateCpu(Tensor parameter, Tensor gradient, float learningRate, CompiledNode node) {
        if (parameter.getDataType() != gradient.getDataType()) {
            throw new IllegalStateException("Parameter and gradient dtype differ for nodeId=" + node.id());
        }
        switch (parameter.getDataType()) {
            case FLOAT32 -> {
                float[] p = TensorInternalAccess.float32Data(parameter);
                float[] g = TensorInternalAccess.float32Data(gradient);
                for (int i = 0; i < p.length; i++) {
                    p[i] -= learningRate * g[i];
                }
            }
            case FLOAT64 -> {
                double[] p = TensorInternalAccess.float64Data(parameter);
                double[] g = TensorInternalAccess.float64Data(gradient);
                for (int i = 0; i < p.length; i++) {
                    p[i] -= (double) learningRate * g[i];
                }
            }
            case BFLOAT16 -> {
                short[] p = TensorInternalAccess.bfloat16Data(parameter);
                short[] g = TensorInternalAccess.bfloat16Data(gradient);
                for (int i = 0; i < p.length; i++) {
                    float updated = CpuDTypeOps.fromBFloat16Bits(p[i])
                            - learningRate * CpuDTypeOps.fromBFloat16Bits(g[i]);
                    p[i] = CpuDTypeOps.toBFloat16Bits(updated);
                }
            }
            case INT32, BOOL -> throw new UnsupportedOperationException(
                    "SGD supports floating trainable parameters only; got " + parameter.getDataType()
            );
        }
    }

    private boolean nativeEligible(OptimizerStepContext context, TrainableParameterRef ref, int gradientNodeId) {
        if (context.runtimeConfig().cpuStorageProfile() != CpuStorageProfile.CPU_NATIVE) {
            return false;
        }
        if (ref.parameterNode().dataType() != DataType.FLOAT32
                && !nativeBf16Experimental(context, ref)) {
            return false;
        }
        Tensor gradient = context.executionContext().runtimeTensorForNodeId(gradientNodeId);
        return gradient.getDataType() == ref.parameterNode().dataType()
                && java.util.Arrays.equals(ref.parameterNode().shape(), gradient.getShapeUnsafe());
    }

    private OwnedNativeParameter nativeParameterFor(TrainableParameterRef ref) {
        Tensor source = ref.parameterTensor();
        OwnedNativeParameter owned = nativeParameters.get(source);
        if (owned == null || owned.storage().closed() || owned.storage().getSize() != source.getFlatDataSize()) {
            if (owned != null) {
                owned.storage().close();
            }
            NativeTensorStorage storage = nativeStorageFactory.allocate(
                    source.getDataType(),
                    source.getFlatDataSize(),
                    "optimizer-sgd-" + source.getDataType().name().toLowerCase(java.util.Locale.ROOT) + ":" + source.getLabel()
            );
            NativeCpuMaterializer.arrayToNative(source, storage);
            owned = new OwnedNativeParameter(source, storage);
            nativeParameters.put(source, owned);
        }
        return owned;
    }

    private static void updateNativeF32(NativeFloat32Storage parameter, NativeFloat32Storage gradient, float learningRate) {
        MemorySegment p = parameter.segment();
        MemorySegment g = gradient.segment();
        for (int i = 0; i < parameter.getSize(); i++) {
            long offset = (long) i * Float.BYTES;
            p.set(JAVA_FLOAT, offset, p.get(JAVA_FLOAT, offset) - learningRate * g.get(JAVA_FLOAT, offset));
        }
    }

    private static void updateNativeBF16(NativeBFloat16Storage parameter, NativeBFloat16Storage gradient, float learningRate) {
        MemorySegment p = parameter.segment();
        MemorySegment g = gradient.segment();
        for (int i = 0; i < parameter.getSize(); i++) {
            long offset = (long) i * Short.BYTES;
            float updated = CpuDTypeOps.fromBFloat16Bits(p.get(JAVA_SHORT, offset))
                    - learningRate * CpuDTypeOps.fromBFloat16Bits(g.get(JAVA_SHORT, offset));
            p.set(JAVA_SHORT, offset, CpuDTypeOps.toBFloat16Bits(updated));
        }
    }

    private static boolean nativeBf16Experimental(OptimizerStepContext context, TrainableParameterRef ref) {
        return ref.parameterNode().dataType() == DataType.BFLOAT16
                && context.runtimeConfig().bfloat16TrainingPolicy() == BFloat16TrainingPolicy.PARAMS_BF16_EXPERIMENTAL;
    }

    private static final class OwnedNativeParameter {
        private final Tensor source;
        private final NativeTensorStorage storage;

        private OwnedNativeParameter(Tensor source, NativeTensorStorage storage) {
            this.source = source;
            this.storage = storage;
        }

        private Tensor source() {
            return source;
        }

        private NativeTensorStorage storage() {
            return storage;
        }

        private NativeTensorStorage view() {
            return switch (storage.getType()) {
                case FLOAT32 -> new NativeFloat32Storage(storage.getSize(), storage.allocation(), storage.byteOffset(), false);
                case BFLOAT16 -> new NativeBFloat16Storage(storage.getSize(), storage.allocation(), storage.byteOffset(), false);
                default -> throw new IllegalStateException("Unsupported native SGD parameter dtype: " + storage.getType());
            };
        }
    }
}
