package training.optimizer;

import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.buffer.MetalBufferAllocator;
import backend.metal.buffer.MetalBufferBinding;
import graph.CompiledNode;
import backend.cpu.kernels.CpuDTypeOps;
import tensor.DataType;
import tensor.Tensor;

import java.util.Collection;

/**
 * Stochastic gradient descent optimizer for trainable parameters.
 */
public final class SgdOptimizer extends AbstractTrainableOptimizer {
    private final float learningRate;

    public SgdOptimizer(float learningRate) {
        this(null, learningRate);
    }

    public SgdOptimizer(Collection<Tensor> parameters, float learningRate) {
        super(parameters);
        if (!(learningRate > 0.0f) || !Float.isFinite(learningRate)) {
            throw new IllegalArgumentException("learningRate must be a finite positive value.");
        }
        this.learningRate = learningRate;
    }

    public float learningRate() {
        return learningRate;
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
        parameter.markStorageModified();
        ref.parameterNode().sourceTensor().markStorageModified();
        context.executionContext().markCpuCurrent(ref.parameterNode().id(), "optimizer CPU SGD update");
    }

    private static void updateCpu(Tensor parameter, Tensor gradient, float learningRate, CompiledNode node) {
        if (parameter.getDataType() != gradient.getDataType()) {
            throw new IllegalStateException("Parameter and gradient dtype differ for nodeId=" + node.id());
        }
        switch (parameter.getDataType()) {
            case FLOAT32 -> {
                float[] p = parameter.getFloat32Data();
                float[] g = gradient.getFloat32Data();
                for (int i = 0; i < p.length; i++) {
                    p[i] -= learningRate * g[i];
                }
            }
            case FLOAT64 -> {
                double[] p = parameter.getFloat64Data();
                double[] g = gradient.getFloat64Data();
                for (int i = 0; i < p.length; i++) {
                    p[i] -= (double) learningRate * g[i];
                }
            }
            case BFLOAT16 -> {
                short[] p = parameter.getBFloat16Data();
                short[] g = gradient.getBFloat16Data();
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
}
