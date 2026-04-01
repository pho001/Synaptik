package graph.codegen;

import backend.kernels.cpu.FusedExecutionProfiler;
import backend.kernels.cpu.fused.CompiledFusedKernel;
import operations.FusedOperation;
import utils.CustomClassLoader;

import java.lang.reflect.Constructor;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public final class CompiledFusedKernelFactory {
    private static final AtomicInteger CLASS_COUNTER = new AtomicInteger();

    private final ConcurrentHashMap<FusedKernelCacheKey, Constructor<? extends CompiledFusedKernel>> cache =
            new ConcurrentHashMap<>();

    public CompiledFusedKernel create(FusedOperation descriptor) {
        if (descriptor == null) {
            throw new IllegalArgumentException("descriptor cannot be null");
        }

        FusedKernelCacheKey key = new FusedKernelCacheKey(
                descriptor.getSchedulerSignature(),
                descriptor.getPrecisionMode()
        );

        try {
            Constructor<? extends CompiledFusedKernel> ctor = cache.computeIfAbsent(
                    key,
                    ignored -> compileConstructor(descriptor)
            );
            return ctor.newInstance();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to instantiate compiled fused kernel", e);
        }
    }

    @SuppressWarnings("unchecked")
    private Constructor<? extends CompiledFusedKernel> compileConstructor(FusedOperation descriptor) {
        long t0 = FusedExecutionProfiler.enabled() ? System.nanoTime() : 0L;
        try {
            int id = CLASS_COUNTER.incrementAndGet();
            String binaryName = "backend.kernels.cpu.fused.GeneratedFusedKernel" + id;
            String internalName = binaryName.replace('.', '/');

            byte[] bytecode = FusedKernelGeneratorRouter.generate(
                    internalName,
                    descriptor.getPlan(),
                    descriptor.getPrecisionMode()
            );

            CustomClassLoader loader = new CustomClassLoader();
            Class<?> generatedClass = loader.define(binaryName, bytecode);
            Constructor<?> ctor = generatedClass.getConstructor();

            if (FusedExecutionProfiler.enabled()) {
                FusedExecutionProfiler.recordCompile(
                        descriptor.getSchedulerSignature(),
                        descriptor.getExpression(),
                        descriptor.getPlan().nodeCount(),
                        descriptor.getPlan().inputCount(),
                        descriptor.getPrecisionMode(),
                        descriptor.isLowCostHint(),
                        System.nanoTime() - t0
                    );
            }

            return (Constructor<? extends CompiledFusedKernel>) ctor;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to compile fused kernel", e);
        }
    }
}
