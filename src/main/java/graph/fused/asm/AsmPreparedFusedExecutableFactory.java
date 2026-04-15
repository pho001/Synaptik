package graph.fused.asm;

import backend.kernels.cpu.fused.FusedExecutionProfiler;
import graph.codegen.FusedAsmSpecializationKind;
import graph.codegen.FusedAsmSpecializationMatcher;
import graph.codegen.FusedKernelGeneratorRouter;
import graph.codegen.FusedKernelCacheKey;
import graph.fused.FusedExecutionPlan;
import graph.fused.PreparedFusedExecutable;
import operations.FusedOperation;
import utils.CustomClassLoader;

import java.lang.reflect.Constructor;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

public final class AsmPreparedFusedExecutableFactory {
    private static final AtomicInteger CLASS_COUNTER = new AtomicInteger();

    private final ConcurrentHashMap<FusedKernelCacheKey, Constructor<? extends PreparedFusedExecutable>> cache =
            new ConcurrentHashMap<>();

    public PreparedFusedExecutable create(FusedExecutionPlan plan) {
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        FusedOperation descriptor = plan.descriptor();
        FusedAsmSpecializationKind specializationKind =
                FusedAsmSpecializationMatcher.match(descriptor.getPlan(), descriptor.getPrecisionMode());

        FusedKernelCacheKey key = new FusedKernelCacheKey(
                descriptor.getSchedulerSignature(),
                descriptor.getPrecisionMode(),
                plan.asmVectorWidth(),
                specializationKind
        );

        try {
            Constructor<? extends PreparedFusedExecutable> ctor = cache.computeIfAbsent(
                    key,
                    ignored -> compileConstructor(descriptor, key.vectorWidth(), key.specializationKind())
            );
            return ctor.newInstance();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to instantiate ASM fused executable", e);
        }
    }

    @SuppressWarnings("unchecked")
    private Constructor<? extends PreparedFusedExecutable> compileConstructor(
            FusedOperation descriptor,
            int vectorWidth,
            FusedAsmSpecializationKind specializationKind
    ) {
        long t0 = FusedExecutionProfiler.enabled() ? System.nanoTime() : 0L;
        try {
            int id = CLASS_COUNTER.incrementAndGet();
            String binaryName = "graph.fused.asm.GeneratedFusedExecutable"
                    + id
                    + "_"
                    + specializationKind.cacheToken()
                    + "W"
                    + Math.max(1, vectorWidth);
            String internalName = binaryName.replace('.', '/');

            byte[] bytecode = FusedKernelGeneratorRouter.generate(
                    internalName,
                    descriptor.getPlan(),
                    descriptor.getPrecisionMode(),
                    vectorWidth,
                    specializationKind
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

            return (Constructor<? extends PreparedFusedExecutable>) ctor;
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to compile ASM fused executable", e);
        }
    }
}
