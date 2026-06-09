package backend.cpu1.kernels.fused.codegen;

import backend.cpu1.fused.ir.Cpu1FusedNodePlan;
import backend.cpu1.kernels.fused.Cpu1FusedElementwiseRangeRunner;
import backend.cpu1.kernels.fused.codegen.asm.Cpu1FusedAsmClassEmitter;
import backend.cpu1.kernels.fused.codegen.asm.Cpu1FusedGeneratedClassLoader;

import java.lang.reflect.Constructor;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Prepare-time factory for cpu1 fused generated kernels.
 */
public final class Cpu1FusedCodegenKernelFactory {
    private static final ConcurrentMap<Cpu1FusedCodegenClassSignature, CachedTemplate> CACHE =
            new ConcurrentHashMap<>();
    private static final AtomicInteger CLASS_COUNTER = new AtomicInteger();

    private Cpu1FusedCodegenKernelFactory() {
    }

    public static Cpu1FusedCodegenKernel prepareKernel(Cpu1FusedCodegenPlan plan) {
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        Cpu1FusedCodegenRejectionReason rejectionReason = plan.rejectionReason();
        if (rejectionReason != Cpu1FusedCodegenRejectionReason.NONE) {
            throw rejection(plan, rejectionReason);
        }
        CachedTemplate template = CACHE.computeIfAbsent(plan.classSignature(),
                ignored -> compileTemplate(plan));
        return new Cpu1FusedCodegenKernel(
                plan.classSignature(),
                template.binaryClassName(),
                template.newRunner(scalarValuesF32(plan), scalarValuesF64(plan))
        );
    }

    public static UnsupportedOperationException rejection(
            Cpu1FusedCodegenPlan plan,
            Cpu1FusedCodegenRejectionReason rejectionReason
    ) {
        if (plan == null) {
            throw new IllegalArgumentException("plan cannot be null");
        }
        if (rejectionReason == null) {
            throw new IllegalArgumentException("rejectionReason cannot be null");
        }
        return new UnsupportedOperationException("cpu1 fused ASM codegen rejected: "
                + rejectionReason + " (" + rejectionReason.description() + "), signature="
                + plan.classSignature().canonicalSignature());
    }

    private static CachedTemplate compileTemplate(Cpu1FusedCodegenPlan plan) {
        try {
            String binaryName = "backend.cpu1.kernels.fused.codegen.generated.Cpu1GeneratedFusedKernel"
                    + CLASS_COUNTER.incrementAndGet();
            Cpu1FusedAsmClassEmitter.EmittedClass emitted = Cpu1FusedAsmClassEmitter.emit(binaryName, plan);
            Class<?> generatedClass = new Cpu1FusedGeneratedClassLoader()
                    .defineGenerated(emitted.binaryName(), emitted.bytecode());
            if (!Cpu1FusedElementwiseRangeRunner.class.isAssignableFrom(generatedClass)) {
                throw new IllegalStateException("Generated class does not implement Cpu1FusedElementwiseRangeRunner: "
                        + generatedClass.getName());
            }
            @SuppressWarnings("unchecked")
            Constructor<? extends Cpu1FusedElementwiseRangeRunner> constructor =
                    (Constructor<? extends Cpu1FusedElementwiseRangeRunner>)
                            generatedClass.getConstructor(float[].class, double[].class);
            return new CachedTemplate(binaryName, constructor);
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException("Failed to compile cpu1 fused ASM kernel", e);
        }
    }

    private static float[] scalarValuesF32(Cpu1FusedCodegenPlan plan) {
        float[] values = new float[scalarCount(plan)];
        int index = 0;
        for (Cpu1FusedNodePlan node : plan.expressionPlan().nodes()) {
            if (node.scalarParameter().present()) {
                values[index++] = node.scalarParameter().f32();
            }
        }
        return values;
    }

    private static double[] scalarValuesF64(Cpu1FusedCodegenPlan plan) {
        double[] values = new double[scalarCount(plan)];
        int index = 0;
        for (Cpu1FusedNodePlan node : plan.expressionPlan().nodes()) {
            if (node.scalarParameter().present()) {
                values[index++] = node.scalarParameter().f64();
            }
        }
        return values;
    }

    private static int scalarCount(Cpu1FusedCodegenPlan plan) {
        int count = 0;
        for (Cpu1FusedNodePlan node : plan.expressionPlan().nodes()) {
            if (node.scalarParameter().present()) {
                count++;
            }
        }
        return count;
    }

    private record CachedTemplate(
            String binaryClassName,
            Constructor<? extends Cpu1FusedElementwiseRangeRunner> constructor
    ) {
        private CachedTemplate {
            if (binaryClassName == null || binaryClassName.isBlank()) {
                throw new IllegalArgumentException("binaryClassName cannot be blank");
            }
            if (constructor == null) {
                throw new IllegalArgumentException("constructor cannot be null");
            }
        }

        private Cpu1FusedElementwiseRangeRunner newRunner(float[] f32Scalars, double[] f64Scalars) {
            try {
                return constructor.newInstance(f32Scalars, f64Scalars);
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("Failed to instantiate cpu1 fused ASM kernel", e);
            }
        }
    }
}
