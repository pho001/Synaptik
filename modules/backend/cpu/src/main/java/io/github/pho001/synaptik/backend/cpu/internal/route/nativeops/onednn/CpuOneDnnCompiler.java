package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.Objects;

/** Closed native realization boundary; provider availability never widens Model semantics. */
public final class CpuOneDnnCompiler {
    private CpuOneDnnCompiler() { }

    /** Expected cold native-candidate rejection, never an invocation failure or fallback. */
    public static final class UnsupportedRealizationException extends IllegalArgumentException {
        /**
         * Identifies why native representation or implementation qualification failed.
         *
         * @param message non-null rejection reason
         * @throws NullPointerException if message is null
         */
        UnsupportedRealizationException(String message) {
            super("oneDNN candidate: " + Objects.requireNonNull(message, "message"));
        }
    }

    /**
     * Admits typed literal operations structurally, before native implementation qualification.
     * Opaque convolution, contraction bias and activation remain diagnostic only. Each pipeline
     * step is independently qualified; no input values or runtime numeric certificates are used.
     *
     * @param spec non-null common numeric projection; no tensor contents are inspected
     * @return whether the recipe has a supported F32/BOOL form and safe literal geometry
     * @throws NullPointerException if spec is null
     */
    public static boolean qualified(CpuOneDnnKernelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        if (spec instanceof CpuOneDnnKernelSpec.Pipeline pipeline) {
            for (var input : pipeline.inputs()) if (input.dataType() != DataType.FLOAT32) return false;
            for (var step : pipeline.steps()) if (!qualified(step.spec())) return false;
            return true;
        }
        for (var input : spec.inputs()) if (input.dataType() != DataType.FLOAT32) return false;
        boolean predicate = spec instanceof CpuOneDnnKernelSpec.Binary binary
                && switch (binary.algorithm()) {
                    case LT, LE, EQ, NE -> true;
                    default -> false;
                };
        if (spec.output().dataType() != (predicate ? DataType.BOOL : DataType.FLOAT32)) return false;
        if (spec instanceof CpuOneDnnKernelSpec.Contraction contraction) {
            if (contraction.operation() != CpuOneDnnKernelSpec.Operation.MATMUL
                    || contraction.bias() != CpuOneDnnKernelSpec.Bias.NONE
                    || contraction.activation() != CpuOneDnnKernelSpec.Activation.NONE
                    || contraction.inputs().getFirst().dimensions().getLast() <= 1
                    || !densePermutation(contraction.output())) return false;
            for (var input : contraction.inputs()) if (!densePermutation(input)) return false;
            return true;
        }
        if (spec instanceof CpuOneDnnKernelSpec.Pooling pooling) return qualifiedPooling(pooling);
        return true;
    }

    private static boolean densePermutation(CpuOneDnnKernelSpec.Tensor tensor) {
        int axes = 0;
        for (long extent : tensor.dimensions()) if (extent > 1) axes++;
        long expectedStride = 1;
        for (int visited = 0; visited < axes; visited++) {
            int matching = -1;
            for (int axis = 0; axis < tensor.dimensions().size(); axis++) {
                if (tensor.dimensions().get(axis) > 1 && tensor.strides().get(axis) == expectedStride) {
                    if (matching >= 0) return false;
                    matching = axis;
                }
            }
            if (matching < 0) return false;
            expectedStride = Math.multiplyExact(expectedStride, tensor.dimensions().get(matching));
        }
        return true;
    }

    private static boolean qualifiedPooling(CpuOneDnnKernelSpec.Pooling pool) {
        long divisor = 1;
        boolean hasInteriorWindow = true;
        for (int axis = 0; axis < pool.kernel().size(); axis++) {
            long kernel = pool.kernel().get(axis), dilation = pool.dilations().get(axis);
            long step = pool.strides().get(axis), left = pool.paddingLeft().get(axis);
            long input = pool.input().dimensions().get(axis + 2);
            long output = pool.output().dimensions().get(axis + 2);
            divisor = Math.multiplyExact(divisor, kernel);
            long reach = Math.multiplyExact(kernel - 1, dilation);
            long available = Math.subtractExact(Math.addExact(Math.addExact(input, left),
                    pool.paddingRight().get(axis)), Math.addExact(reach, 1));
            if (available < 0 || available / step + 1 != output) return false;
            long firstInterior = left / step + (left % step == 0 ? 0 : 1);
            long lastStart = Math.subtractExact(Math.addExact(input - 1, left), reach);
            boolean interior = lastStart >= 0 && firstInterior <= Math.min(output - 1, lastStart / step);
            hasInteriorWindow &= interior;
        }
        return !hasInteriorWindow && (divisor & (divisor - 1)) == 0;
    }

    /**
     * Queries exact native descriptor facts without retaining native state in analysis.
     * Primitive queries create no executable; Graph temporarily compiles, queries and closes.
     *
     * @param library non-null open exact library borrowed for this query
     * @param spec non-null qualified numeric recipe, with no retained tensor data
     * @param api non-null resolved PRIMITIVE or GRAPH API, never AUTO
     * @return non-null handle-free scratch size and implementation identity
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if the API is unresolved or the recipe is unqualified/unsupported
     * @throws IllegalStateException if the library is closed or native querying/cleanup fails
     */
    public static CpuOneDnnKernelMetadata describe(CpuOneDnnLibrary library,
            CpuOneDnnKernelSpec spec, CpuOneDnnRoutePlan.Api api) {
        requireQualified(spec, api);
        if (spec instanceof CpuOneDnnKernelSpec.Pipeline pipeline) {
            return CpuOneDnnPipeline.describe(library, pipeline, api);
        }
        if (api == CpuOneDnnRoutePlan.Api.PRIMITIVE) {
            return CpuOneDnnPrimitives.describe(library, spec);
        }
        // Graph exposes exact scratch only after compilation. Query and close transactionally;
        // no native handle escapes analysis. Finalization recompiles the exact retained recipe.
        try (CpuOneDnnKernel kernel = CpuOneDnnGraph.compile(library, spec)) {
            return new CpuOneDnnKernelMetadata(kernel.scratchpadBytes(), kernel.implementation());
        }
    }

    /**
     * Freshly realizes the selected recipe after shared resource assignment.
     *
     * @param library non-null open library retained by the returned kernel
     * @param spec non-null qualified numeric recipe to recompile exactly
     * @param api non-null resolved PRIMITIVE or GRAPH API, never AUTO
     * @return non-null caller-owned persistent kernel; the finalizer must verify analyzed facts
     *     and transfer its close duty through PreparedResource ownership or close on rollback
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if the API is unresolved or the recipe is unqualified/unsupported
     * @throws IllegalStateException if the library is closed or native construction/cleanup fails
     */
    public static CpuOneDnnKernel compile(CpuOneDnnLibrary library, CpuOneDnnKernelSpec spec,
            CpuOneDnnRoutePlan.Api api) {
        requireQualified(spec, api);
        if (spec instanceof CpuOneDnnKernelSpec.Pipeline pipeline) {
            return CpuOneDnnPipeline.compile(library, pipeline, api);
        }
        return api == CpuOneDnnRoutePlan.Api.PRIMITIVE
                ? CpuOneDnnPrimitives.compile(library, spec) : CpuOneDnnGraph.compile(library, spec);
    }

    private static void requireQualified(CpuOneDnnKernelSpec spec, CpuOneDnnRoutePlan.Api api) {
        Objects.requireNonNull(api, "api");
        if (api == CpuOneDnnRoutePlan.Api.AUTO || !qualified(spec)) {
            throw new IllegalArgumentException("oneDNN realization is not qualified for the current Model contract");
        }
        if (api == CpuOneDnnRoutePlan.Api.GRAPH) requireQualifiedGraph(spec);
    }

    private static void requireQualifiedGraph(CpuOneDnnKernelSpec spec) {
        if (spec instanceof CpuOneDnnKernelSpec.Pipeline pipeline) {
            for (var step : pipeline.steps()) requireQualifiedGraph(step.spec());
        } else if (spec instanceof CpuOneDnnKernelSpec.Unary unary
                && (unary.algorithm() == CpuOneDnnKernelSpec.UnaryAlgorithm.ABS
                        || unary.algorithm() == CpuOneDnnKernelSpec.UnaryAlgorithm.EXP)) {
            throw new UnsupportedRealizationException("Graph does not expose the selected elementary implementation");
        } else if (spec instanceof CpuOneDnnKernelSpec.Binary binary) {
            switch (binary.algorithm()) {
                case MIN, MAX -> throw new UnsupportedRealizationException("Graph does not expose the selected extrema implementation");
                case LT, LE, EQ, NE -> throw new UnsupportedRealizationException("Graph 3.14 has no exact predicate operation");
                default -> { }
            }
        } else if (spec instanceof CpuOneDnnKernelSpec.Pooling) {
            throw new UnsupportedRealizationException("Graph padded-average implementation is not qualified");
        }
    }
}
