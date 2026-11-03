package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import java.util.Objects;

/** Closed native realization boundary; provider availability never widens Model semantics. */
public final class CpuOneDnnCompiler {
    private CpuOneDnnCompiler() { }

    /**
     * Admits only the qualified bare F32 contraction. Convolution zero signs, external bias
     * rounding, and activation special values are not interchangeable with Model's contracts.
     * Those ordinary units remain explicitly planned portable work, never a late fallback.
     *
     * @param spec non-null common numeric projection; no tensor contents are inspected
     * @return true only for bare MATMUL without bias/activation and with K greater than one;
     *     layout, dtype and whole-plan eligibility require the surrounding analysis checks
     * @throws NullPointerException if spec is null
     */
    public static boolean qualified(CpuOneDnnKernelSpec spec) {
        Objects.requireNonNull(spec, "spec");
        return spec.operation() == CpuOneDnnKernelSpec.Operation.MATMUL
                && spec.bias() == CpuOneDnnKernelSpec.Bias.NONE
                && spec.activation() == CpuOneDnnKernelSpec.Activation.NONE
                && spec.inputs().getFirst().dimensions().getLast() > 1;
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
        return api == CpuOneDnnRoutePlan.Api.PRIMITIVE
                ? CpuOneDnnPrimitives.compile(library, spec) : CpuOneDnnGraph.compile(library, spec);
    }

    private static void requireQualified(CpuOneDnnKernelSpec spec, CpuOneDnnRoutePlan.Api api) {
        Objects.requireNonNull(api, "api");
        if (api == CpuOneDnnRoutePlan.Api.AUTO || !qualified(spec)) {
            throw new IllegalArgumentException("oneDNN realization is not qualified for the current Model contract");
        }
    }
}
