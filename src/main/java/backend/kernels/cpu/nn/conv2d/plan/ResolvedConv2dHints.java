package backend.kernels.cpu.nn.conv2d.plan;

import backend.blas.BlasProvider;

import java.util.Objects;

public record ResolvedConv2dHints(
        boolean useBlas,
        BlasProvider provider,
        int m,
        int n,
        int k,
        long work
) {
    public ResolvedConv2dHints {
        provider = Objects.requireNonNullElse(provider, BlasProvider.NONE);
        m = Math.max(0, m);
        n = Math.max(0, n);
        k = Math.max(0, k);
        work = Math.max(0L, work);
    }
}
