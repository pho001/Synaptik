package graph.fused;

public interface FusedExecutionBackend {
    boolean supports(FusedExecutionPlan plan);

    PreparedFusedExecutable prepare(FusedExecutionPlan plan);

    String name();
}
