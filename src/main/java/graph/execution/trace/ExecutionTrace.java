package graph.execution.trace;

public record ExecutionTrace(
        CompileTrace compile,
        PrepareTrace prepare,
        RunTrace run
) {
    public ExecutionTrace {
        compile = compile == null ? CompileTrace.skipped() : compile;
        prepare = prepare == null ? PrepareTrace.skipped() : prepare;
        run = run == null ? RunTrace.empty(backend.runtime.ExecutionMode.FORWARD) : run;
    }
}
