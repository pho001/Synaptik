package graph.execution.trace;

/**
 * Combined lifecycle trace for compile, prepare, and run.
 *
 * @param compile compile-stage trace
 * @param prepare prepare-stage trace
 * @param run run-stage trace
 */
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
