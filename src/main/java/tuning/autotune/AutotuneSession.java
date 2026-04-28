package tuning.autotune;

import tuning.measure.DefaultMeasurementEngine;
import tuning.measure.MeasurementEngine;
import tuning.search.ExhaustiveSearchStrategy;
import tuning.search.SearchStrategy;
import tuning.store.JsonFileBestProfileStore;
import tuning.store.JsonFileTuningHistoryStore;
import tuning.store.BestProfileStore;
import tuning.store.TuningHistoryStore;
import tuning.validate.DefaultValidationEngine;
import tuning.validate.ValidationEngine;

/**
 * Executes an {@link AutotuneRequest} from candidate generation through optional
 * validation, measurement, search refinement, and persistence.
 *
 * <p>Sessions are single-use orchestration objects. Implementations may hold
 * mutable progress state during {@link #run()}, so callers should create a new
 * session for each request and avoid invoking {@code run()} concurrently on the
 * same session instance.</p>
 */
public interface AutotuneSession {
    /**
     * Runs the full autotune workflow.
     *
     * <p>Side effects are limited to workload execution, progress listener calls,
     * and writes requested by the request's persistence policy. Candidate
     * validation failures and measurement exceptions are captured as failed
     * reports instead of aborting the whole run.</p>
     *
     * @return immutable result containing finalists, summary details, and
     * persistence status
     */
    TuningResult run();

    /**
     * Creates a session with the default search strategy, measurement engine,
     * validation engine, and JSON file stores.
     *
     * @param request non-null autotune request
     * @return new session instance
     */
    static AutotuneSession create(AutotuneRequest request) {
        return create(
                request,
                AutotuneDefaultStrategySelector.select(request),
                new DefaultMeasurementEngine(),
                new DefaultValidationEngine(),
                new JsonFileBestProfileStore(),
                new JsonFileTuningHistoryStore()
        );
    }

    /**
     * Creates a session with caller-supplied collaborators, useful for tests,
     * alternate search strategies, or custom persistence.
     *
     * @param request non-null autotune request
     * @param searchStrategy strategy that selects and optionally refines candidates
     * @param measurementEngine engine that compiles/prepares/executes workloads
     * @param validationEngine engine that checks candidate outputs before measurement
     * @param bestProfileStore store used when best-profile persistence is enabled
     * @param historyStore store used when history persistence is enabled
     * @return new session instance
     */
    static AutotuneSession create(
            AutotuneRequest request,
            SearchStrategy searchStrategy,
            MeasurementEngine measurementEngine,
            ValidationEngine validationEngine,
            BestProfileStore bestProfileStore,
            TuningHistoryStore historyStore
    ) {
        return new DefaultAutotuneSession(request, searchStrategy, measurementEngine, validationEngine, bestProfileStore, historyStore);
    }
}
