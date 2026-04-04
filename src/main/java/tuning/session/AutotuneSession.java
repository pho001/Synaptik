package tuning.session;

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

public interface AutotuneSession {
    TuningResult run();

    static AutotuneSession create(AutotuneRequest request) {
        return create(
                request,
                new ExhaustiveSearchStrategy(),
                new DefaultMeasurementEngine(),
                new DefaultValidationEngine(),
                new JsonFileBestProfileStore(),
                new JsonFileTuningHistoryStore()
        );
    }

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
