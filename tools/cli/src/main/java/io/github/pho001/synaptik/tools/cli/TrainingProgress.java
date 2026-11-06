package io.github.pho001.synaptik.tools.cli;

import java.io.PrintStream;
import java.util.Objects;
import java.util.function.LongSupplier;
import org.jline.terminal.Terminal;
import org.jline.utils.Signals;

/**
 * CLI-owned progress and cooperative SIGINT lifetime. The refresh thread reads detached scalar
 * telemetry only; it never calls or closes a training owner. No terminal raw mode is entered.
 * Rendering failures are retained until close, so a primary training failure remains primary
 * under try-with-resources. A final record is attempted synchronously even on very short runs.
 */
final class TrainingProgress implements AutoCloseable {
    enum Outcome { COMPLETED, CANCELLED, FAILED }

    /** Capture seam; production implementations own only presentation resources. */
    interface Sink extends AutoCloseable {
        void render(TrainingProgressStats.Snapshot snapshot, Outcome outcome, String detail);
        @Override
        void close();
    }

    private final TrainingProgressStats stats;
    private final Sink sink;
    private final long refreshMillis;
    private final int epochs;
    private final Object renderLock = new Object();
    private volatile boolean cancellationRequested;
    private boolean finished;
    private boolean closed;
    private volatile Throwable failure;
    private Throwable deliveredFailure;
    private Thread refresher;
    private Thread shutdownHook;
    private Terminal terminal;
    private Terminal.SignalHandler previousTerminalHandler;
    private Object previousPlainHandler;

    /**
     * Opens a stderr monitor with a dedicated daemon refresh thread and cooperative interrupt
     * callback. Terminal/provider limitations select plain logs rather than a different provider.
     * @param error borrowed stderr destination, never closed by this owner
     * @param plain true to prohibit cursor controls
     * @param refreshMillis positive refresh interval in milliseconds
     * @param epochs positive total epoch count
     * @param stepsPerEpoch positive expected steps in each epoch
     * @param examplesPerEpoch positive examples in each epoch
     * @param backendLabel detached backend description
     * @return monitor to finish and close on the runner thread
     * @throws IllegalArgumentException if measurement geometry or cadence is invalid
     */
    static TrainingProgress open(PrintStream error, boolean plain, long refreshMillis, int epochs,
            long stepsPerEpoch, long examplesPerEpoch, String backendLabel) {
        Objects.requireNonNull(error, "error");
        validate(refreshMillis, epochs, stepsPerEpoch, examplesPerEpoch);
        TrainingProgressDisplay display = TrainingProgressDisplay.open(error, plain, epochs,
                stepsPerEpoch, examplesPerEpoch, backendLabel);
        TrainingProgress progress = new TrainingProgress(System::nanoTime, display, refreshMillis, epochs);
        try {
            progress.terminal = display.terminal();
            if (progress.terminal != null) {
                progress.previousTerminalHandler = progress.terminal.handle(Terminal.Signal.INT,
                        signal -> progress.requestCancellation());
            } else {
                // Public JLine 3.30.6 utility returns the prior JVM handler, or null when
                // unsupported. DumbTerminal.handle does not install a native SIGINT handler.
                progress.previousPlainHandler = Signals.register("INT", progress::requestCancellation);
            }
            progress.shutdownHook = new Thread(progress::shutdownCleanup, "synaptik-progress-cleanup");
            Runtime.getRuntime().addShutdownHook(progress.shutdownHook);
            progress.refresh();
            progress.refresher = new Thread(progress::refreshLoop, "synaptik-progress-refresh");
            progress.refresher.setDaemon(true);
            progress.refresher.start();
            return progress;
        } catch (RuntimeException | Error problem) {
            try {
                progress.close();
            } catch (RuntimeException | Error cleanup) {
                if (cleanup != problem) {
                    problem.addSuppressed(cleanup);
                }
            }
            throw problem;
        }
    }

    /**
     * Deterministic clock/capture seam without threads, terminal acquisition, or signal changes.
     * @param clock monotonic nanosecond clock
     * @param sink owned detached snapshot consumer
     * @param refreshMillis positive cadence, used when production starts the refresher
     * @param epochs positive configured epoch count
     */
    TrainingProgress(LongSupplier clock, Sink sink, long refreshMillis, int epochs) {
        validate(refreshMillis, epochs, 1, 1);
        this.stats = new TrainingProgressStats(Objects.requireNonNull(clock, "clock"));
        this.sink = Objects.requireNonNull(sink, "sink");
        this.refreshMillis = refreshMillis;
        this.epochs = epochs;
    }

    private static void validate(long refreshMillis, int epochs, long steps, long examples) {
        if (refreshMillis <= 0 || epochs <= 0 || steps <= 0 || examples <= 0) {
            throw new IllegalArgumentException("Positive refresh interval and training geometry required");
        }
    }

    /** @param oneBasedEpoch next one-based epoch, no greater than configured epochs */
    void beginEpoch(int oneBasedEpoch) {
        if (oneBasedEpoch > epochs) {
            throw new IllegalArgumentException("Epoch exceeds configured count");
        }
        stats.beginEpoch(oneBasedEpoch);
    }

    /**
     * Records only a successful, committed update; a nonfinite objective is still legal.
     * @param preUpdateLoss actual detached objective before the successful update
     * @param examples positive number of actual examples consumed
     * @param inputTargetBytes nonnegative logical decoded input-plus-target payload bytes
     */
    void step(double preUpdateLoss, long examples, long inputTargetBytes) {
        stats.step(preUpdateLoss, examples, inputTargetBytes);
    }

    /** Records the actual completion time of the currently active epoch. */
    void endEpoch() {
        stats.endEpoch();
    }

    /** @return whether SIGINT requested no further admissions; throws a known presentation failure */
    boolean cancellationRequested() {
        if (failure != null) {
            synchronized (renderLock) {
                rethrowFailure();
            }
        }
        return cancellationRequested;
    }

    /** Signal callback and deterministic seam: only publishes a flag, never closes an owner. */
    void requestCancellation() {
        cancellationRequested = true;
    }

    /**
     * Flushes the latest successful counters and final status exactly once. Rendering failures
     * are deferred to close, where Java resource suppression preserves the runner's primary error.
     * @param outcome actual completion, cancellation, or failure
     * @param detail optional diagnostic text, sanitized and bounded by the presentation adapter
     */
    void finish(Outcome outcome, String detail) {
        Objects.requireNonNull(outcome, "outcome");
        synchronized (renderLock) {
            if (finished || closed) {
                return;
            }
            finished = true;
            renderLock.notifyAll();
            render(outcome, detail);
        }
    }

    /** Samples actual successful telemetry synchronously; also used by deterministic tests. */
    void refresh() {
        synchronized (renderLock) {
            if (!finished && !closed) {
                render(null, null);
            }
        }
    }

    private void render(Outcome outcome, String detail) {
        try {
            sink.render(stats.snapshot(), outcome, detail);
        } catch (RuntimeException | Error problem) {
            remember(problem);
        }
    }

    private void refreshLoop() {
        synchronized (renderLock) {
            while (!finished && !closed) {
                try {
                    renderLock.wait(refreshMillis);
                } catch (InterruptedException interrupted) {
                    // Only the UI thread is interrupted; the training thread never is.
                    Thread.currentThread().interrupt();
                    return;
                }
                if (!finished && !closed) {
                    render(null, null);
                }
            }
        }
    }

    private void remember(Throwable problem) {
        if (failure == null) {
            failure = problem;
        } else if (failure != problem) {
            // Keep failure retention bounded if a broken output is sampled repeatedly.
            if (failure.getSuppressed().length < 4) {
                failure.addSuppressed(problem);
            }
        }
    }

    private void rethrowFailure() {
        if (failure instanceof RuntimeException runtime) {
            deliveredFailure = failure;
            throw runtime;
        }
        if (failure instanceof Error error) {
            deliveredFailure = failure;
            throw error;
        }
    }

    private void restoreSignal() {
        if (terminal != null && previousTerminalHandler != null) {
            terminal.handle(Terminal.Signal.INT, previousTerminalHandler);
            previousTerminalHandler = null;
        } else if (previousPlainHandler != null) {
            Signals.unregister("INT", previousPlainHandler);
            previousPlainHandler = null;
        }
    }

    private void shutdownCleanup() {
        // Reuse the same all-cleanup-attempted and primary-error-preserving owner closure.
        close();
    }

    /**
     * Stops future redraws and restores signal/cursor/terminal resources without touching training
     * owners. An already delivered presentation failure is not thrown again by resource closure:
     * Java try-with-resources must never attempt to suppress a throwable onto itself.
     * @throws RuntimeException if rendering or terminal cleanup failed
     */
    @Override
    public void close() {
        synchronized (renderLock) {
            if (!closed) {
                if (!finished) {
                    finished = true;
                    render(cancellationRequested ? Outcome.CANCELLED : Outcome.FAILED,
                            "training closed before completion");
                }
                closed = true;
                renderLock.notifyAll();
                try {
                    restoreSignal();
                } catch (RuntimeException | Error problem) {
                    remember(problem);
                }
                try {
                    sink.close();
                } catch (RuntimeException | Error problem) {
                    remember(problem);
                }
                if (shutdownHook != null) {
                    try {
                        Runtime.getRuntime().removeShutdownHook(shutdownHook);
                    } catch (IllegalStateException shuttingDown) {
                        // Shutdown has already started; the hook sees closed and does nothing.
                    } catch (RuntimeException | Error problem) {
                        remember(problem);
                    }
                }
            }
            if (failure != deliveredFailure) {
                rethrowFailure();
            }
        }
    }
}
