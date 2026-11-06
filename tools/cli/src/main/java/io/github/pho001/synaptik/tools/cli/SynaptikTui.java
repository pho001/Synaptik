package io.github.pho001.synaptik.tools.cli;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayDeque;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

/**
 * Outer interactive controller. One platform worker admits application workflows and owns all
 * numerical lifetimes. The UI sees detached metadata/measurements and may only request a flag.
 */
final class SynaptikTui {
    private SynaptikTui() { }

    /**
     * Runs on a qualified local terminal, restoring it only after active numerical owners close.
     * @param error borrowed terminal/diagnostic stream, never closed
     * @return zero on menu quit, the active operation's status on quit while working, or 130 on INT
     * @throws RuntimeException if terminal acquisition, rendering or restoration fails
     */
    static int run(PrintStream error) {
        TuiConsole console = TuiConsole.open(error);
        Session session = null;
        Throwable primary = null;
        try {
            session = new Session(error);
            boolean dirty = true;
            boolean wasRunning = false;
            int columns = -1;
            int rows = -1;
            long renderedAt = System.nanoTime();
            for (;;) {
                if (console.interruptionRequested()) session.interrupt();
                session.tick();
                if (session.finished()) break;
                var size = console.size();
                boolean running = session.job != null && !session.job.done;
                long now = System.nanoTime();
                long cadence = session.job == null ? Long.MAX_VALUE : session.job.refreshNanos;
                if (dirty || running != wasRunning || columns != size.getColumns() || rows != size.getRows()
                        || (running && now - renderedAt >= cadence)) {
                    columns = size.getColumns();
                    rows = size.getRows();
                    var frame = session.frame(columns, rows);
                    console.render(frame.lines(), frame.cursor());
                    renderedAt = now;
                    dirty = false;
                }
                wasRunning = running;
                var key = console.readKey(100);
                session.handle(key);
                dirty |= key.kind() != TuiConsole.Kind.TIMEOUT;
            }
            return session.exitStatus;
        } catch (RuntimeException | Error failure) {
            primary = failure;
            throw failure;
        } finally {
            Throwable cleanup = null;
            if (session != null) {
                try {
                    session.stopForClosure();
                    session.close();
                } catch (RuntimeException | Error failure) {
                    cleanup = failure;
                }
            }
            try {
                console.close();
            } catch (RuntimeException | Error failure) {
                if (cleanup == null) cleanup = failure;
                else suppress(cleanup, failure);
            }
            if (primary != null) {
                if (session != null && session.job != null) suppress(primary, session.job.failure);
                suppress(primary, cleanup);
            } else {
                if (cleanup != null && session != null && session.job != null)
                    suppress(cleanup, session.job.failure);
                rethrow(cleanup);
            }
        }
    }

    private static void suppress(Throwable primary, Throwable other) {
        if (other != null && other != primary) primary.addSuppressed(other);
    }

    private static void rethrow(Throwable failure) {
        if (failure instanceof RuntimeException runtime) throw runtime;
        if (failure instanceof Error error) throw error;
    }

    /** UI-only navigation state; tests can drive real workflows without acquiring a terminal. */
    static final class Session implements AutoCloseable {
        private final PrintStream error;
        private final ExecutorService worker = Executors.newSingleThreadExecutor(
                Thread.ofPlatform().name("synaptik-tui-worker").factory());
        private int selected;
        private TuiForm form;
        private String notice;
        private ActiveJob job;
        private boolean exit;
        private boolean back;
        private int exitStatus;

        /** @param error borrowed workflow diagnostics; detached monitors never write to it */
        Session(PrintStream error) { this.error = error; }

        /** @return whether all admitted work is closed and the UI may exit */
        boolean finished() { return exit && (job == null || job.done); }

        /** Samples detached measurements only, and applies pending navigation after closure. */
        void tick() {
            if (job == null) return;
            job.refresh();
            if (!job.done) return;
            if (job.failure instanceof Error && !(job.failure instanceof LinkageError)
                    && !(job.failure instanceof java.io.IOError)) rethrow(job.failure);
            if (exit) exitStatus = job.status;
            else if (back) returnFromJob();
        }

        /**
         * Builds the active menu, editor or detached job screen without numerical-owner access.
         * @param columns actual terminal width in display cells
         * @param rows actual terminal height in rows
         * @return immutable detached screen content fitted to the current terminal geometry
         */
        TuiView.Frame frame(int columns, int rows) {
            if (job != null) return TuiView.job(currentJob(), job.events(), columns, rows);
            if (form != null) return TuiView.form(form, columns, rows, notice);
            return TuiView.menu(selected, columns, rows, notice);
        }

        /** @return detached current operation state, or null while browsing/editing */
        TuiView.Job currentJob() { return job == null ? null : job.view(); }

        /** @param key decoded terminal key; EOF and INT never interrupt the worker */
        void handle(TuiConsole.Key key) {
            if (key.kind() == TuiConsole.Kind.INTERRUPT) { interrupt(); return; }
            if (key.kind() == TuiConsole.Kind.EOF) { requestExit(); return; }
            if (job != null) {
                if (character(key, 'q')) requestExit();
                else if (job.done && (key.kind() == TuiConsole.Kind.ENTER
                        || key.kind() == TuiConsole.Kind.ESCAPE)) returnFromJob();
                else if (!job.done && key.kind() == TuiConsole.Kind.ESCAPE) {
                    back = true;
                    job.requestStop("Back requested");
                } else if (!job.done && character(key, 's') && job.training()) {
                    job.requestStop("Stop requested");
                }
                return;
            }
            if (form != null) {
                try {
                    switch (form.handle(key)) {
                        case BACK -> { form = null; notice = null; }
                        case CHANGED -> notice = null;
                        case SUBMIT -> submit(CliArguments.parse(form.arguments()));
                        case UNCHANGED -> { }
                    }
                } catch (IllegalArgumentException failure) {
                    notice = diagnostic(failure);
                }
                return;
            }
            if (character(key, 'q') || key.kind() == TuiConsole.Kind.ESCAPE) requestExit();
            else if (key.kind() == TuiConsole.Kind.UP || key.kind() == TuiConsole.Kind.BACK_TAB)
                selected = Math.floorMod(selected - 1, 4);
            else if (key.kind() == TuiConsole.Kind.DOWN || key.kind() == TuiConsole.Kind.TAB)
                selected = (selected + 1) % 4;
            else if (character(key, 't')) openForm(0);
            else if (character(key, 'p')) openForm(1);
            else if (character(key, 'i')) openForm(2);
            else if (key.kind() == TuiConsole.Kind.ENTER) openForm(selected);
        }

        private void openForm(int action) {
            selected = action;
            notice = null;
            if (action == 3) requestExit();
            else form = new TuiForm(TuiForm.Action.values()[action]);
        }

        /**
         * Admits one actual shared application workflow, never a CLI subprocess or text adapter.
         * @param args syntax-admitted train/predict/inspect options; prediction requires a file
         */
        void submit(CliArguments args) {
            if (job != null && !job.done) throw new IllegalStateException("An operation is already active");
            if (!List.of("train", "predict", "inspect").contains(args.command()))
                throw new IllegalArgumentException("Choose train, predict or inspect");
            if (args.command().equals("predict")) args.requiredPath("output");
            ActiveJob next = new ActiveJob(args, error);
            worker.execute(next);
            job = next;
            back = false;
            notice = null;
        }

        private void returnFromJob() {
            if (job.failure != null && form != null) notice = diagnostic(job.failure);
            else { form = null; notice = null; }
            job = null;
            back = false;
        }

        /** Native INT and raw Ctrl-C stop then exit, only after worker-owned resources close. */
        void interrupt() {
            exitStatus = 130;
            requestExit();
        }

        private void requestExit() {
            exit = true;
            if (job != null && !job.done) job.requestStop("Exit requested");
            else if (job != null) exitStatus = job.status;
        }

        private void stopForClosure() {
            if (job != null && !job.done) job.requestStop("Terminal closing");
        }

        /**
         * Stops new admissions and waits uninterruptibly for the one worker's ordinary closure.
         * An interrupted UI restores its own interrupt flag; it never interrupts the worker.
         */
        @Override public void close() {
            stopForClosure();
            worker.shutdown();
            boolean interrupted = false;
            for (;;) {
                try {
                    if (worker.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS)) break;
                } catch (InterruptedException failure) {
                    interrupted = true;
                }
            }
            if (interrupted) Thread.currentThread().interrupt();
        }
    }

    private static boolean character(TuiConsole.Key key, int value) {
        return key.kind() == TuiConsole.Kind.CHARACTER && key.character() == value;
    }

    private static String diagnostic(Throwable failure) {
        String message = failure.getClass().getSimpleName() + ": " + failure.getMessage();
        return message.length() <= 512 ? message : message.substring(0, 512);
    }

    /**
     * One admitted operation. Only run() accesses sources/Model/Engine/batch owners. Other methods
     * exchange detached snapshots or request a stop; done is published after all owners close.
     */
    static final class ActiveJob implements Runnable {
        private static final int EVENT_CAPACITY = 16;
        private final CliArguments args;
        private final PrintStream error;
        private final long started = System.nanoTime();
        private final ArrayDeque<String> eventLog = new ArrayDeque<>();
        private final Object admission = new Object();
        private volatile TrainingProgress monitor;
        private volatile TuiView.Progress progress;
        private volatile boolean stopRequested;
        private volatile boolean done;
        private volatile int status;
        private volatile String detail = "Validating input and admitting workflow";
        private volatile List<String> inspection = List.of();
        private volatile Throwable failure;
        private volatile long ended;
        private volatile long refreshNanos = TimeUnit.MILLISECONDS.toNanos(CliDefaults.REFRESH_MILLIS);
        private long sampled;
        private int recordedEpoch;

        /** @param args admitted command syntax @param error borrowed diagnostics */
        ActiveJob(CliArguments args, PrintStream error) {
            this.args = args;
            this.error = error;
            event("Started " + args.command());
        }

        boolean training() { return args.command().equals("train"); }

        /** Applies requests made during validation atomically when the detached monitor appears. */
        void requestStop(String reason) {
            synchronized (admission) {
                if (done || stopRequested) return;
                stopRequested = true;
                if (training()) {
                    if (monitor != null) monitor.requestCancellation();
                    if (progress == null || progress.outcome() == null)
                        detail = reason + "; waiting for the current admitted call and normal cleanup";
                } else {
                    detail = reason + "; waiting for admitted " + args.command()
                            + " to finish normally (no cancellation/rollback)";
                }
                event(detail);
            }
        }

        TrainingProgress openProgress(PrintStream destination, boolean plain, long refreshMillis,
                int epochs, long stepsPerEpoch, long examplesPerEpoch, String backend) {
            TrainingProgress created = new TrainingProgress(System::nanoTime, new TrainingProgress.Sink() {
                @Override public void render(TrainingProgressStats.Snapshot snapshot,
                        TrainingProgress.Outcome outcome, String finalDetail) {
                    progress = new TuiView.Progress(epochs, stepsPerEpoch, examplesPerEpoch,
                            backend, snapshot, outcome, finalDetail);
                    for (var epoch : snapshot.recentEpochs()) {
                        if (epoch.number() > recordedEpoch) {
                            recordedEpoch = epoch.number();
                            event("Completed epoch " + epoch.number() + ": " + epoch.steps()
                                    + " updates, mean pre-update loss " + epoch.meanLoss());
                        }
                    }
                    if (outcome != null) {
                        detail = finalDetail;
                        event(outcome + ": " + finalDetail);
                    }
                }
                @Override public void close() { /* Detached data only: no terminal/core resources. */ }
            }, refreshMillis, epochs);
            synchronized (admission) {
                refreshNanos = TimeUnit.MILLISECONDS.toNanos(refreshMillis);
                if (stopRequested) created.requestCancellation();
                monitor = created;
                event("Admitted training: " + epochs + " epochs, " + stepsPerEpoch
                        + " updates/epoch, " + examplesPerEpoch + " examples/epoch; " + backend);
            }
            created.refresh();
            return created;
        }

        /** Called by UI at its cadence; reads only the detached monitor. */
        void refresh() {
            long now = System.nanoTime();
            TrainingProgress current = monitor;
            if (current != null && (sampled == 0 || now - sampled >= refreshNanos)) {
                sampled = now;
                current.refresh();
            }
        }

        @Override public void run() {
            try {
                switch (args.command()) {
                    case "train" -> status = CsvRegressionWorkflow.train(args, error, this::openProgress);
                    case "predict" -> {
                        // File output is mandatory even for package-local programmatic admission.
                        var destination = args.requiredPath("output");
                        try (var discard = new PrintStream(OutputStream.nullOutputStream())) {
                            status = CsvRegressionWorkflow.predict(args, discard);
                        }
                        detail = "Predictions saved to " + destination;
                        event(detail);
                    }
                    case "inspect" -> {
                        inspection = CliModel.describe(CliModel.load(args.requiredPath("model")));
                        detail = "Public graph metadata loaded; no Engine acquired";
                        event(detail);
                    }
                    default -> throw new AssertionError("Unexpected operation");
                }
            } catch (IOException | RuntimeException | Error problem) {
                failure = problem;
                status = problem instanceof IllegalArgumentException ? 2 : 1;
                detail = diagnostic(problem);
                event(detail);
                for (Throwable suppressed : problem.getSuppressed()) event("cleanup: " + diagnostic(suppressed));
            } finally {
                synchronized (admission) {
                    if (failure == null) {
                        if (training() && progress != null) detail = progress.detail();
                        else if (args.command().equals("predict"))
                            detail = "Predictions saved to " + args.requiredPath("output");
                        else if (args.command().equals("inspect"))
                            detail = "Public graph metadata loaded; no Engine acquired";
                    }
                    ended = System.nanoTime();
                    done = true;
                }
            }
        }

        private synchronized void event(String value) {
            if (eventLog.size() == EVENT_CAPACITY) eventLog.removeFirst();
            eventLog.addLast(value == null ? "" : value.length() <= 512 ? value : value.substring(0, 512));
        }

        synchronized List<String> events() { return List.copyOf(eventLog); }

        TuiView.Job view() {
            boolean completed = done;
            return new TuiView.Job(args.command(), !completed, stopRequested, status, detail,
                    inspection, progress, TrainingProgressStats.duration(completed ? ended : System.nanoTime(), started));
        }
    }
}
