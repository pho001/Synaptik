package io.github.pho001.synaptik.tools.benchmarks;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
import io.github.pho001.synaptik.engine.PreparedExecution;
import io.github.pho001.synaptik.engine.RunResult;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.convolution.Conv2dAttrs;
import io.github.pho001.synaptik.model.operation.convolution.Conv3dAttrs;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

/** Runnable, report-only CPU Engine lifecycle baseline using fixed public Tensor and Engine APIs. */
public final class CpuLifecycleBenchmark {
    private CpuLifecycleBenchmark() {}

    /** Runs the selected profile and writes one JSON report to stdout. */
    public static void main(String[] args) {
        String profile = args.length == 0 ? "smoke" : args[0].toLowerCase(Locale.ROOT);
        if (!profile.equals("smoke") && !profile.equals("baseline")) {
            throw new IllegalArgumentException("profile must be smoke or baseline");
        }
        int warmups = profile.equals("smoke") ? 1 : 3;
        int measurements = profile.equals("smoke") ? 2 : 10;
        List<Workload> workloads = workloads();
        long started = System.nanoTime();
        List<String> reports = new ArrayList<>();
        for (Workload workload : workloads) {
            try {
                reports.add(run(workload, warmups, measurements));
            } catch (RuntimeException failure) {
                reports.add("{\"name\":\"" + workload.name + "\",\"status\":\"unavailable\",\"error\":\""
                        + esc(failure.getMessage() == null ? failure.getClass().getSimpleName() : failure.getMessage()) + "\"}");
            }
        }
        long elapsed = System.nanoTime() - started;
        System.out.println("{\"schema\":1,\"profile\":\"" + profile + "\",\"warmupIterations\":"
                + warmups + ",\"measurementIterations\":" + measurements
                + ",\"java\":\"" + esc(System.getProperty("java.version"))
                + "\",\"os\":\"" + esc(System.getProperty("os.name") + " " + System.getProperty("os.arch"))
                + "\",\"elapsedNanos\":" + elapsed + ",\"workloads\":["
                + String.join(",", reports) + "]}");
    }

    private static String run(Workload workload, int warmups, int measurements) {
        long compileNanos;
        long prepareNanos;
        long oneShotNanos = 0;
        List<Long> repeated = new ArrayList<>();
        long checksum = 0;
        try (Engine engine = Engine.standard()) {
            long begin = System.nanoTime();
            var graph = engine.compile(List.of(workload.output));
            compileNanos = System.nanoTime() - begin;
            begin = System.nanoTime();
            try (PreparedExecution prepared = engine.prepare(graph)) {
                prepareNanos = System.nanoTime() - begin;
                for (int i = 0; i < warmups; i++) checksum ^= runPrepared(engine, prepared, workload.inputs);
                for (int i = 0; i < measurements; i++) {
                    begin = System.nanoTime();
                    checksum ^= runPrepared(engine, prepared, workload.inputs);
                    repeated.add(System.nanoTime() - begin);
                }
            }
            for (int i = 0; i < warmups; i++) checksum ^= compute(engine, workload.output);
            for (int i = 0; i < measurements; i++) {
                begin = System.nanoTime();
                checksum ^= compute(engine, workload.output);
                oneShotNanos += System.nanoTime() - begin;
            }
        }
        return "{\"name\":\"" + workload.name + "\",\"inputs\":" + workload.inputs.size()
                + ",\"shape\":\"" + workload.output.descriptor().shape() + "\",\"compileNanos\":"
                + compileNanos + ",\"prepareNanos\":" + prepareNanos + ",\"repeatedRunNanos\":"
                + longs(repeated) + ",\"oneShotComputeTotalNanos\":" + oneShotNanos
                + ",\"checksum\":" + checksum + "}";
    }

    private static long runPrepared(Engine engine, PreparedExecution prepared, List<Tensor> inputs) {
        try (RunResult result = engine.run(prepared, inputs)) {
            HostTensorValue value = result.materialize(result.publications().getFirst(), Long.MAX_VALUE);
            return checksum(value);
        }
    }

    private static long compute(Engine engine, Tensor output) {
        HostTensorValue value = engine.compute(output);
        return checksum(value);
    }

    private static long checksum(HostTensorValue value) {
        ByteBuffer bytes = value.bytes();
        long hash = 0xcbf29ce484222325L;
        while (bytes.hasRemaining()) hash = (hash ^ (bytes.get() & 0xffL)) * 0x100000001b3L;
        return hash;
    }

    private static String longs(List<Long> values) {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < values.size(); i++) { if (i > 0) out.append(','); out.append(values.get(i)); }
        return out.append(']').toString();
    }

    private static String esc(String value) { return value.replace("\\", "\\\\").replace("\"", "\\\""); }

    private static List<Workload> workloads() {
        return List.of(
                binaryAdd("pointwise", tensor(Shape.of(2, 3)), tensor(Shape.of(2, 3))),
                unary("matmul", tensor(Shape.of(2, 3)), tensor(Shape.of(3, 4)), true),
                unary("conv2d", tensor(Shape.of(1, 2, 8, 8)), tensor(Shape.of(3, 2, 3, 3)), false),
                unary("conv3d", tensor(Shape.of(1, 2, 5, 5, 5)), tensor(Shape.of(3, 2, 3, 3, 3)), false),
                reduction("reduction", tensor(Shape.of(4, 16))),
                normalization("normalization", tensor(Shape.of(4, 16))));
    }
    private static Workload binaryAdd(String name, Tensor input, Tensor second) {
        return finish(name, input.add(second), input, second);
    }

    private static Tensor inputData(Tensor template) {
        Shape shape = template.descriptor().shape();
        int count = Math.toIntExact(shape.knownElementCount().orElseThrow());
        float[] data = new float[count];
        for (int i = 0; i < count; i++) data[i] = ((i * 17) % 101 - 50) / 101.0f;
        return TensorFactory.fromFlatArray(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(shape)), false),
                Optional.empty(), data);
    }

    private static Workload finish(String name, Tensor output, Tensor... inputs) {
        List<Tensor> data = new ArrayList<>(inputs.length);
        for (Tensor input : inputs) data.add(inputData(input));
        return new Workload(name, output, List.copyOf(data));
    }

    private static Workload unary(String name, Tensor input, Tensor second, boolean matmul) {
        Tensor output = matmul ? input.matmul(second) : name.equals("conv2d")
                ? input.conv2d(second, Conv2dAttrs.defaults())
                : input.conv3d(second, Conv3dAttrs.defaults());
        return finish(name, output, input, second);
    }

    private static Workload reduction(String name, Tensor input) { return finish(name, input.sum(), input); }
    private static Workload normalization(String name, Tensor input) {
        return finish(name, input.layerNorm(Shape.of(16),
                io.github.pho001.synaptik.model.datatype.ScalarValue.float32(1e-5f)), input);
    }

    private static Tensor tensor(Shape shape) {
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(io.github.pho001.synaptik.model.layout.LayoutDescriptor.contiguous(shape)), false));
    }

    private record Workload(String name, Tensor output, List<Tensor> inputs) {}
}
