package io.github.pho001.synaptik.backend.cpu.internal.route.nativeops.onednn;

import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuKernelSpecialization;
import io.github.pho001.synaptik.backend.cpu.internal.cache.CpuLoweringFingerprint;
import io.github.pho001.synaptik.backend.cpu.internal.codegen.emit.CpuGeneratedKernel;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuKernelIr;
import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuPointwiseOpcode;
import io.github.pho001.synaptik.backend.cpu.internal.prepare.CpuPartitionPreparationPlan;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.lang.invoke.MethodHandles;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * Cold recipe for the existing generated CAST at a native low result boundary. Native arithmetic
 * writes declared F32 storage; this ordinary scalar emitter performs the only final RNE narrowing.
 * Concrete dimensions, strides, segments and origins remain outside the generated class identity.
 */
public final class CpuOneDnnOutputNarrowing {
    private final CpuOneDnnProjection projection;
    private final CpuKernelIr ir;
    private final CpuKernelSpecialization specialization;
    private final long elementCount;

    private CpuOneDnnOutputNarrowing(CpuOneDnnProjection projection) {
        this.projection = projection;
        var tensor = projection.spec().output();
        int rank = tensor.dimensions().size();
        var roles = Collections.nCopies(rank, CpuAccessPlan.AxisRole.STRIDED);
        var read = new CpuAccessPlan(CpuAccessPlan.AccessKind.READ,
                CpuAccessPlan.Regime.GENERAL_ODOMETER, rank, roles, 0);
        var write = new CpuAccessPlan(CpuAccessPlan.AccessKind.WRITE,
                CpuAccessPlan.Regime.GENERAL_ODOMETER, rank, roles, 0);
        ir = new CpuKernelIr(List.of(
                new CpuKernelIr.Value(0, DataType.FLOAT32, CpuKernelIr.Value.Kind.INPUT, read),
                new CpuKernelIr.Value(1, projection.logicalOutputType(), CpuKernelIr.Value.Kind.OUTPUT, write)),
                List.of(new CpuKernelIr.Instruction(CpuPointwiseOpcode.CAST, List.of(0), 1)),
                new CpuKernelIr.Loop("start", "end"), List.of(new CpuKernelIr.Store(1, 0)));
        specialization = new CpuKernelSpecialization(CpuLoweringFingerprint.fromHex(ir.structuralKey()),
                CpuPartitionPreparationPlan.ExecutionStrategy.SCALAR,
                List.of(DataType.FLOAT32, projection.logicalOutputType()),
                List.of(CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT,
                        CpuKernelSpecialization.CarrierAccess.MEMORY_SEGMENT),
                0, -1, List.of(), false, CpuKernelSpecialization.LOW_PRECISION_ARITHMETIC_SCHEMA);
        long count = 1;
        for (long extent : tensor.dimensions()) count = Math.multiplyExact(count, extent);
        elementCount = count;
    }

    /**
     * Constructs a handle-free final CAST recipe without generating code or allocating run storage.
     * @param projection non-null selected projection with a low logical result and F32 native result
     * @return immutable recipe for the ordinary artifact store and cold binding
     * @throws NullPointerException if projection is null
     * @throws IllegalArgumentException if this projection has no output narrowing
     * @throws ArithmeticException if its exact logical count overflows
     */
    public static CpuOneDnnOutputNarrowing from(CpuOneDnnProjection projection) {
        Objects.requireNonNull(projection, "projection");
        if (!projection.requiresOutputNarrowing()) {
            throw new IllegalArgumentException("oneDNN direct result has no final CAST");
        }
        return new CpuOneDnnOutputNarrowing(projection);
    }

    /** @return exact ordinary generated CAST specialization, with no baked native resources */
    public CpuKernelSpecialization specialization() { return specialization; }
    /** @return existing pointwise CAST IR, not an independent conversion implementation */
    public CpuKernelIr ir() { return ir; }
    /** @return checked logical result count, never a physical span including gaps */
    public long elementCount() { return elementCount; }

    /**
     * Binds the ordinary pointwise numeric ABI once; no allocation occurs in the returned call.
     * @param artifact non-null generated CAST with this exact specialization
     * @param source borrowed F32 stage sliced to the native descriptor span at origin zero
     * @param destination borrowed writable logical result sliced to its exact span at origin zero
     * @return fixed synchronous call retaining these run-owned segments without ownership transfer
     * @throws IllegalArgumentException if the artifact or exact segment spans disagree
     */
    Runnable bind(CpuGeneratedKernel artifact, MemorySegment source, MemorySegment destination) {
        if (!artifact.specialization().equals(specialization)
                || source.byteSize() != projection.spec().output().byteSize()
                || destination.byteSize() != projection.logicalOutputByteSize()
                || destination.isReadOnly()) {
            throw new IllegalArgumentException("oneDNN output CAST artifact or spans disagree");
        }
        var tensor = projection.spec().output();
        int rank = tensor.dimensions().size();
        // Ordinary pointwise pack: extents, initial coordinates, bases, strides, inner positions/sizes.
        long[] geometry = new long[4 * rank + 6];
        for (int axis = 0; axis < rank; axis++) {
            geometry[axis] = tensor.dimensions().get(axis);
            geometry[2 * rank + 2 + axis] = tensor.strides().get(axis);
            geometry[3 * rank + 2 + axis] = tensor.strides().get(axis);
        }
        geometry[4 * rank + 4] = 1;
        geometry[4 * rank + 5] = 1;
        MethodHandle call = MethodHandles.insertArguments(artifact.entryPoint(), 0,
                source, destination, geometry);
        return () -> {
            try { call.invokeExact(0L, elementCount); }
            catch (RuntimeException | Error failure) { throw failure; }
            catch (Throwable failure) {
                throw new IllegalStateException("generated oneDNN output CAST failed", failure);
            }
        };
    }
}
