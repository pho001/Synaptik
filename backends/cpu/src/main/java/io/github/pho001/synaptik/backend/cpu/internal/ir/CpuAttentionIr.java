package io.github.pho001.synaptik.backend.cpu.internal.ir;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable code-shaping identity for one direct scaled-dot-product-attention row body.
 *
 * <p>Semantic input roles retain their occurrence positions even when two or three roles share one
 * read-only boundary. Concrete ranks, extents, scale, layouts, addresses, range bounds, and
 * workspace identity remain cold invocation geometry.
 *
 * @param queryType exact query representation
 * @param keyType exact key representation
 * @param valueType exact value representation
 * @param resultType promoted output and optional weights representation
 * @param masked whether semantic role three is a canonical BOOL mask
 * @param causal whether top-left causal eligibility is active
 * @param outputCount one for output only or two for output plus weights
 * @param roleBoundaryPositions query, key, value, and optional mask positions in the ordered
 *     unique-input boundary prefix
 * @param boundaryTypes ordered unique-input then output boundary types
 * @param boundaryAccesses ordered direct read/write access identities
 */
public record CpuAttentionIr(
    DataType queryType,
    DataType keyType,
    DataType valueType,
    DataType resultType,
    boolean masked,
    boolean causal,
    int outputCount,
    List<Integer> roleBoundaryPositions,
    List<DataType> boundaryTypes,
    List<CpuAccessPlan> boundaryAccesses)
    implements CpuPortableKernelIr {

  /**
   * Creates a checked direct-scalar attention identity.
   *
   * @throws NullPointerException if a required type, role list, boundary-type list, or access list
   *     is null
   * @throws IllegalArgumentException if the types, role-to-boundary mapping, or ordered read/write
   *     boundary facts do not describe one supported attention occurrence
   */
  public CpuAttentionIr {
    Objects.requireNonNull(queryType, "queryType");
    Objects.requireNonNull(keyType, "keyType");
    Objects.requireNonNull(valueType, "valueType");
    Objects.requireNonNull(resultType, "resultType");
    roleBoundaryPositions = List.copyOf(roleBoundaryPositions);
    boundaryTypes = List.copyOf(boundaryTypes);
    boundaryAccesses = List.copyOf(boundaryAccesses);
    int roles = masked ? 4 : 3;
    int uniqueInputs = boundaryTypes.size() - outputCount;
    if (!floating(queryType)
        || !floating(keyType)
        || !floating(valueType)
        || !floating(resultType)
        || outputCount < 1
        || outputCount > 2
        || roleBoundaryPositions.size() != roles
        || uniqueInputs < 1
        || boundaryAccesses.size() != boundaryTypes.size()
        || roleBoundaryPositions.stream().anyMatch(p -> p == null || p < 0 || p >= uniqueInputs)
        || boundaryAccesses.subList(0, uniqueInputs).stream()
            .anyMatch(a -> a.accessKind() != CpuAccessPlan.AccessKind.READ)
        || boundaryAccesses.subList(uniqueInputs, boundaryAccesses.size()).stream()
            .anyMatch(a -> a.accessKind() != CpuAccessPlan.AccessKind.WRITE)
        || boundaryTypes.get(roleBoundaryPositions.get(0)) != queryType
        || boundaryTypes.get(roleBoundaryPositions.get(1)) != keyType
        || boundaryTypes.get(roleBoundaryPositions.get(2)) != valueType
        || masked && boundaryTypes.get(roleBoundaryPositions.get(3)) != DataType.BOOL
        || boundaryTypes.subList(uniqueInputs, boundaryTypes.size()).stream()
            .anyMatch(t -> t != resultType)) {
      throw new IllegalArgumentException("attention IR facts disagree");
    }
  }

  /**
   * Encodes every generated-code fact and no cold geometry.
   *
   * @return a new immutable generic kernel identity for the same attention specialization; never
   *     {@code null}
   */
  public CpuKernelIr encodedKernelIr() {
    var values = new ArrayList<CpuKernelIr.Value>(boundaryTypes.size());
    int uniqueInputs = boundaryTypes.size() - outputCount;
    for (int i = 0; i < boundaryTypes.size(); i++)
      values.add(
          new CpuKernelIr.Value(
              i,
              boundaryTypes.get(i),
              i < uniqueInputs ? CpuKernelIr.Value.Kind.INPUT : CpuKernelIr.Value.Kind.OUTPUT,
              boundaryAccesses.get(i)));
    var stores = new ArrayList<CpuKernelIr.Store>(outputCount);
    for (int i = 0; i < outputCount; i++) stores.add(new CpuKernelIr.Store(uniqueInputs + i, i));
    String identity =
        "attention:q="
            + queryType
            + ":k="
            + keyType
            + ":v="
            + valueType
            + ":result="
            + resultType
            + ":masked="
            + masked
            + ":causal="
            + causal
            + ":outputs="
            + outputCount
            + ":roles="
            + roleBoundaryPositions
            + ":acc="
            + (resultType == DataType.FLOAT64 ? "F64" : "F32")
            + ":scratch=ATTENTION_ROW_STATE:realization=DIRECT_SCALAR";
    return new CpuKernelIr(
        values, List.of(), new CpuKernelIr.Loop("start", "end"), stores, identity);
  }

  /**
   * Returns the deterministic structural key.
   *
   * @return the non-null structural key derived from the complete code-shaping identity
   */
  @Override
  public String structuralKey() {
    return encodedKernelIr().structuralKey();
  }

  private static boolean floating(DataType type) {
    return type == DataType.BFLOAT16 || type == DataType.FLOAT32 || type == DataType.FLOAT64;
  }
}
