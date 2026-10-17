import Lean
import Task0053Generated

namespace Task0053.Integer

open Task0053.Generated

set_option maxRecDepth 100000

private def u64Max : Nat := 2^64 - 1
private def i64Max : Int := 2^63 - 1
private def i64Min : Int := -(2^63)

private def magnitude (word : Nat) : Nat := word % 2^31
private def negative (word : Nat) : Bool := decide (2^31 ≤ word % 2^32)
private def exponentField (word : Nat) : Nat := magnitude word / 2^23
private def fraction (word : Nat) : Nat := magnitude word % 2^23

private def saturatingMulU64 (left right : Nat) : Nat :=
  if right ≠ 0 ∧ u64Max / right < left then u64Max else left * right

private def saturatingAddI64 (left right : Int) : Int :=
  if 0 < right ∧ i64Max - right < left then i64Max
  else if right < 0 ∧ left < i64Min - right then i64Min
  else left + right

private def saturatingSubI64 (left right : Int) : Int :=
  saturatingAddI64 left (-right)

private def i64FromMagnitude (value : Nat) (isNegative : Bool) : Int :=
  if isNegative then
    if 2^63 ≤ value then i64Min else -(Int.ofNat value)
  else if 2^63 ≤ value then i64Max else Int.ofNat value

def roundDivEvenNat (value divisor : Nat) : Nat :=
  if divisor = 0 then u64Max
  else
    let quotient := value / divisor
    let remainder := value % divisor
    quotient + if divisor - remainder < remainder ∨
        (remainder = divisor - remainder ∧ quotient % 2 = 1) then 1 else 0

def natDistance (left right : Nat) : Nat :=
  if left ≤ right then right - left else left - right

theorem roundDivEvenNat_error (value divisor : Nat) (positive : 0 < divisor) :
    2 * natDistance (roundDivEvenNat value divisor * divisor) value ≤ divisor := by
  have division := Nat.mod_add_div value divisor
  have remainder_lt := Nat.mod_lt value positive
  have quotient_le : divisor * (value / divisor) ≤ value := by omega
  have next_gt : value < divisor * (value / divisor + 1) := by
    rw [Nat.mul_add]
    simp only [Nat.mul_one]
    omega
  have next_not_le : ¬ divisor * (value / divisor + 1) ≤ value :=
    Nat.not_le_of_gt next_gt
  have quotient_distance :
      value - divisor * (value / divisor) = value % divisor := by omega
  have next_distance :
      divisor * (value / divisor + 1) - value = divisor - value % divisor := by
    rw [Nat.mul_add]
    simp only [Nat.mul_one]
    omega
  unfold roundDivEvenNat
  simp only [ite_eq_right (Nat.ne_of_gt positive)]
  split <;>
    simp [natDistance, Nat.mul_comm, quotient_le, next_not_le,
      quotient_distance, next_distance] <;>
    omega

theorem roundDivEvenNat_monotone (left right divisor : Nat)
    (ordered : left ≤ right) (positive : 0 < divisor) :
    roundDivEvenNat left divisor ≤ roundDivEvenNat right divisor := by
  have divided := Nat.div_le_div_right ordered (c := divisor)
  have left_division := Nat.mod_add_div left divisor
  have right_division := Nat.mod_add_div right divisor
  have left_remainder := Nat.mod_lt left positive
  have right_remainder := Nat.mod_lt right positive
  by_cases same : left / divisor = right / divisor
  · rw [same] at left_division
    have remainders : left % divisor ≤ right % divisor := by omega
    unfold roundDivEvenNat
    simp only [ite_eq_right (Nat.ne_of_gt positive)]
    split <;> split <;> omega
  · have separated : left / divisor + 1 ≤ right / divisor := by omega
    unfold roundDivEvenNat
    simp only [ite_eq_right (Nat.ne_of_gt positive)]
    split <;> split <;> omega
def roundShiftEven (value shift : Nat) : Nat :=
  if shift = 0 then value
  else if 64 ≤ shift then 0
  else roundDivEvenNat value (2^shift)

theorem roundShiftEven_monotone (left right shift : Nat) (ordered : left ≤ right) :
    roundShiftEven left shift ≤ roundShiftEven right shift := by
  unfold roundShiftEven
  split
  · exact ordered
  · split
    · exact Nat.le_refl 0
    · exact roundDivEvenNat_monotone left right (2^shift) ordered
        (Nat.pow_pos (by decide))

private def roundSignedDiv (value : Int) (divisor : Nat) : Int :=
  if divisor = 0 then if value < 0 then i64Min else i64Max
  else i64FromMagnitude (roundDivEvenNat value.natAbs divisor) (decide (value < 0))

private def truncDiv (value : Int) (divisor : Nat) : Int :=
  if divisor = 0 then 0
  else i64FromMagnitude (value.natAbs / divisor) (decide (value < 0))

private def truncRem (value : Int) (divisor : Nat) : Int :=
  if divisor = 0 then value
  else i64FromMagnitude (value.natAbs % divisor) (decide (value < 0))

def saturatingShiftLeft (value shift : Nat) : Nat :=
  if 64 ≤ shift ∨ u64Max / 2^shift < value then u64Max
  else value * 2^shift

theorem saturatingShiftLeft_monotone (left right shift : Nat) (ordered : left ≤ right) :
    saturatingShiftLeft left shift ≤ saturatingShiftLeft right shift := by
  unfold saturatingShiftLeft
  by_cases wide : 64 ≤ shift
  · simp [wide]
  · have narrow : ¬64 ≤ shift := wide
    by_cases left_overflow : u64Max / 2^shift < left
    · have right_overflow : u64Max / 2^shift < right := by omega
      simp [narrow, left_overflow, right_overflow]
    · by_cases right_overflow : u64Max / 2^shift < right
      · simp [narrow, left_overflow, right_overflow]
        exact (Nat.le_div_iff_mul_le (Nat.pow_pos (by decide))).mp (by omega)
      · simp [narrow, left_overflow, right_overflow]
        exact Nat.mul_le_mul_right (2^shift) ordered

def normalizedSignificand (inputMagnitude : Nat) : Nat :=
  fraction (magnitude inputMagnitude) + 2^23

theorem normalizedSignificand_monotone_same_exponent (left right : Nat)
    (ordered : left ≤ right) (right_bound : right < 2^31)
    (same_exponent :
      exponentField (magnitude left) = exponentField (magnitude right)) :
    normalizedSignificand left ≤ normalizedSignificand right := by
  have left_bound : left < 2^31 := Nat.lt_of_le_of_lt ordered right_bound
  have left_magnitude : magnitude left = left := by
    exact Nat.mod_eq_of_lt left_bound
  have right_magnitude : magnitude right = right := by
    exact Nat.mod_eq_of_lt right_bound
  unfold exponentField at same_exponent
  simp only [left_magnitude, right_magnitude] at same_exponent
  have left_division := Nat.mod_add_div left (2^23)
  have right_division := Nat.mod_add_div right (2^23)
  rw [same_exponent] at left_division
  unfold normalizedSignificand fraction
  rw [left_magnitude, right_magnitude]
  omega

def magnitudeToQ48 (inputMagnitude : Nat) : Nat :=
  let word := magnitude inputMagnitude
  let exponent := exponentField word
  let significand := normalizedSignificand word
  if 102 ≤ exponent then
    let shift := exponent - 102
    saturatingShiftLeft significand shift
  else roundShiftEven significand (102 - exponent)

theorem magnitudeToQ48_monotone_same_exponent (left right : Nat)
    (ordered : left ≤ right) (right_bound : right < 2^31)
    (same_exponent :
      exponentField (magnitude left) = exponentField (magnitude right)) :
    magnitudeToQ48 left ≤ magnitudeToQ48 right := by
  have left_bound : left < 2^31 := Nat.lt_of_le_of_lt ordered right_bound
  have left_magnitude : magnitude left = left := Nat.mod_eq_of_lt left_bound
  have right_magnitude : magnitude right = right := Nat.mod_eq_of_lt right_bound
  have exponents : exponentField left = exponentField right := by
    simpa only [left_magnitude, right_magnitude] using same_exponent
  have significands := normalizedSignificand_monotone_same_exponent
    left right ordered right_bound same_exponent
  unfold magnitudeToQ48
  simp only [left_magnitude, right_magnitude]
  rw [exponents]
  split
  · exact saturatingShiftLeft_monotone _ _ _ significands
  · exact roundShiftEven_monotone _ _ _ significands

def rangeQuotientMagnitude (inputMagnitude : Nat) : Nat :=
  roundDivEvenNat (magnitudeToQ48 inputMagnitude) ln2Over128Q48

theorem rangeQuotientMagnitude_monotone_same_exponent (left right : Nat)
    (ordered : left ≤ right) (right_bound : right < 2^31)
    (same_exponent :
      exponentField (magnitude left) = exponentField (magnitude right)) :
    rangeQuotientMagnitude left ≤ rangeQuotientMagnitude right := by
  exact roundDivEvenNat_monotone _ _ ln2Over128Q48
    (magnitudeToQ48_monotone_same_exponent left right ordered right_bound same_exponent)
    (by decide)

theorem rangeQuotientMagnitude_constant_between
    (left word right : Nat)
    (left_ordered : left ≤ word) (right_ordered : word ≤ right)
    (right_bound : right < 2^31)
    (left_exponent :
      exponentField (magnitude left) = exponentField (magnitude word))
    (right_exponent :
      exponentField (magnitude word) = exponentField (magnitude right))
    (endpoints :
      rangeQuotientMagnitude left = rangeQuotientMagnitude right) :
    rangeQuotientMagnitude word = rangeQuotientMagnitude left := by
  have lower := rangeQuotientMagnitude_monotone_same_exponent
    left word left_ordered (Nat.lt_of_le_of_lt right_ordered right_bound) left_exponent
  have upper := rangeQuotientMagnitude_monotone_same_exponent
    word right right_ordered right_bound right_exponent
  omega

private def rawToQ48 (word : Nat) : Int :=
  i64FromMagnitude (magnitudeToQ48 word) (negative word)

private def mulQ32 (left right : Int) : Int :=
  i64FromMagnitude
    (roundShiftEven (saturatingMulU64 left.natAbs right.natAbs) 32)
    (decide ((left < 0) != (right < 0)))

private def packQ31 (initial : Nat) (initialExponent : Int) : Nat :=
  let (value, exponent) :=
    if initial < 0x80000000 then
      (min u64Max (initial * 2), saturatingSubI64 initialExponent 1)
    else if 0x100000000 ≤ initial then
      (roundShiftEven initial 1, saturatingAddI64 initialExponent 1)
    else (initial, initialExponent)
  let rounded := roundShiftEven value 8
  let (significand, exponent) := if rounded = 0x01000000 then
      (rounded / 2, saturatingAddI64 exponent 1)
    else (rounded, exponent)
  let biased := saturatingAddI64 exponent 127
  if 255 ≤ biased then 0x7f800000
  else if biased ≤ 0 then 0
  else biased.natAbs * 2^23 + significand % 2^23

private def expWord (word : Nat) : Nat :=
  let raw := word % 2^32
  let mag := magnitude raw
  let exponent := exponentField raw
  let frac := fraction raw
  if exponent = 255 then
    if frac ≠ 0 then raw ||| 0x00400000
    else if negative raw then 0 else 0x7f800000
  else if mag = 0 ∨ exponent = 0 then 0x3f800000
  else if !negative raw ∧ overflowFirst ≤ mag then 0x7f800000
  else if negative raw ∧ subnormalFirstMagnitude ≤ mag then 0
  else
    let x := rawToQ48 raw
    let quotient := roundSignedDiv x ln2Over128Q48
    let scaledMagnitude := saturatingMulU64 quotient.natAbs ln2Over128Q48
    let scaled := i64FromMagnitude scaledMagnitude (decide (quotient < 0))
    let residualQ48 := saturatingSubI64 x scaled
    let residual := roundSignedDiv residualQ48 0x10000
    let p5 := saturatingAddI64 35791394 (mulQ32 residual 5965232)
    let p4 := saturatingAddI64 178956971 (mulQ32 residual p5)
    let p3 := saturatingAddI64 715827883 (mulQ32 residual p4)
    let p2 := saturatingAddI64 2147483648 (mulQ32 residual p3)
    let p1 := saturatingAddI64 4294967296 (mulQ32 residual p2)
    let polynomial := saturatingAddI64 4294967296 (mulQ32 residual p1)
    let scaleExponent := truncDiv quotient 128
    let rawIndex := truncRem quotient 128
    let (tableIndex, scaleExponent) := if rawIndex < 0 then
        (rawIndex + 128, saturatingSubI64 scaleExponent 1)
      else (rawIndex, scaleExponent)
    if tableIndex < 0 ∨ 128 ≤ tableIndex ∨ polynomial ≤ 0 then 0
    else
      let table := exp2Q31.getD tableIndex.natAbs 0
      let reconstructed := roundShiftEven
        (saturatingMulU64 table polynomial.natAbs) 32
      packQ31 reconstructed scaleExponent

private def addOne (value : Nat) : Nat :=
  if value = 0 then 0x3f800000
  else if value = 0x3f800000 then 0x40000000
  else
    let exponent := value / 2^23 % 256
    if exponent = 0 then 0x3f800000
    else if 255 ≤ exponent then 0x7f800000
    else if 127 < exponent then value
    else
      let significand := value % 2^23 + 2^23
      let sum := 2^23 + roundShiftEven significand (127 - exponent)
      let (sum, outputExponent) := if 2^24 ≤ sum then
          (roundShiftEven sum 1, 128)
        else (sum, 127)
      outputExponent * 2^23 + sum % 2^23

private def dividePositive (numerator denominator : Nat) : Nat :=
  if numerator = 0 then 0
  else
    let numeratorExponent := numerator / 2^23 % 256
    let denominatorExponent := denominator / 2^23 % 256
    if numeratorExponent = 0 ∨ denominatorExponent = 0 then 0
    else if denominatorExponent = 255 then 0
    else if numeratorExponent = 255 ∨ denominator = 0 then 0x7f800000
    else
      let initialExponent : Int := Int.ofNat numeratorExponent - Int.ofNat denominatorExponent
      let initialLeft := numerator % 2^23 + 2^23
      let right := denominator % 2^23 + 2^23
      let (left, exponent) := if initialLeft < right then
          (initialLeft * 2, saturatingSubI64 initialExponent 1)
        else (initialLeft, initialExponent)
      let scaled := left * 2^23
      let quotient := scaled / right
      let remainder := scaled % right
      let significand := quotient + if right - remainder < remainder ∨
          (remainder = right - remainder ∧ quotient % 2 = 1) then 1 else 0
      let (significand, exponent) := if 2^24 ≤ significand then
          (significand / 2, saturatingAddI64 exponent 1)
        else (significand, exponent)
      if exponent < -126 then 0
      else if 127 < exponent then 0x7f800000
      else (exponent + 127).natAbs * 2^23 + significand % 2^23

private def sigmoidWord (word : Nat) : Nat :=
  let raw := word % 2^32
  if exponentField raw = 255 ∧ fraction raw ≠ 0 then raw ||| 0x00400000
  else
    let isNegative := negative raw
    let exponential := expWord (if isNegative then raw else raw ^^^ 0x80000000)
    let denominator := addOne exponential
    let numerator := if isNegative then exponential else 0x3f800000
    dividePositive numerator denominator

-- Public exact-model entry points consumed by the finite certificate checker.
def rangeQuotient (inputMagnitude : Nat) (isNegative : Bool) : Int :=
  let word := inputMagnitude + if isNegative then 2^31 else 0
  roundSignedDiv (rawToQ48 word) ln2Over128Q48

def expCandidate (word : Nat) : Nat := expWord word

def sigmoidCandidate (word : Nat) : Nat := sigmoidWord word

-- These exact executable examples bind the Lean definitions to retained boundary identities.
theorem exp_positive_zero : expWord 0x00000000 = 0x3f800000 := by decide
theorem exp_negative_zero : expWord 0x80000000 = 0x3f800000 := by decide
theorem exp_overflow_boundary : expWord 0x42b17218 = 0x7f800000 := by decide
theorem exp_ftz_boundary : expWord 0xc2aeac50 = 0x00000000 := by decide
theorem sigmoid_positive_infinity : sigmoidWord 0x7f800000 = 0x3f800000 := by decide
theorem sigmoid_negative_infinity : sigmoidWord 0xff800000 = 0x00000000 := by decide

-- The generated certificate kernel-checks the retained numeric width inequalities.
theorem retained_width_certificate : widthCertificateChecks = true :=
  width_certificate_checks

-- Every directed table interval lies strictly inside the retained nearest-even cell.
theorem retained_table_cells : directedTableChecks = true :=
  directed_table_checks

-- MPFR's directed numerators are replayed as exact rational 128th-root enclosures.
theorem retained_table_algebraic_enclosures : algebraicTableChecks = true :=
  algebraic_table_checks

theorem retained_table_words_match_algorithm : directedTableWordsMatch = true :=
  directed_table_words_match

-- The retained finite facts do not supply mathematical exponential semantics.
-- A no-axiom constructive-real/rational-Cauchy exp-series and range-reduction bridge,
-- source-equivalence theorem, and SIGMOID RNE-or-FTZ site theorem remain external gates.

end Task0053.Integer
