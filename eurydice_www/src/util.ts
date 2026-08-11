/**
 * Per-field display schema, mirroring the engine's `FieldSchema`. Categorical
 * fields carry their labels; the flat outcome storage holds raw ints (category
 * members as ordinals). The engine uses this representation for all-symbol
 * fields and mixed number/symbol fields.
 */
export type FieldSchema =
  { kind: "int" } | { kind: "categorical"; labels: string[] };

/**
 * One output field, mirroring the engine's `Field`: how its values are
 * rendered, plus the name an `output ... labeled` clause gave it. Keeping the
 * name beside the schema is what guarantees they agree on the field count.
 */
export interface Field {
  name?: string;
  schema: FieldSchema;
}

/**
 * The canonical representation of every output distribution. A scalar has one
 * field and one value per outcome; a tuple has two or more. This is the shape
 * the engine serializes, used as-is with no normalization step.
 */
export interface Distribution {
  fields: Field[];
  /** Outcomes concatenated in row-major order. */
  values: number[];
  /** One probability per outcome in `values`. */
  probabilities: number[];
}

/**
 * A one-field refinement of the canonical distribution shape. Keeping scalar
 * outputs in their canonical representation avoids maintaining a second,
 * chart-specific copy of field metadata.
 */
export interface ScalarDistribution extends Distribution {
  fields: [Field];
}

/** The value of one field in one outcome. */
export function outcomeValue(
  distribution: Distribution,
  outcome: number,
  field: number,
): number {
  return distribution.values[outcome * distribution.fields.length + field];
}

/** Copies one outcome out of the distribution's flat row-major storage. */
export function outcomeValues(
  distribution: Distribution,
  outcome: number,
): number[] {
  const arity = distribution.fields.length;
  const start = outcome * arity;
  return distribution.values.slice(start, start + arity);
}

export type NamedDistribution = [string, Distribution];
export type NamedScalarDistribution = [string, ScalarDistribution];

export function isScalarDistribution(
  distribution: Distribution,
): distribution is ScalarDistribution {
  return distribution.fields.length === 1;
}

export function isNamedScalarDistribution(
  output: NamedDistribution,
): output is NamedScalarDistribution {
  return isScalarDistribution(output[1]);
}
