/**
 * Per-field display schema, mirroring the engine's `FieldSchema`. Categorical
 * fields carry their labels; the outcome vectors themselves store raw ints
 * (category members as ordinals). The engine uses this representation for
 * all-symbol fields and mixed number/symbol fields.
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
  entries: [number[], number][];
}

/**
 * A one-field refinement of the canonical distribution shape. Keeping scalar
 * outputs in their canonical representation avoids maintaining a second,
 * chart-specific copy of field metadata.
 */
export interface ScalarDistribution extends Distribution {
  fields: [Field];
  entries: [[number], number][];
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
