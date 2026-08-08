/**
 * Per-field output schema, mirroring the engine's
 * `FieldSchema`. Enum fields carry their member labels; the outcome
 * vectors themselves store raw ints (enum members as ordinals).
 */
export type FieldSchema =
  | { kind: "int" }
  | { kind: "enum"; labels: string[] };

/**
 * The canonical representation of every output distribution. A scalar has one
 * field and one value per outcome; a tuple has two or more.
 */
export interface Distribution {
  fields: FieldSchema[];
  fieldNames?: string[];
  probabilities: [number[], number][];
}

/**
 * A one-field refinement of the canonical distribution shape. Keeping scalar
 * outputs in their canonical representation avoids maintaining a second,
 * chart-specific copy of field metadata.
 */
export type ScalarDistribution = Omit<
  Distribution,
  "fields" | "probabilities"
> & {
  fields: [FieldSchema];
  probabilities: [[number], number][];
};

export type NamedDistribution = [string, Distribution];
export type NamedScalarDistribution = [string, ScalarDistribution];

export function isScalarDistribution(
  distribution: Distribution
): distribution is ScalarDistribution {
  return distribution.fields.length === 1;
}

export function isNamedScalarDistribution(
  output: NamedDistribution
): output is NamedScalarDistribution {
  return isScalarDistribution(output[1]);
}
