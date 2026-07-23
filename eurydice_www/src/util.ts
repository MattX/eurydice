export interface ScalarDistribution {
  probabilities: [number, number][];
  enum_name?: string;
  labels?: string[];
}

/**
 * Per-field output schema, mirroring the engine's
 * `FieldSchema`. Enum fields carry their member labels; the outcome
 * vectors themselves store raw ints (enum members as ordinals).
 */
export type FieldSchema =
  | { kind: "int" }
  | { kind: "enum"; enumName: string; labels: string[] };

/**
 * The canonical representation of every output distribution. A scalar has one
 * field and one value per outcome; a tuple has two or more.
 */
export interface Distribution {
  fields: FieldSchema[];
  fieldNames?: string[];
  probabilities: [number[], number][];
}

/** Projects a canonical one-field distribution into the existing chart view. */
export function asScalarDistribution(
  distribution: Distribution
): ScalarDistribution {
  if (distribution.fields.length !== 1) {
    throw new Error("expected a one-field distribution");
  }
  const scalar: ScalarDistribution = {
    probabilities: distribution.probabilities.map(([values, probability]) => [
      values[0],
      probability,
    ]),
  };
  const field = distribution.fields[0];
  if (field.kind === "enum") {
    scalar.enum_name = field.enumName;
    scalar.labels = field.labels;
  }
  return scalar;
}
