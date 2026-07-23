export interface Distribution {
  probabilities: [number, number][];
  enum_name?: string;
  labels?: string[];
}

/**
 * Per-field schema for a tuple output, mirroring the engine's
 * `TupleFieldSchema`. Enum fields carry their member labels; the outcome
 * vectors themselves store raw ints (enum members as ordinals).
 */
export type TupleFieldSchema =
  | { kind: "int" }
  | { kind: "enum"; enumName: string; labels: string[] };

/**
 * A tuple-valued (joint) distribution. Each outcome is a vector of raw ints,
 * one per field, paired with its probability. A tuple element or a tuple list
 * is normalized into this same shape (a single outcome, or a uniform mix).
 */
export interface TupleDistribution {
  fields: TupleFieldSchema[];
  fieldNames?: string[];
  probabilities: [number[], number][];
}
