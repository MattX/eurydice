import { FieldSchema, ScalarDistribution } from "../util";

export function scalarDistribution(
  entries: [number, number][],
  schema: FieldSchema = { kind: "int" },
): ScalarDistribution {
  return {
    fields: [{ schema }],
    values: entries.map(([outcome]) => outcome),
    probabilities: entries.map(([, probability]) => probability),
  };
}
