import { FieldSchema, ScalarDistribution } from "../util";

export function scalarDistribution(
  entries: [number, number][],
  schema: FieldSchema = { kind: "int" },
): ScalarDistribution {
  return {
    fields: [{ schema }],
    entries: entries.map(([outcome, probability]) => [[outcome], probability]),
  };
}
