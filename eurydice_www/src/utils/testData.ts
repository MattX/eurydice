import {
  FieldSchema,
  ScalarDistribution,
} from "../util";

export function scalarDistribution(
  entries: [number, number][],
  field: FieldSchema = { kind: "int" }
): ScalarDistribution {
  return {
    fields: [field],
    entries: entries.map(([outcome, probability]) => [
      [outcome],
      probability,
    ]),
  };
}
