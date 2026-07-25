import {
  FieldSchema,
  ScalarDistribution,
} from "../util";

export function scalarDistribution(
  probabilities: [number, number][],
  field: FieldSchema = { kind: "int" }
): ScalarDistribution {
  return {
    fields: [field],
    probabilities: probabilities.map(([outcome, probability]) => [
      [outcome],
      probability,
    ]),
  };
}
