import { describe, expect, it } from "vitest";

import { applyTextEdits, TextEdit } from "./diagnostics";

describe("structured diagnostics", () => {
  it("applies multi-edit fixes without shifting later ranges", () => {
    const edit = (start: number, end: number, replacement: string): TextEdit => ({
      range: { source: 0, range: { start, end } },
      replacement,
    });

    expect(
      applyTextEdits("output 1 == 1;", [
        edit(10, 11, ""),
        edit(13, 14, ""),
      ]),
    ).toBe("output 1 = 1");
  });
});
