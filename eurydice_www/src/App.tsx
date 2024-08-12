import CodeMirror, { ViewUpdate } from "@uiw/react-codemirror";
import React from "react";
import { Line } from "react-chartjs-2";
import { parser } from "./grammar/eurydice";
import {
  foldNodeProp,
  foldInside,
  indentNodeProp,
  LRLanguage,
  LanguageSupport,
} from "@codemirror/language";
import { githubLight } from "@uiw/codemirror-theme-github";
import { styleTags, tags as t } from "@lezer/highlight";

// TODO: don't register everything
import { Chart, registerables } from "chart.js";
import { printTree } from "./printTree";
Chart.register(...registerables);

const worker = new Worker(new URL("./worker.js", import.meta.url));

const parserWithMetadata = parser.configure({
  props: [
    styleTags({
      Reference: t.variableName,
      Number: t.number,
      String: t.string,
      Comment: t.blockComment,
      "over result output print": t.keyword,
      "if else loop": t.controlKeyword,
      "function": t.definitionKeyword,
      "( )": t.paren,
      "{ }": t.brace,
      "[ ]": t.squareBracket,
    }),
    indentNodeProp.add({
      Application: (context) =>
        context.column(context.node.from) + context.unit,
    }),
    foldNodeProp.add({
      Application: foldInside,
    }),
  ],
});

const language = LRLanguage.define({
  parser: parserWithMetadata,
  languageData: {
    commentTokens: { block: { open: "\\", close: "\\" } },
  },
});

const languageSupport = new LanguageSupport(language);

function App() {
  const [value, setValue] = React.useState("output 1d6 + 8");
  const [chartData, setChartData] = React.useState(
    new Map<string, Distribution>()
  );
  const [errors, setErrors] = React.useState<string[]>([]);

  worker.onmessage = (event: MessageEvent<EurydiceMessage>) => {
    if (event.data.Err) {
      setErrors([event.data.Err]);
    } else {
      setErrors([]);
      const chartData = new Map<string, Distribution>();
      for (const [key, value] of event.data.Ok!) {
        if (value.Distribution !== undefined) {
          chartData.set(key, value.Distribution);
        } else if (value.Int !== undefined) {
          chartData.set(key, { probabilities: [[value.Int, 1]] });
        } else if (value.List !== undefined) {
          const length = value.List.length;
          chartData.set(key, {
            probabilities: value.List.map((x) => [x, 1.0 / length]),
          });
        }
      }
      setChartData(chartData);
    }
  };

  const onChange = React.useCallback((val: string, viewUpdate: ViewUpdate) => {
    setValue(val);
    worker.postMessage(val);
  }, []);

  // Compute the range of outcomes
  const outcomes = Array.from(chartData.values()).flatMap((dist) => {
    return dist.probabilities.map(([x, _]) => x);
  });
  const min_outcome = Math.min(...outcomes);
  const max_outcome = Math.max(...outcomes);
  const range = Array.from(
    { length: max_outcome - min_outcome + 1 },
    (_, i) => i + min_outcome
  );
  let datasets = [];
  const rng = splitmix32(2);
  for (const entry of chartData.entries()) {
    const [name, dist] = entry;
    const distMap = new Map(dist.probabilities);
    const data = range.map((x) => (distMap.get(x) ?? 0) * 100);
    const color = `rgba(${Math.floor(rng() * 256)}, ${Math.floor(
      rng() * 256
    )}, ${Math.floor(rng() * 256)}, 1.0)`;
    datasets.push({ label: name, data, borderColor: color });
  }

  return (
    <>
      <div className="flex flex-col md:flex-row w-full">
        <div className="w-full md:w-1/2 p-4">
          <CodeMirror
            value={value}
            height="400px"
            onChange={onChange}
            extensions={[languageSupport]}
            theme={githubLight}
          />
        </div>
        <div className="w-full md:w-1/2 p-4">
          <div>{errors}</div>
          <Line
            data={{
              labels: range,
              datasets,
            }}
            options={{
              scales: {
                y: {
                  ticks: {
                    callback: (value) => `${value}%`,
                  },
                },
              },
            }}
          />
          <div><pre>
            {printTree(parserWithMetadata.parse(value), value)}
            </pre>
          </div>
        </div>
      </div>
    </>
  );
}

export default App;

interface EurydiceMessage {
  Ok: [string, DistributionWrapper][] | undefined;
  Err: string | undefined;
}

interface DistributionWrapper {
  Distribution: Distribution | undefined;
  Int: number | undefined;
  List: number[] | undefined;
}

interface Distribution {
  probabilities: [number, number][];
}

/// A simple seedable random number generator
/// https://stackoverflow.com/a/47593316
function splitmix32(a: number) {
  return function () {
    a |= 0;
    a = (a + 0x9e3779b9) | 0;
    let t = a ^ (a >>> 16);
    t = Math.imul(t, 0x21f0aaad);
    t = t ^ (t >>> 15);
    t = Math.imul(t, 0x735a2d97);
    return ((t = t ^ (t >>> 15)) >>> 0) / 4294967296;
  };
}
