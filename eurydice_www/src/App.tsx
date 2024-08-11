import CodeMirror, { ViewUpdate } from "@uiw/react-codemirror";
import React from "react";
import { Line } from "react-chartjs-2";

const worker = new Worker(new URL('./worker.js', import.meta.url));
worker.onmessage = (event) => { console.log(event.data); };

function App() {
  console.log(import.meta.url);
  // worker.postMessage({ a: 1, b: 2 });
  const [value, setValue] = React.useState("output 1d6 + 8");
  const [chartData, setChartData] = React.useState([]);
  const onChange = React.useCallback((val: string, viewUpdate: ViewUpdate) => {
    setValue(val);
    worker.postMessage(val);
  }, []);

  return (
    <>
      <div className="flex flex-col md:flex-row w-full">
        <div className="w-full md:w-1/2 p-4">
          <CodeMirror
            value={value}
            height="400px"
            onChange={onChange}
          />
        </div>
        <div className="w-full md:w-1/2 p-4">
          <Line data={{datasets: chartData, }} />
        </div>
      </div>
    </>
  );
}

export default App;
