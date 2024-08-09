import init, { greet } from "eurydice_wasm";
import CodeMirror, { ViewUpdate } from "@uiw/react-codemirror";
import React from "react";

function App() {
  // init().then(() => greet());
  const [value, setValue] = React.useState("output 1d6 + 8");
  const onChange = React.useCallback((val: string, viewUpdate: ViewUpdate) => {
    console.log("val:", val);
    setValue(val);
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
          <h2 className="text-xl font-bold mb-2">Right Pane</h2>
          <p>
            This is the content for the right pane (or bottom pane on smaller
            screens).
          </p>
        </div>
      </div>
    </>
  );
}

export default App;
