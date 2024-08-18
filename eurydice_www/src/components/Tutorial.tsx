import React, { ReactElement, useEffect } from "react";

export default function Tutorial(props: TutorialProps) {
  const [step, setStep] = React.useState(0);

  useEffect(() => {
    props.setEditorText(steps[step].editorText);
  }, []);

  const prevButton =
    step > 0 ? (
      <button
        className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 mx-1 rounded"
        onClick={() => {
          props.setEditorText(steps[step - 1].editorText);
          setStep(step - 1);
        }}
      >
        ← Previous
      </button>
    ) : null;
  const nextButton =
    step < steps.length - 1 ? (
      <button
        className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 mx-1 rounded"
        onClick={() => {
          props.setEditorText(steps[step + 1].editorText);
          setStep(step + 1);
        }}
      >
        Next →
      </button>
    ) : null;

  return (
    <div className="p-1">
      <div className="p-2">{steps[step].text}</div>
      <div className="float-end">
        {prevButton}
        {nextButton}
        <button
          className="border-2 border-blue-500 hover:border-blue-700 py-1 px-2 mx-1 rounded"
          onClick={props.closeTutorial}
        >
          Close tutorial
        </button>
      </div>
    </div>
  );
}

export interface TutorialProps {
  setEditorText: (text: string) => void;
  closeTutorial: () => void;
}

const steps: { text: ReactElement; editorText: string }[] = [
  {
    text: (
      <span>
        Welcome to Eurydice! This tutorial will guide you through the basics of
        using Eurydice.
      </span>
    ),
    editorText: "\\\\\\ Code will appear in the editor. Feel free to edit it!",
  },
  {
    text: (
      <span>
        As you type, your program will be executed, and the output will appear
        on the right. Typing <code>output</code> followed by a die or dice pool
        description will output probabilities for this pool.
      </span>
    ),
    editorText: "output d6",
  },
  {
    text: (
      <span>
        Most standard formulas you might see in a game or D&D manual can be
        typed in as is. Also try different output options on the right!
      </span>
    ),
    editorText: "output 2d6 + 3d8 - 5",
  },
  {
    text: (
      <span>
        Keeping only the lowest or highest die can also be done easily.
      </span>
    ),
    editorText: `output [highest 3 of 4d6]
output [lowest 2 of 3d8]
output {1, 3, 5}@5d10 \\\\\\ Sums the lowest, middle, and highest of 5 d10s.`,
  },
  {
    text: <span>Loops can be used to avoid repetitive code.</span>,
    editorText: `loop SIDES over {4, 6, 8, 10, 12, 20} {
  output 3dSIDES named "3d[SIDES]"
}`,
  },
  {
    text: <span>You can implement abritrary transformations of dice pools with functions.</span>,
    editorText: `\\\\\\ This function returns the highest die in ATTACK that
\\\\\\ does not have an equivalent die in DEFENSE.
function: cancel ATTACK:s with DEFENSE:s {
  \\\\\\ ATTACK is sorted in descending order by default.
  loop VALUE over ATTACK {
    if ![DEFENSE contains VALUE] {
      result: VALUE
    }
  }
  result: 0
}

output [cancel 4d6 with 2d6]
output [highest 1 of 4d6]`,
  }
];
