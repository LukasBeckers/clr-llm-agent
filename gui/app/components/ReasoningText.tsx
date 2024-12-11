// components/ReasoningText.tsx
import React, { useState, forwardRef, useImperativeHandle } from "react";

interface ReasoningTextProps {
  text: string; // initial text to be displayed
}

export interface ReasoningTextHandle {
  addText: (text: string) => void; // method to append new text
}

const ReasoningText = forwardRef<ReasoningTextHandle, ReasoningTextProps>(
  ({ text }, ref) => {
    const [currentText, setCurrentText] = useState<string>(text);

    // Expose the addText method to parent components
    useImperativeHandle(ref, () => ({
      addText: (newText: string) => {
        setCurrentText((prevText) => prevText + newText);
      },
    }));

    return (
      <div
        className="relative bg-reasoning_bg rounded-[16px]  max-w-[867px] min-w-[200px] w-auto pb-4"
        style={{ wordWrap: "break-word" }}
      >
        {/* "Reasoning" Label */}
        <div
          className="absolute subpixel-antialiased top-2 right-5 font-sans text-reasoning_label text-base font-bold"
        >
          Reasoning
        </div>

        {/* Display the entire reasoning text */}
        <div className="mt-9 ml-8 mr-4 font-sans font-normal subpixel-antialiased text-base text-gray-700">
          {currentText}
        </div>
      </div>
    );
  }
);

export default ReasoningText;
