// components/ResultMessage.tsx
import React, { useState, forwardRef, useImperativeHandle } from "react";

interface ResultMessageProps {
  text: string; // initial text to be displayed
}

export interface ResultMessageHandle {
  addText: (text: string) => void; // method to append new text
}

const ResultMessage = forwardRef<ResultMessageHandle, ResultMessageProps>(
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
        className="relative bg-result_bg rounded-[16px]  max-w-[867px] min-w-[200px] w-auto pb-4"
        style={{ wordWrap: "break-word" }}
      >
        {/* "Results" Label */}
        <div
          className="absolute subpixel-antialiased top-2 right-5 font-sans text-result_label text-base font-bold"
        >
          Results
        </div>

        {/* Display the entire result text */}
        <div className="mt-9 ml-8 mr-4 font-sans font-normal subpixel-antialiased text-base text-gray-700">
          {currentText}
        </div>
      </div>
    );
  }
);

export default ResultMessage;
