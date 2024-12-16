// components/ResultMessage.tsx
import React, { useState, forwardRef, useImperativeHandle } from "react";

interface ResultMessageProps {
  text: string; // initial text to be displayed
}

const ResultMessage: React.FC<ResultMessageProps> = (
  ({ text }) => {
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
        <div className="mt-9 ml-8 mr-4 font-sans font-normal subpixel-antialiased text-base text-gray-700 whitespace-pre-wrap">
          {text}
        </div>
      </div>
    );
  }
);

export default ResultMessage;
