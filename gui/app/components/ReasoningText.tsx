// ReasoningText.tsx 
import React from "react";

interface ReasoningTextProps {
  text: string; // the full text from the parent
}

const ReasoningText: React.FC<ReasoningTextProps> = ({ text }) => {
  return (
    <div 
      className="relative bg-reasoning_bg rounded-[16px] max-w-[867px] min-w-[200px] w-auto pb-4" 
      style={{ wordWrap: "break-word" }}
    >
      <div className="absolute subpixel-antialiased top-2 right-5 font-sans text-reasoning_label text-base font-bold">
        Reasoning
      </div>
      <div 
        className="mt-9 ml-8 mr-4 font-sans font-normal subpixel-antialiased text-base text-gray-700 whitespace-pre-wrap" 
        // Added 'whitespace-pre-wrap' class here
      >
        {text}
      </div>
    </div>
  );
};

export default ReasoningText;
