// components/ReasoningText.tsx
import React, { useState, forwardRef, useImperativeHandle } from "react";

interface ReasoningTextProps {
}

export interface ReasoningTextHandle {
  appendText: (text: string) => void;
}

const ReasoningText = forwardRef<ReasoningTextHandle, ReasoningTextProps>((props, ref) => {
  const [texts, setTexts] = useState<string[]>([]);

  // Expose the appendText method to parent components
  useImperativeHandle(ref, () => ({
    appendText: (text: string) => {
      setTexts((prevTexts) => [...prevTexts, text]);
    },
  }));

  return (
    <div
      className="relative bg-reasoning_bg rounded-[16px] max-w-[867px] w-auto p-4"
      style={{ wordWrap: "break-word" }}
    >
      {/* "Reasoning" Label */}
      <div
        className="absolute top-2 right-5 font-sans text-reasoning_label text-base font-bold"
        // Tailwind's text-base corresponds to 16px
      >
        Reasoning
      </div>

      {/* Render appended texts */}
      <div className="mt-9 ml-8 mr-4">
        {texts.map((text, index) => (
          <p
            key={index}
            className="mt-[36px] ml-[32px] mb-[16px] mr-[16px] text-base text-gray-700"
            // Adjust text styles as needed
          >
            {text}
          </p>
        ))}
      </div>
    </div>
  );
});

export default ReasoningText;
