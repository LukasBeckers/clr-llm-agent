// components/InputField.tsx
import React from "react";
import Image from "next/image";

interface InputFieldProps {
  placeholder?: string;
  onConfirm?: (text: string) => void;
}

const InputField: React.FC<InputFieldProps> = ({
  placeholder = "Write a message to the assistant, write an empty message to continue!",
  onConfirm = (text: string) => {
    console.log("Submitted Text:", text);
  },
}) => {
  const textareaRef = React.useRef<HTMLTextAreaElement>(null);

  const handleInput = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      // Reset height to auto to calculate new scrollHeight properly
      textarea.style.height = "auto";
      // Set height based on content, constrained by max-height via Tailwind
      textarea.style.height = `${textarea.scrollHeight}px`;
    }
  };

  const handleConfirmClick = () => {
    const textarea = textareaRef.current;
    if (textarea) {
      const text = textarea.value.trim();
      if (text !== "") {
        onConfirm(text);
        textarea.value = "";
        // Reset height to the minimum height
        textarea.style.height = "72px";
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault(); // Prevent inserting a newline
      handleConfirmClick();
    }
  };

  return (
    <div className="relative subpixel-antialiased flex font-sans w-[768px]">
      <textarea
        ref={textareaRef}
        onInput={handleInput}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        className={`
          w-full
          h-[72px]
          min-h-[72px]
          max-h-[400px]
          bg-inputfield_main_color
          text-inputfield_text_color
          p-4
          pr-10
          rounded-[16px]
          focus:outline-none
          overflow-y-auto
          resize-none
        `}
        style={{ lineHeight: "1.5" }}
      />
      {/* Confirm Message Icon */}
      <button
        onClick={handleConfirmClick}
        className="absolute bottom-4 right-4 cursor-pointer focus:outline-none"
        aria-label="Confirm Message"
      >
        <Image
          src="/confirm_message.svg" // Ensure the path is correct
          alt="Confirm Message"
          width={24}
          height={24}
        />
      </button>
    </div>
  );
};

export default InputField;
