// components/ContentsInUse.tsx
import React, { useEffect, useState, useRef } from "react";
import UserMessage from "./UserMessage";
import ReasoningText from "./ReasoningText";
import ResultMessage from "./ResultMessage";
import InputField from "./InputField";
import Image from "next/image";
// Removed unused import: import Result from "postcss/lib/result";

// Mock function to check for existing chat history
async function checkChatHistory() {
  // In a real app, fetch or load local storage / backend data here
  return [];
}

interface ContentsInUseProps {
  messages: { type: "user" | "reasoning" | "result"; text: string }[];
  onUserMessage: (text: string) => void; // Callback when new user messages are confirmed
  addMessage: (message: {
    type: "user" | "reasoning" | "result";
    text: string;
  }) => void;
}

const ContentsInUse: React.FC<ContentsInUseProps> = ({
  messages,
  onUserMessage,
  addMessage
}) => {
  // State to hold chat messages
  // Store objects with type: 'user' | 'reasoning' | 'result'
  // and their corresponding text.

  // Container ref for scrolling
  const containerRef = useRef<HTMLDivElement>(null);

  const handleConfirm = (text: string) => {
    // Callback
    onUserMessage(text);
    // Add User Message to the messages array
    addMessage({ type: "user", text: text });
  };

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages]);

  const renderMessage = (
    message: { type: string; text: string },
    index: number
  ) => {
    if (message.type === "user") {
      let style: React.CSSProperties = {};
      let containerClasses = "relative flex justify-end";

      if (index === 0) {
        // First user message
        style = { marginTop: "72px", marginRight: "72px", marginLeft: "144px" };
      } else {
        // Subsequent user messages
        style = { marginTop: "16px", marginRight: "72px", marginLeft: "144px" };
      }

      return (
        <div key={index} className={containerClasses} style={style}>
          <UserMessage text={message.text} />
        </div>
      );
    } else if (message.type === "reasoning") {
      let style: React.CSSProperties = {};
      let containerClasses = "relative flex justify-start";

      style = { marginTop: "16px", marginRight: "144px", marginLeft: "72px" };

      return (
        <div key={index} className={containerClasses} style={style}>
          <ReasoningText ref={null} text={message.text} />
        </div>
      );
    } else if (message.type === "result") {
      let style: React.CSSProperties = {};
      let containerClasses = "relative flex justify-start";

      style = { marginTop: "16px", marginRight: "144px", marginLeft: "72px" };

      return (
        <div key={index} className={containerClasses} style={style}>
          <ResultMessage ref={null} text={message.text} />
        </div>
      );
    }

    return null;
  };

  return (
    <div className="relative flex-1 h-screen bg-background_main overflow-hidden">
      {/* Main content area with scrollable messages */}
      <div
        className="relative z-10 h-screen flex flex-col overflow-auto mb-[14px] pb-[400px]"
        ref={containerRef}
        style={{
          // Add top and bottom gradient fades:
          scrollSnapType: "y proximity",
        }}
      >
        {/* messages */}
        {messages.map((m, i) => renderMessage(m, i))}
      </div>

      {/* InputField fixed at bottom-right */}
      <div
        className="fixed z-20"
        style={{
          bottom: "72px",
          right: "72px",
        }}
      >
        <InputField onConfirm={handleConfirm} />
      </div>
    </div>
  );
};

export default ContentsInUse;
