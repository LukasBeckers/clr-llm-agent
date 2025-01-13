// components/MainWindow.tsx

import React, { useState } from "react";
import ContentsStartPage1 from "./ContentsStartPage1";
import ContentsInUse from "./ContentsInUse";

interface MainwindowProps {
  activeStep: number;
}

const Mainwindow: React.FC<MainwindowProps> = ({ activeStep }) => {
  const [initialUserMessage, setInitialUserMessage] = useState<string | null>(
    null
  );

  console.log("ActiveStep in Main Window", activeStep);

  const handleFirstMessageCommit = (message: string) => {
    if (message !== "Go on, do your thing!") {
      // Store the first committed user message and trigger a re-render
      setInitialUserMessage(message);
      setStepMessages((prev) => ({
        ...prev,
        0: {
          ...prev[0],
          messages: [
            ...prev[0].messages,
            { type: "user", text: message, step: 0 },
          ],
        },
      }));
      const payload = {
        text: message,
        step: activeStep,
      };

      // Log the payload to verify its structure and types
      console.log("Sending payload:", payload);
      console.log("Type of 'text':", typeof message);
      console.log("Type of 'activeStep':", typeof activeStep);

      // Send the POST request
      fetch("http://127.0.0.1:8000/user_message", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });
    }
  };

  const updateLastMessageInStep = (stepId: number, chunk: string) => {
    setStepMessages((prev) => {
      const stepData = prev[stepId];
      if (!stepData || stepData.messages.length === 0) return prev;

      const newMessages = [...stepData.messages];
      const lastIndex = newMessages.length - 1;

      // Create a new message object, ensuring a new reference:
      newMessages[lastIndex] = {
        ...newMessages[lastIndex],
        text: newMessages[lastIndex].text + chunk,
      };

      // Return a new state object with updated references
      const newState = {
        ...prev,
        [stepId]: {
          ...stepData,
          messages: newMessages,
        },
      };

      return newState;
    });
  };

  const [stepMessages, setStepMessages] = useState<{
    [stepId: number]: {
      initialUserMessage?: string;
      messages: {
        type: "user" | "reasoning" | "result" | "error" | "image";
        text: string;
        step: number;
      }[];
    };
  }>({
    0: { messages: [] },
    1: { messages: [] },
    2: { messages: [] },
    3: { messages: [] },
    4: { messages: [] },
    5: { messages: [] },
  });
  console.log("stepMessages changed:", stepMessages);

  const addMessageToStep = (
    stepId: number,
    newMessage: { type: string; text: string }
  ) => {
    setStepMessages((prev) => ({
      ...prev,
      [stepId]: {
        ...prev[stepId],
        messages: [...prev[stepId].messages, newMessage],
      },
    }));
  };

  return (
    <div className="relative flex-1 min-h-screen bg-background_main overflow-hidden">
      {/* Ellipse */}
      <div
        className={`absolute z-0 transform rotate-[-30deg] bg-background_style_color_start_page opacity-50`}
        style={{
          width: "2028.7px",
          height: "1011.26px",
          left: "-250.01px",
          top: "10px",
          borderRadius: "50% / 50%",
          filter: "blur(200px)",
        }}
        aria-hidden="true"
      ></div>

      {/* Mainwindow Content */}
      <div
        className={`relative z-10 flex ${
          initialUserMessage === null
            ? "justify-center items-center h-screen"
            : "justify-end items-start"
        }`}
      >
        {initialUserMessage === null ? (
          // If no message has been committed yet, show the start page
          <ContentsStartPage1 onFirstMessageCommit={handleFirstMessageCommit} />
        ) : (
          // Once the first message is committed, show the ContentsInUse component
          <ContentsInUse
            activeStep={activeStep}
            messages={stepMessages[activeStep]?.messages || []}
            onUserMessage={(text) => {
              console.log("New user message:", text);
            }}
            addMessage={(newMessage, step? : number) => {
              const targetStep = step !== undefined ? step: activeStep;
              addMessageToStep(targetStep, newMessage)
            }}
            updateLastMessage={(chunk, step? : number) => {
              const targetStep = step !== undefined ? step: activeStep;
              updateLastMessageInStep(targetStep, chunk)
            }}
          />
        )}
      </div>
    </div>
  );
};

export default Mainwindow;
