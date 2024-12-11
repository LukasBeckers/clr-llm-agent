// components/MainWindow.tsx

import React, { useState } from "react";
import ContentsStartPage1 from "./ContentsStartPage1";
import ContentsInUse from "./ContentsInUse";

interface MainwindowProps {
  styling_color: string;
}

const Mainwindow: React.FC<MainwindowProps> = ({ styling_color }) => {
  const [initialUserMessage, setInitialUserMessage] = useState<string | null>(
    null
  );

  const handleFirstMessageCommit = (message: string) => {
    // Store the first committed user message and trigger a re-render
    setInitialUserMessage(message);
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
      <div className={`relative z-10 flex ${initialUserMessage === null ? "justify-center items-center h-screen" : "justify-end items-start"}`}>
        {initialUserMessage === null ? (
          // If no message has been committed yet, show the start page
          <ContentsStartPage1 onFirstMessageCommit={handleFirstMessageCommit} />
        ) : (
          // Once the first message is committed, show the ContentsInUse component
          <ContentsInUse
            initialUserMessage={initialUserMessage}
            onUserMessage={(text) => {
              console.log("New user message:", text);
            }}
          />
        )}
      </div>
    </div>
  );
};

export default Mainwindow;
