// components/ContentsInUse.tsx

import React, { useEffect, useRef, useState } from "react";
import UserMessage from "./UserMessage";
import ReasoningText from "./ReasoningText";
import ResultMessage from "./ResultMessage";
import InputField from "./InputField";
import ResultImage from "./ResultsImage";


interface Message {
  id?: string;
  step: number;
  type: "user" | "reasoning" | "result" | "error" | "image";
  start_answer_token?: string;
  stop_answer_token?: string;
  text: string;
  finished?: boolean;
  url?: string;
}

interface ContentsInUseProps {
  activeStep: number;
  messages: Message[];
  onUserMessage: (text: string) => void;
  addMessage: (message: Message, step?: number) => void;
  updateLastMessage: (chunk: string, step?: number) => void; // New prop
}

const ContentsInUse: React.FC<ContentsInUseProps> = ({
  activeStep,
  messages,
  onUserMessage,
  addMessage,
  updateLastMessage,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTo({
        top: containerRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [messages]);

  const handleConfirm = (text: string) => {
    onUserMessage(text);
    addMessage({ step: activeStep, type: "user", text: text });

    const payload = {
      text: text,
      step: activeStep,
    };

    console.log("Sending payload:", payload);
    console.log("Type of 'text':", typeof text);
    console.log("Type of 'activeStep':", typeof activeStep);

    fetch("http://127.0.0.1:8000/user_message", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    })
      .then(async (response) => {
        if (!response.ok) {
          const errorData = await response.json();
          console.error("Error response from server:", errorData);
          throw new Error(`Error: ${response.status} ${response.statusText}`);
        }
        return response.json();
      })
      .then((data) => {
        console.log("Message successfully sent:", data);
      })
      .catch((error) => {
        console.error("Failed to send message:", error);
        addMessage({
          step: activeStep,
          type: "result",
          text: "Failed to send message. Please try again.",
        });
      });
  };

  const pollCurrentMessage = async () => {
    if (isStreaming) return;

    try {
      const response = await fetch("http://127.0.0.1:8000/current_message");
      if (!response.ok) {
        // Handle non-OK responses if necessary
        return;
      }

      const data: Message = await response.json();

      // Check if there's a valid message
      if (data && typeof data.step === "number" && data.type !== undefined) {
        // image datatypes are handled separately
        if (data.type === "image") {
          console.log("Image message detected", data.url, data.text);

          const newMessage: Message = {
            id: data.id,
            step: data.step,
            type: data.type,
            text: data.text,
            url: data.url,
          };
          addMessage(newMessage, data.step);
        } else {
          const messageType =
            data.type === "reasoning" ? "reasoning" : "result";

          // Create a new message and add it to the messages array
          console.log(
            "data in poll current message (step, message type, activeStep)",
            data.step,
            messageType,
            activeStep
          );
          const newMessage: Message = {
            id: data.id,
            step: data.step,
            type: messageType,
            text: "", // Initialize with empty text
          };
          addMessage(newMessage, data.step);
          setIsStreaming(true);

          // Start streaming tokens from /stream_current_message
          streamTokens(data.step);
        }
      }
    } catch (error) {
      console.error("Error polling current_message:", error);
    }
  };

  const streamTokens = async (step: number) => {
    try {
      console.log("Stream current Message called!")
      const response = await fetch(
        "http://127.0.0.1:8000/stream_current_message"
      );
      if (!response.ok) {
        throw new Error(`Error streaming: ${response.statusText}`);
      }
      console.log("Stream Message recieved!")
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("No readable stream.");
      }

      const decoder = new TextDecoder("utf-8");

      let done = false;
      while (!done) {
        console.log("In while loop of stream Tokens")
        const { value, done: readerDone } = await reader.read();
        done = readerDone;
        if (value) {
          const chunk = decoder.decode(value, {stream: true});
          console.log("CHUNK!::", chunk)
          appendToLastMessage(chunk, step);
        }
      }

      setIsStreaming(false);
    } catch (error) {
      console.error("Error streaming tokens:", error);
      setIsStreaming(false);
      addMessage({
        step: activeStep,
        type: "error",
        text: "Error receiving stream. Please try again.",
      });
    }
  };

  const appendToLastMessage = (chunk: string, step?: number) => {
    console.log("Chunk in appendToLastMessage", chunk)
    updateLastMessage(chunk, step);
  };

  // Set up polling every 100 ms
  useEffect(() => {
    if (!isStreaming && pollingIntervalRef.current === null) {
      pollingIntervalRef.current = setInterval(pollCurrentMessage, 1000);
    }

    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    };
  }, [isStreaming]);

  const renderMessage = (message: Message, index: number) => {
    console.log(
      "Rendering message at index",
      index,
      "with text:",
      message.text
    );
    if (message.type === "user") {
      if (!message.text || message.text.trim() === "") {
        return null;
      }

      let style: React.CSSProperties = {};
      let containerClasses = "relative flex justify-end";

      if (index === 0) {
        style = { marginTop: "72px", marginRight: "72px", marginLeft: "144px" };
      } else {
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
      if (message.text !== "") {
        return (
          <div key={index} className={containerClasses} style={style}>
            <ReasoningText text={message.text} />
          </div>
        );
      }
    } else if (message.type === "result") {
      let style: React.CSSProperties = {};
      let containerClasses = "relative flex justify-start";
      style = { marginTop: "16px", marginRight: "144px", marginLeft: "72px" };
      if (message.text !== "") {
      return (
        <div key={index} className={containerClasses} style={style}>
          <ResultMessage text={message.text} />
        </div>
      );}
    } else if (message.type === "image") {
      let style: React.CSSProperties = {};
      let containerClasses = "relative flex justify-start";
      style = { marginTop: "16px", marginRight: "144px", marginLeft: "72px" };

      return <div key={index} className={containerClasses} style={style}>
        <ResultImage url={message.url} altText="Result Image failed to load!"/>
      </div>;
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
