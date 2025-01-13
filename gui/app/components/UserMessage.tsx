import React from "react";

interface UserMessageProps {
  text: string;
}

const UserMessage: React.FC<UserMessageProps> = ({ text }) => {
  return (
    <div
      className="relative subpixel-antialiased bg-user_bg rounded-[16px] max-w-[867px] w-auto pb-4 min-w-[200px]"
      // Ensure a minimum width to fit the label and padding
      style={{ wordWrap: "break-word" }}
    >
      {/* "Message" Label */}
      <div
        className="absolute subpixel-antialiased top-2 right-5 font-sans text-user_label text-base font-bold ml-4"
        // Tailwind's text-base corresponds to 16px
      >
        User Message
      </div>

      {/* Render appended texts */}
      <div className="mt-9 ml-8 mr-4 font-sans subpixel-antialiased text-normal">
        {text}
      </div>
    </div>
  );
};

export default UserMessage;
