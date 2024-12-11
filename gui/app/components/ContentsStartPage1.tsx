// components/ContentsStartPage1.tsx

import React from "react";
import InputField from "./InputField";
import Image from "next/image";

interface ContentsStartPage1Props {
  onFirstMessageCommit: (message: string) => void;
}

const ContentsStartPage1: React.FC<ContentsStartPage1Props> = ({
  onFirstMessageCommit,
}) => {
  const handleConfirm = (text: string) => {
    // When a message is confirmed, notify Mainwindow
    onFirstMessageCommit(text);
  };

  return (
    <div className="flex flex-col items-center">
      {/*spacer*/}
      <div className="h-full"></div>
      {/*header*/}
      <div className="flex items-center font-sans text-[32px] font-bold">
        <Image
          src="CLR Agent.svg"
          alt=""
          width={48}
          height={48}
          className="mr-[32px]"
        />
        What is your research question?
      </div>
      {/*spacer*/}
      <div className="h-[48px]"></div>

      <InputField
        placeholder="Write your research question!"
        onConfirm={handleConfirm}
      />
      {/*spacer*/}
      <div className="h-full"></div>
    </div>
  );
};

export default ContentsStartPage1;
