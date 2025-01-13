import React from "react";
import Image from "next/image";

interface LogoElementProps {
  iconSrc: string;
  text: string;
}

const LogoElement: React.FC<LogoElementProps> = ({ iconSrc, text }) => {
  return (
    <div className="flex items-center">
      <Image
        src={iconSrc}
        alt=""
        width={64}
        height={64}
        className="text-sidebar_bright_white"
      />
      <span className="font-sans ml-6 mr-5 text-[20px] subpixel-antialiased font-bold text-sidebar_bright_white whitespace-nowrap">
        {text}
      </span>
    </div>
  );
};

export default LogoElement;
