import React, { useState } from "react";
import Image from "next/image";

interface SettingsProps {
  id: number;
  isActive: boolean;
  icon: string;
  label: string;
  onToggle: (id: number) => void;
}

const SettingsElement: React.FC<SettingsProps> = ({
  id,
  isActive,
  icon,
  label,
  onToggle,
}) => {
  return (
    <div
      onClick={() => onToggle(id)}
      className="flex items-center justify-center cursor-pointer bg-sidebar_background relative h-[72px] w-full"
    >
      {/*Active Indicator*/}
      <div
        className={`absolute left-0 w-2 h-full rounded-tr-[16px] rounded-br-[16px] ${
          isActive ? "bg-sidebar_gray_white" : "bg-sidebar_background"
        }`}
      ></div>
      {/*Settings Icon*/}
      <Image src={icon} alt="" width={24} height={24} className="" />
      {/*Text*/}
      <span
        className={`subpixel-antialiased text-[14px] font-semibold font-sans ${
          isActive ? "text-sidebar_bright_white" : "text-sidebar_gray_white"
        } ml-4`}
      >
        {label}
      </span>
    </div>
  );
};

export default SettingsElement;
