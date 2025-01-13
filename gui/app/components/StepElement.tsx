import React, { useState } from "react";
import Image from "next/image";

interface StepElementProps {
  id: number;
  isActive: boolean;
  isCompleted: boolean;
  isSelectable: boolean;
  lock_closed: string; // Icon for closed lock
  lock_opened: string; // Icon for opened lock
  lock_opened_white: string; // Icon for opened lock and selected
  check_mark_true: string; // Icon for checkmark true
  check_mark_false: string; // same but for closed
  label: string; // Displayed Label of the StepElement
  onToggle: (id: number) => void;
}

const StepElement: React.FC<StepElementProps> = ({
  id,
  isActive,
  isCompleted,
  isSelectable,
  lock_closed,
  lock_opened,
  lock_opened_white,
  check_mark_true,
  check_mark_false,
  label,
  onToggle,
}) => {
  return (
    <div
      onClick={() => onToggle(id)}
      className="flex items-center justify-between cursor-pointer bg-sidebar_background relative h-[72px] w-full"
    >
      {/*Active Indicator*/}
      <div
        className={`absolute left-0 w-2 h-full rounded-tr-[16px] rounded-br-[16px] ${
          isActive ? "bg-sidebar_gray_white" : "bg-sidebar_background"
        }`}
      ></div>
      {/*Lock Icon*/}
      {isActive && (
        <Image
          src={lock_opened_white}
          alt=""
          width={24}
          height={24}
          className="ml-6"
        />
      )}
      {!isSelectable && (
        <Image
          src={lock_closed}
          alt=""
          width={24}
          height={24}
          className="ml-6"
        />
      )}
      {isSelectable && !isActive && (
        <Image
          src={lock_opened}
          alt=""
          width={24}
          height={24}
          className="ml-6"
        />
      )}

      {/*Text*/}
      <span
        className={`subpixel-antialiased text-[14px] font-semibold font-sans ${
          isActive ? "text-sidebar_bright_white" : "text-sidebar_gray_white"
        } ml-4`}
      >
        {label}
      </span>
      {/*Checkbox*/}
      {isCompleted && (
        <Image
          src={check_mark_true}
          alt=""
          width={20}
          height={20}
          className="mr-8 ml-auto"
        />
      )}
      {!isCompleted && (
        <Image
          src="Check (False).svg"
          alt=""
          width={20}
          height={20}
          className="mr-8 ml-auto"
        />
      )}
    </div>
  );
};

export default StepElement;
