import React, { useState } from "react";
import LogoElement from "./LogoElement";
import StepElement from "./StepElement";
import SettingsElement from "./Settings";

interface SidebarProps {
  children?: React.ReactNode;
  onStepChange: (stepId: number) => void;
}

// Define the type for step items
interface StepItem {
  id: number;
  isActive: boolean;
  isCompleted?: boolean; // Optional for SettingsElement
  isSelectable?: boolean; // Optional, defaults to true if not provided
  icon?: string; // For SettingsElement
  lock_closed?: string;
  lock_opened?: string;
  lock_opened_white?: string;
  check_mark_true?: string;
  check_mark_false?: string;
  label: string;
  isSettings?: boolean; // Flag to identify SettingsElement
}

const Sidebar: React.FC<SidebarProps> = ({ children, onStepChange }) => {
  // Initial steps data
  const initialSteps: StepItem[] = [
    {
      id: 0,
      isActive: true,
      isCompleted: true,
      isSelectable: true,
      lock_closed: "Lock (closed).svg",
      lock_opened: "Lock (open) gray.svg",
      lock_opened_white: "Lock (open) white.svg",
      check_mark_true: "Check (True).svg",
      check_mark_false: "Check (False).svg",
      label: "Set Goals",
    },
    {
      id: 1,
      isActive: false,
      isCompleted: false,
      isSelectable: true,
      lock_closed: "Lock (closed).svg",
      lock_opened: "Lock (open) gray.svg",
      lock_opened_white: "Lock (open) white.svg",
      check_mark_true: "Check (True).svg",
      check_mark_false: "Check (False).svg",
      label: "Define Scope",
    },
    {
      id: 2,
      isActive: false,
      isCompleted: false,
      isSelectable: false, // Not selectable
      lock_closed: "Lock (closed).svg",
      lock_opened: "Lock (open) gray.svg",
      lock_opened_white: "Lock (open) white.svg",
      check_mark_true: "Check (True).svg",
      check_mark_false: "Check (False).svg",
      label: "Select Algorithms",
    },
    {
      id: 3,
      isActive: false,
      isCompleted: false,
      isSelectable: false, // Not selectable
      lock_closed: "Lock (closed).svg",
      lock_opened: "Lock (open) gray.svg",
      lock_opened_white: "Lock (open) white.svg",
      check_mark_true: "Check (True).svg",
      check_mark_false: "Check (False).svg",
      label: "Perform Analysis",
    },
    {
      id: 4,
      isActive: false,
      isCompleted: false,
      isSelectable: false, // Not selectable
      lock_closed: "Lock (closed).svg",
      lock_opened: "Lock (open) gray.svg",
      lock_opened_white: "Lock (open) white.svg",
      check_mark_true: "Check (True).svg",
      check_mark_false: "Check (False).svg",
      label: "Generate Insights",
    },
    {
      id: 5,
      isActive: false,
      isCompleted: false,
      isSelectable: false, // Not selectable
      lock_closed: "Lock (closed).svg",
      lock_opened: "Lock (open) gray.svg",
      lock_opened_white: "Lock (open) white.svg",
      check_mark_true: "Check (True).svg",
      check_mark_false: "Check (False).svg",
      label: "Present Findings",
    },
    {
      id: 7,
      isActive: false,
      label: "Settings",
      icon: "Settings.svg",
      isSettings: true, // Custom flag to identify settings
    },
  ];

  // State to manage steps
  const [steps, setSteps] = useState<StepItem[]>(initialSteps);

  const handleStepToggle = (id: number) => {
    const clickedStep = steps.find((step) => step.id === id);
    const isSelectable = clickedStep?.isSelectable !== false;
    if (!isSelectable) return;

    setSteps((prevSteps) =>
      prevSteps.map((step) =>
        step.id === id ? { ...step, isActive: true } : { ...step, isActive: false }
      )
    );

    // Call the parent callback to inform about step change
    onStepChange(id);
  };
  

  return (
    <div className="sidebar">
      <aside className="bg-sidebar_background text-white h-full flex flex-col">
        {/* Logo Element */}
        <div className="mt-6 ml-6">
          <LogoElement iconSrc="Logo.svg" text="CLR Assistant" />
        </div>

        {/* Spacing */}
        <div className="h-[72px]"></div>

        {/* Step Elements */}
        {/* Step 1 */}
        <StepElement
          key={steps[0].id}
          id={steps[0].id}
          isActive={steps[0].isActive}
          isCompleted={steps[0].isCompleted!}
          isSelectable={steps[0].isSelectable!}
          lock_closed={steps[0].lock_closed!}
          lock_opened={steps[0].lock_opened!}
          lock_opened_white={steps[0].lock_opened_white!}
          check_mark_true={steps[0].check_mark_true!}
          check_mark_false={steps[0].check_mark_false!}
          label={steps[0].label}
          onToggle={handleStepToggle}
        />
        {/* Step 2 */}
        <StepElement
          key={steps[1].id}
          id={steps[1].id}
          isActive={steps[1].isActive}
          isCompleted={steps[1].isCompleted!}
          isSelectable={steps[1].isSelectable!}
          lock_closed={steps[1].lock_closed!}
          lock_opened={steps[1].lock_opened!}
          lock_opened_white={steps[1].lock_opened_white!}
          check_mark_true={steps[1].check_mark_true!}
          check_mark_false={steps[1].check_mark_false!}
          label={steps[1].label}
          onToggle={handleStepToggle}
        />
        {/* Step 3 */}
        <StepElement
          key={steps[2].id}
          id={steps[2].id}
          isActive={steps[2].isActive}
          isCompleted={steps[2].isCompleted!}
          isSelectable={steps[2].isSelectable!}
          lock_closed={steps[2].lock_closed!}
          lock_opened={steps[2].lock_opened!}
          lock_opened_white={steps[2].lock_opened_white!}
          check_mark_true={steps[2].check_mark_true!}
          check_mark_false={steps[2].check_mark_false!}
          label={steps[2].label}
          onToggle={handleStepToggle}
        />
        {/* Step 4 */}
        <StepElement
          key={steps[3].id}
          id={steps[3].id}
          isActive={steps[3].isActive}
          isCompleted={steps[3].isCompleted!}
          isSelectable={steps[3].isSelectable!}
          lock_closed={steps[3].lock_closed!}
          lock_opened={steps[3].lock_opened!}
          lock_opened_white={steps[3].lock_opened_white!}
          check_mark_true={steps[3].check_mark_true!}
          check_mark_false={steps[3].check_mark_false!}
          label={steps[3].label}
          onToggle={handleStepToggle}
        />
        {/* Step 5 */}
        <StepElement
          key={steps[4].id}
          id={steps[4].id}
          isActive={steps[4].isActive}
          isCompleted={steps[4].isCompleted!}
          isSelectable={steps[4].isSelectable!}
          lock_closed={steps[4].lock_closed!}
          lock_opened={steps[4].lock_opened!}
          lock_opened_white={steps[4].lock_opened_white!}
          check_mark_true={steps[4].check_mark_true!}
          check_mark_false={steps[4].check_mark_false!}
          label={steps[4].label}
          onToggle={handleStepToggle}
        />
        {/* Step 6 */}
        <StepElement
          key={steps[5].id}
          id={steps[5].id}
          isActive={steps[5].isActive}
          isCompleted={steps[5].isCompleted!}
          isSelectable={steps[5].isSelectable!}
          lock_closed={steps[5].lock_closed!}
          lock_opened={steps[5].lock_opened!}
          lock_opened_white={steps[5].lock_opened_white!}
          check_mark_true={steps[5].check_mark_true!}
          check_mark_false={steps[5].check_mark_false!}
          label={steps[5].label}
          onToggle={handleStepToggle}
        />
        {/* Spacing */}
        <div className="flex-grow"></div>
        {/*Settings*/}
        <div className="mb-[48px]">
          <SettingsElement
            id={steps[steps.length - 1].id}
            isActive={steps[steps.length - 1].isActive}
            icon={steps[steps.length - 1].icon!}
            label={steps[steps.length - 1].label}
            onToggle={handleStepToggle}
          />
        </div>
      </aside>
    </div>
  );
};

export default Sidebar;
