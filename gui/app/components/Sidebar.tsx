// components/Sidebar.tsx

import React, { useState, useEffect, useRef } from "react";
import LogoElement from "./LogoElement";
import StepElement from "./StepElement";
import SettingsElement from "./Settings";
import axios from "axios";

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
      isCompleted: false,
      isSelectable: false,
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
      isSelectable: false,
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

  // Polling function to fetch status
  const isFetchingRef = useRef(false);

  const pollStatus = async () => {
    if (isFetchingRef.current) return; // Prevent overlapping calls
    isFetchingRef.current = true;

    try {
      const response = await fetch("http://localhost:8000/status");
      const data: Record<string, { isallowed: boolean; finished: boolean }> = await response.json();
      console.log("Fetched status data:", data);

      // Update steps based on parsed data
      setSteps((prevSteps) => {
        const updatedSteps = prevSteps.map((step) => {
          if (step.isSettings) return step; // Skip SettingsElement
      
          const status = data[step.id];
          if (status) {
            return {
              ...step,
              isSelectable: status.isallowed,
              isCompleted: status.finished,
            };
          }
          return step;
        });
        console.log("Updated steps:", updatedSteps);
        return updatedSteps;
      });

    } catch (error) {
      console.error("Error fetching status:", error);
    } finally {
      isFetchingRef.current = false;
    }
  };

  useEffect(() => {
    // Start polling every 100ms to reduce load
    const intervalId = setInterval(pollStatus, 100); // Adjust as needed

    // Initial fetch to populate state immediately
    pollStatus();

    // Cleanup on unmount
    return () => clearInterval(intervalId);
  }, []);

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
        {steps.map((step) =>
          step.isSettings ? (
            <SettingsElement
              key={step.id}
              id={step.id}
              isActive={step.isActive}
              icon={step.icon!}
              label={step.label}
              onToggle={handleStepToggle}
            />
          ) : (
            <StepElement
              key={step.id}
              id={step.id}
              isActive={step.isActive}
              isCompleted={step.isCompleted!}
              isSelectable={step.isSelectable!}
              lock_closed={step.lock_closed!}
              lock_opened={step.lock_opened!}
              lock_opened_white={step.lock_opened_white!}
              check_mark_true={step.check_mark_true!}
              check_mark_false={step.check_mark_false!}
              label={step.label}
              onToggle={handleStepToggle}
            />
          )
        )}

        {/* Spacing */}
        <div className="flex-grow"></div>
      </aside>
    </div>
  );
};

export default Sidebar;
