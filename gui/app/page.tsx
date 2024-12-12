"use client";

import { useState } from "react";
import React from "react";
import Mainwindow from "./components/MainWindow";
import Sidebar from "./components/Sidebar";

export default function Home() {
  const [activeStep, setActiveStep] = useState(0);

  return (
    <div className="flex min-h-screen">
      <Sidebar onStepChange={setActiveStep}></Sidebar>
      <Mainwindow activeStep={activeStep}></Mainwindow>
    </div>
  );
}
