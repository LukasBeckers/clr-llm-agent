"use client";

import React from "react";
import Mainwindow from "./components/MainWindow";
import Sidebar from "./components/Sidebar";

export default function Home() {
  const handleClick = () => {
    alert("Button is Clicked");
  };

  return (
    <div className="flex min-h-screen">
      <Sidebar></Sidebar>
      <Mainwindow styling_color="background_style_color_start_page"></Mainwindow>
    </div>
  );
}
