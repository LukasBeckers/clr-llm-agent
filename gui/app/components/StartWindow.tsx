// components/StartWindow

import React from "react";
import ContentsStartPage1 from "./ContentsStartPage1";


interface MainwindowProps {
  styling_color: string;
}

const Mainwindow: React.FC<MainwindowProps> = ({ styling_color }) => {
  return (
    <div className="relative flex-1 min-h-screen bg-background_main overflow-hidden">
      {/* Ellipse */}
      <div
        className={`absolute z-0 transform rotate-[-30deg] bg-background_style_color_start_page opacity-50`}
        style={{
          width: "2028.7px",
          height: "1011.26px",
          left: "-250.01px",
          top: "10px",
          borderRadius: "50% / 50%",
          filter: "blur(200px)",
        }}
        aria-hidden="true"
      ></div>

      {/* Mainwindow Content */}
      <div className="relative z-10 flex justify-center items-center h-screen">
      <ContentsStartPage1></ContentsStartPage1>
      </div>
    </div>
  );
};

export default Mainwindow;
