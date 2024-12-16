// components/ResultImage.tsx
import React from "react";

interface ResultImageProps {
  url: string;        // URL to the PNG image
  altText?: string;   // Optional alt text for accessibility
}

const ResultImage: React.FC<ResultImageProps> = ({ url, altText = "Result Image" }) => {
  return (
    <div
      className="relative bg-result_bg rounded-[16px] max-w-[867px] min-w-[200px] w-auto pb-4"
      style={{ wordWrap: "break-word" }}
    >
      {/* "Results" Label */}
      <div
        className="absolute subpixel-antialiased top-2 right-5 font-sans text-result_label text-base font-bold"
      >
        Results
      </div>

      {/* Display the image */}
      <div className="mt-9 ml-8 mr-4 flex justify-center items-center">
        <img
          src={url}
          alt={altText}
          className="max-w-full h-auto rounded-md"
          onError={(e) => {
            (e.target as HTMLImageElement).src = "C:\\Users\\Lukas\\PycharmProjects\\clr-llm-agent\\visualizations\\TopicWordsOverTime.png"; 
          }}
        />
      </div>
    </div>
  );
};

export default ResultImage;
