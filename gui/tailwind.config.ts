import type { Config } from "tailwindcss";

export default {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['var(--font-poppins)', 'Arial', 'Helvetica', 'sans-serif'],
      },
      fontWeight: {
        thin: "100", // Thin → 100
        extralight: "200", // Extra Light → 200
        light: "300", // Light → 300
        normal: "400", // Regular (Normal) → 400
        medium: "500", // Medium → 500
        semibold: "600", // Semi Bold → 600
        bold: "700", // Bold → 700
        extrabold: "800", // Extra Bold → 800
        black: "900", // Black → 900
      },
      colors: {
        sidebar_background: "#5A57FF",
        sidebar_bright_white: "#FFFFFF",
        sidebar_gray_white: "#C2C0FF",
        sidebar_green: "#6BFF57",
        sidebar_black: "#353535",
        background_main: "rgba(194, 192, 255, 0.5)",
        background_style_color_start_page: "#FFC0EA",
        inputfield_main_color: "#C2C0FF",
        input_field_items: "#353535",
      },
    },
  },
  variants: {
    extend: {
      fontSmoothing: ["responsive"],
    },
  },
  plugins: [],
} satisfies Config;
