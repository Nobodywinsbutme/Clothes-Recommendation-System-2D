import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Fashion Suggest",
  description: "Gợi ý trang phục dựa trên DeepFashion",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
