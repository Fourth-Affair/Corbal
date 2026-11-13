import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Planform · Building intelligence",
  description: "Explore your building with an interactive digital floor plan.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
