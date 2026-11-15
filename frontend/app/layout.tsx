import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Corbal",
  description: "Corbal: ask questions about construction drawings and get answers that point to the evidence.",
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
