import type { Metadata } from "next";
import "./globals.css";
import Navbar from "@/components/Navbar";

export const metadata: Metadata = {
  title: { default: "MedQA AI — Medical Intelligence Platform", template: "%s | MedQA AI" },
  description: "AI-powered medical Q&A, symptom checker, and drug encyclopedia. Fine-tuned Mistral-7B on 10K medical pairs using QLoRA.",
  keywords: ["medical AI", "symptom checker", "drug database", "Mistral-7B", "QLoRA"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
      </head>
      <body>
        <Navbar />
        <main style={{ paddingTop: "64px" }}>{children}</main>
      </body>
    </html>
  );
}
