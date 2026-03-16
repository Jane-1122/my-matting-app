import "./globals.css";
import type { ReactNode } from "react";

export const metadata = {
  title: "Jing's Video Matting Studio",
  description: "High-quality video background removal by Jing",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-CN">
      <body className="min-h-screen bg-gray-50 text-gray-900 antialiased">
        <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 py-6">
          <header className="mb-6 flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary">
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-5 w-5 text-white"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  strokeWidth={2}
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    d="m15.75 10.5 4.72-4.72a.75.75 0 0 1 1.28.53v11.38a.75.75 0 0 1-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 0 0 2.25-2.25v-9a2.25 2.25 0 0 0-2.25-2.25h-9A2.25 2.25 0 0 0 2.25 7.5v9a2.25 2.25 0 0 0 2.25 2.25Z"
                  />
                </svg>
              </div>
              <span className="text-lg font-bold tracking-tight text-gray-900">
                Jing&apos;s Video Matting Studio
              </span>
            </div>
          </header>
          <main className="flex-1">{children}</main>
          <footer className="mt-8 border-t border-gray-200 pt-4 text-center text-xs text-gray-400">
            © {new Date().getFullYear()} Jing&apos;s Video Matting Studio
          </footer>
        </div>
      </body>
    </html>
  );
}
