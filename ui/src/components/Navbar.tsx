"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, useEffect } from "react";

const links = [
  { href: "/", label: "Home" },
  { href: "/chat", label: "AI Chat" },
  { href: "/symptoms", label: "Symptom Checker" },
  { href: "/drugs", label: "Drug Database" },
  { href: "/metrics", label: "Research" },
  { href: "/about", label: "About" },
];

export default function Navbar() {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const [apiStatus, setApiStatus] = useState<"loading"|"online"|"offline">("loading");

  useEffect(() => {
    fetch("/api/health", { signal: AbortSignal.timeout(3000) })
      .then(r => r.ok ? setApiStatus("online") : setApiStatus("offline"))
      .catch(() => setApiStatus("offline"));
  }, []);

  return (
    <nav className="navbar">
      <div className="nav-inner">
        <Link href="/" className="nav-logo">
          <span className="nav-logo-icon">🧬</span>
          <span>MedQA <span className="nav-logo-accent">AI</span></span>
        </Link>
        <div className="nav-links">
          {links.map(l => (
            <Link key={l.href} href={l.href} className={`nav-item${path === l.href ? " nav-active" : ""}`}>{l.label}</Link>
          ))}
        </div>
        {/* API status badge */}
        <div style={{display:"flex",alignItems:"center",gap:6,fontSize:12,color:"var(--t3)",marginRight:8}}>
          <span className={`status-dot ${apiStatus}`} />
          <span style={{display:"none"}} className="api-status-label">
            {apiStatus === "online" ? "API Online" : apiStatus === "offline" ? "API Offline" : "…"}
          </span>
        </div>
        <Link href="/chat" className="nav-cta">Try for Free →</Link>
        <button className="nav-hamburger" onClick={() => setOpen(o => !o)} aria-label="Toggle menu">☰</button>
      </div>
      {open && (
        <>
          <div style={{position:"fixed",inset:0,zIndex:99}} onClick={() => setOpen(false)} />
          <div className="nav-mobile" style={{position:"relative",zIndex:101}}>
            {links.map(l => (
              <Link key={l.href} href={l.href}
                className={`nav-mobile-item${path === l.href ? " nav-active" : ""}`}
                onClick={() => setOpen(false)}>{l.label}</Link>
            ))}
          </div>
        </>
      )}
    </nav>
  );
}
