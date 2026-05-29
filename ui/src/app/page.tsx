"use client";
import Link from "next/link";

const features = [
  { icon: "💬", color: "#6382ff", bg: "rgba(99,130,255,.12)", title: "Medical AI Chat", desc: "Ask any medical question and get evidence-based answers powered by our fine-tuned Mistral-7B model trained on 10K clinical Q&A pairs." },
  { icon: "🩺", color: "#34d399", bg: "rgba(52,211,153,.12)", title: "Symptom Checker", desc: "Select your symptoms and get a differential diagnosis with urgency levels based on clinical pattern matching algorithms." },
  { icon: "💊", color: "#a78bfa", bg: "rgba(167,139,250,.12)", title: "Drug Encyclopedia", desc: "Comprehensive database of medications with mechanism of action, dosing, contraindications, and side-effect profiles." },
  { icon: "📊", color: "#22d3ee", bg: "rgba(34,211,238,.12)", title: "Research Metrics", desc: "Transparent model benchmarks: ROUGE-L, BERTScore, and perplexity comparisons between base and fine-tuned models." },
  { icon: "⚡", color: "#fbbf24", bg: "rgba(251,191,36,.12)", title: "Sub-500ms Inference", desc: "Powered by vLLM continuous batching for 3-4× higher throughput than standard HuggingFace pipelines at the same GPU." },
  { icon: "🔒", color: "#f87171", bg: "rgba(248,113,113,.12)", title: "Clinical Safety", desc: "Built with medical accuracy in mind. Every response includes appropriate clinical disclaimers and references to licensed care." },
];

const stats = [
  { num: "10K+", label: "Training Samples" },
  { num: "+18.4%", label: "ROUGE-L Gain" },
  { num: "420ms", label: "Avg Latency" },
  { num: "2.19%", label: "Params Trained" },
];

export default function Home() {
  return (
    <div className="page">
      <div className="bg-scene"><div className="bg-grid" /><div className="orb orb1" /><div className="orb orb2" /><div className="orb orb3" /></div>

      {/* ── Hero ── */}
      <section className="hero">
        <div>
          <div className="hero-badge">🧬 Fine-tuned Mistral-7B · QLoRA · Medical AI</div>
          <h1 className="hero-title">
            Medical Intelligence<br />
            <span>Powered by AI</span>
          </h1>
          <p className="hero-sub">Evidence-based medical Q&A, symptom analysis, and drug information — powered by Mistral-7B fine-tuned on 10,000 clinical Q&A pairs with QLoRA.</p>
          <div className="hero-actions">
            <Link href="/chat"><button className="btn-primary">💬 Start AI Chat →</button></Link>
            <Link href="/symptoms"><button className="btn-secondary">🩺 Check Symptoms</button></Link>
          </div>
        </div>
      </section>

      {/* ── Stats ── */}
      <div className="stats-bar" style={{ position: "relative", zIndex: 1 }}>
        <div className="stats-row">
          {stats.map(s => (
            <div key={s.label}>
              <div className="stat-num">{s.num}</div>
              <div className="stat-label2">{s.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Features ── */}
      <section className="features" style={{ position: "relative", zIndex: 1 }}>
        <div className="container">
          <div className="text-center mb-8">
            <div className="section-tag">✦ Platform Features</div>
            <h2 className="section-title">Everything you need for<br />clinical decision support</h2>
            <p className="section-sub" style={{ margin: "0 auto" }}>From real-time medical chat to comprehensive drug lookups — all in one unified platform.</p>
          </div>
          <div className="features-grid">
            {features.map(f => (
              <div key={f.title} className="feat-card">
                <div className="feat-icon" style={{ background: f.bg, color: f.color }}>{f.icon}</div>
                <div className="feat-title">{f.title}</div>
                <p className="feat-desc">{f.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section style={{ position: "relative", zIndex: 1, padding: "80px 0" }}>
        <div className="container text-center">
          <div style={{ background: "linear-gradient(135deg,rgba(99,130,255,.12),rgba(167,139,250,.08))", border: "1px solid rgba(99,130,255,.25)", borderRadius: "24px", padding: "60px 40px" }}>
            <h2 className="section-title">Ready to explore medical AI?</h2>
            <p style={{ color: "var(--t2)", marginBottom: "32px", fontSize: "16px" }}>Start with a medical question or explore our drug database.</p>
            <div className="hero-actions">
              <Link href="/chat"><button className="btn-primary">Get Started Free</button></Link>
              <Link href="/drugs"><button className="btn-secondary">Browse Drugs →</button></Link>
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ── */}
      <footer className="footer" style={{ position: "relative", zIndex: 1 }}>
        <div className="container">
          <div className="footer-logo">🧬 MedQA <span className="text-p">AI</span></div>
          <p className="footer-sub">Mistral-7B · QLoRA fine-tuning · medalpaca/medical_meadow_medqa</p>
          <div className="footer-links">
            {[["Home","/"],["AI Chat","/chat"],["Symptoms","/symptoms"],["Drugs","/drugs"],["Research","/metrics"],["About","/about"]].map(([l,h]) => (
              <Link key={h} href={h} className="footer-link">{l}</Link>
            ))}
          </div>
          <p style={{ color: "var(--t3)", fontSize: "12px", marginTop: "24px" }}>
            Built by Vantaku Jagadeesh · B.Tech CS (Data Science & AI) · CSVTU 2026 · ⚠️ Not for clinical use
          </p>
        </div>
      </footer>
    </div>
  );
}
