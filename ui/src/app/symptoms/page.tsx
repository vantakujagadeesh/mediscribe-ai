"use client";
import { useState } from "react";

const API = "";  // Next.js proxy rewrites /api/* → backend
const ALL_SYMPTOMS = ["fever","headache","chest pain","shortness of breath","abdominal pain","cough","fatigue","nausea","dizziness","back pain","joint pain","rash","weight loss","palpitations"];

type Result = { conditions: {condition:string;match_score:number;urgency:string}[]; matched_symptoms: string[]; advice: string };

export default function SymptomsPage() {
  const [selected, setSelected] = useState<string[]>([]);
  const [custom, setCustom] = useState("");
  const [result, setResult] = useState<Result|null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function toggle(s: string) {
    setSelected(p => p.includes(s) ? p.filter(x=>x!==s) : [...p,s]);
  }

  function addCustom() {
    const v = custom.trim().toLowerCase();
    if (v && !selected.includes(v)) { setSelected(p=>[...p,v]); setCustom(""); }
  }

  async function analyze() {
    if (!selected.length) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const r = await fetch(`${API}/api/symptoms`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({symptoms:selected})});
      setResult(await r.json());
    } catch { setError("Cannot reach API server. Start: python3 api.py"); }
    finally { setLoading(false); }
  }

  const urgencyClass = (u:string) => u==="High"?"urgency-high":u==="Medium"?"urgency-medium":"urgency-low";

  return (
    <div className="page" style={{minHeight:"100vh"}}>
      <div className="bg-scene"><div className="bg-grid"/><div className="orb orb1"/><div className="orb orb2"/></div>
      <div className="container" style={{position:"relative",zIndex:1,paddingTop:"48px",paddingBottom:"80px"}}>
        <div className="page-header text-center">
          <div className="section-tag">🩺 AI-Powered</div>
          <h1 className="section-title">Symptom Checker</h1>
          <p style={{color:"var(--t2)",maxWidth:520,margin:"0 auto",fontSize:15,lineHeight:1.7}}>Select your symptoms below. Our clinical pattern-matching algorithm will generate a ranked differential diagnosis with urgency levels.</p>
          <p style={{color:"var(--t3)",fontSize:12,marginTop:12}}>⚠️ This tool is for educational purposes only. Not a substitute for professional medical advice.</p>
        </div>

        <div className="grid-2 mt-8">
          {/* Left: symptom selector */}
          <div className="card">
            <h3 style={{fontWeight:700,marginBottom:16,fontSize:15}}>Select Symptoms</h3>
            <div className="symptom-tags">
              {ALL_SYMPTOMS.map(s=>(
                <button key={s} className={`symptom-tag${selected.includes(s)?" selected":""}`} onClick={()=>toggle(s)}>
                  {selected.includes(s)?"✓ ":""}{s}
                </button>
              ))}
            </div>
            <div style={{marginTop:16,display:"flex",gap:8}}>
              <input value={custom} onChange={e=>setCustom(e.target.value)} onKeyDown={e=>e.key==="Enter"&&addCustom()} placeholder="Add custom symptom…" className="search-bar" style={{flex:1,padding:"10px 14px"}} />
              <button onClick={addCustom} className="send-btn" style={{width:42,height:42}}>+</button>
            </div>

            {selected.length>0&&(
              <div style={{marginTop:16}}>
                <p style={{fontSize:12,color:"var(--t3)",marginBottom:8}}>Selected ({selected.length}):</p>
                <div className="flex flex-wrap gap-2">
                  {selected.map(s=>(
                    <span key={s} style={{padding:"4px 12px",borderRadius:999,background:"rgba(99,130,255,.15)",border:"1px solid rgba(99,130,255,.3)",fontSize:12,color:"var(--p)",cursor:"pointer"}} onClick={()=>toggle(s)}>
                      {s} ✕
                    </span>
                  ))}
                </div>
              </div>
            )}

            <button onClick={analyze} disabled={!selected.length||loading} className="btn-primary w-full" style={{marginTop:20,width:"100%",fontSize:14}}>
              {loading?"Analyzing…":"🔍 Analyze Symptoms"}
            </button>
            {error&&<p style={{color:"var(--r)",fontSize:13,marginTop:12}}>{error}</p>}
          </div>

          {/* Right: results */}
          <div>
            {!result&&!loading&&(
              <div className="card" style={{textAlign:"center",padding:"48px 24px"}}>
                <div style={{fontSize:48,marginBottom:16}}>🏥</div>
                <p style={{color:"var(--t2)",fontSize:14}}>Select symptoms on the left and click "Analyze Symptoms" to see your differential diagnosis.</p>
              </div>
            )}
            {loading&&(
              <div className="card" style={{textAlign:"center",padding:"48px 24px"}}>
                <div className="typing" style={{justifyContent:"center"}}><div className="dot"/><div className="dot"/><div className="dot"/></div>
                <p style={{color:"var(--t2)",fontSize:13,marginTop:12}}>Running clinical pattern analysis…</p>
              </div>
            )}
            {result&&(
              <div className="fade-in">
                <div className="card mb-4">
                  <h3 style={{fontWeight:700,marginBottom:4,fontSize:15}}>Differential Diagnosis</h3>
                  <p style={{fontSize:12,color:"var(--t3)",marginBottom:16}}>Matched symptoms: {result.matched_symptoms.join(", ") || "none"}</p>
                  {result.conditions.length===0&&<p style={{color:"var(--t2)"}}>No matches found. Try different symptoms.</p>}
                  {result.conditions.map((c,i)=>(
                    <div key={i} className="condition-card mb-4">
                      <div>
                        <div style={{fontWeight:600,fontSize:14}}>{c.condition}</div>
                        <div style={{fontSize:12,color:"var(--t3)",marginTop:3}}>Match score: {c.match_score}%</div>
                      </div>
                      <span className={urgencyClass(c.urgency)}>{c.urgency}</span>
                    </div>
                  ))}
                </div>
                <div className="card" style={{background:"rgba(251,191,36,.06)",borderColor:"rgba(251,191,36,.2)"}}>
                  <div style={{fontWeight:700,marginBottom:6,fontSize:13,color:"var(--y)"}}>⚠️ Clinical Advice</div>
                  <p style={{fontSize:13,color:"var(--t2)",lineHeight:1.6}}>{result.advice}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
