"use client";
import { useState, useRef, useEffect } from "react";

const API = "";  // use Next.js proxy rewrites — falls back to same origin
const QUICK = [
  "What are symptoms of appendicitis?",
  "How does insulin resistance develop?",
  "What is the mechanism of beta-blockers?",
  "Explain hypertension treatment guidelines",
  "What is aspirin used for?",
  "Difference between MRI and CT scan?",
  "How is stroke treated?",
  "Explain asthma management steps",
];

type Msg = { id: string; role: "user"|"ai"; text: string; latency?: number; ts: string };

export default function ChatPage() {
  const [msgs, setMsgs] = useState<Msg[]>([{id:"w",role:"ai",text:"Hello! I'm MedQA AI — your evidence-based medical assistant. Ask me about symptoms, conditions, medications, or clinical concepts.\n\n⚠️ Always consult a licensed physician for personal medical decisions.",ts:new Date().toISOString()}]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [online, setOnline] = useState<boolean|null>(null);
  const endRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => { endRef.current?.scrollIntoView({behavior:"smooth"}); }, [msgs]);
  useEffect(() => {
    fetch(`${API}/api/health`,{signal:AbortSignal.timeout(4000)}).then(r=>r.json()).then(d=>setOnline(d.status==="ok")).catch(()=>setOnline(false));
  }, []);

  async function send(q: string) {
    if (!q.trim()||loading) return;
    setMsgs(p=>[...p,{id:Date.now()+"",role:"user",text:q,ts:new Date().toISOString()}]);
    setInput(""); setLoading(true);
    if(taRef.current) taRef.current.style.height="auto";
    try {
      const r = await fetch(`${API}/api/chat`,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({question:q})});
      const d = await r.json();
      setMsgs(p=>[...p,{id:(Date.now()+1)+"",role:"ai",text:d.answer,latency:d.latency_ms,ts:d.timestamp}]);
    } catch {
      setMsgs(p=>[...p,{id:(Date.now()+1)+"",role:"ai",text:"⚠️ Cannot reach the API server. Make sure `python3 api.py` is running.",ts:new Date().toISOString()}]);
    } finally { setLoading(false); }
  }

  return (
    <div className="page" style={{minHeight:"100vh"}}>
      <div className="bg-scene"><div className="bg-grid"/><div className="orb orb1"/><div className="orb orb2"/></div>
      <div className="container" style={{position:"relative",zIndex:1,paddingTop:"40px",paddingBottom:"40px"}}>
        <div className="chat-wrap">
          <div className="flex justify-between items-center mb-6">
            <div>
              <h1 style={{fontSize:"24px",fontWeight:800,letterSpacing:"-.02em"}}>Medical AI Chat</h1>
              <p style={{fontSize:"13px",color:"var(--t2)",marginTop:"4px"}}>Mistral-7B · QLoRA fine-tuned on 10K medical pairs</p>
            </div>
            <span className={`badge ${online===true?"badge-g":online===false?"badge-r":"badge-y"}`}>
              ● {online===true?"API Online":online===false?"API Offline":"Connecting…"}
            </span>
          </div>

          {/* Quick questions */}
          <div className="quick-questions">
            {QUICK.map(q=>(
              <button key={q} className="quick-q" onClick={()=>send(q)}>{q}</button>
            ))}
          </div>

          <div className="card" style={{display:"flex",flexDirection:"column",height:"calc(100vh - 340px)",minHeight:"420px"}}>
            <div className="chat-box" style={{flex:1}}>
              {msgs.map(m=>(
                <div key={m.id} className={`msg ${m.role}`}>
                  <div className={`msg-av ${m.role}`}>{m.role==="user"?"👤":"🧬"}</div>
                  <div className="msg-body">
                    <div className="msg-role">{m.role==="user"?"You":"MedQA AI"}</div>
                    <div className="msg-text">{m.text}</div>
                    {m.latency&&<div className="msg-meta"><span>⚡ {m.latency.toFixed(0)}ms</span><span>{new Date(m.ts).toLocaleTimeString()}</span></div>}
                  </div>
                </div>
              ))}
              {loading&&(
                <div className="msg ai">
                  <div className="msg-av ai">🧬</div>
                  <div className="msg-body">
                    <div className="msg-role">MedQA AI</div>
                    <div className="typing"><div className="dot"/><div className="dot"/><div className="dot"/><span style={{fontSize:12,color:"var(--t3)",marginLeft:6}}>Analyzing…</span></div>
                  </div>
                </div>
              )}
              <div ref={endRef}/>
            </div>
            <div className="chat-input-area">
              <textarea ref={taRef} className="chat-input" rows={1} placeholder="Ask a medical question… (Enter to send, Shift+Enter for newline)" value={input}
                onChange={e=>{setInput(e.target.value);const el=e.target;el.style.height="auto";el.style.height=Math.min(el.scrollHeight,120)+"px"}}
                onKeyDown={e=>{if(e.key==="Enter"&&!e.shiftKey){e.preventDefault();send(input)}}}
              />
              <button className="send-btn" onClick={()=>send(input)} disabled={!input.trim()||loading}>↑</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
