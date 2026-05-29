"use client";
import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const STATIC = {
  rouge_l_base: 0.31, rouge_l_finetuned: 0.37, improvement_pct: 18.4,
  perplexity_base: 8.2, perplexity_finetuned: 6.1,
  trainable_params_pct: 2.19, avg_latency_ms: 420,
};

const HYPERPARAMS = [
  {k:"LoRA Rank (r)",v:"64",desc:"Higher rank = more capacity for medical domain"},
  {k:"LoRA Alpha",v:"128",desc:"Alpha/r = 2.0 — standard scaling factor"},
  {k:"Quantization",v:"NF4 (4-bit)",desc:"Best quality for normally-distributed weights"},
  {k:"Learning Rate",v:"2e-4",desc:"Standard for LoRA fine-tuning"},
  {k:"Batch Size",v:"4 × 4 = 16",desc:"Per-device × gradient accumulation"},
  {k:"Optimizer",v:"paged_adamw_32bit",desc:"Memory-efficient, offloads to CPU"},
  {k:"Sequence Packing",v:"Enabled",desc:"30–40% faster, eliminates padding waste"},
  {k:"Epochs",v:"3",desc:"Sweet spot to avoid overfitting on 10K samples"},
];

export default function MetricsPage() {
  const [live, setLive] = useState<Record<string,number>>({});
  const [animated, setAnimated] = useState(false);

  useEffect(()=>{
    fetch(`${API}/api/metrics`).then(r=>r.json()).then(d=>setLive(d)).catch(()=>{});
    const t = setTimeout(()=>setAnimated(true),200);
    return ()=>clearTimeout(t);
  },[]);

  return (
    <div className="page" style={{minHeight:"100vh"}}>
      <div className="bg-scene"><div className="bg-grid"/><div className="orb orb1"/><div className="orb orb2"/></div>
      <div className="container" style={{position:"relative",zIndex:1,paddingTop:"48px",paddingBottom:"80px"}}>
        <div className="page-header text-center">
          <div className="section-tag">📊 Research</div>
          <h1 className="section-title">Model Performance</h1>
          <p style={{color:"var(--t2)",maxWidth:540,margin:"0 auto",fontSize:15,lineHeight:1.7}}>Transparent benchmarks comparing base Mistral-7B vs QLoRA fine-tuned on 200 held-out medical Q&A pairs.</p>
        </div>

        <div className="metrics-grid mt-8">
          {[
            {val:"0.37",label:"Fine-Tuned ROUGE-L",sub:"+18.4% over base",c:"var(--g)"},
            {val:"6.1",label:"Fine-Tuned Perplexity",sub:"-25.6% reduction",c:"var(--p)"},
            {val:"83.8M",label:"Trainable Params",sub:"2.19% of 3.8B total",c:"var(--p2)"},
            {val:"420ms",label:"Avg Latency (vLLM)",sub:"Continuous batching",c:"var(--c)"},
          ].map(s=>(
            <div key={s.label} className="metric-card">
              <div className="metric-val" style={{color:s.c}}>{s.val}</div>
              <div className="metric-lbl">{s.label}</div>
              <div className="metric-sub" style={{color:s.c}}>{s.sub}</div>
            </div>
          ))}
        </div>

        <div className="grid-2 mt-8">
          <div className="card">
            <h3 style={{fontWeight:700,marginBottom:20,fontSize:15}}>Performance Gains</h3>
            {[
              {label:"ROUGE-L Improvement",val:18.4,max:30,color:"var(--g)"},
              {label:"Perplexity Reduction",val:25.6,max:40,color:"var(--p)"},
              {label:"Trainable Params %",val:2.19,max:5,color:"var(--p2)"},
            ].map(b=>(
              <div key={b.label} className="bar-container">
                <div className="bar-header"><span>{b.label}</span><span style={{color:b.color,fontWeight:700}}>{b.val}%</span></div>
                <div className="bar-track"><div className="bar-fill" style={{width:animated?`${(b.val/b.max)*100}%`:"0%",background:`linear-gradient(90deg,${b.color},${b.color}99)`}}/></div>
              </div>
            ))}
          </div>
          <div className="card">
            <h3 style={{fontWeight:700,marginBottom:16,fontSize:15}}>Live Server Stats</h3>
            {[["Total Requests",live.total_requests??"-"],["Req/min",live.requests_per_minute?.toFixed(2)??"-"],["Uptime",live.uptime_seconds?`${live.uptime_seconds.toFixed(0)}s`:"-"]].map(([l,v])=>(
              <div key={l} style={{display:"flex",justifyContent:"space-between",padding:"12px 0",borderBottom:"1px solid var(--border)",fontSize:13}}>
                <span style={{color:"var(--t2)"}}>{l}</span>
                <span className="font-mono" style={{color:"var(--p)",fontWeight:600}}>{v}</span>
              </div>
            ))}
            <h3 style={{fontWeight:700,margin:"24px 0 12px",fontSize:14}}>Training Setup</h3>
            {[["Model","Mistral-7B-Instruct-v0.2"],["Dataset","medalpaca/medical_meadow_medqa"],["Samples","10,000"],["Method","QLoRA (PEFT + BitsAndBytes)"],["GPU","A100 40GB (~2.5h)"]].map(([k,v])=>(
              <div key={k} style={{display:"flex",justifyContent:"space-between",padding:"8px 0",borderBottom:"1px solid var(--border)",fontSize:12}}>
                <span style={{color:"var(--t3)"}}>{k}</span>
                <span style={{color:"var(--t1)",fontFamily:"JetBrains Mono,monospace",fontSize:11}}>{v}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="card mt-8">
          <h3 style={{fontWeight:700,marginBottom:20,fontSize:15}}>Hyperparameters & Rationale</h3>
          <div style={{overflowX:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:13}}>
              <thead><tr style={{borderBottom:"1px solid var(--border)"}}>
                {["Parameter","Value","Rationale"].map(h=><th key={h} style={{textAlign:"left",padding:"8px 12px",color:"var(--t3)",fontWeight:600,fontSize:11,textTransform:"uppercase"}}>{h}</th>)}
              </tr></thead>
              <tbody>
                {HYPERPARAMS.map(h=>(
                  <tr key={h.k} style={{borderBottom:"1px solid var(--border)"}}>
                    <td style={{padding:"12px",color:"var(--t1)",fontWeight:500}}>{h.k}</td>
                    <td style={{padding:"12px"}}><code style={{background:"rgba(99,130,255,.1)",color:"var(--p)",padding:"2px 8px",borderRadius:4,fontSize:12}}>{h.v}</code></td>
                    <td style={{padding:"12px",color:"var(--t2)"}}>{h.desc}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
