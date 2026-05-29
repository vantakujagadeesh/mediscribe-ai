"use client";
import Link from "next/link";

const PIPELINE = [
  {icon:"📥",tag:"datasets",title:"1. Load Dataset",desc:"medalpaca/medical_meadow_medqa — 10K Q&A pairs. Shuffled, 95/5 train/eval split. Formatted as Mistral chat template: <s>[INST]…[/INST]…</s>"},
  {icon:"🔢",tag:"bitsandbytes",title:"2. 4-bit NF4 Quantization",desc:"BitsAndBytes NF4 quantization compresses Mistral-7B from ~14GB → ~4GB VRAM. Double-quant saves an additional 0.4GB. Makes A100 fine-tuning feasible."},
  {icon:"🔗",tag:"peft",title:"3. Attach LoRA Adapters",desc:"r=64, alpha=128 applied to all linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj). Only 83.8M / 3.8B params are trained (2.19%)."},
  {icon:"🏋️",tag:"trl",title:"4. SFT Training",desc:"TRL SFTTrainer with sequence packing (35% faster), cosine LR schedule, paged_adamw_32bit optimizer. Gradient checkpointing reduces VRAM by 40%."},
  {icon:"🔀",tag:"safetensors",title:"5. Merge & Push",desc:"merge_and_unload() fuses LoRA Δ weights into base model. Saved as safetensors shards and pushed to HuggingFace Hub."},
  {icon:"📐",tag:"rouge-score",title:"6. Evaluate",desc:"ROUGE-L + BERTScore on 200 held-out samples. Base: 0.31 → Fine-tuned: 0.37 (+18.4%). Perplexity: 8.2 → 6.1 (-25.6%)."},
  {icon:"🚀",tag:"vllm",title:"7. Serve with vLLM",desc:"FastAPI + vLLM continuous batching. 3-4× higher throughput vs HuggingFace pipeline. Sub-500ms at 420ms avg. HF fallback for CPU/dev."},
];

const INTERVIEW = [
  "Reduced trainable parameters from 7B to 83M (2.19%) using QLoRA without significant quality loss",
  "NF4 quantization compressed the model from ~14GB → ~4GB VRAM, enabling fine-tuning on consumer hardware",
  "Sequence packing eliminated padding waste and improved training throughput by ~35%",
  "vLLM continuous batching gives 3-4× higher throughput than HuggingFace pipeline at the same GPU",
  "ROUGE-L improved from 0.31 → 0.37 (+18%) on 200 held-out medical Q&A pairs",
  "Achieved sub-500ms inference (420ms avg) using vLLM on A100 with 90% GPU memory utilization",
];

export default function AboutPage() {
  return (
    <div className="page" style={{minHeight:"100vh"}}>
      <div className="bg-scene"><div className="bg-grid"/><div className="orb orb1"/><div className="orb orb2"/></div>
      <div className="container" style={{position:"relative",zIndex:1,paddingTop:"48px",paddingBottom:"80px"}}>

        {/* Header */}
        <div className="page-header text-center">
          <div className="section-tag">⚙️ Technical Deep-Dive</div>
          <h1 className="section-title">About MedQA AI</h1>
          <p style={{color:"var(--t2)",maxWidth:580,margin:"0 auto",fontSize:15,lineHeight:1.7}}>A production-grade QLoRA fine-tuning pipeline for medical Q&A, built as an end-to-end ML project showcasing modern PEFT techniques.</p>
        </div>

        {/* Pipeline */}
        <h2 style={{fontWeight:800,fontSize:20,marginBottom:24,letterSpacing:"-.01em"}}>Full Pipeline</h2>
        <div style={{display:"flex",flexDirection:"column",gap:12,marginBottom:48}}>
          {PIPELINE.map((s,i)=>(
            <div key={i} className="card" style={{display:"flex",gap:16,alignItems:"flex-start",padding:"20px 24px"}}>
              <div style={{fontSize:24,width:40,flexShrink:0,textAlign:"center",marginTop:2}}>{s.icon}</div>
              <div style={{flex:1}}>
                <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:6}}>
                  <span style={{fontWeight:700,fontSize:14}}>{s.title}</span>
                  <span className="badge badge-b">{s.tag}</span>
                </div>
                <p style={{fontSize:13,color:"var(--t2)",lineHeight:1.65}}>{s.desc}</p>
              </div>
            </div>
          ))}
        </div>

        {/* Tech stack + Interview */}
        <div className="grid-2 mb-8">
          <div className="card">
            <h3 style={{fontWeight:700,marginBottom:20,fontSize:15}}>Tech Stack</h3>
            {[["Base model","Mistral-7B-Instruct-v0.2"],["Fine-tuning","QLoRA (PEFT + BitsAndBytes)"],["Trainer","TRL SFTTrainer"],["Tracking","Weights & Biases"],["Evaluation","ROUGE-L, BERTScore"],["Inference","vLLM + FastAPI"],["Frontend","Next.js 16 + TypeScript"],["Hosting","HuggingFace Hub"]].map(([k,v])=>(
              <div key={k} style={{display:"flex",justifyContent:"space-between",padding:"10px 0",borderBottom:"1px solid var(--border)",fontSize:13}}>
                <span style={{color:"var(--t3)"}}>{k}</span>
                <span style={{color:"var(--t1)",fontWeight:500,fontFamily:"JetBrains Mono,monospace",fontSize:12}}>{v}</span>
              </div>
            ))}
          </div>

          <div className="card">
            <h3 style={{fontWeight:700,marginBottom:16,fontSize:15}}>💼 Interview Talking Points</h3>
            {INTERVIEW.map((p,i)=>(
              <div key={i} style={{display:"flex",gap:10,marginBottom:14}}>
                <span style={{color:"var(--g)",fontWeight:700,flexShrink:0,fontSize:14}}>→</span>
                <p style={{fontSize:13,color:"var(--t2)",lineHeight:1.6}}>{p}</p>
              </div>
            ))}
          </div>
        </div>

        {/* GPU Costs */}
        <div className="card mb-8">
          <h3 style={{fontWeight:700,marginBottom:16,fontSize:15}}>GPU & Cost Estimates (3 epochs, 10K samples)</h3>
          <div style={{overflowX:"auto"}}>
            <table style={{width:"100%",borderCollapse:"collapse",fontSize:13}}>
              <thead><tr style={{borderBottom:"1px solid var(--border)"}}>
                {["Platform","GPU","VRAM","Est. Time","Cost"].map(h=><th key={h} style={{textAlign:"left",padding:"8px 12px",color:"var(--t3)",fontWeight:600,fontSize:11,textTransform:"uppercase"}}>{h}</th>)}
              </tr></thead>
              <tbody>
                {[["Colab Pro","A100","40GB","~2.5 hrs","~₹900/mo"],["RunPod","A100","40GB","~2.5 hrs","~$3"],["Kaggle","T4 ×2","16GB","~8-10 hrs","Free"],["Local","RTX 3090","24GB","~5 hrs","Electricity"]].map(r=>(
                  <tr key={r[0]} style={{borderBottom:"1px solid var(--border)"}}>
                    {r.map((c,i)=><td key={i} style={{padding:"12px",color:i===0?"var(--t1)":"var(--t2)"}}>{c}</td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Author */}
        <div className="card text-center" style={{maxWidth:520,margin:"0 auto",padding:"40px"}}>
          <div style={{width:80,height:80,borderRadius:"50%",background:"linear-gradient(135deg,var(--p),var(--p2))",display:"flex",alignItems:"center",justifyContent:"center",fontSize:36,margin:"0 auto 16px"}}>👨‍💻</div>
          <h3 style={{fontWeight:800,fontSize:20,marginBottom:4}}>Vantaku Jagadeesh</h3>
          <p style={{color:"var(--p)",fontSize:13,marginBottom:12}}>B.Tech CS (Data Science & AI) · CSVTU 2026</p>
          <p style={{color:"var(--t2)",fontSize:13,lineHeight:1.7,marginBottom:20}}>Building production ML systems at the intersection of NLP, fine-tuning, and scalable deployment.</p>
          <div style={{display:"flex",gap:12,justifyContent:"center"}}>
            <Link href="https://linkedin.com/in/your-profile" target="_blank" className="btn-secondary" style={{padding:"8px 20px",fontSize:13,borderRadius:8}}>LinkedIn</Link>
            <Link href="https://github.com/your-username" target="_blank" className="btn-secondary" style={{padding:"8px 20px",fontSize:13,borderRadius:8}}>GitHub</Link>
            <Link href="/chat" className="btn-primary" style={{padding:"8px 20px",fontSize:13,borderRadius:8}}>Try the Demo</Link>
          </div>
        </div>
      </div>
    </div>
  );
}
