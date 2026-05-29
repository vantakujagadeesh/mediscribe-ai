"use client";
import { useState, useEffect } from "react";

const API = "";  // Next.js proxy rewrites /api/* → backend
type Drug = { name:string; class:string; uses:string; mechanism:string; side_effects:string; dose:string };

export default function DrugsPage() {
  const [drugs, setDrugs] = useState<Drug[]>([]);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<Drug|null>(null);
  const [loading, setLoading] = useState(false);

  async function search(q: string) {
    setLoading(true);
    try {
      const r = await fetch(`${API}/api/drugs?q=${encodeURIComponent(q)}`);
      const d = await r.json();
      setDrugs(d.drugs || []);
    } catch { setDrugs([]); }
    finally { setLoading(false); }
  }

  useEffect(() => { search(""); }, []);
  useEffect(() => {
    const t = setTimeout(() => search(query), 300);
    return () => clearTimeout(t);
  }, [query]);

  return (
    <div className="page" style={{minHeight:"100vh"}}>
      <div className="bg-scene"><div className="bg-grid"/><div className="orb orb1"/><div className="orb orb2"/></div>
      <div className="container" style={{position:"relative",zIndex:1,paddingTop:"48px",paddingBottom:"80px"}}>
        <div className="page-header text-center">
          <div className="section-tag">💊 Drug Encyclopedia</div>
          <h1 className="section-title">Drug Database</h1>
          <p style={{color:"var(--t2)",maxWidth:520,margin:"0 auto",fontSize:15,lineHeight:1.7}}>Search our clinical drug database for mechanism of action, dosing, indications, and side-effect profiles.</p>
        </div>

        <div style={{maxWidth:600,margin:"0 auto 32px"}}>
          <input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search by name, class, or indication…" className="search-bar" />
        </div>

        {loading&&<p style={{textAlign:"center",color:"var(--t2)"}}>Searching…</p>}

        {selected ? (
          <div className="card fade-in" style={{maxWidth:700,margin:"0 auto"}}>
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 style={{fontWeight:800,fontSize:22}}>{selected.name}</h2>
                <div style={{color:"var(--p)",fontSize:13,fontWeight:600,marginTop:4}}>{selected.class}</div>
              </div>
              <button onClick={()=>setSelected(null)} style={{background:"rgba(99,130,255,.1)",border:"1px solid var(--border)",borderRadius:8,padding:"8px 14px",color:"var(--t2)",cursor:"pointer",fontSize:13}}>← Back</button>
            </div>
            {[
              ["💊 Uses / Indications", selected.uses],
              ["⚙️ Mechanism of Action", selected.mechanism],
              ["⚠️ Side Effects", selected.side_effects],
              ["💉 Dosing", selected.dose],
            ].map(([label,val])=>(
              <div key={label} style={{marginBottom:20,padding:"16px 18px",background:"rgba(8,15,35,.6)",borderRadius:12,border:"1px solid var(--border)"}}>
                <div style={{fontWeight:700,fontSize:13,marginBottom:8,color:"var(--t1)"}}>{label}</div>
                <p style={{fontSize:14,color:"var(--t2)",lineHeight:1.7}}>{val}</p>
              </div>
            ))}
          </div>
        ) : (
          <div className="drugs-grid">
            {drugs.map(d=>(
              <div key={d.name} className="drug-card" onClick={()=>setSelected(d)}>
                <div className="drug-name">{d.name}</div>
                <div className="drug-class">{d.class}</div>
                <div className="drug-row"><span className="drug-label">Uses</span><span className="drug-val">{d.uses}</span></div>
                <div className="drug-row"><span className="drug-label">Dose</span><span className="drug-val font-mono" style={{fontSize:12}}>{d.dose}</span></div>
                <div style={{marginTop:12,fontSize:12,color:"var(--p)",fontWeight:600}}>Click for full details →</div>
              </div>
            ))}
            {!loading&&drugs.length===0&&<p style={{color:"var(--t2)",gridColumn:"1/-1",textAlign:"center",padding:"40px 0"}}>No drugs found for "{query}"</p>}
          </div>
        )}
      </div>
    </div>
  );
}
