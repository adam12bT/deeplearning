import React from 'react';
import {createRoot} from 'react-dom/client';
import {useRef,useState} from 'react';
import {post} from './api';
import type {Plan,Result} from './api';
import './style.css';

const examples = ['A moonlit village beside a mountain lake','A volcano by the sea at sunset','A river winding through grassy mountains with trees'];

function App(){
  const [prompt,setPrompt]=useState(examples[0]);
  const [plan,setPlan]=useState<Plan|null>(null);
  const [result,setResult]=useState<Result|null>(null);
  const [phase,setPhase]=useState<'idle'|'reading'|'drawing'>('idle');
  const [error,setError]=useState('');const [keepSeed,setKeepSeed]=useState(false);
  const [variation,setVariation]=useState(1);const [zoom,setZoom]=useState(4);
  const controller=useRef<AbortController|null>(null);
  const busy=phase!=='idle';
  async function imagine(){
    setError('');setPlan(null);setPhase('reading');
    const active=new AbortController();controller.current=active;
    const timer=window.setTimeout(()=>active.abort(),180000);
    try{
      const interpreted=await post<Plan>('/v1/interpret',{prompt},active.signal);setPlan(interpreted);
      if(interpreted.unsupported.length){setError('This model cannot reliably draw: '+interpreted.unsupported.join(', ')+'. Try a supported landscape.');return;}
      setPhase('drawing');
      const next=await post<Result>('/v1/generations',{scene:interpreted.scene,seed:keepSeed&&result ? result.seed:null,temperature:variation},active.signal);
      setResult(next);
    }catch(e){setError(e instanceof Error && e.name==='AbortError' ? 'The request was cancelled or took too long. A queued GPU request may still finish.' : e instanceof Error ? e.message : 'Something went wrong.');}
    finally{clearTimeout(timer);setPhase('idle');controller.current=null;}
  }
  const image=result ? `data:${result.mime_type};base64,${result.image_base64}`:'/example.png';
  return <div className="app">
    <header><a className="brand" href="/" aria-label="Small Worlds home"><span className="mark">▦</span> small worlds<span className="brand-note">/ landscape studio</span></a><span className="badge"><i/>FROM SCRATCH, PIXEL BY PIXEL</span></header>
    <main>
      <div className="intro"><span className="eyebrow">A LITTLE PLACE TO GET LOST</span><h1>Your words.<br/><span>A small world.</span></h1><p>Describe a landscape. Let a tiny Transformer imagine it, one pixel patch at a time.</p></div>
      <div className="workspace">
        <section className="composer" aria-label="Scene description">
          <div className="section-title"><span className="step">01</span><h2>Set the scene</h2></div>
          <label htmlFor="prompt">What does your world look like?</label>
          <textarea id="prompt" maxLength={1000} value={prompt} onChange={e=>{setPrompt(e.target.value);setPlan(null);setError('');}} disabled={busy} placeholder="A quiet village under a crescent moon…"/>
          <div className="prompt-footer"><span>Landscapes, not limits of imagination.</span><span>{prompt.length}/1000</span></div>
          <div className="examples"><span>NEED A START?</span>{examples.map((text,i)=><button key={text} disabled={busy} onClick={()=>{setPrompt(text);setPlan(null);setError('');}}>{['☾ Moonlit village','△ Sunset volcano','≈ Mountain river'][i]}<span>↗</span></button>)}</div>
          <details><summary>Fine-tune the feeling <span>+</span></summary><label className="range-label" htmlFor="variation">Variation <span>{variation.toFixed(2)}</span></label><input id="variation" type="range" min="0.5" max="1.2" step="0.05" value={variation} disabled={busy} onChange={e=>setVariation(Number(e.target.value))}/><label className="check"><input type="checkbox" checked={keepSeed} disabled={!result||busy} onChange={e=>setKeepSeed(e.target.checked)}/>Keep the last seed</label><small>A fixed seed reuses the starting randomness. Changing the scene or variation still changes the image.</small></details>
          <button className="generate" disabled={busy||prompt.trim().length<3} onClick={imagine}>{busy ? <><span className="spinner"/>{phase==='reading'?'Reading your scene…':'Drawing your world…'}</>:<>Imagine this world <span>↗</span></>}</button>
          {busy&&<button className="cancel" onClick={()=>controller.current?.abort()}>Cancel request</button>}
          {error&&<div className="error" role="alert">{error}</div>}
          {plan&&<div className="plan"><span className="eyebrow">YOUR SCENE, INTERPRETED</span><p>{plan.explanation}</p><div className="chips">{Object.entries(plan.scene).map(([key,value])=><span key={key} title={key}>{value}</span>)}</div></div>}
          <p className="scope">Made for skies, mountains, volcanoes, rivers, lakes, coasts, trees and villages. New creatures and precise layouts aren’t supported yet.</p>
        </section>
        <section className="output" aria-label="Generated image">
          <div className="output-top"><span><i/>{result?'YOUR LITTLE WORLD':'A GLIMPSE OF WHAT’S POSSIBLE'}</span><span>64 × 64 / PNG</span></div>
          <div className={`canvas ${busy?'working':''}`} aria-busy={busy}><div className="image-border"><img src={image} width={64*zoom} height={64*zoom} alt={result ? `Generated ${result.scene.sky} landscape with ${result.scene.landform}`:'Example generated pixel-art mountain lake at night'}/></div>{busy&&<span className="working-label">{phase==='reading'?'Finding the scene':'Waiting for the GPU · then drawing'}</span>}</div>
          <div className="output-tools"><div className="zooms" aria-label="Preview zoom">{[1,2,4].map(n=><button key={n} aria-pressed={zoom===n} onClick={()=>setZoom(n)}>{n}×</button>)}</div><a className="download" href={image} download={`small-world-${result?.seed??'example'}.png`}>Download PNG <span>↓</span></a></div>
          <div className="metadata"><span>{result?`SEED ${result.seed}`:'EXAMPLE IMAGE'}</span><span>{result?result.model_version:'LARGER TRANSFORMER'}</span><span>{result?`${result.seconds.toFixed(1)}s`:'NATIVE PIXEL ART'}</span></div>
          <p className="caption">{result?'A new interpretation, not a guaranteed blueprint. Some requested features may be missing.':'Every image starts with a seed. Yours will be chosen at random.'}</p>
        </section>
      </div>
    </main><footer><span>SMALL WORLDS, MANY POSSIBILITIES.</span><span>Text → scene controls → your own image model</span></footer>
  </div>;
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
