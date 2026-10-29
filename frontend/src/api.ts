export type Scene = {sky:string; landform:string; ground:string; water:string; objects:string};
export type Plan = {scene:Scene; unsupported:string[]; explanation:string};
export type Result = {image_base64:string; mime_type:string; seed:number; model_version:string; seconds:number; scene:Scene};
const base = (import.meta.env.VITE_API_BASE_URL || '/api').replace(/\/$/,'');

export async function post<T>(path:string,body:unknown,signal:AbortSignal):Promise<T> {
  const response = await fetch(base+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body),signal});
  const data = await response.json().catch(()=>null);
  if(!response.ok) throw new Error(data?.error?.message || (response.status===422 ? 'Please check your prompt and settings.' : 'The service is unavailable. Please try again.'));
  return data as T;
}
