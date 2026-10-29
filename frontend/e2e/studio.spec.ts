import {test,expect} from '@playwright/test';
import {readFileSync} from 'node:fs';
const scene={sky:'night',landform:'mountains',ground:'rock',water:'lake',objects:'settlement'};
const image=readFileSync('public/example.png').toString('base64');

test('text interpretation, generation, download and seed reuse',async({page})=>{
  const seeds:(number|null)[]=[];
  await page.route('**/api/v1/interpret',route=>route.fulfill({json:{scene,unsupported:[],explanation:'A moonlit village by a lake.'}}));
  await page.route('**/api/v1/generations',route=>{
    seeds.push(route.request().postDataJSON().seed);
    return route.fulfill({json:{image_base64:image,mime_type:'image/png',seed:42,model_version:'landscape-large-v1',seconds:2.7,scene}});
  });
  await page.goto('/');await page.getByRole('button',{name:'Imagine this world'}).click();
  await expect(page.getByText('SEED 42')).toBeVisible();
  await expect(page.getByText('landscape-large-v1')).toBeVisible();
  await expect(page.getByRole('link',{name:'Download PNG'})).toHaveAttribute('download','small-world-42.png');
  await page.getByText('Fine-tune the feeling').click();
  await page.getByLabel('Keep the last seed').check();
  await page.getByRole('button',{name:'Imagine this world'}).click();
  await expect.poll(()=>seeds.length).toBe(2);expect(seeds).toEqual([null,42]);
  await page.screenshot({path:'../runs/frontend_review/desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:'../runs/frontend_review/mobile.png',fullPage:true});
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBeTruthy();
});

test('unsupported concepts do not consume a GPU request',async({page})=>{
  let generated=false;
  await page.route('**/api/v1/interpret',route=>route.fulfill({json:{scene,unsupported:['dragon'],explanation:'A lake, but dragons are unsupported.'}}));
  await page.route('**/api/v1/generations',route=>{generated=true;return route.abort();});
  await page.goto('/');await page.getByLabel('What does your world look like?').fill('A dragon by the lake');
  await page.getByRole('button',{name:'Imagine this world'}).click();
  await expect(page.getByRole('alert')).toContainText('dragon');expect(generated).toBeFalsy();
});

test('GPU quota is shown as an actionable error',async({page})=>{
  await page.route('**/api/v1/interpret',route=>route.fulfill({json:{scene,unsupported:[],explanation:'A lake.'}}));
  await page.route('**/api/v1/generations',route=>route.fulfill({status:429,json:{error:{code:'gpu_quota',message:'GPU quota is exhausted. Wait for its reset.'}}}));
  await page.goto('/');await page.getByRole('button',{name:'Imagine this world'}).click();
  await expect(page.getByRole('alert')).toContainText('quota is exhausted');
  await expect(page.getByRole('button',{name:'Imagine this world'})).toBeEnabled();
});
