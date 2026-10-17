// Browser smoke checks against the production build. All generation is mocked.
// Run: npm run build && node scripts/smoke-ui.mjs (requires Chrome and Node >=22).
import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {mkdtemp, rm, writeFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {preview} from 'vite';

const profile = await mkdtemp(join(tmpdir(), 'sartiq-ui-'));
const server = await preview({preview: {host: '127.0.0.1', port: 4179, strictPort: true}});
const chrome = spawn(process.env.CHROME_BIN || 'google-chrome', [
  '--headless=new', '--no-sandbox', '--disable-gpu', '--disable-background-networking',
  '--disable-extensions', '--disable-sync', '--no-first-run', '--no-default-browser-check',
  '--autoplay-policy=no-user-gesture-required',
  '--remote-debugging-port=0', `--user-data-dir=${profile}`, 'about:blank',
], {stdio: ['ignore', 'ignore', 'pipe']});
let socket;
try {
  const endpoint = await new Promise((resolve, reject) => {
    let stderr = '';
    const timer = setTimeout(() => reject(new Error('Chrome did not start')), 10000);
    chrome.once('error', error => {clearTimeout(timer); reject(error);});
    chrome.stderr.on('data', chunk => {
      stderr += chunk;
      const match = stderr.match(/DevTools listening on (ws:\/\/\S+)/);
      if (match) {clearTimeout(timer); resolve(match[1]);}
    });
  });
  socket = new WebSocket(endpoint);
  await new Promise(resolve => socket.addEventListener('open', resolve, {once: true}));
  let nextId = 0;
  const pending = new Map();
  const listeners = new Map();
  const send = (method, params = {}, sessionId) => new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, {resolve, reject});
    socket.send(JSON.stringify({id, method, params, sessionId}));
  });
  socket.addEventListener('message', ({data}) => {
    const message = JSON.parse(data);
    if (message.id) {
      const handler = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) handler?.reject(new Error(JSON.stringify(message.error)));
      else handler?.resolve(message.result);
    } else listeners.get(message.sessionId)?.(message);
  });
  const fixture = `
    window.__cameraRequests = 0; window.__stopped = 0;
    const canvas = document.createElement('canvas'); canvas.width=480; canvas.height=640;
    const context=canvas.getContext('2d'); context.fillStyle='#e8e8e1'; context.fillRect(0,0,480,640);
    context.fillStyle='#6a8060'; context.beginPath();
    context.moveTo(160,150); context.lineTo(80,210); context.lineTo(100,290);
    context.lineTo(155,255); context.lineTo(155,490); context.lineTo(325,490);
    context.lineTo(325,255); context.lineTo(380,290); context.lineTo(400,210);
    context.lineTo(320,150); context.closePath(); context.fill();
    window.__fixture = canvas.toDataURL('image/png');
    Object.defineProperty(navigator.mediaDevices, 'getUserMedia', {value: async () => {
      window.__cameraRequests++;
      if (window.__denyCamera) throw new DOMException('Denied','NotAllowedError');
      const stream=canvas.captureStream(5);
      const tick=setInterval(()=>{context.fillStyle='#e8e8e1';context.fillRect(0,0,2,2);},100);
      for (const track of stream.getTracks()) {
        const stop=track.stop.bind(track); track.stop=()=>{window.__stopped++; clearInterval(tick); stop();};
      }
      return stream;
    }});
  `;
  for (const mode of ['desktop', 'narrow-desktop', 'mobile', 'mobile-denied', 'desktop-unverified']) {
    const mobile = mode.startsWith('mobile');
    const {targetId} = await send('Target.createTarget', {url: 'about:blank'});
    const {sessionId} = await send('Target.attachToTarget', {targetId, flatten: true});
    const command = (method, params) => send(method, params, sessionId);
    const errors = [];
    let requests = 0;
    let filePickers = 0;
    listeners.set(sessionId, event => {
      if (event.method === 'Page.fileChooserOpened') filePickers++;
      if (event.method === 'Runtime.exceptionThrown') errors.push(event.params.exceptionDetails.text);
      if (event.method === 'Fetch.requestPaused') {
        const {requestId, request} = event.params;
        const isEvents = request.url.endsWith('/events');
        if (!isEvents) requests++;
        const unverified = mode === 'desktop-unverified';
        const job = {job_id: 'mock', status: unverified ? 'review_required' : 'complete', result_url: resultImage,
          quality: {single_product:{status:unverified ? 'unverified' : 'passed'}},
          images: [{id:'final', label:'Final output', url:resultImage}],
          execution_trace: [{id:'restyle', label:'Creating photo', model:'Mock model',
            libraries:{}, python:'3', status:'complete', seconds:.1, details:{}}]};
        void command('Fetch.fulfillRequest', {requestId, responseCode: isEvents ? 200 : 202,
          responseHeaders: [{name:'Content-Type', value:isEvents ? 'text/event-stream' : 'application/json'}],
          body:Buffer.from(isEvents ? `data: ${JSON.stringify(job)}\n\n` : JSON.stringify({job_id:'mock'})).toString('base64')});
      }
    });
    await command('Page.enable'); await command('Runtime.enable');
    await command('Page.setInterceptFileChooserDialog', {enabled:true});
    await command('Fetch.enable', {patterns:[{urlPattern:'*://*/api/*'}]});
    await command('Emulation.setDeviceMetricsOverride', {
      width:mode.startsWith('desktop') ? 1440 : 390, height:mode.startsWith('desktop') ? 1100 : 844,
      deviceScaleFactor:1, mobile,
    });
    if (mobile) {
      await command('Emulation.setUserAgentOverride', {userAgent:'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Version/17.0 Mobile/15E148 Safari/604.1', platform:'iPhone'});
      await command('Emulation.setTouchEmulationEnabled', {enabled:true});
    }
    await command('Page.addScriptToEvaluateOnNewDocument', {
      source:(mode === 'mobile-denied' ? 'window.__denyCamera=true;' : '') + fixture,
    });
    const evaluate = async expression => {
      const result = await command('Runtime.evaluate', {expression, returnByValue:true, awaitPromise:true, userGesture:true});
      if (result.exceptionDetails) throw new Error(`${mode}: ${result.exceptionDetails.exception?.description || result.exceptionDetails.text}\n${expression}`);
      return result.result.value;
    };
    const waitFor = async expression => {
      const deadline = Date.now()+8000;
      while (Date.now()<deadline) {
        if (await evaluate(expression)) return;
        await new Promise(resolve => setTimeout(resolve, 40));
      }
      throw new Error(`${mode}: timed out waiting for ${expression}`);
    };
    await command('Page.navigate', {url:'http://127.0.0.1:4179'});
    await command('Page.bringToFront');
    await waitFor('!!document.querySelector(".studio")');
    const resultImage = await evaluate('window.__fixture');
    if (mobile) {
      await waitFor('document.querySelector(".camera-screen")?.open && window.__cameraRequests > 0');
      if (mode === 'mobile-denied') {
        await waitFor('!!document.querySelector(".camera-placeholder .secondary")');
        assert.equal(await evaluate('!!document.querySelector(".camera-library")'), true);
        await evaluate('document.querySelector(".camera-library").click()');
        await waitFor('!document.querySelector(".camera-screen")');
        assert.equal(filePickers, 1, 'Denied camera must still allow photo library');
      } else {
        await waitFor('!document.querySelector(".shutter").disabled');
        const cameraScreenshot = await command('Page.captureScreenshot', {format:'png'});
        await writeFile(join(tmpdir(), 'sartiq-camera.png'), Buffer.from(cameraScreenshot.data, 'base64'));
        await evaluate('document.querySelector(".shutter").click()');
        await waitFor('!document.querySelector(".camera-screen") && document.querySelectorAll(".view-thumbnail").length === 1');
        assert.ok(await evaluate('window.__stopped > 0'), 'Camera tracks must stop after capture');
      }
    } else {
      assert.equal(await evaluate('window.__cameraRequests'), 0);
      assert.equal(await evaluate('document.querySelectorAll("[capture]").length'), 0);
      assert.equal(await evaluate('document.querySelectorAll(".camera-screen").length'), 0);
      await evaluate('document.querySelector(".upload").click()');
      assert.equal(filePickers, 1, 'Desktop must open the file picker');
    }
    // Upload two local synthetic photos, then exercise actual UI/SSE transitions.
    await evaluate(`(async()=>{
      const blob=await (await fetch(window.__fixture)).blob(); const transfer=new DataTransfer();
      transfer.items.add(new File([blob],'front.png',{type:'image/png'}));
      transfer.items.add(new File([blob],'detail.png',{type:'image/png'}));
      const input=document.querySelector('input[type=file]:not([capture])');
      input.files=transfer.files; input.dispatchEvent(new Event('change',{bubbles:true}));
    })()`);
    await waitFor('document.querySelectorAll(".view-thumbnail").length===2');
    await evaluate(`document.querySelector('[aria-label="Use photo 2 as main"]').click()`);
    await waitFor('document.querySelector(".filename").textContent==="detail.png"');
    await evaluate('document.querySelector(".primary").click()');
    await waitFor(`document.querySelector(".ready")?.textContent === ${JSON.stringify(mode === 'desktop-unverified' ? 'Review' : 'Ready')}`);
    if (mode === 'desktop-unverified') {
      assert.ok(await evaluate('document.querySelector(".review-note")?.textContent.includes("Product count could not be verified")'));
    }
    assert.equal(requests, 1);
    await evaluate('document.querySelector(".actions .secondary").click()');
    await waitFor('document.querySelector(".comparison-screen")?.open');
    await evaluate(`document.querySelector('[aria-label="Close comparison"]').click()`);
    await evaluate('document.querySelector(".history-toggle").click()');
    await waitFor('!!document.querySelector("#image-history")');
    assert.equal(await evaluate('document.documentElement.scrollWidth <= innerWidth'), true, 'No horizontal page overflow');
    if (!mobile) assert.equal(await evaluate('window.__cameraRequests'), 0);
    assert.deepEqual(errors, [], 'No browser runtime exceptions');
    const screenshot = await command('Page.captureScreenshot', {format:'png'});
    await writeFile(join(tmpdir(), `sartiq-${mode}.png`), Buffer.from(screenshot.data, 'base64'));
    console.log(`PASS ${mode}: input, camera policy, multi-photo, generation, SSE, comparison, history`);
    await send('Target.closeTarget', {targetId}); listeners.delete(sessionId);
  }
} finally {
  socket?.close();
  chrome.kill('SIGTERM');
  await new Promise(resolve => server.httpServer.close(resolve));
  await rm(profile, {recursive:true, force:true, maxRetries:5, retryDelay:100});
}
