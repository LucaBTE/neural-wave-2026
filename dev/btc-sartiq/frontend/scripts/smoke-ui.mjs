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
  '--headless=new', '--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--disable-background-networking',
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
    Object.defineProperty(navigator.mediaDevices, 'enumerateDevices', {value: async () => window.__optics ? [
      {kind:'videoinput', deviceId:'rear', label:'Back Camera'},
      {kind:'videoinput', deviceId:'ultra', label:'Back Ultra Wide Camera'},
      {kind:'videoinput', deviceId:'broken', label:'Unavailable lens'},
    ] : []});
    Object.defineProperty(navigator.mediaDevices, 'getUserMedia', {value: async (options) => {
      window.__cameraRequests++;
      if (window.__denyCamera) throw new DOMException('Denied','NotAllowedError');
      const deviceId = options.video.deviceId?.exact || 'rear';
      if (deviceId === 'broken') throw new DOMException('Unavailable','NotReadableError');
      window.__lastDevice = deviceId;
      const stream=canvas.captureStream(5);
      const cameraTrack = stream.getVideoTracks()[0];
      if (window.__optics) {
        let zoom = 2;
        cameraTrack.getCapabilities = () => ({zoom:{min:.5, max:4, step:.1}});
        const settings = cameraTrack.getSettings.bind(cameraTrack);
        cameraTrack.getSettings = () => ({...settings(), deviceId, zoom});
        cameraTrack.applyConstraints = async (constraints) => {
          if (window.__zoomError) throw new DOMException('Unsupported','OverconstrainedError');
          const values = constraints.advanced.filter(item => item.zoom !== undefined);
          if (values.length !== 1) throw new Error('Conflicting zoom constraints');
          zoom = values[0].zoom; window.__appliedZoom = zoom;
        };
      } else cameraTrack.getCapabilities = () => ({});
      const tick=setInterval(()=>{context.fillStyle='#e8e8e1';context.fillRect(0,0,2,2);},100);
      for (const track of stream.getTracks()) {
        const stop=track.stop.bind(track); track.stop=()=>{window.__stopped++; clearInterval(tick); stop();};
      }
      return stream;
    }});
  `;
  let cameraFrameBaseline;
  for (const mode of ['desktop', 'desktop-laptop', 'narrow-desktop', 'mobile', 'mobile-multi', 'mobile-denied', 'mobile-zoom', 'mobile-zoom-error', 'mobile-reduced-motion', 'fault-load', 'fault-decode', 'fault-unmount', 'desktop-unverified']) {
    const mobile = mode.startsWith('mobile');
    const {targetId} = await send('Target.createTarget', {url: 'about:blank'});
    const {sessionId} = await send('Target.attachToTarget', {targetId, flatten: true});
    const command = (method, params) => send(method, params, sessionId);
    const errors = [];
    let requests = 0;
    let filePickers = 0;
    let failure = null;
    listeners.set(sessionId, event => {
      if (event.method === 'Page.fileChooserOpened') filePickers++;
      if (event.method === 'Runtime.exceptionThrown') errors.push(event.params.exceptionDetails.text);
      if (event.method === 'Fetch.requestPaused') {
        const {requestId, request} = event.params;
        if (request.url.endsWith('/test-output.png')) {
          setTimeout(() => void command('Fetch.fulfillRequest', {requestId, responseCode:200,
            responseHeaders:[{name:'Content-Type',value:'image/png'},{name:'Cache-Control',value:'public, max-age=3600'}],
            body:resultImage.split(',')[1]}), 1200);
          return;
        }
        const isEvents = request.url.endsWith('/events');
        if (!isEvents) requests++;
        if (!isEvents && failure) {
          void command('Fetch.fulfillRequest', {requestId, responseCode:failure.status,
            responseHeaders:[{name:'Content-Type', value:failure.contentType || 'text/html'}],
            body:Buffer.from(failure.body).toString('base64')});
          return;
        }
        const job = {job_id: 'mock', status: mode === 'desktop-unverified' ? 'review_required' : 'complete', quality: {single_product:{status:mode === 'desktop-unverified' ? 'unverified' : 'passed'}}, result_url: mode === 'fault-load' ? 'data:image/png;base64,broken' : mode === 'desktop' ? '/test-output.png' : resultImage,
          images: [{id:'final', label:'Final output', url:resultImage}],
          execution_trace: [{id:'restyle', label:'Creating photo', model:'Mock model',
            libraries:{}, python:'3', status:'complete', seconds:.1, details:{}}]};
        setTimeout(() => void command('Fetch.fulfillRequest', {requestId, responseCode: isEvents ? 200 : 202,
          responseHeaders: [{name:'Content-Type', value:isEvents ? 'text/event-stream' : 'application/json'}],
          body:Buffer.from(isEvents ? `data: ${JSON.stringify(job)}\n\n` : JSON.stringify({job_id:'mock'})).toString('base64')}), isEvents && mode === 'desktop' ? 2500 : 250);
      }
    });
    await command('Page.enable'); await command('Runtime.enable');
    if (mode === 'mobile-reduced-motion') await command('Emulation.setEmulatedMedia', {features:[{name:'prefers-reduced-motion',value:'reduce'}]});
    await command('Page.setInterceptFileChooserDialog', {enabled:true});
    await command('Fetch.enable', {patterns:[{urlPattern:'*://*/api/*'},{urlPattern:'*://*/test-output.png'}]});
    await command('Emulation.setDeviceMetricsOverride', {
      width:(mode === 'desktop' || mode === 'desktop-unverified') ? 1440 : mode === 'desktop-laptop' ? 1280 : 390, height:(mode === 'desktop' || mode === 'desktop-unverified') ? 1100 : mode === 'desktop-laptop' ? 800 : 844,
      deviceScaleFactor:1, mobile,
    });
    if (mobile) {
      await command('Emulation.setUserAgentOverride', {userAgent:'Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Version/17.0 Mobile/15E148 Safari/604.1', platform:'iPhone'});
      await command('Emulation.setTouchEmulationEnabled', {enabled:true});
    }
    await command('Page.addScriptToEvaluateOnNewDocument', {
      source:(mode === 'mobile-denied' ? 'window.__denyCamera=true;' : '') + (mode.startsWith('mobile-zoom') ? `window.__optics=true;window.__zoomError=${mode === 'mobile-zoom-error'};` : '') + fixture,
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
    const resultImage = await evaluate(`(() => {const canvas=document.createElement('canvas'); canvas.width=600; canvas.height=900; const ctx=canvas.getContext('2d'); ctx.fillStyle='#efedf2'; ctx.fillRect(0,0,600,900); return canvas.toDataURL('image/png');})()`);
    assert.equal(await evaluate('window.__cameraRequests'), 0, 'Home must not request the camera');
    assert.equal(await evaluate('!!document.querySelector(".editorial-showcase")'), true);
    assert.equal(await evaluate('document.querySelector(".editorial-showcase h1").textContent'), 'Your photo.A better presentation');
    if (mobile) {
      const homeLayout = await evaluate(`(() => {const button=document.querySelector('.home-upload button').getBoundingClientRect(); const slider=document.querySelector('.editorial-reveal').getBoundingClientRect(); return {below:button.top>slider.bottom,camera:!!document.querySelector('.home-upload .lucide-camera')};})()`);
      assert.deepEqual(homeLayout, {below:true,camera:true});
      const mobileHomeShot = await command('Page.captureScreenshot', {format:'png'});
      await writeFile(join(tmpdir(), `sartiq-home-${mode}.png`), Buffer.from(mobileHomeShot.data, 'base64'));
    }
    assert.equal(await evaluate('!!document.querySelector(".camera-screen")'), false);
    const pipelineHome = await evaluate(`(() => {const pipeline=document.querySelector('.home-pipeline').getBoundingClientRect(); const slider=document.querySelector('.editorial-reveal').getBoundingClientRect(); const button=document.querySelector('.home-upload').getBoundingClientRect(); return {below:pipeline.top>=slider.bottom,above:pipeline.bottom<=button.top,steps:document.querySelectorAll('.home-pipeline li').length};})()`);
    if (mode.startsWith('desktop')) {
      const homeColumns = await evaluate(`(() => {const slider=document.querySelector('.editorial-reveal').getBoundingClientRect();return Array.from(document.querySelectorAll('.editorial-showcase h1,.home-pipeline,.home-upload,.desktop-upload-hint')).every(el=>el.getBoundingClientRect().right<slider.left);})()`);
      assert.equal(homeColumns,true,'Desktop home keeps only the slider on the right');
    } else assert.deepEqual(pipelineHome, {below:true,above:true,steps:3}, 'Mobile home pipeline sits between the comparison slider and upload button');
    for (const position of [0, 25, 100]) {
      await evaluate(`(() => {const slider=document.querySelector('.editorial-reveal input'); Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(slider, '${position}'); slider.dispatchEvent(new Event('input',{bubbles:true}));})()`);
      assert.equal(await evaluate('document.querySelector(".editorial-reveal__result").style.clipPath'), `inset(0px 0px 0px ${position}%)`);
      assert.equal(await evaluate('document.querySelector(".editorial-reveal__line").style.left'), `${position}%`);
    }
    if (mode.startsWith('desktop')) {
      assert.equal(await evaluate('document.querySelector(".editorial-showcase").getBoundingClientRect().bottom < innerHeight - 24'), true, 'Home card leaves a bottom margin');
      const homeShot = await command('Page.captureScreenshot', {format:'png'});
      await writeFile(join(tmpdir(), `sartiq-home-${mode}.png`), Buffer.from(homeShot.data, 'base64'));
    }
    if (mobile) {
      await evaluate('document.querySelector(".home-upload button").click()');
      await waitFor('document.querySelector(".camera-screen")?.open && window.__cameraRequests > 0');
      assert.equal(await evaluate('document.activeElement === document.querySelector(".camera-screen")'), true, 'Opening the camera must not autofocus a mode button');
      assert.equal(await evaluate('getComputedStyle(document.querySelector(".camera-mode button")).outlineStyle'), 'none');
      if (mode === 'mobile-denied') {
        await waitFor('!!document.querySelector(".camera-placeholder .secondary")');
        assert.equal(await evaluate('!!document.querySelector(".camera-library")'), true);
        await evaluate('document.querySelector(".camera-library").click()');
        assert.equal(filePickers, 1, 'Denied camera must still allow photo library');
        assert.equal(await evaluate('document.querySelector(".camera-screen").open'), true, 'Canceling the library must leave the camera open');
        await evaluate(`(async()=>{const blob=await (await fetch(window.__fixture)).blob(); const transfer=new DataTransfer(); transfer.items.add(new File([blob],'library.png',{type:'image/png'})); const input=document.querySelector('.camera-screen input[type=file]:not([capture])'); input.files=transfer.files; input.dispatchEvent(new Event('change',{bubbles:true}));})()`);
        await waitFor('!document.querySelector(".camera-screen") && document.querySelectorAll(".view-thumbnail").length === 1');
      } else {
        await waitFor('!document.querySelector(".shutter").disabled');
        if (mode.startsWith('mobile-zoom')) {
          await waitFor('document.querySelectorAll(".camera-zoom button").length === 3');
          assert.equal(await evaluate('!!document.querySelector(".camera-screen input[type=range], .camera-lens, .camera-optics")'), false);
          await evaluate(`document.querySelector('[aria-label="Zoom 0.5×"]').click()`);
          if (mode === 'mobile-zoom-error') {
            await waitFor('document.querySelector(".camera-control-error")?.textContent.includes("Unable to adjust zoom")');
            assert.equal(await evaluate('document.querySelector(".camera-zoom button[aria-pressed=true]").textContent'), '2×');
          } else {
            await waitFor('window.__appliedZoom === .5');
            assert.equal(await evaluate('document.querySelector(".camera-zoom button[aria-pressed=true]").textContent'), '0.5×');
            await evaluate(`document.querySelector('[aria-label="Zoom 1×"]').click()`);
            await waitFor('window.__appliedZoom === 1');
          }
          assert.equal(await evaluate('document.querySelector(".camera-zoom").parentElement.className'), 'camera-viewport');
        } else assert.equal(await evaluate('!!document.querySelector(".camera-zoom")'), false);
        assert.equal(await evaluate('getComputedStyle(document.querySelector(".camera-viewport video")).objectFit'), 'cover');
        const frameSize = await evaluate('(() => {const frame=document.querySelector(".camera-viewport").getBoundingClientRect(); return {width:frame.width,height:frame.height};})()');
        if (mode === 'mobile') cameraFrameBaseline = frameSize;
        if (mode.startsWith('mobile-zoom')) assert.deepEqual(frameSize, cameraFrameBaseline, 'Zoom controls and errors must not change the camera frame dimensions');
        const cameraRatio = await evaluate('(() => {const frame=document.querySelector(".camera-viewport").getBoundingClientRect(); return frame.width/frame.height;})()');
        const cameraScreenshot = await command('Page.captureScreenshot', {format:'png'});
        await writeFile(join(tmpdir(), 'sartiq-camera.png'), Buffer.from(cameraScreenshot.data, 'base64'));
        if (mode === 'mobile-multi') {
          const modeStart = await evaluate('getComputedStyle(document.querySelector(".camera-mode"), "::before").transform');
          assert.equal(await evaluate('getComputedStyle(document.querySelector(".camera-mode"), "::before").transitionDuration'), '0.33s');
          await evaluate('document.querySelectorAll(".camera-mode button")[1].click()');
          await waitFor(`getComputedStyle(document.querySelector('.camera-mode'), '::before').transform !== ${JSON.stringify(modeStart)}`);
          for (let count = 1; count <= 2; count++) {
            await evaluate('document.querySelector(".shutter").click()');
            await waitFor(`document.querySelector('.camera-badge').textContent === '${count}'`);
          }
          assert.equal(requests, 0, 'Multi-photo mode waits for confirmation');
          await evaluate('document.querySelector(".camera-deck").click()');
          await waitFor('document.querySelectorAll(".camera-review img").length === 2');
          await evaluate('document.querySelector(".camera-confirm").click()');
        } else await evaluate('document.querySelector(".shutter").click()');
        await waitFor('!document.querySelector(".camera-screen") && document.querySelector(".ready")?.textContent === "Ready"');
        assert.equal(requests, 1, 'Capture must start generation once');
        await waitFor('document.querySelector(".photo.upload img")?.naturalWidth > 0');
        const capturedRatio = await evaluate('(() => {const img=document.querySelector(".photo.upload img"); return img.naturalWidth/img.naturalHeight;})()');
        assert.ok(Math.abs(capturedRatio - cameraRatio) < .01, 'Captured crop must match the visible camera frame');
        assert.ok(await evaluate('window.__stopped > 0'), 'Camera tracks must stop after capture');
      }
    } else {
      assert.equal(await evaluate('window.__cameraRequests'), 0);
      assert.equal(await evaluate('document.querySelectorAll("[capture]").length'), 0);
      assert.equal(await evaluate('document.querySelectorAll(".camera-screen").length'), 0);
      await evaluate('document.querySelector(".upload").click()');
      assert.equal(filePickers, 1, 'Desktop must open the file picker');
    }
    await evaluate('document.querySelector(".site-header .brand").click()');
    await waitFor('!!document.querySelector(".editorial-showcase")');
    requests = 0;
    // Upload two local synthetic photos, then exercise actual UI/SSE transitions.
    await evaluate(`(async()=>{
      const blob=await (await fetch(window.__fixture)).blob(); const transfer=new DataTransfer();
      transfer.items.add(new File([blob],'front.png',{type:'image/png'}));
      const detailCanvas=document.createElement('canvas'); detailCanvas.width=768; detailCanvas.height=512;
      const bitmap=await createImageBitmap(blob); detailCanvas.getContext('2d').drawImage(bitmap,0,0,768,512); bitmap.close();
      const detailBlob=await new Promise(resolve=>detailCanvas.toBlob(resolve,'image/png'));
      transfer.items.add(new File([detailBlob],'detail.png',{type:'image/png'}));
      const input=document.querySelector('input[type=file]:not([capture])');
      if (${JSON.stringify(mode)} === 'desktop-laptop') {
        document.querySelector('.studio').dispatchEvent(new DragEvent('drop',{bubbles:true,cancelable:true,dataTransfer:transfer}));
      } else {input.files=transfer.files; input.dispatchEvent(new Event('change',{bubbles:true}));}
    })()`);
    await waitFor('document.querySelectorAll(".view-thumbnail").length===2');
    if (mode.startsWith('desktop')) {
      assert.equal(await evaluate('getComputedStyle(document.querySelector(".workbench .comparison")).display'), 'grid');
      assert.equal(await evaluate('!!document.querySelector(".desktop-workspace-heading p, .desktop-workspace-heading button")'), false);
      assert.equal(await evaluate('getComputedStyle(document.querySelector(".desktop-workspace-heading")).textAlign'), 'center');
      const stages = await evaluate(`(() => {const original=document.querySelector('.reference-panel').getBoundingClientRect(); const result=document.querySelector('.result-panel').getBoundingClientRect(); const connector=document.querySelector('.pipeline-connector').getBoundingClientRect(); return {equal:Math.abs(original.width-result.width)<1 && Math.abs(original.height-result.height)<1, between:connector.left>original.right && connector.right<result.left};})()`);
      assert.deepEqual(stages, {equal:true,between:true}, 'Pipeline stages must be equal with the generate action between them');
      const ratios = await evaluate(`Array.from(document.querySelectorAll('.reference-panel .photo,.result-panel .photo')).map(el=>{const r=el.getBoundingClientRect();return r.width/r.height;})`);
      ratios.forEach(ratio => assert.ok(Math.abs(ratio - .75) < .005, 'Desktop image frames are portrait 3:4'));
      assert.equal(await evaluate('getComputedStyle(document.querySelector(".reference-panel .photo img")).objectFit'), 'contain');
      const thumbnailRail = await evaluate(`(() => {const thumbnails=Array.from(document.querySelectorAll('.view-thumbnail')).map(el=>el.getBoundingClientRect()); const card=document.querySelector('.reference-panel').getBoundingClientRect();return thumbnails[0].right<card.left && thumbnails[1].top>thumbnails[0].bottom && Math.abs(thumbnails[0].left-thumbnails[1].left)<1;})()`);
      assert.equal(thumbnailRail,true,'Reference thumbnails form a vertical rail to the left of the original');
      const prepareShot = await command('Page.captureScreenshot', {format:'png'});
      await writeFile(join(tmpdir(), `sartiq-prepare-${mode}.png`), Buffer.from(prepareShot.data, 'base64'));
    }
    await evaluate(`document.querySelector('[aria-label="Use photo 2 as main"]').click()`);
    await waitFor('document.querySelector(".filename").textContent==="detail.png"');
    if (mode.startsWith('desktop')) {
      await waitFor('Math.abs(Number(document.querySelector(".workbench").style.getPropertyValue("--reference-aspect")) - 1.5) < .001');
      const frames = await evaluate(`Array.from(document.querySelectorAll('.reference-panel .photo,.result-panel .photo')).map(el=>{const r=el.getBoundingClientRect();return {width:r.width,ratio:r.width/r.height};})`);
      frames.forEach(frame => {assert.ok(Math.abs(frame.ratio-1.5)<.005, 'Frames follow the exact landscape input ratio');assert.ok(frame.width>=320, 'Desktop photo remains large enough to inspect');});
      const landscapeShot = await command('Page.captureScreenshot', {format:'png'});
      await writeFile(join(tmpdir(), `sartiq-landscape-${mode}.png`), Buffer.from(landscapeShot.data, 'base64'));
    }
    if (mode === 'desktop') {
      for (const testCase of [
        {status:500, body:'', expected:'Generation service unavailable (HTTP 500)'},
        {status:502, body:'<html>Bad gateway</html>', expected:'Generation service unavailable (HTTP 502)'},
        {status:413, body:'{"detail":"Maximum 12 MB per photo"}', contentType:'application/json', expected:'Maximum 12 MB per photo'},
        {status:200, body:'<html>Wrong proxy target</html>', expected:'invalid response'},
        {status:202, body:'{}', contentType:'application/json', expected:'invalid response'},
      ]) {
        failure = testCase;
        await evaluate('document.querySelector(".pipeline-generate").click()');
        await waitFor(`document.querySelector('[role=alert]')?.textContent.includes(${JSON.stringify(testCase.expected)})`);
        assert.equal(await evaluate('!!document.querySelector(".processing")'), false);
      }
      failure = null; requests = 0;
    }
    if (mode === 'fault-decode') await evaluate(`HTMLImageElement.prototype.decode=()=>Promise.reject(new Error('Decode failed'));`);
    if (mode === 'fault-unmount') await evaluate(`HTMLImageElement.prototype.decode=function(){return new Promise(resolve=>window.__finishPendingDecode=resolve);};`);
    if (mode === 'desktop') await evaluate(`window.__decodeOriginal=HTMLImageElement.prototype.decode; HTMLImageElement.prototype.decode=async function(){await window.__decodeOriginal.call(this); if(this.classList.contains('result-final')) {window.__decodingImage=this;await new Promise(resolve=>window.__releaseDecode=resolve);}};`);
    await evaluate(`document.querySelector(${JSON.stringify(mode.startsWith('desktop') ? '.pipeline-generate' : '.mobile-generate')}).click()`);
    await waitFor('!!document.querySelector(".processing .generation-particles")');
    if (mode.startsWith('fault-')) {
      if (mode === 'fault-unmount') {
        await waitFor('!!window.__finishPendingDecode');
        await evaluate('document.querySelector(".brand").click(); window.__finishPendingDecode()');
        await waitFor('!!document.querySelector(".editorial-showcase")');
        assert.equal(await evaluate('!!document.querySelector(".photo.result")'), false);
      } else {
        await waitFor('document.querySelector(".generation-overlay")?.dataset.phase === "error"');
        assert.equal(await evaluate('!!document.querySelector(".result-final--visible,.ready")'), false);
        assert.ok(await evaluate('document.querySelector("[role=alert]").textContent.includes("could not be loaded")'));
        assert.equal(await evaluate('getComputedStyle(document.querySelector(".generation-overlay")).opacity'), '1');
      }
      assert.deepEqual(errors, []);
      console.log(`PASS ${mode}: safe failure or navigation during decoding`);
      await send('Target.closeTarget', {targetId}); listeners.delete(sessionId);
      continue;
    }
    if (mode === 'desktop') await evaluate(`window.__progressSamples=[]; window.__progressTimer=setInterval(()=>{const bar=document.querySelector('[role=progressbar]');if(bar)window.__progressSamples.push({percent:Number(bar.getAttribute('aria-valuenow')),decoded:!!document.querySelector('.result-final--visible')});},50);`);
    assert.equal(await evaluate('document.querySelector(".processing .generation-source").alt'), 'Original product being transformed');
    assert.equal(await evaluate('!!document.querySelector(".progress-panel")'),false,'Progress is only shown in the image overlay');
    if (mode.startsWith('desktop')) {
      assert.equal(await evaluate(`(() => {const original=document.querySelector('.reference-panel').getBoundingClientRect();const result=document.querySelector('.result-panel').getBoundingClientRect();const rail=document.querySelector('.input-views').getBoundingClientRect();return Math.abs(original.width-result.width)<1 && Math.abs(original.height-result.height)<1 && rail.right<original.left;})()`),true,'Cards keep their geometry beside the thumbnail rail during generation');
    }
    assert.match(await evaluate('document.querySelector(".live-caption .generation-elapsed").textContent'), /^\d+:\d{2}$/);
    if (mode === 'mobile-reduced-motion') assert.equal(await evaluate('!!document.querySelector(".generation-particles canvas")'), false);
    if (mode === 'desktop') {
      await waitFor('document.querySelector(".generation-particles")?.dataset.ready === "true"');
      assert.equal(await evaluate('getComputedStyle(document.querySelector(".pipeline-connector--active"), "::before").animationName'), 'pipeline-aura');
      const generationShot = await command('Page.captureScreenshot', {format:'png'});
      await writeFile(join(tmpdir(), 'sartiq-generation-desktop.png'), Buffer.from(generationShot.data, 'base64'));
    }
    if (mode === 'desktop') {
      await waitFor('!!document.querySelector(".result-final") && !document.querySelector(".result-final").naturalWidth');
      assert.equal(await evaluate('!!document.querySelector(".generation-particles canvas")'), true, 'Particles remain alive until the final image is loaded');
      assert.ok(await evaluate('Number(document.querySelector("[role=progressbar]").getAttribute("aria-valuenow")) < 100'), 'No 100% before output decode');
      await waitFor('!!window.__releaseDecode');
      assert.equal(await evaluate('document.querySelector(".generation-overlay").dataset.phase'), 'decoding');
      assert.equal(await evaluate('!!document.querySelector(".result-final--visible")'), false, 'Loaded but undecoded photo stays covered');
      await evaluate('window.__revealedImage=document.querySelector(".result-final");window.__sourceImage=document.querySelector(".generation-source");window.__frameRect=document.querySelector(".photo.result").getBoundingClientRect().toJSON();window.__releaseDecode()');
      await waitFor('!!document.querySelector(".photo.is-revealing .result-final--visible")');
      assert.equal(await evaluate('window.__decodingImage === window.__revealedImage'), true, 'Decode belongs to the displayed element');
      assert.equal(await evaluate('document.querySelector(".generation-overlay").getAnimations().length'), 1);

      assert.equal(await evaluate('document.querySelector(".result-final").naturalWidth > 0 && !!document.querySelector(".generation-particles canvas")'), true, 'Decoded photo blends over the existing particle canvas');
      assert.equal(await evaluate('getComputedStyle(document.querySelector(".result-final")).transitionDuration'), '0s');
    }
    await waitFor(`document.querySelector('.ready')?.textContent === ${JSON.stringify(mode === 'desktop-unverified' ? 'Review' : 'Ready')}`);
    assert.equal(await evaluate('!!document.querySelector(".execution-details,.review-note")'), false, 'Result preserves the simplified interface');
    if (mode === 'desktop') {
      assert.equal(await evaluate('window.__revealedImage === document.querySelector(".result-final") && window.__sourceImage === document.querySelector(".generation-source")'), true, 'Both image layers survive completion');
      assert.equal(await evaluate('JSON.stringify(window.__frameRect) === JSON.stringify(document.querySelector(".photo.result").getBoundingClientRect().toJSON())'), true, 'No frame shift at completion');
      assert.equal(await evaluate('getComputedStyle(document.querySelector(".generation-overlay")).opacity'), '0');
      const samples = await evaluate('clearInterval(window.__progressTimer); window.__progressSamples');
      const processingSamples = samples.filter(sample => !sample.decoded).map(sample => sample.percent);
      assert.ok(new Set(processingSamples).size >= 3, 'Progress advances between pipeline updates');
      for (let index=1;index<processingSamples.length;index++) {
        assert.ok(processingSamples[index] >= processingSamples[index-1], 'Progress never goes backwards');
        assert.ok(processingSamples[index]-processingSamples[index-1] < 10, 'Progress avoids large instantaneous jumps');
      }
      assert.ok(samples.every(sample=>sample.decoded || sample.percent < 100));
    }
    assert.equal(requests, 1);
    assert.equal(await evaluate('getComputedStyle(document.querySelector(".has-result")).backgroundColor'), 'rgb(221, 189, 253)');
    if (mode.startsWith('desktop')) {
      const layout = await evaluate(`(() => {const original=document.querySelector('.reference-panel').getBoundingClientRect(); const result=document.querySelector('.result-panel').getBoundingClientRect(); const actions=document.querySelector('.actions').getBoundingClientRect(); return {beside:original.right < result.left, aligned:Math.abs(original.top-result.top)<1, actionsVisible:actions.bottom<=innerHeight};})()`);
      const resultShot = await command('Page.captureScreenshot', {format:'png'});
      await writeFile(join(tmpdir(), `sartiq-result-${mode}.png`), Buffer.from(resultShot.data, 'base64'));
      assert.deepEqual(layout, {beside:true, aligned:true, actionsVisible:true}, 'Desktop reference, result and actions should fit the workspace');
    }
    const resultFrame = await evaluate('(() => {const frame=document.querySelector(".photo.result").getBoundingClientRect(); return {width:frame.width,height:frame.height};})()');
    assert.equal(await evaluate('!!document.querySelector(".pipeline-connector--active")'), false, 'Pipeline halo stops when the result is ready');
    await evaluate('document.querySelector(".actions .secondary").click()');
    await waitFor('!!document.querySelector(".comparison-slider")');
    assert.equal(await evaluate('!!document.querySelector(".comparison-screen__top")'), false);
    assert.equal(await evaluate(`!!document.querySelector('[aria-label="Close comparison"]')`), false);
    assert.equal(await evaluate('document.body.style.overflow'), '');
    await waitFor('document.querySelector(".comparison-slider__result img").complete');
    const dimensions = await evaluate(`(() => {const frame=document.querySelector('.comparison-slider').getBoundingClientRect(); const original=document.querySelector('.comparison-slider > img').getBoundingClientRect(); const result=document.querySelector('.comparison-slider__result img').getBoundingClientRect(); return {ratio:frame.width/frame.height, originalWidth:original.width, resultWidth:result.width, originalHeight:original.height, resultHeight:result.height};})()`);
    const comparedFrame = await evaluate('(() => {const frame=document.querySelector(".photo.result").getBoundingClientRect(); return {width:frame.width,height:frame.height};})()');
    assert.deepEqual(comparedFrame, resultFrame, 'Compare must preserve the exact result frame');
    assert.ok(Math.abs(dimensions.ratio - resultFrame.width/resultFrame.height) < .02, 'Slider fills the result frame');
    assert.equal(dimensions.originalWidth, dimensions.resultWidth);
    assert.equal(dimensions.originalHeight, dimensions.resultHeight);
    if (mode.startsWith('desktop')) assert.equal(await evaluate('document.querySelector(".actions").getBoundingClientRect().bottom <= innerHeight'), true, 'Comparison actions stay visible on desktop');
    const comparisonScreenshot = await command('Page.captureScreenshot', {format:'png'});
    await writeFile(join(tmpdir(), `sartiq-compare-${mode}.png`), Buffer.from(comparisonScreenshot.data, 'base64'));
    await evaluate('document.querySelector(".actions .secondary").click()');
    await waitFor('!document.querySelector(".comparison-slider")');
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
