import {useEffect, useRef} from 'react';

/** A temporary image-derived point cloud. It never modifies the generated photo. */
export function GenerationParticles({source}: {source: string}) {
  const host = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const container = host.current;
    if (!container || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    let disposed = false;
    let release: (() => void) | undefined;
    const image = new Image();
    image.onload = async () => {
      try {
        const THREE = await import('three');
        if (disposed) return;
        const renderer = new THREE.WebGLRenderer({alpha: true, antialias: false, powerPreference: 'low-power'});
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
        renderer.setClearColor(0x000000, 0);
        release = () => {renderer.dispose(); renderer.domElement.remove();};
        const scene = new THREE.Scene();
        const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, .1, 10);
        camera.position.z = 3;
        const aspect = image.naturalWidth / image.naturalHeight;
        const rows = Math.max(1, Math.min(240, Math.floor(Math.sqrt(50_000 / aspect))));
        const columns = Math.max(1, Math.min(1000, Math.round(rows * aspect)));
        const sample = document.createElement('canvas');
        sample.width = columns; sample.height = rows;
        const context = sample.getContext('2d');
        if (!context) {renderer.dispose(); return;}
        context.drawImage(image, 0, 0, columns, rows);
        const pixels = context.getImageData(0, 0, columns, rows).data;
        const positions = new Float32Array(columns * rows * 3);
        const colors = new Float32Array(positions.length);
        const seeds = new Float32Array(columns * rows);
        const color = new THREE.Color();
        for (let y = 0; y < rows; y++) for (let x = 0; x < columns; x++) {
          const index = y * columns + x;
          const offset = index * 3;
          const pixel = index * 4;
          const hash = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
          const seed = hash - Math.floor(hash);
          positions[offset] = ((x + .5 + (seed - .5) * .7) / columns * 2 - 1) * aspect;
          positions[offset + 1] = 1 - (y + .5 + (seed - .5) * .7) / rows * 2;
          positions[offset + 2] = (pixels[pixel] + pixels[pixel + 1] + pixels[pixel + 2]) / 765 * .15 + seed * .035;
          color.setRGB(pixels[pixel] / 255, pixels[pixel + 1] / 255, pixels[pixel + 2] / 255, THREE.SRGBColorSpace);
          colors[offset] = color.r; colors[offset + 1] = color.g; colors[offset + 2] = color.b;
          seeds[index] = seed;
        }
        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('seed', new THREE.BufferAttribute(seeds, 1));
        const material = new THREE.ShaderMaterial({
          transparent: true, depthWrite: false, vertexColors: true,
          uniforms: {time: {value: 0}, pointSize: {value: 3}},
          vertexShader: `attribute float seed; uniform float time; uniform float pointSize; varying vec3 particleColor;
            void main() {
              particleColor = color;
              vec3 p = position;
              float speed = 1.2 + seed * 1.4;
              float wave = sin(p.y * 6.0 + time * speed + seed * 14.0);
              p.x += wave * .018 + sin(time * .9 + seed * 31.0) * .006;
              p.y += cos(p.x * 8.0 + time * speed * .8 + seed * 21.0) * .016;
              p.z += wave * .05;
              gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
              gl_PointSize = pointSize * (.8 + seed * .2 + p.z * 1.5) * (1.0 + sin(time * 1.6 + seed * 25.0) * .18);
            }`,
          fragmentShader: `varying vec3 particleColor;
            void main() {
              float radius = length(gl_PointCoord - .5);
              float alpha = (1.0 - smoothstep(.22, .5, radius)) * .88;
              gl_FragColor = vec4(particleColor, alpha);
              #include <tonemapping_fragment>
              #include <colorspace_fragment>
            }`,
        });
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        const resize = () => {
          const {width, height} = container.getBoundingClientRect();
          if (!width || !height) return;
          const frameAspect = width / height;
          const cover = window.matchMedia('(min-width:960px)').matches
            ? Math.min(1, frameAspect / aspect) : Math.max(1, frameAspect / aspect);
          camera.left = -frameAspect / cover; camera.right = frameAspect / cover;
          camera.top = 1 / cover; camera.bottom = -1 / cover; camera.updateProjectionMatrix();
          renderer.setSize(width, height);
          material.uniforms.pointSize.value = height / rows * cover * renderer.getPixelRatio() * 1.65;
        };
        const observer = new ResizeObserver(resize);
        let frame = 0;
        let previous = 0;
        const start = performance.now();
        // Interpolate random targets to avoid jumps in position, scale and rotation.
        const drift = (time: number, channel: number) => {
          const key = Math.floor(time);
          const fraction = time - key;
          const blend = fraction * fraction * (3 - 2 * fraction);
          const random = (index: number) => {
            const value = Math.sin(index * 127.1 + channel * 311.7) * 43758.5453;
            return (value - Math.floor(value)) * 2 - 1;
          };
          return random(key) * (1 - blend) + random(key + 1) * blend;
        };
        const render = (now: number) => {
          frame = requestAnimationFrame(render);
          if (document.hidden || now - previous < 33) return;
          previous = now;
          const time = (now - start) / 1000;
          material.uniforms.time.value = time * 1.8;
          const motionTime = time / 1.4;
          points.rotation.y = drift(motionTime, 1) * .04;
          points.rotation.x = drift(motionTime, 2) * .025;
          points.rotation.z = drift(motionTime, 3) * .012;
          points.position.x = drift(motionTime, 4) * .016;
          points.position.y = drift(motionTime, 5) * .014;
          points.scale.setScalar(1.035 + drift(motionTime, 6) * .022);
          renderer.render(scene, camera);
        };
        release = () => {
          cancelAnimationFrame(frame); observer.disconnect();
          geometry.dispose(); material.dispose(); renderer.dispose();
          renderer.domElement.remove(); container.removeAttribute('data-ready');
        };
        renderer.domElement.addEventListener('webglcontextlost', event => {
          event.preventDefault(); cancelAnimationFrame(frame); container.removeAttribute('data-ready');
        });
        container.appendChild(renderer.domElement); observer.observe(container); resize();
        renderer.render(scene, camera); container.dataset.ready = 'true';
        frame = requestAnimationFrame(render);
      } catch {
        release?.(); release = undefined;
        // The original image and CSS texture remain available without WebGL.
      }
    };
    image.src = source;
    return () => {disposed = true; image.onload = null; image.src = ''; release?.();};
  }, [source]);
  return <div ref={host} className="generation-particles" aria-hidden="true"/>;
}
