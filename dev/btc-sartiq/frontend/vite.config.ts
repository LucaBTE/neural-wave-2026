import {networkInterfaces} from 'node:os';
import {defineConfig, type Plugin} from 'vite';
import react from '@vitejs/plugin-react';

function mobileInstructions(): Plugin {
  return {
    name: 'sartiq-mobile-instructions',
    configureServer(server) {
      const printUrls = server.printUrls.bind(server);
      server.printUrls = () => {
        printUrls();
        const networkUrls = server.resolvedUrls?.network || [];
        const interfaces = Object.entries(networkInterfaces())
          .filter(([name]) => !/^(docker|veth|br-|virbr|tun|tap|tailscale|wg)/i.test(name))
          .sort(([a], [b]) => Number(!/^wl|^wi/i.test(a)) - Number(!/^wl|^wi/i.test(b)));
        const addresses = interfaces.flatMap(([, entries]) => (entries || [])
          .filter(entry => !entry.internal && entry.family === 'IPv4')
          .map(entry => entry.address));
        const mobileUrls = [...new Set(addresses.flatMap(address =>
          networkUrls.filter(url => new URL(url).hostname === address)))];
        console.log('\n  Sartiq · Open on your phone\n');
        if (mobileUrls.length) {
          for (const url of mobileUrls) console.log(`  Mobile:  ${url}`);
          console.log('\n  Connect your phone to the same Wi-Fi and open a Mobile URL.');
          console.log('  Keep the backend running on this computer (port 8000).');
          console.log('  On HTTP, tap “Take photo” to use the device camera, or “Photos”.');
          console.log('  The live camera preview requires an HTTPS URL.\n');
        } else {
          console.log('  No LAN address is available. Connect this computer to Wi-Fi');
          console.log('  and run: npm run dev -- --host 0.0.0.0\n');
        }
      };
    },
  };
}

export default defineConfig({
  plugins: [react(), mobileInstructions()],
  server: {
    host: '0.0.0.0',
    proxy: {'/api': {
      target: 'http://127.0.0.1:8000',
      configure(proxy) {
        proxy.on('error', (_error, _request, response) => {
          if ('writeHead' in response && !response.headersSent) {
            response.writeHead(502, {'Content-Type': 'application/json'});
            response.end(JSON.stringify({detail: 'Generation service unavailable. Start the backend on port 8000 and try again.'}));
          }
        });
      },
    }},
    allowedHosts: ['.trycloudflare.com'],
  },
});
