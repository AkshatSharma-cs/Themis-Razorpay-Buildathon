// frontend/scripts/inject-config.js
const fs = require('fs');
const path = require('path');

const template = path.join(__dirname, '..', 'config.template.js');   // frontend/config.template.js
const output = path.join(__dirname, '..', 'config.js');              // frontend/config.js
const apiBase = process.env.THEMIS_API_BASE || 'https://themis-razorpay-buildathon.onrender.com';

const contents = fs.readFileSync(template, 'utf8')
  .replace(/window\.THEMIS_API_BASE\s*=.*/, `window.THEMIS_API_BASE = "${apiBase}";`);

fs.writeFileSync(output, contents);
console.log(`[inject-config] wrote frontend/config.js -> ${apiBase}`);