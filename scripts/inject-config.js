const fs = require('fs');
const path = require('path');

const apiBase = process.env.API_BASE_URL || '';
const templatePath = path.join(__dirname, '..', 'frontend', 'config.template.js');
const outputPath = path.join(__dirname, '..', 'frontend', 'config.js');

const template = fs.readFileSync(templatePath, 'utf8');
fs.writeFileSync(outputPath, template.replace('%%API_BASE_URL%%', apiBase));

console.log(`frontend/config.js written with API_BASE_URL="${apiBase || '(empty)'}"`);