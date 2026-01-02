// src/index.js
require('dotenv').config({ path: __dirname + '/../../.env' }); // ensure env is loaded first
const app = require('./app');

const PORT = process.env.PORT || 8000;

app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});
