const { Pool } = require('pg');
require('dotenv').config();

const pool = new Pool({
  user: process.env.POSTGRES_USER,
  host: process.env.POSTGRES_HOST,
  database: process.env.POSTGRES_DB,
  password: process.env.POSTGRES_PASSWORD,
  port: process.env.POSTGRES_PORT,
});

// Only log in non-test environments
if (process.env.NODE_ENV !== 'test') {
  pool.query('SELECT NOW()')
    .then(() => console.log('✅ Connected to Postgres at startup'))
    .catch(err => console.error('❌ Postgres connection error:', err));
}

pool.on('error', err => console.error('Postgres runtime error', err));


module.exports = { pool };
