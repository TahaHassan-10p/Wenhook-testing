require("dotenv").config();
const path = require("path");
const express = require("express");
const cors = require("cors");

console.log("🚀 Starting backend server from:", __filename);
console.log("OPENAI_API_KEY loaded:", process.env.OPENAI_API_KEY);

const app = express(); // <-- define app first

// Serve uploads folder as static
app.use("/uploads", express.static(path.join(__dirname, "../uploads")));

// Update the CORS configuration
app.use(
  cors({
    origin: ["http://localhost:3000", "http://127.0.0.1:3000"],
    credentials: true,
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allowedHeaders: ["Content-Type", "Authorization", "x-api-key"],
  })
);

// Body parsers
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Debug logger
app.use((req, _res, next) => {
  console.log(`[${req.method}] ${req.originalUrl}`);
  next();
});

// Route imports
const { router: authRouter } = require("../routes/auth");
const requisitionsRouter = require("../routes/requisitions");
const candidatesRouter = require("../routes/candidates");

// Route registrations
app.use("/auth", authRouter);
app.use("/requisitions", requisitionsRouter);
app.use("/candidates", candidatesRouter);

// Health check
app.get("/", (_req, res) => res.send("Backend is running"));

// Log registered routes - FIXED: Check if router exists first
console.log("Registered routes:");
if (app._router && app._router.stack) {
  app._router.stack.forEach((middleware) => {
    if (middleware.route) {
      console.log(
        `  ${Object.keys(middleware.route.methods)[0].toUpperCase()} ${
          middleware.route.path
        }`
      );
    } else if (
      middleware.name === "router" &&
      middleware.handle &&
      middleware.handle.stack
    ) {
      const basePath = middleware.regexp.source
        .replace(/\\\//g, "/")
        .split("(?")[0];
      middleware.handle.stack.forEach((handler) => {
        if (handler.route) {
          const path = basePath + handler.route.path;
          console.log(
            `  ${Object.keys(handler.route.methods)[0].toUpperCase()} ${path}`
          );
        }
      });
    }
  });
} else {
  console.log("  Router not initialized yet");
}

module.exports = app;
