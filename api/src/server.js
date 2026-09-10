// Starts the API. Run it with `npm run dev` from the api/ folder, or press Run
// on api/run.js.
//
// What this program is: a translator. A browser cannot talk to Postgres - no
// browser can, and that is deliberate, because it would mean shipping database
// credentials to every visitor. So this sits in between. It listens for web
// requests, runs the matching SQL, and sends rows back as JSON.

import cors from "cors";
import express from "express";
import { router as buildingsRouter } from "./routes/buildings.js";
import { router as runsRouter } from "./routes/runs.js";
import { pool } from "./db.js";

// `app` is the whole API. Everything below either adds a step every request
// passes through, or attaches a set of URLs it knows how to answer.
const app = express();
const PORT = process.env.API_PORT ?? 3001;

// CORS is a browser rule: a page may only call the server it came from, unless
// that server says otherwise. In development the web app runs on port 5173 and
// this API on 3001, which counts as different servers, so without this the
// browser would block every request.
app.use(cors());

// Log each request with how long it took. Useful when a page feels slow: it
// tells you immediately whether the API or the browser is the reason.
app.use((req, res, next) => {
  const started = Date.now();
  res.on("finish", () => {
    console.log(`${res.statusCode} ${req.method} ${req.originalUrl} ${Date.now() - started}ms`);
  });
  next();
});

// Every route lives under /api, so the web app's dev server can forward
// anything starting with /api here and leave everything else alone.
app.use("/api", runsRouter);
app.use("/api", buildingsRouter);

// Unknown path. Without this, a typo in a URL returns Express's default HTML
// error page, and a frontend expecting JSON fails confusingly.
app.use((req, res) => {
  res.status(404).json({ error: "not found" });
});

// One error handler for the whole API. Routes call next(err) and end up here,
// so no route has to repeat the same try/catch response logic.
//
// Express identifies this as the error handler by its four arguments, so `next`
// must stay in the signature even though it is unused.
// eslint-disable-next-line no-unused-vars
app.use((err, req, res, next) => {
  const status = err.status ?? 500;
  if (status >= 500) {
    console.error(err);
  }
  res.status(status).json({
    error: status >= 500 ? "database unavailable" : err.message,
  });
});

const server = app.listen(PORT, () => {
  console.log(`API listening on http://localhost:${PORT}`);
  console.log(`Try it: http://localhost:${PORT}/api/health`);
});

// Close the database pool cleanly on Ctrl+C, so Postgres is not left holding
// connections open from a killed process.
for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, () => {
    console.log("\nshutting down");
    server.close(() => pool.end().then(() => process.exit(0)));
  });
}