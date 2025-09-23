Deploying ATR Backend (Render) + Frontend (GitHub Pages)

Backend on Render (free tier)
1) Fork/Push this repo (done). Ensure it contains Dockerfile and render.yaml.
2) Go to Render dashboard → New → Web Service.
3) Connect your GitHub and select this repository.
4) If prompted, choose “Use Docker” (Render auto-detects Dockerfile). Alternatively, click “Use Render YAML” and point to render.yaml.
5) Set environment variables:
   - PORT=8000
   - DISABLE_TTS=1
   - WHISPER_MODEL=base
   - ALLOWED_ORIGINS=https://<your-frontend-domain>
6) Click Create Web Service. After the build, open the URL and check /docs.

Frontend on GitHub Pages (free)
1) In this repo, host the static files from ATR Model: index.html, script.js, styles.css.
   - Option A (root): Move/copy them to a /docs folder or root. Or create a separate repo for the frontend.
   - Option B (Render Static Site): Use a Render Static Site pointing to the repository path containing index.html.
2) If you use GitHub Pages:
   - Settings → Pages → Build and deployment → Source: Deploy from a branch.
   - Select main and the folder (root or /docs).
   - Save. Pages URL will be shown (e.g., https://<user>.github.io/<repo>/).
3) In script.js, set API_BASE to your Render backend URL. Or set ALLOWED_ORIGINS on the backend to include your Pages URL and keep API_BASE pointing there.

Notes
- Free tier constraints: keep DISABLE_TTS=1 and WHISPER_MODEL=base for lower memory.
- CORS: Ensure ALLOWED_ORIGINS includes your Pages/Static site URL(s).
- Health check: /docs can serve as a simple health endpoint on Render.

.env (optional for local/dev)
- Create a file named .env in ATR Model/ (same folder as backend/):

  PORT=8000
  DISABLE_TTS=1
  WHISPER_MODEL=base
  ALLOWED_ORIGINS=http://127.0.0.1:5500,http://localhost:5500
  SUPABASE_URL=your_supabase_project_url
  SUPABASE_KEY=your_supabase_anon_public_key

- The backend auto-loads ATR Model/.env at startup if present. On Render, set these as Environment Variables in the dashboard instead of using a file.



