# Render Deployment Guide

## Status 128 Error Fix

Status 128 typically means a Git or build error. Here's how to fix it:

### Option 1: Using Docker (Current Setup)

1. **Make sure all files are committed:**
   ```bash
   git add .
   git commit -m "Fix deployment"
   git push
   ```

2. **Check Render logs** for specific error messages

3. **Common fixes:**
   - Ensure `requirements.txt` has all dependencies (✅ Fixed)
   - Ensure Dockerfile is correct (✅ Fixed)
   - Check if build is timing out (TensorFlow is large)

### Option 2: Switch to Native Python (Faster Build)

If Docker keeps failing, you can switch to native Python:

1. In Render dashboard, change:
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -b 0.0.0.0:$PORT app:app`

2. Update `render.yaml`:
   ```yaml
   services:
     - type: web
       name: attendance-system
       env: python
       plan: free
       buildCommand: pip install -r requirements.txt
       startCommand: gunicorn -b 0.0.0.0:$PORT app:app
   ```

### Environment Variables Needed

Make sure these are set in Render dashboard:
- `DATABASE_URL` - Your Neon PostgreSQL URL
- `BREVO_API_KEY` - Your Brevo API key
- `ADMIN_EMAIL` - Email for sending reports
- `CRON_SECRET` - Secret for manual cron triggers (optional)
- `PORT` - Automatically set by Render

### Troubleshooting

1. **Build timeout**: TensorFlow is large. Render free tier might timeout.
   - Solution: Upgrade to paid tier or optimize Dockerfile

2. **Memory issues**: Free tier has 512MB RAM limit
   - Solution: TensorFlow needs ~1GB, consider upgrading

3. **Git errors**: Status 128
   - Check: `.gitignore` doesn't exclude needed files
   - Check: All files are committed
   - Check: Repository is properly connected

### Quick Test

Test locally first:
```bash
docker build -t attendance-system .
docker run -p 10000:10000 -e PORT=10000 attendance-system
```

Then visit: http://localhost:10000
