import os
import sys
import pwd

APP_USER = "app"
APP_GROUP = "app"

DIRS = ["/app/.mlruns", "/app/.registry"]

def main():
    app_uid = pwd.getpwnam(APP_USER).pw_uid
    app_gid = pwd.getpwnam(APP_GROUP).pw_gid

    for d in DIRS:
        os.makedirs(d, exist_ok=True)
        os.chown(d, app_uid, app_gid)
    
    os.setgid(app_gid)
    os.setuid(app_uid)

    # Custom message
    print("\n" + "="*50, flush=True)
    print("  🚀 RuneLog UI is ready!", flush=True)
    print("  You can now view your app in your browser.", flush=True)
    print("  URL: http://localhost:8501", flush=True)
    print("="*50 + "\n", flush=True)

    # Redirect stdout and stderr to /dev/null to suppress app output
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    
    os.execvp(sys.argv[1], sys.argv[1:])

if __name__ == "__main__":
    main()