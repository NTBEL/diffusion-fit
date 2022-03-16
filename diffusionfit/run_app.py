import os
import sys
from streamlit import cli as stcli

APP_DIR = os.path.dirname(__file__)
APP_FILE = os.path.join(APP_DIR, "streamlit_app.py")

# Launching the streamlit app with sys and call to stcli.main was adapted from
# https://stackoverflow.com/a/62775219
if __name__ == '__main__':
    sys.argv = ["streamlit","run",APP_FILE, '--server.maxUploadSize', '500']
    sys.exit(stcli.main())
