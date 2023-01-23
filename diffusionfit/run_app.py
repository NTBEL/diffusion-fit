import os
import sys
try:
    # Older versions of streamlit, such as 1.7.0.
    from streamlit import cli as stcli
except:
    # streamlit 1.17.0 compatibility. 
    from streamlit.web import cli as stcli    

APP_DIR = os.path.dirname(__file__)
APP_FILE = os.path.join(APP_DIR, "streamlit_app.py")

# Launching the streamlit app with sys and call to stcli.main was adapted from
# https://stackoverflow.com/a/62775219
if __name__ == '__main__':
    sys.argv = ["streamlit","run",APP_FILE, '--server.maxUploadSize', '500']
    sys.exit(stcli.main())
