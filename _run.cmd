IF NOT EXIST "venv\" (
    echo create Virtual Environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
pip install -r requirements.txt
cmd /k
