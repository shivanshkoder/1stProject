echo [$(date)]: "STARTING INIT SETUP"
echo [$(date)]: "Creating env file"
conda create --prefix ./env python=3.8 -y
echo [$(date)]: "Activating env"
source activate ./env
echo [$(date)]: "Installing requirements"
pip install -r requirements.txt
echo [$(date)]: "DONE"

