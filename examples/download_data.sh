# Find the directory where this script is stored
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing zenodo_get"
pip3 install zenodo_get

echo "Downloading files"
zenodo_get 20055890 -o $SCRIPT_DIR

echo "Unzipping"
unzip $SCRIPT_DIR/dataset.zip -d $SCRIPT_DIR
