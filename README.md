# simple_image_search_via_embeddinggemma-2
Just a simple proof of concept image indexing/search with embeddinggemma-2.

Install with:  pip install -r requirements.txt
           or:  uv pip install -r requirements.txt
On Windows, plain PyPI "torch" is the CPU build, which is what you have.
For an NVIDIA GPU, install torch + torchvision from the CUDA index first, e.g.:
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
and then run pip install -r requirements.txt (the torch lines below are already satisfied).

Index folder via: python imgsearch.py index /some_path/
Start webui: python webui.py /some_path/ --port 8080 
Afterwards go in your browser http://localhost:8080 to use it.
