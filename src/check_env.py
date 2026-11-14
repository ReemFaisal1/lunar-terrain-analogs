import numpy, scipy, rasterio, torch, sklearn, earthaccess
print("numpy", numpy.__version__, "| rasterio", rasterio.__version__)
print("torch", torch.__version__, "| Apple GPU (MPS):", torch.backends.mps.is_available())