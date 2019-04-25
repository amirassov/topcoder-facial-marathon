from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_pool',
    ext_modules=[
        CUDAExtension('roi_pool_cuda', [
            'detection/roi_pool_cuda.cpp',
            'detection/roi_pool_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension})
