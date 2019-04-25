from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_align_cuda',
    ext_modules=[
        CUDAExtension('roi_align_cuda', [
            'detection/roi_align_cuda.cpp',
            'detection/roi_align_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
