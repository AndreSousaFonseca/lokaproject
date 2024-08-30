from setuptools import find_packages.setup

setup(
    name ='locacode',
    version ='0.0.1',
    author ='andre fonseca',
    install_requires = ['aiofiles','asyncio','nest_asyncio','transformers','torch','bs4','re','tqdm','logging','faiss','numpy'],
    packages = find_packages()
)