from setuptools import find_packages, setup

setup(
    name ='lokacode',
    version ='0.0.1',
    author ='andre fonseca',
    install_requires = ['aiofiles','asyncio','nest_asyncio','transformers','torch','bs4','tqdm','numpy', 'python-dotenv'],
    packages = find_packages()
)