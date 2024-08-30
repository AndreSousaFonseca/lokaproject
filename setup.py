from setuptools import find_packages, setup

setup(
    name ='lokacode',
    version ='0.0.1',
    author ='andre fonseca',
    install_requires = ['aiofiles','asyncio','nest_asyncio','sentence_transformers',
                        'numpy','transformers','torch','markdown','bs4','nltk','rouge_score ',
                        'sklearn','python-dotenv'],
    packages = find_packages()
)

