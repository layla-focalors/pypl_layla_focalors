from setuptools import setup, find_packages
setup(name='layla_focalors',
version='1.0.0',
description='layla-focalors, AI Preset python',
author='layla-focalors',
author_email='layla-focalors@arisia.space',
url='https://github.com/layla-focalors',
license='MIT', 
py_modules=['cyclegan'],
python_requires='>=3',
install_requires=['numpy', 'pandas', 'scipy', 'scikit-learn', 'pymysql', 'cryptography','selenium','uvicorn[standard]','torch','torchvision','dominate','visdom','wandb'], 
packages=['layla_focalors'] 
)