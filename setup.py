from setuptools import setup,find_packages

setup(
    name               = 'Chattiori-Model-Merger'
    , version          = '3.2'
    , license          = 'Apache License'
    , author           = "Team-C"
    , author_email     = 'havocai69@gmail.com'
    , packages         = find_packages('src')
    , package_dir      = {'': 'src'}
    , url              = 'https://github.com/Faildes/Chattiori-Model-Merger'
    , keywords         = 'stable-diffusion merge-model'
    , install_requires = [
        'torch'
        , 'safetensors'
        , 'diffusers'
        , 'lora'
    ]
    , include_package_data=True
)
