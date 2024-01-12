## Movie Recommendation System based on Latent Factor Model (LFM) Algorithm

### Introduction
This study presents a personalized movie recommendation system leveraging a Latent Factor Model (LFM) to mitigate the decision fatigue faced by consumers amidst a plethora of digital content. Utilizing a comprehensive MovieLens dataset and a robust implementation process, we constructed a model capable of providing tailored recommendations.
For details, please refer to the [notebook]().

 ### Getting Started

 #### Prerequisites
 - Ensure you have Python installed on your system. You can download Python from python.org.(Python Version >=3.9)
 - [Optional] Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
#### Installing the Project
The project can be installed directly using pip which will also handle the installation of all required dependencies. 
```bash
pip install .
```
This command tells **pip** to install the current package (denoted by .) along with its dependencies. The dependencies and any other installation instructions are defined in the **pyproject.toml** file.


#### Running 
```bash
make data  # processing the raw data 
make train # training  and validation 
make test  # test and evaluate the model 
```

### License
This project uses the following license: MIT.










