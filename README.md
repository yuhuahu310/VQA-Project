# VQA Project

We're here to tackle the VizWiz-VQA task. For more information, visit https://vizwiz.org/tasks-and-datasets/vqa/.

## Development Guidelines

### Dataset Placement
Download the dataset from the VizWiz official webpage linked above.

Unzip and place the Annotations as `data/Annotations/{test,train,val}.json`. 

Unzip and place the images as `data/Images/{train,val,test}/VizWiz_{train,val,test}_XXX.jpg`.

### Relative Imports

All Python scripts should assume they will be run from the directory they are in (as opposed to the project root). For relative imports to work, referenced modules need to be added to the systems path.

For example, if you want to import the `VQADataset` class in `dataloader/dataset.py` in `unimodal_baseline/language/BERT.py`, you need to do

```
import sys
sys.path.insert(0, '../../dataloader')
from dataset import VQADataset
```

Note that this only inserts into the systems path at runtime and will not change your `PYTHONPATH` environment variable. For certian IDEs to correctly resolve imports, you may need to manually add paths to workspace settings. For VSCode Pylance to work, note that additional configuration has been added in `.vscode/settings.json`.

### Code Structure

Always make your code as modularized and extensible as possible to encourage reuse. Refrain from hard coding things. Use json and/or yaml configuration files for parameters/hyperparameters when possible.

NEVER write code on top level of a file. Always wrap them in functions or `if __name__ == '__main__':`. Otherwise, pain will escalate quickly.

### Naming

PEP 8 naming conventions should be followed. In particular, 
- Directory, file, variable, and function names should follow the `lower_case_with_underscores` format. 
- Class names should follow the `CamelCase` format. 
- Constants should follow the `UPPER_CASE_WITH_UNDERSCORES` format. 
- Weak internal indicator formatted as `_single_leading_underscore` should be used for class methods that are internal use only.

### Version Control

- Code and useful figures should be checked into git. Note that `.gitignore` is properly set up.
- Take caution when checking in huge/binary objects. 
- Meaningful, informative commit messages should be included.
- Always pull from remote before starting to work to avoid conflicts.
- Prefer `rebase` to `merge` when local and remote branches differ to avoid unecessary git log messages. Concretely, do `git pull --rebase`.
- Start a new feature branch, develop there, and submit a Pull Request once done if you anticipate your changes will potentially affect others' work.

### Packages

PyPi packages required should be added to `requirements.txt`. They can be installed using `pip install -r requirements.txt`.

## Authors
~~Video~~ Visual QA Ninjas @ CMU 11-777 MMML